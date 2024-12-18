from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutput, CausalLMOutputWithPast, BaseModelOutputWithPast
import strategies
import asyncio
from collections import deque
import multiprocessing as mp


# Base class for outputs of decoder-only generation models.
@dataclass
class DecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    acceptance_count: int = None
    draft_token_count: int = None
    invocation_count: int = None


# Base class for draft outputs of decoder-only generation models using speculative decoding.
@dataclass
class DecoderOnlyDraftOutput(ModelOutput):
    sequences: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cand_probs: Optional[Tuple[torch.FloatTensor]] = None


# Base class for verification outputs of decoder-only generation models using speculative decoding.
@dataclass
class DecoderOnlyVerificationOutput(ModelOutput):
    sequences: torch.LongTensor = None
    target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    acceptance_count: Optional[int] = None


# Serial Base Engine of decoder-only generation models.
class BaseEngine:
    def __init__(
        self,
        model,
        eos_token_id: int,
        max_new_tokens: int = 128,
        temp: float = 1,
    ) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.temp = temp

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        past_key_values = None
        invocation_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
            else:
                pruned_input_ids = input_ids

            outputs: CausalLMOutputWithPast = self.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            batch_num, seq_len, _ = logits.size()

            if self.temp == 0:
                _, ground_tokens = logits.topk(k=1, dim=-1)  # batch x seq_len x 1
            else:
                ground_probs = torch.softmax(
                    logits / self.temp, dim=-1
                )  # batch x seq_len x hidden_dim

                ground_tokens = torch.multinomial(
                    ground_probs.view(batch_num * seq_len, -1), num_samples=1
                )  # batch x seq_len x 1
            ground_tokens = ground_tokens.view(batch_num, seq_len)

            input_ids = torch.cat(
                (input_ids, ground_tokens[:, -1:].to(input_ids)), dim=1
            )

            invocation_count += 1

            if (
                self.eos_token_id in input_ids[0, -1:]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(sequences=input_ids, invocation_count=invocation_count)


# Serial Speculative Engine of decoder-only generation models using speculative decoding.
class SpeculativeEngine:
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        tree_attn: bool = True,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.strategy: strategies.Strategy = None

        if tree_attn:
            self.strategy = strategies.TreeStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
            )
        else:
            self.strategy = strategies.BatchStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
            )

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        target_model_past_key_values = None
        draft_model_past_key_values = None

        invocation_count = 0
        acceptance_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            draft_output = self.strategy.generate_draft(
                input_ids,
                past_key_values=draft_model_past_key_values,
            )

            draft_model_past_key_values = draft_output.past_key_values

            verification_output = self.strategy.verify(
                input_ids=draft_output.sequences,
                target_model_past_key_values=target_model_past_key_values,
                draft_model_past_key_values=draft_output.past_key_values,
                cand_probs=draft_output.cand_probs,
            )

            input_ids = verification_output.sequences

            draft_model_past_key_values = (
                verification_output.draft_model_past_key_values
            )
            target_model_past_key_values = (
                verification_output.target_model_past_key_values
            )

            invocation_count += 1
            acceptance_count += verification_output.acceptance_count

            if (
                self.eos_token_id in input_ids[0, -self.strategy.max_draft_len :]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(
            sequences=input_ids,
            acceptance_count=acceptance_count,
            draft_token_count=invocation_count * self.strategy.max_draft_len,
            invocation_count=invocation_count,
        )


# Scheduled Speculative Engine of decoder-only generation models using speculative decoding(Parallel Draft and Serial Verify).
class ScheduledSpeculativeEngine:
    def __init__(
        self,
        draft_models: List,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        max_new_tokens: int = 128,
        draft_model_temp: float = 0,
        target_model_temp: float = 0,
        replacement: bool = False,
        speculative_sampling: bool = True,
        tree_attn: bool = True,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.draft_models = draft_models
        self.target_model = target_model
        
        self.draft_mapping = {} # record draft_model info
        self.generate_queue = deque() # control parallel draft
        self.verify_queue = deque() # control FCFS(scheduled verify)

        if tree_attn:
            for idx, draft_model in enumerate(draft_models):
                draft_info = {
                    'draft_model': draft_model,
                    'draft_model_past_key_values': None,
                    'target_model_past_key_values': None,
                    'strategy': strategies.TreeStrategy(
                        draft_model=draft_model,
                        target_model=target_model,
                        k_config=k_config,
                        draft_model_temp=draft_model_temp,
                        target_model_temp=target_model_temp,
                        replacement=replacement,
                        speculative_sampling=speculative_sampling,
                    ),
                    'input_ids': None,
                    'invocation_count': 0,
                    'acceptance_count': 0,
                    'enabled': True,
                    'stop': False
                }
                self.draft_mapping[idx] = draft_info
                self.generate_queue.append(idx)
        else:
            for idx, draft_model in enumerate(draft_models):
                draft_info = {
                    'draft_model': draft_model,
                    'draft_model_past_key_values': None,
                    'target_model_past_key_values': None,
                    'strategy': strategies.BatchStrategy(
                        draft_model=draft_model,
                        target_model=target_model,
                        k_config=k_config,
                        draft_model_temp=draft_model_temp,
                        target_model_temp=target_model_temp,
                        replacement=replacement,
                        speculative_sampling=speculative_sampling,
                    ),
                    'input_ids': None,
                    'invocation_count': 0,
                    'acceptance_count': 0,
                    'enabled': True,
                    'stop': False
                }
                self.draft_mapping[idx] = draft_info
                self.generate_queue.append(idx)
        
        self.complete = 0 # record complete number
        self.process_pool = mp.Pool(processes=len(self.draft_mapping)) # create muti-processes

    def clear(self):
        # self.generate_queue.clear()
        # self.verify_queue.clear()
        for draft_info in self.draft_mapping.values():
            draft_info['draft_model_past_key_values'] = None
            draft_info['target_model_past_key_values'] = None
            draft_info['input_ids'] = None
            draft_info['invocation_count'] = 0
            draft_info['acceptance_count'] = 0
            draft_info['enabled'] = True
            draft_info['stop'] = False
        for idx in self.draft_mapping.keys():
            self.generate_queue.append(idx)
        self.complete = 0
    
    def start_processes(self):
        for idx in range(len(self.draft_mapping)): 
            self.process_pool.apply_async(self.process_draft, args=(idx,))
        asyncio.create_task(self.verify_listener())
        asyncio.create_task(self.generate_listener())

    def process_draft(self, draft_id):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.process_draft_loop(draft_id))

    async def process_draft_loop(self, draft_id):
        while True:
            # print(self.complete)
            if self.complete == len(self.draft_mapping):
                break
            draft_info = self.draft_mapping[draft_id]
            if draft_info['enabled'] and draft_info['stop']==False:
                await self.generate_draft_output(draft_id, draft_info)

    async def generate_draft_output(self, draft_id, draft_info):
        # print("draft:"+str(draft_id))
        strategy = draft_info['strategy']
        draft_output = strategy.generate_draft(draft_info['input_ids'], past_key_values=draft_info['draft_model_past_key_values'])
        draft_info['draft_model_past_key_values'] = draft_output.past_key_values
        draft_info['stop'] = True
        self.verify_queue.append((draft_id, draft_output))

    async def verify_draft_output(self, draft_id, draft_output):
        # print("verify:"+str(draft_id))
        draft_info = self.draft_mapping[draft_id]
        strategy = draft_info['strategy']
        verification_output = strategy.verify(
            input_ids=draft_output.sequences,
            target_model_past_key_values=draft_info['target_model_past_key_values'],
            draft_model_past_key_values=draft_output.past_key_values,
            cand_probs=draft_output.cand_probs,
        )
        draft_info['input_ids'] = verification_output.sequences
        draft_info['draft_model_past_key_values'] = verification_output.draft_model_past_key_values
        draft_info['target_model_past_key_values'] = verification_output.target_model_past_key_values
        draft_info['invocation_count'] += 1
        draft_info['acceptance_count'] += verification_output.acceptance_count
        if self.finish(draft_info):
            draft_info['enabled'] = False
            # print("finish:"+str(draft_id))
            self.complete += 1
        draft_info['stop'] = False
        self.generate_queue.append(draft_id)

    def finish(self, draft_info):
        if (
            self.eos_token_id in draft_info['input_ids'][0, -draft_info['strategy'].max_draft_len:]
            or draft_info['input_ids'].size(-1) - draft_info['init_input_len'] >= self.max_new_tokens
        ):
            return True
        else:
            return False
    
    async def verify_listener(self):
        while True:
            if self.complete == len(self.draft_mapping):
                break
            if self.verify_queue:
                draft_id, draft_output = self.verify_queue.popleft()
                draft_info = self.draft_mapping[draft_id]
                if draft_info['enabled']:
                    await self.verify_draft_output(draft_id, draft_output)
                self.generate_queue.append(draft_id)
            await asyncio.sleep(0)

    async def generate_listener(self):
        while True:
            if self.complete == len(self.draft_mapping):
                break
            if self.generate_queue:
                draft_id = self.generate_queue.popleft()
                draft_info = self.draft_mapping[draft_id]
                if draft_info['stop'] == True:
                    self.generate_queue.append(draft_id)
                    continue
                if draft_info['enabled']:
                    await self.generate_draft_output(draft_id, draft_info)
            await asyncio.sleep(0)

    def reset(self, input_ids_list):
        for idx, input_ids in enumerate(input_ids_list):
            self.draft_mapping[idx]['input_ids'] = input_ids
            self.draft_mapping[idx]['init_input_len'] = input_ids.size(-1)

    async def generate_and_verify(self, input_ids_list):
        self.start_processes()
        await self.wait_for_processes_to_finish()
        return

    async def wait_for_processes_to_finish(self):
        while self.complete < len(self.draft_mapping):
            await asyncio.sleep(0)
        if self.complete == len(self.draft_mapping):
            return