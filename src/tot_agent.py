import asyncio
import re
import time
import numpy as np
from prompts import gsm8k_thought_prompt, gsm8k_answer_prompt, gsm8k_evaluate_prompt, gsm8k_solution_prompt

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return float(0)

def extract_first_num_in_range10(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\1\2", text)  # 处理形如 123,456 的数字
    # 匹配0-10之间的数字，包括整数和浮点数
    res = re.findall(r"\b(?:[0-9]|10)(?:\.\d+)?\b", text)
    if len(res) > 0:
        num_str = res[0]
        return float(num_str)
    else:
        return float(0)

class ToTAgent:
    def __init__(
        self,
        llm_engine,
        tokenizer,
        decoding_strategy: str = "autoregressive",
        prompt_type: str = "cot",
        evaluation_strategy: str = "value",
    ):
        self.llm_engine = llm_engine
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.evaluation_strategy = evaluation_strategy
        self.decoding_strategy = decoding_strategy

    def response(self, task: str):
        """Generate text from prompt using llm_engine"""
        inputs = self.tokenizer(task, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        if self.decoding_strategy=="autoregressive" or self.decoding_strategy=="speculative":
            start_time = time.time()
            response = self.llm_engine.generate(input_ids)
            end_time = time.time()
        response_text = self.tokenizer.batch_decode(
            response.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0][len(task):]
        run_time = end_time-start_time
        if self.decoding_strategy=="autoregressive":
            return response_text, run_time
        elif self.decoding_strategy=="speculative":
            acceptance_rate = response.acceptance_count/(response.invocation_count * 3)
            return response_text, run_time, acceptance_rate
    
    def response_parallel(self, task_list: list):
        """Generate text from prompt_list using llm_engine"""
        input_ids_list = []
        for task in task_list:
            inputs = self.tokenizer(task, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            input_ids_list.append(input_ids)
        response_text_list = []
        self.llm_engine.reset(input_ids_list)
        start_time = time.time()
        asyncio.run(self.llm_engine.generate_and_verify(input_ids_list))
        end_time = time.time()
        acceptance_rate_list = []
        for i in range(len(task_list)):
            acceptance_rate = self.llm_engine.draft_mapping[i]['acceptance_count']/(self.llm_engine.draft_mapping[i]['invocation_count']*len(task_list))
            acceptance_rate_list.append(acceptance_rate)
            response_text = self.tokenizer.batch_decode(
                self.llm_engine.draft_mapping[i]['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0][len(task):]
            response_text_list += [response_text]
        self.llm_engine.clear()
        run_time = end_time-start_time
        mean_acceptance_rate = np.mean(acceptance_rate_list)
        
        return response_text_list, run_time, mean_acceptance_rate

    def generate_thoughts(
        self, state, k, initial_prompt
    ):
        if type(state) == str:
            state_text = state
        else:
            state_text = ""

        if state_text == initial_prompt:
            state_text = ""
        
        prompt = gsm8k_thought_prompt.format(initial_prompt=initial_prompt, state_text=state_text)

        all_time = 0
        thoughts = []  # Ensure thoughts is initialized here
        
        if self.decoding_strategy == "autoregressive":
            for _ in range(k):
                thought, run_time = self.response(prompt)
                match = re.search(r'([^?]*\?)', thought)
                if match:
                    thought = match.group(1)
                all_time += run_time
                thoughts += [thought]
            mean_acc = 0
        elif self.decoding_strategy == "speculative":
            mean_acc = 0
            for _ in range(k):
                thought, run_time, acc_rate = self.response(prompt)
                match = re.search(r'([^?]*\?)', thought)
                if match:
                    thought = match.group(1)
                all_time += run_time
                mean_acc += acc_rate
                thoughts += [thought]
            mean_acc = float(mean_acc / k)
        elif self.decoding_strategy == "multi_speculative":
            prompts = [prompt] * k
            temp_thoughts, run_time, mean_acc = self.response_parallel(prompts)
            for thought in temp_thoughts:
                match = re.search(r'([^?]*\?)', thought)
                if match:
                    thought = match.group(1)
                thoughts.append(thought)
            all_time += run_time
        else:
            # Add an else clause to handle unexpected decoding strategies
            raise ValueError(f"Invalid decoding strategy: {self.decoding_strategy}")

        return thoughts, all_time, mean_acc


    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        # not computing time for generating solution, because it is consistent in every methods
        if self.decoding_strategy=="autoregressive" or self.decoding_strategy=="speculative":
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                if not state:
                    state = ""
                state_text = state
            prompt = gsm8k_solution_prompt.format(initial_prompt=initial_prompt, state_text=state_text)
            if self.decoding_strategy=="autoregressive":
                response, _ = self.response(prompt)
            else:
                response, _, _ = self.response(prompt)
            answer = extract_last_num(str(response))
            answer = state_text
        else:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                if not state:
                    state = ""
                state_text = state
            prompt = gsm8k_solution_prompt.format(initial_prompt=initial_prompt, state_text=state_text)
            answer = extract_last_num(str(state_text))
            answer = state_text
            # TODO: SEED solution ()
        return answer

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}
        
        if self.evaluation_strategy == "value":
            state_values = {}
            if self.decoding_strategy=="autoregressive" or self.decoding_strategy=="speculative":
                all_time = 0
                mean_acc = 0
                # print(states)
                for state in states:
                    if type(state) == str:
                        state_text = state
                    else:
                        state_text = ""

                    prompt = gsm8k_answer_prompt.format(initial_prompt=initial_prompt, state_text=state_text)
                    
                    if self.decoding_strategy=="speculative":
                        response, run_time, acc_rate1 = self.response(prompt)
                        all_time += run_time
                        answer = extract_last_num(response)
                        state += "##"+ str(answer)
                        state_text += "##"+ str(answer)
                        response, run_time, acc_rate2 = self.response(gsm8k_evaluate_prompt.format(initial_prompt=initial_prompt, state_text=state_text, answer=answer))
                        all_time += run_time
                        value = extract_first_num_in_range10(response)
                        mean_acc += (acc_rate1 + acc_rate2)/2
                    else:
                        response, run_time = self.response(prompt)
                        all_time += run_time
                        answer = extract_last_num(response)
                        state += "##" + str(answer)
                        state_text += "##"+ str(answer)
                        response, run_time = self.response(gsm8k_evaluate_prompt.format(initial_prompt=initial_prompt, state_text=state_text, answer=answer))
                        all_time += run_time
                        value = extract_first_num_in_range10(response)
                    state_values[state] = value
                mean_acc = float(mean_acc/len(states))
                return state_values, all_time, mean_acc
            elif self.decoding_strategy=="multi_speculative":
                # print(states)
                prompt_list = []
                for state in states:
                    if type(state) == str:
                        state_text = state
                    else:
                        state_text = ""
                    
                    prompt = gsm8k_answer_prompt.format(initial_prompt=initial_prompt, state_text=state_text)
                    prompt_list.append(prompt)
            
                response_list, all_time1, mean_acc1 = self.response_parallel(prompt_list)

                # print(response_list)

                prompt2_list = []
                for response, state in zip(response_list, states):
                    if type(state) == str:
                        state_text = state
                    else:
                        state_text = ""
                    answer = extract_last_num(response)
                    state += "##" + str(answer)
                    state_text += "##"+ str(answer)
                    prompt2 = gsm8k_evaluate_prompt.format(initial_prompt=initial_prompt, state_text=state_text, answer=answer)
                    prompt2_list.append(prompt2)

                response_list, all_time2, mean_acc2 = self.response_parallel(prompt2_list)
                # print(response_list)

                for response in response_list:
                    value = extract_first_num_in_range10(response)
                    state_values[state] = value
                all_time = all_time1 + all_time2
                mean_acc = (mean_acc1 + mean_acc2)/2
                return state_values, all_time, mean_acc

        else:
            raise ValueError(
                "Invalid evaluation strategy. Choose 'value' or 'vote'."
            )
