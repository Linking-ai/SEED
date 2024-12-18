import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_tree_attn.modeling_llama import LlamaForCausalLM
from llama_tree_attn.tokenization_llama import LlamaTokenizer
import json
import re
import logging
from datetime import datetime
from LLM_engine import BaseEngine, SpeculativeEngine, ScheduledSpeculativeEngine
from tot_search import BFS
from tot_agent import ToTAgent

# Config environment
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TOKENIZERS_PARALLELISM'] = "true"

logging_time = datetime.now().strftime("%m%d")
logging.basicConfig(
    filename=f'logs/run_tot_llama2_68m_7b_gsm8k_{logging_time}_211_temp_0_0.log',                        
    level=logging.INFO,                                 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_model_and_tokenizer(model_type, tokenizer_path, target_model_path, draft_model_path, draft_model_num, temperature, muti_candidate_draft):
    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path) if muti_candidate_draft else AutoTokenizer.from_pretrained(tokenizer_path)
        target_model = LlamaForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, local_files_only=True, device_map="auto") if muti_candidate_draft else AutoModelForCausalLM.from_pretrained(target_model_path, local_files_only=True, device_map="auto")
        draft_models = [LlamaForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.float16, local_files_only=True, device_map="auto") for _ in range(draft_model_num)]
    
    # AR, SD, MCSD, SD+SEED, MCSD+SEED Engine Initialization
    if muti_candidate_draft:
        engines = {
            'AR': BaseEngine(target_model, eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, temp=temperature),
            'SD': SpeculativeEngine(draft_models[0], target_model, k_config=(2, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, replacement=True, target_model_temp=temperature),
            'SD_SEED': ScheduledSpeculativeEngine(draft_models, target_model, k_config=(2, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, replacement=True, target_model_temp=temperature),
            'MCSD': SpeculativeEngine(draft_models[0], target_model, k_config=(2, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, target_model_temp=temperature),
            'MCSD_SEED': ScheduledSpeculativeEngine(draft_models, target_model, k_config=(2, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, target_model_temp=temperature)
        }
    else:
        engines = {
            'AR': BaseEngine(target_model, eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, temp=temperature),
            'SD': SpeculativeEngine(draft_models[0], target_model, k_config=(1, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, replacement=True, target_model_temp=temperature),
            'SD_SEED': ScheduledSpeculativeEngine(draft_models, target_model, k_config=(1, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, replacement=True, target_model_temp=temperature),
            'MCSD': SpeculativeEngine(draft_models[0], target_model, k_config=(1, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, target_model_temp=temperature),
            'MCSD_SEED': ScheduledSpeculativeEngine(draft_models, target_model, k_config=(1, 1, 1), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100, tree_attn=True, target_model_temp=temperature)
        }
    return tokenizer, engines

def log_metrics(prefix, gen_time, gen_tht_time, gen_eval_time, gen_acc, gen_tht_acc, gen_eval_acc, ar_time=None):
    logging.info(f"******************")
    logging.info(f"{prefix}_gen_time: {gen_time}")
    logging.info(f"{prefix}_gen_tht_time: {gen_tht_time}")
    logging.info(f"{prefix}_gen_eval_time: {gen_eval_time}")
    logging.info(f"{prefix}_gen_tht_acc: {gen_tht_acc}")
    logging.info(f"{prefix}_gen_eval_acc: {gen_eval_acc}")
    logging.info(f"{prefix}_gen_acc: {gen_acc}")
    
    if ar_time is not None:
        speedup = ar_time / gen_time
        logging.info(f"{prefix}_speedup_ratio: {speedup:.2f}")

def main():
    # Load datasets
    if args.dataset_name == "gsm8k":
        with open('data/gsm8k_test.jsonl', "r") as f:
            gsm8k = [json.loads(line) for line in f][args.start_eid:args.end_eid]
    
    # Initialize models and engines
    tokenizer, engines = initialize_model_and_tokenizer(
        args.model_type, args.tokenizer_path, args.target_model_path, args.draft_model_path, args.num_thoughts, args.temperature, args.muti_candidate_draft
    )

    # Main loop
    total_times = {key: 0 for key in engines.keys()}
    total_thoughts_times = {key: 0 for key in engines.keys()}
    total_eval_times = {key: 0 for key in engines.keys()}
    total_accuracies = {key: 0 for key in engines.keys()}
    total_thoughts_accuracies = {key: 0 for key in engines.keys()}
    total_eval_accuracies = {key: 0 for key in engines.keys()}

    for idx, example in tqdm(enumerate(gsm8k)):
        logging.info(f'--------------{idx}--------------')
        initial_prompt = example['question']
        
        # Initialize ToT Agents
        tot_agents = {
            key: ToTAgent(
                llm_engine=gen,
                tokenizer=tokenizer,
                decoding_strategy = (
                    "multi_speculative" if "SEED" in key else
                    "autoregressive" if key == "AR" else
                    "speculative"
                ),
                prompt_type="cot",
                evaluation_strategy="value"
            )
            for key, gen in engines.items()
        }

        # Process and log each generation
        for key, agent in tot_agents.items():
            gen_time, gen_tht_time, gen_eval_time, gen_acc, gen_tht_acc, gen_eval_acc, solution = BFS(agent).solve(
                initial_prompt=initial_prompt,
                num_thoughts=args.num_thoughts,
                max_steps=args.max_steps,
                max_states=args.max_states,
                consistency=args.consistency,
            )
            print(solution)
            log_metrics(key, gen_time, gen_tht_time, gen_eval_time, gen_acc, gen_tht_acc, gen_eval_acc)

            total_times[key] += gen_time
            total_thoughts_times[key] += gen_tht_time
            total_eval_times[key] += gen_eval_time
            total_accuracies[key] += gen_acc
            total_thoughts_accuracies[key] += gen_tht_acc
            total_eval_accuracies[key] += gen_eval_acc

            torch.cuda.empty_cache()

    # Final logging of averages
    for key in engines.keys():
        logging.info(f"{key} - Time: {total_times[key]}, Thoughts Time: {total_thoughts_times[key]}, Eval Time: {total_eval_times[key]}")
        logging.info(f"{key} - Accuracy: {total_accuracies[key]}, Thoughts Accuracy: {total_thoughts_accuracies[key]}, Eval Accuracy: {total_eval_accuracies[key]}")

     # Calculate and log speedup ratios compared to AR
    ar_total_time = total_times["AR"]
    for method in ["SD", "SD_SEED", "MCSD", "MCSD_SEED"]:
        speedup = ar_total_time / total_times[method]
        logging.info(f"{method} - Speedup ratio compared to AR: {speedup:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="llama", type=str)
    parser.add_argument('--tokenizer_path', default="../../LLM/Llama-68M-Chat-v1", type=str)
    parser.add_argument('--draft_model_path', default="../../LLM/Llama-68M-Chat-v1", type=str)
    parser.add_argument('--target_model_path', default="../../LLM/Llama-2-7b-chat-hf", type=str)
    parser.add_argument('--muti_candidate_draft', default=True, type=bool)
    parser.add_argument('--temperature', default=0.2, type=float, help='the diversity of generated text')
    parser.add_argument('--dataset_name', default="gsm8k", type=str, required=False)
    parser.add_argument('--start_eid', default=0, type=int, required=False)
    parser.add_argument('--end_eid', default=100, type=int, required=False)
    parser.add_argument('--num_thoughts', default=3, type=int, required=False, help='')
    parser.add_argument('--max_steps', default=1, type=int, required=False, help='')
    parser.add_argument('--max_states', default=1, type=int, required=False, help='')
    parser.add_argument('--consistency', default=1, type=int, required=False, help='')
    args = parser.parse_args()

    main()