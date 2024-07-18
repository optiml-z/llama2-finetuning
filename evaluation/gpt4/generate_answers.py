import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from fastchat.model import get_conversation_template


def run_eval(model_path, model_id, question_file, answer_file, num_gpus, args):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers(
                model_path, model_id, ques_jsons[i : i + chunk_size], args
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ans_handle)

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model


@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, args):

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = get_llm(model_id, args.cache_dir)

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        conv = get_conversation_template(model_id)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.0,
            max_new_tokens=256,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-id", default = "meta-llama/Llama-2-7b", type=str)
    parser.add_argument("--question-file", type=str, default="./json_utils/new_question.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cache_dir", default="./llm_weights", type=str )
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.1, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", default="magnitude" ,type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    args = parser.parse_args()
    import numpy as np
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    
    file_name = f"./compressed_answers/llama2-7b/{args.prune_method}_answer_{args.sparsity_ratio}.jsonl" # N:M
    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        file_name,
        args.num_gpus,
        args
    )
