import os
import json
import random
import shortuuid
from datasets import load_dataset

question_dict = []
golden_ans = []
count = 1

ds = load_dataset("mrm8488/unnatural-instructions-full")

# Convert dataset to list and randomly sample 100 elements
sampled_data = random.sample(list(ds['train']), 100)

print(len(sampled_data))

for item in sampled_data:
    for instance in item['instances']:
        instruction = item['instruction']
        input = instance['input']
        output = instance['output']
        question = {
            "question_id": count,
            "text": "Below is an instruction that describes a task, paired with an input Write a response that appropriately completes the request.\n\n### Instruction:\n" + 
            instruction + "\n\n### Input:\n" + input + "\n\n### Response:",
            "category": "small"
        }
        question_dict.append(question)
        # Collect the answers
        golden_ans.append({
            "question_id": count,
            "answer_id": shortuuid.uuid(),
            "text": output
        })
        count += 1

print(len(question_dict))   
questions = [json.dumps(q) for q in question_dict]
with open("./json_utils/new_question.jsonl", "w") as fo:   
    fo.write("\n".join(questions))

answers = [json.dumps(ans) for ans in golden_ans]
with open("./json_utils/golden_answer.jsonl", "w") as fo:
    fo.write("\n".join(answers))