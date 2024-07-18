import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
from tqdm import tqdm
from ignite.metrics import JSDivergence
from ignite.engine import Engine

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:{output}"
    ),
}

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def compute_js_divergence(original_model, compressed_model, dataloader, tokenizer):
    def process_function(engine, batch):
        if batch.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(batch)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(batch)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Get the logits from the original model
        with torch.no_grad():
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits
        
        # Get the logits from the compressed model
        with torch.no_grad():
            compressed_outputs = compressed_model(**inputs)
            compressed_logits = compressed_outputs.logits
        
        return compressed_logits, original_logits

    evaluator = Engine(process_function)
    js_divergence_metric = JSDivergence()
    js_divergence_metric.attach(evaluator, 'js-div')

    for batch in tqdm(dataloader, desc="Processing batches..."):
        evaluator.run([batch])

    avg_js_divergence = evaluator.state.metrics['js-div']
    return avg_js_divergence

def get_llm(model_name, cache_dir="llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def load_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return TextDataset(data[:5000])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Jensen-Shannon Divergence between original and compressed models")
    parser.add_argument('--original_model', type=str, required=True, help="Path to the original model")
    parser.add_argument('--compressed_model', type=str, required=True, help="Path to the compressed model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader")
    args = parser.parse_args()

    original_model = get_llm(args.original_model)
    compressed_model = get_llm(args.compressed_model)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(args.original_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the dataset and dataloader
    dataset = load_dataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Compute Jensen-Shannon Divergence
    avg_js_divergence = compute_js_divergence(original_model, compressed_model, dataloader, tokenizer)
    print(f"\nAverage Jensen-Shannon Divergence between original and compressed model({args.compressed_model}): {avg_js_divergence}")