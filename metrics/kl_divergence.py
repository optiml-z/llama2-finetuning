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

def compute_kl_divergence(original_model, compressed_model, dataloader, tokenizer):
    kl_divergences = []
    
    for batch in tqdm(dataloader, desc="Processing batches..."):
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
        
        # Compute the softmax to get probabilities
        original_probs = F.softmax(original_logits, dim=-1)
        compressed_probs = F.softmax(compressed_logits, dim=-1)
        
        # Compute the KL Divergence
        kl_div = F.kl_div(compressed_probs.log(), original_probs, reduction='batchmean')
        kl_divergences.append(kl_div.item())
    
    # Compute the average KL Divergence
    avg_kl_divergence = sum(kl_divergences) / len(kl_divergences)
    return avg_kl_divergence

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
    parser = argparse.ArgumentParser(description="Compute KL Divergence between original and compressed models")
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

    # Compute KL Divergence
    avg_kl_divergence = compute_kl_divergence(original_model, compressed_model, dataloader, tokenizer)
    print(f"\nAverage KL Divergence between original and compressed model: {avg_kl_divergence}")