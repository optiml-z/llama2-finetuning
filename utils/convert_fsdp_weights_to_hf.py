import fire
import os
import yaml

from transformers import LlamaTokenizer
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import FileSystemReader
from llama_recipes.inference.model_utils import  load_llama_from_config


def load_sharded_model_single_gpu(model,model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model

def main(
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    ):
    
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file
        with open(train_params_path, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    #load the HF model definition from config
    model_def = load_llama_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path
    tokenizer = LlamaTokenizer.from_pretrained(HF_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")

if __name__ == "__main__":
    fire.Fire(main)