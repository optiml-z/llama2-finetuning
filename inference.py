import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from utils.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

  def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, **kwargs,):
    print(f"User prompt:\n{user_prompt}")
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
    if is_xpu_available():
        batch = {k: v.to("xpu") for k, v in batch.items()}
    else:
        batch = {k: v.to("cuda") for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"The inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return output_text

  if prompt_file is not None:
      assert os.path.exists(
          prompt_file
      ), f"Provided Prompt file does not exist {prompt_file}"
      with open(prompt_file, "r") as f:
          user_prompt = "\n".join(f.readlines())
      output = inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
      print(f"LLM Response:\n{output}")
  
  if not sys.stdin.isatty():
      user_prompt = "\n".join(sys.stdin.readlines())
      output = inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
      print(f"LLM Response:\n{output}")

if __name__ == "__main__":
    fire.Fire(main)
