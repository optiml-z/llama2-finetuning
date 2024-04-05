from pathlib import Path
import torch
import json
from typing import Optional

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
)
from llama_utils import ModelArgs, Tokenizer, Transformer  # replace with your actual module

def load_model(ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        ):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert model_parallel_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    
    return model, tokenizer