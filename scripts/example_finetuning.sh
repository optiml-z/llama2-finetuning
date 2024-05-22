torchrun --nnodes 1 --nproc_per_node 2  finetuning.py --enable_fsdp --model_name models/llama-2-7b-hf/ --dist_checkpoint_root_folder finetuned_model/model_checkpoints --use_fast_kernels --output_dir finetuned_model --save_metrics
torchrun --nnodes 1 --nproc_per_node 2  finetuning.py --enable_fsdp --model_name models/llama-2-7b-hf/ --use_peft --peft_method lora --output_dir lora_model
