python evaluation/eval.py --model hf --model_args pretrained=models/llama-2-7b-hf/,dtype="float" --task unnatural --device cuda:0 --batch_size 8 --output_path eval_results/pretrained

python evaluation/eval.py --model hf --model_args pretrained=finetuned_model/hf/,dtype="float" --task unnatural --device cuda:0 --batch_size 8 --output_path eval_results/finetuned

python evaluation/eval.py --model hf --model_args pretrained=compressed_model/,dtype="float" --task unnatural --device cuda:0 --batch_size 8 --output_path eval_results/compressed

python evaluation/eval.py --model hf --model_args pretrained=models/llama-2-7b-hf/,dtype="float",peft=lora_model --task unnatural --device cuda:0 --batch_size 8 --output_path eval_results/lora
