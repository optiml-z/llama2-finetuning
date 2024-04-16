# Llama 2 finetuning

To get started with finetuning Llama language models -- ranging from 7B to 70B parameters, firstly download model weights. 
In order to download the model weights and tokenizer, please visit the [Meta website](https://llama.meta.com/llama-downloads/) and accept their License.
Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have wget and md5sum installed. Then run the script: ./download.sh.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as 403: Forbidden, it is likely due to expired 
links. You can always re-request a link.

Then, setup up rest of the requirements as following:

```bash
python3 -m venv venv-llama
source venv-llama/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

This finetuning is highly relied on meta-llama/llama-recipe and hence required llama-recipe as a package.

Once all the setup is completed, we can begin finetuning the llama2 models. The default dataset used for finetuning is [vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4). Download the dataset from [here](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) and update the 
[data path](./configs/datasets.py) accordingly. Then use the following scripts for various tasks:

### Finetuning

```bash
python3 finetuning.py  [--use_peft] [--peft_method lora] [--quantization] [--use_fp16] --model_name path_to_model_folder/7B --output_dir path_to_save_PEFT_model
```

### Inference

```bash
python3 inference.py --model_name <training_config.model_name> --peft_model <training_config.output_dir> --prompt_file <test_prompt_file>
```

### Evaluation

```bash
python evaluation/eval.py --model [hf] --model_args pretrained=<training_config.model_name>,dtype="float",peft=<training_config.output_dir> 
--task hellaswag --device cuda:0 --batch_size 8 --output_path eval_results
```