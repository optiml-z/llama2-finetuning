# Llama 2 finetuning

To get started with finetuning Llama language models -- ranging from 7B to 70B parameters, firstly download model weights. 
In order to download the model weights and tokenizer, please visit the [Meta website](https://llama.meta.com/llama-downloads/) and accept their License.
Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have wget and md5sum installed. Then run the script: ./download.sh.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as 403: Forbidden, it is likely due to expired 
links. You can always re-request a link.

Then, setup up rest of the requirements as following:

```bash
python -m venv venv-llama
source venv-llama/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

This finetuning is highly relied on meta-llama/llama-recipe and hence required llama-recipe as a package.
 