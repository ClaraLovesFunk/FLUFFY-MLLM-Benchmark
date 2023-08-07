# Implementing LLaVA for benchmark

## Step 1: Clone this repository and navigate to LLaVA folder

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```


## Step 2: Install Packages

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
``````

## Step 3: Upgrade to the latest code base

```bash
git pull
pip uninstall transformers
pip install -e .
``````

## Step 4: Modify scripts
1. `LLaVA/llava/eval/run_llava.py``
    - Replace “device = cuda” with “device = device” and determine which device to use at the top.
    - Add preferred cache and set the environment variable at the very beginning of the script, e.g.:

        ```python
        CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        ``````

2. `LLaVA/llava/model/builder.py``
    - Replace “device = cuda” with “device = device” and determine which device to use at the top.
    - Add preferred cache and set the environment variable at the beginning of the script, e.g.:

        ```python
        CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        Add the cache to all methods “.from_pretrained()”, e.g. lora_cfg_pretrained = AutoConfig.from_pretrained(model_path, cache_dir=CACHE_DIR)
        ``````
