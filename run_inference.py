#!/usr/bin/env python3

import os

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR 

import sys
import argparse
import subprocess

def run_inference(model_name, dataset_name):
    
    base_model_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/models"
    base_dataset_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets"
    venv_base_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/venvs" 

    model_path = os.path.join(base_model_dir, model_name)
    #dataset_path = os.path.join(base_dataset_dir, dataset_name, "dev.json")
    venv_path = os.path.join(venv_base_dir, model_name, "bin", "python") # Using the Python interpreter directly

    # Ensure model, dataset directories/files and virtual environment exist
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return

    '''if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return'''

    if not os.path.exists(venv_path):
        print(f"Virtual environment for {model_name} does not exist!")
        return

    # Run inference script using the Python interpreter from the virtual environment
    cmd = [venv_path, os.path.join(model_path, "inference.py"), "--dataset", dataset_name]
    env = os.environ.copy()  # Get the current environment variables.
    env["TRANSFORMERS_CACHE"] = CACHE_DIR
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a model with a given dataset.")
    parser.add_argument("-model", type=str, required=True, help="Name of the model.")
    parser.add_argument("-dataset", type=str, required=True, help="Name of the dataset.")
    
    args = parser.parse_args()

    run_inference(args.model, args.dataset)



# python3 run_inference.py -model llava -dataset mami