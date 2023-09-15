#!/usr/bin/env python3

import os
import itertools
import time

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR 

import argparse
import subprocess

# Defining all models and datasets
ds_name_all = ['aokvqa', 'okvqa', 'hateful_memes', 'mami', 'mvsa', 'esnlive', 'scienceqa', 'clevr', 'gqa']
model_name_all = ['llava']  

def run_inference(model_name, dataset_name):
    
    base_model_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/models"
    base_dataset_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets"
    venv_base_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/venvs" 

    model_path = os.path.join(base_model_dir, model_name)
    venv_path = os.path.join(venv_base_dir, model_name, "bin", "python") # Using the Python interpreter directly

    # Ensure model, dataset directories/files and virtual environment exist
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return

    if not os.path.exists(venv_path):
        print(f"Virtual environment for {model_name} does not exist!")
        return

    # Run inference script using the Python interpreter from the virtual environment
    cmd = [venv_path, os.path.join(model_path, "inference.py"), "--dataset", dataset_name]
    env = os.environ.copy()  # Get the current environment variables.
    env["TRANSFORMERS_CACHE"] = CACHE_DIR
    subprocess.run(cmd, env=env)

def run_all_inferences(model_names, dataset_names):
    for model, dataset in itertools.product(model_names, dataset_names):
        try:
            run_inference(model, dataset)
            
        except Exception as e:
            print(f"Error running inference for model {model} on dataset {dataset}. Error: {e}")

if __name__ == "__main__":

    start_time= time.time()

    parser = argparse.ArgumentParser(description="Run inference on a model with a given dataset.")
    parser.add_argument("-models", type=str, nargs='+', required=True, help="List of model names separated by spaces. Use 'all' for all models.")
    parser.add_argument("-datasets", type=str, nargs='+', required=True, help="List of dataset names separated by spaces. Use 'all' for all datasets.")
    
    args = parser.parse_args()

    # Replace 'all' with the respective list of all models or datasets
    if 'all' in args.models:
        args.models = model_name_all
    if 'all' in args.datasets:
        args.datasets = ds_name_all

    run_all_inferences(args.models, args.datasets)

    end_time= time.time()
    run_time = int((end_time - start_time)/60)
    print(f'runtime (min): {run_time}')


# python3 run_inference.py -models all -datasets all
# python3 run_inference.py -models all -datasets mvsa
# python3 run_inference.py -models openflamingo -datasets aokvqa
