#!/usr/bin/env python3

import os

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR 

import sys
import argparse
import subprocess

def run_inference(model, dataset):
    # Base directories for models and datasets
    base_model_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/models"
    base_dataset_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets"
    venv_base_dir = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/venvs" # Assuming virtual environments are in this directory

    # Construct full paths
    model_path = os.path.join(base_model_dir, model)
    dataset_path = os.path.join(base_dataset_dir, dataset, "dev.json")
    venv_path = os.path.join(venv_base_dir, model, "bin", "python") # Using the Python interpreter directly

    # Ensure model, dataset directories/files and virtual environment exist
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return

    if not os.path.exists(venv_path):
        print(f"Virtual environment for {model} does not exist!")
        return

    # Run inference script using the Python interpreter from the virtual environment
    cmd = [venv_path, os.path.join(model_path, "inference.py"), "--dataset", dataset_path]
    env = os.environ.copy()  # Get the current environment variables.
    env["TRANSFORMERS_CACHE"] = CACHE_DIR
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a model with a given dataset.")
    parser.add_argument("-model", type=str, required=True, help="Name of the model directory.")
    parser.add_argument("-dataset", type=str, required=True, help="Name of the dataset directory.")
    
    args = parser.parse_args()

    run_inference(args.model, args.dataset)
