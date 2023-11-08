import os
import json
import argparse
import sys
from collections import OrderedDict
from PIL import Image, ImageOps

root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_directory)
import utils
import prompts

def insert_prompts_to_output(dataset_name, model_name, run):
    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = utils.get_info(dataset_name=dataset_name, model_name=model_name, run=run)

    # Load the dataset
    with open(ds_file_path, 'r') as f:
        data = json.load(f)
        data_text = data['data']

    # Load the existing output
    with open(output_file_path, 'r') as f:
        pred = json.load(f)

    # Initialize the updated predictions list
    updated_preds = []

    # Iterate over each sample in the dataset
    for sample in data_text:
        text_input_id = sample['text_input_id']

        # Retrieve the corresponding output_sample if it exists
        output_sample = next((item for item in pred if item['text_input_id'] == text_input_id), None)
        if output_sample is None:
            continue

        # Create a new OrderedDict for the updated output sample
        updated_output_sample = OrderedDict([('text_input_id', text_input_id)])

        # Generate and insert prompts followed by outputs in the desired order
        for t in tasks:
            # Generate prompt
            prompt = prompts.zeroshot(test_sample=sample, task=t)
            prompt_name = f'prompt_{t}'  # Removed the redundant dataset name
            updated_output_sample[prompt_name] = prompt

            # Retrieve corresponding output from existing data
            output_key = f'output_{t}'  # Removed the redundant dataset name
            output_value = output_sample.get(output_key)

            # Insert output if it exists
            if output_value is not None:
                updated_output_sample[output_key] = output_value

        # Append the new ordered output sample to the updated predictions list
        updated_preds.append(updated_output_sample)

    # Save the updated output with prompts
    with open(output_file_path, 'w') as f:
        json.dump(updated_preds, f, indent=4)

    print(f"Prompts inserted into output for {dataset_name} using model {model_name} run {run}.")

def read_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--model_name", type=str, default="blip2")
    parser.add_argument("--dataset_names", type=str, nargs='+', default="all", help="List of dataset names or 'all' for all datasets.")
    parser.add_argument("--run", type=str, default='1')
    args = parser.parse_args()

    config = read_config(args.config_path)

    if args.dataset_names == ["all"]:  # If the keyword "all" is provided, use all datasets from the config
        dataset_names = config["dataset_names"]
    else:
        dataset_names = args.dataset_names

    for dataset_name in dataset_names:
        insert_prompts_to_output(dataset_name, args.model_name, args.run)



'''

python3 models/blip2/add_prompts.py --model_name instructblip --dataset_names all

'''