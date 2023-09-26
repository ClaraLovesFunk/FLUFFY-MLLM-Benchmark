import os

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR


import time
import json
import torch
from PIL import Image, ImageOps
import argparse
from lavis.models import load_model_and_preprocess

import sys
root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_directory)
print(sys.path)
import utils 
import prompts 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets_dir = 'datasets'
experiments_dir = 'experiments'


def get_model(model_name, device):
    model_info = utils.ModelInfo(model_name)
    lavis_model_type = model_info.get_lavis_model_type()
    lavis_name = model_info.get_lavis_name()

    model, vis_processors, _ = load_model_and_preprocess(
        name=lavis_name, model_type=lavis_model_type, is_eval=True, device=device)

    model.to(device)
    
    return model, vis_processors


def get_dataset_text(dataset_name):
    dataset_info = utils.DatasetInfo(dataset_name)
    text_dataset_split = dataset_info.get_text_dataset_split()
    dataset_file_path = os.path.join(datasets_dir, dataset_name, text_dataset_split + '.json')
    
    return utils.dataset(dataset_name, dataset_file_path).load()


def gen_output(device, dataset_name, data_text, model, vis_processors):
    dataset_info = utils.DatasetInfo(dataset_name)
    img_dataset_split = dataset_info.get_img_dataset_split() 
    image_dataset_name = dataset_info.get_img_dataset_name() 
    tasks = dataset_info.get_tasks() 

    images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)
    pred = [] 

    for sample in data_text:
        text_input_id = utils.get_text_input_id(dataset_name, sample)
        output_sample = {'text_input_id': text_input_id}

        for t in tasks:
            output_task = 'output_' + t
            image_file_path =  utils.get_img_path(dataset_name, images_dir_path, sample)
            image_raw = Image.open(image_file_path)

            if image_raw.mode != 'RGB':
                if image_raw.mode != 'L':
                    image_raw = image_raw.convert('L')
                image_raw = ImageOps.colorize(image_raw, 'black', 'white')

            image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)
            prompt = prompt.zeroshot(test_sample=sample, task=t)
            output = model.generate({"image": image, "prompt": prompt}, temperature=0)
            output_sample.update({output_task: output[0]})

        pred.append(output_sample)

    return pred


def predict_dataset(model_name, dataset_name, run):
    # Load model and preprocessors
    model, vis_processors, _ = get_model(model_name, device)

    # Load data
    data_text, _ = get_dataset_text(dataset_name)

    # Generate predictions
    pred, _ = gen_output(device, dataset_name, data_text, model, vis_processors)

    # Save output and configuration
    experiment_dir_path = os.path.join(experiments_dir, model_name, dataset_name, 'run' + str(run))
    experiment_output_file_path = os.path.join(experiment_dir_path, 'output.json')
    experiment_config_file_path = os.path.join(experiment_dir_path, 'config.json')

    # Check and create the experiment directory if it doesn't exist
    if not os.path.exists(experiment_dir_path):
        os.makedirs(experiment_dir_path)

    # Save predictions
    with open(experiment_output_file_path, 'w') as f:
        json.dump(pred, f, indent=4)

    # Save configuration
    config = {
        'model': model_name,
        'dataset': dataset_name,
        'run': run,
        # You can add other configuration details here as needed
    }
    with open(experiment_config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    return pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="blip2")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    predict_dataset(args.model_name, args.dataset_name, args.run)


# source venvs/lavis/bin/activate
# cd models/blip2
# python inference.py --dataset hateful_memes