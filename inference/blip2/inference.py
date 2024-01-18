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
from utils.info import get_info
from utils.info import ModelInfo
import prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


datasets_dir = 'datasets'
experiments_dir = 'experiments'


def get_model(model_name, device):
    model_info = ModelInfo(model_name)
    lavis_model_type = model_info.get_lavis_model_type()
    lavis_name = model_info.get_lavis_name()

    model, vis_processors, _ = load_model_and_preprocess(
        name=lavis_name, model_type=lavis_model_type, is_eval=True, device=device)

    model.to(device)
    
    return model, vis_processors



def gen_output(device, dataset_name, data_text, model, vis_processors, image_dir_path, tasks):
    
    pred = [] 

    for sample in data_text: 
        
        output_sample = {'text_input_id': sample['text_input_id']}

        for t in tasks:
            output_task = 'output_' + t
            image_file_path =  os.path.join(image_dir_path, sample['image_id']) 
            image_raw = Image.open(image_file_path)

            if image_raw.mode != 'RGB':
                if image_raw.mode != 'L':
                    image_raw = image_raw.convert('L')
                image_raw = ImageOps.colorize(image_raw, 'black', 'white')

            prompt = prompts.zeroshot(test_sample=sample, task=t)
            image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)
            
            output = model.generate({"image": image, "prompt": prompt}, temperature=0)
            output=output[0]
            #output_sample.update({output_task: output[0]})

            output_name = "output_" + t
            prompt_name = 'prompt_' + t
            
            output_sample.update({prompt_name: prompt})
            output_sample.update({output_name: output})

        pred.append(output_sample)

    return pred


def predict_dataset(model_name, dataset_name, run):

    model, vis_processors = get_model(model_name, device) 

    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = get_info(dataset_name=dataset_name, model_name='blip2', run=run)

    with open(ds_file_path, 'r') as f:
        data = json.load(f)
        data_text = data['data']

    pred = gen_output(device, dataset_name, data_text, model, vis_processors, image_dir_path, tasks) 

    config = {
        'model': model_name,
        'dataset': dataset_name,
        'run': run,
    }

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(output_file_path, 'w') as f:
        json.dump(pred, f, indent=4)

    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="blip2")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--run", type=str, default='1')
    args = parser.parse_args()

    predict_dataset(args.model_name, args.dataset_name, args.run)