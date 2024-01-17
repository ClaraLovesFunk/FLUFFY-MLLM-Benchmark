import os
import sys

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR 

sys.path.insert(0, '/home/users/cwicharz/project/Testing-Multimodal-LLMs/models/llava/repo')
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

sys.path.insert(0, '/home/users/cwicharz/project/Testing-Multimodal-LLMs')
from utils import *
from prompts import *

import argparse
import torch
import json
import pandas as pd
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_info(dataset_name, model_name, run):

    dataset_info = DatasetInfo(dataset_name)
    img_dataset_name = dataset_info.get_img_dataset_name()
    tasks = dataset_info.get_tasks()
    split = dataset_info.get_text_dataset_split()

    base_path = '/home/users/cwicharz/project/Testing-Multimodal-LLMs'
    ds_file_path = os.path.join(base_path, 'datasets', args.dataset, 'ds_benchmark.json')
    image_dir_path = os.path.join(base_path, 'datasets', img_dataset_name) 
    output_dir_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run)
    output_file_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run, 'output.json' )
    config_file_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run, 'config.json' )

    return tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split



def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args, tokenizer, model, image_processor, model_name):



    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device) 

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.001,   #smallest possible temperature     
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]).to(device)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]

    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


def predict_dataset(dataset_name, model_path, run, conv_mode=None):
    
    print(f"Using TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}") 

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    # get infos
    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = get_info(dataset_name = dataset_name, model_name = 'llava', run = run)

    dataset = pd.read_json(ds_file_path) 
    data_list = dataset['data'].tolist()
    
    
   
    
    run_time_inference_start = time.time()

   
    pred = []

    for test_sample in data_list[:2]: 

        output_sample = {'text_input_id': test_sample['text_input_id']}

        for t in tasks:

            prompt = zeroshot(test_sample, t)
            

            args = argparse.Namespace()
            args.model_path = model_path
            args.model_base = None
            
            args.image_file = os.path.join(image_dir_path, test_sample['image_id']) 
            
            args.query = prompt 
            args.conv_mode = conv_mode

            output = eval_model(args, tokenizer, model, image_processor, model_name)
            
            output_name = 'output_' + t
            output_sample.update({output_name: output}) 

        pred.append(output_sample)

    run_time_inference_end = time.time()
    run_time_inference = int(run_time_inference_end - run_time_inference_start)/60
    
    config = {
        'model': 'llava',
        'dataset': dataset_name,
        'split': split,
        'run': run,
        'run time inference':run_time_inference
    }

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(output_file_path, 'w') as f:
        json.dump(pred, f, indent=4)

    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--run", type=str, default="1")
    args = parser.parse_args()
    
    predict_dataset(dataset_name = args.dataset, model_path = args.model_path, run = args.run, conv_mode=args.conv_mode)


# source venvs/llava15/bin/activate
# cd models/llava15
# python inference.py --dataset hateful_memes
