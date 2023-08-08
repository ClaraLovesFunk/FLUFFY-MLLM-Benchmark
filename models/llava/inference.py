import os
import sys

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR 

sys.path.insert(0, '/home/users/cwicharz/project/Testing-Multimodal-LLMs/models/LLaVA/repo')
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

sys.path.insert(0, '/home/users/cwicharz/project/Testing-Multimodal-LLMs')
from utils import *

import argparse
import torch
import json
import pandas as pd
from PIL import Image
import requests
from PIL import Image
from io import BytesIO



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



image_dir_path = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets/hateful_memes/images/all/' #os.path.join(base_path, 'datasets/hateful_memes/images/all/', row['image_path'])
output_path = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/experiments/llava/hateful_memes/run1/output.json' #os.path.join(base_path, 'experiments/llava/hateful_memes/run1/output.json')






def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

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
            temperature=0.1,        
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


def predict_dataset(dataset_path, model_path, conv_mode=None):
    
    dataset = pd.read_json(dataset_path) 
    results = []
    

    
    for i, row in enumerate(dataset.iterrows()):

        #if i >= 2:
        #    break

        test_sample = row[1]
        prompt = prompt_construct(test_sample, 'hate classification')

        args = argparse.Namespace()
        args.model_path = model_path
        args.model_base = None
        
        args.image_file = image_dir_path + test_sample['image_path']  
        
        args.query = prompt #"Classify the following meme as 'hateful' or 'not-hateful'."
        args.conv_mode = conv_mode

        output = eval_model(args)

        results.append({
            "text_input_id": test_sample['id'],
            "output_hate_classification": output,
        })
    
    
    with open(output_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()
    
    predict_dataset(args.dataset, args.model_path, args.conv_mode)



# cd models/LLaVA/llava/eval && python run_llava_clara.py --dataset /home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets/hateful_memes/dev.json

# python inference.py --dataset /home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets/hateful_memes/dev.json
