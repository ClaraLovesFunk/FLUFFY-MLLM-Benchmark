from __future__ import annotations

import os
from io import BytesIO
from typing import Union
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import argparse
import json
import time



CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import sys
root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_directory)
import utils  
import prompts

sys.path.append("models/otter")
from otter_ai import OtterForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





model_name_formal = "luodian/OTTER-Image-MPT7B"
model_name_informal = "otter"


def get_image(file_path: str) -> Image.Image:
    return Image.open(file_path)

def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"

def get_model(device):
    load_bit = "bf16"
    precision = {}
    if load_bit == "bf16":
        precision["torch_dtype"] = torch.bfloat16
    elif load_bit == "fp16":
        precision["torch_dtype"] = torch.float16
    elif load_bit == "fp32":
        precision["torch_dtype"] = torch.float32
    model = OtterForConditionalGeneration.from_pretrained(model_name_formal, device_map="sequential", cache_dir=CACHE_DIR, **precision)
    model.text_tokenizer.padding_side = "left"
    model.eval()
    model.to(device)

    image_processor = transformers.CLIPImageProcessor()
    
    return model, image_processor



def get_response(device, image, prompt: str, model=None, image_processor=None) -> str:
    
    input_data = image

    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype)
        else:
            vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype

    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        temperature = 0,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output

def gen_output(device, data_text, model, image_processor, image_dir_path, tasks):
    pred = []
    
    for sample in data_text: 
        output_sample = {'text_input_id': sample['text_input_id']}
        for task in tasks:
            prompt_generic = prompts.zeroshot(test_sample=sample, task=task)
            prompt_otter = get_formatted_prompt(prompt_generic) 
            image_file_path = os.path.join(image_dir_path, sample['image_id'])

            image = get_image(image_file_path)
            output = get_response(device, image, prompt_otter, model, image_processor)

            output_name = "output_" + task
            prompt_name = 'prompt_' + task
            
            output_sample.update({prompt_name: prompt_otter})
            output_sample.update({output_name: output})

        pred.append(output_sample)

    return pred

def main(dataset_name, run):
    model, image_processor = get_model(device)

    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = utils.get_info(dataset_name=dataset_name, model_name=model_name_informal, run=run)

    with open(ds_file_path, 'r') as f:
        data = json.load(f)
    data_text = data['data']

    run_time_inference_start = time.time()
    pred = gen_output(device, data_text, model, image_processor, image_dir_path, tasks)
    run_time_inference_end = time.time()
    run_time_inference = int(run_time_inference_end - run_time_inference_start) / 60 
    
    config = {
        'model': model_name_informal,
        'dataset': dataset_name,
        "split": split,
        "run": "1",
        "run time inference": run_time_inference
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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run", default="1")
    args = parser.parse_args()

    run = args.run
    dataset_name = args.dataset
    main(dataset_name, run)