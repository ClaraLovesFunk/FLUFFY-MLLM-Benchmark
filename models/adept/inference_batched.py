import os

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import torch
import argparse
import json
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
from tqdm import tqdm
import time
import sys
from PIL import Image

root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_directory)
import utils  
import prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_formal = "adept/fuyu-8b"
model_name_informal = "adept"

def resize_image(image, target_size=(224, 224)):
    return image.resize(target_size)

def get_image(file_path: str) -> Image.Image:
    return Image.open(file_path).convert('RGB')

def get_model(device):
    model = FuyuForCausalLM.from_pretrained(model_name_formal)
    model.to(device)
    processor = FuyuProcessor.from_pretrained(model_name_formal)
    return model, processor

def get_response(device, image, prompt: str, model=None, processor=None) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    generation_output = model.generate(**inputs, max_new_tokens=125) 
    generated_text = processor.batch_decode(generation_output[:, -125:], skip_special_tokens=True) 
    return generated_text[0]

def gen_output(device, data_text, model, processor, image_dir_path, tasks, batch_size=8):
    pred = []
    batch_images = []
    batch_prompts = []
    batch_meta = []

    def process_batch(batch_images, batch_prompts, batch_meta):
        inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        generation_output = model.generate(**inputs, max_new_tokens=15)
        generated_texts = processor.batch_decode(generation_output, skip_special_tokens=True)

        for meta, output in zip(batch_meta, generated_texts):
            sample_index, task = meta
            output_sample = pred[sample_index]
            output_name = "output_" + task
            output_sample[output_name] = output  # Update output for the task
            # No need to reassign output_sample to pred[sample_index] as it's a reference


    for sample_index, sample in enumerate(data_text):
        output_sample = {'text_input_id': sample['text_input_id']}
        for task in tasks:
            prompt_generic = prompts.zeroshot(test_sample=sample, task=task)
            prompt_adept = prompt_generic + '\n'
            image_file_path = os.path.join(image_dir_path, sample['image_id'])
            image = resize_image(get_image(image_file_path))

            prompt_name = 'prompt_' + task
            output_sample[prompt_name] = prompt_generic  # Store the prompt

            batch_images.append(image)
            batch_prompts.append(prompt_adept)
            batch_meta.append((sample_index, task))

            if len(batch_images) == batch_size:
                process_batch(batch_images, batch_prompts, batch_meta)
                batch_images = []
                batch_prompts = []
                batch_meta = []

            '''insert code:
            
            update outputsample with something like this:
            output_sample[output_name] = output
            
            right now, the output does not get added to update_sample and is therefore not saved.'''

        pred.append(output_sample)  # Add the sample to predictions

    if batch_images:
        process_batch(batch_images, batch_prompts, batch_meta)

    return pred

def main(dataset_name, run):

    model, processor = get_model(device)

    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = utils.get_info(dataset_name=dataset_name, model_name=model_name_informal, run=run)

    with open(ds_file_path, 'r') as f:
        data = json.load(f)
    data_text = data['data']

    run_time_inference_start = time.time()
    pred = gen_output(device, data_text, model, processor, image_dir_path, tasks)
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