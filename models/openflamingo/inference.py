import os

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import torch
from open_flamingo import create_model_and_transforms
import time
import argparse
import pandas as pd
import json


import sys
sys.path.append('../../')
import utils 
import prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def eval_model(prompt, image_file):

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4
    )

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.to(device)

    image = utils.load_image(image_file)
    vision_x = image_processor(image).unsqueeze(0).to(device)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    tokenizer.padding_side = "left" 
    lang_x = tokenizer([prompt], return_tensors="pt")
    lang_x.to(device)

    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )

    return tokenizer.decode(generated_text[0])




def predict_dataset(dataset_name, model_path, run, n_ic_samples):
    

    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = utils.get_info(dataset_name = dataset_name, model_name = 'openflamingo', run = run)
    dataset = pd.read_json(ds_file_path) 
    data_list = dataset['data'].tolist()
    
    run_time_inference_start = time.time()
    pred = []

    for test_sample in data_list[:2]:

        output_sample = {'text_input_id': test_sample['text_input_id']}

        for t in tasks:
            prompt = prompts.zeroshot(test_sample, task)
            image_file = os.path.join(image_dir_path, test_sample['image_id'])
            output = eval_model(prompt, image_file)

            output_name = 'output_' + t
            output_sample.update({output_name: output}) 
        pred.append(output_sample)

    run_time_inference_end = time.time()
    run_time_inference = int(run_time_inference_end - run_time_inference_start) / 60
    config = {
        'model': 'openflamingo',
        'dataset': dataset_name,
        'split': split,
        'run': run,
        'run time inference': run_time_inference
    }

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    with open(output_file_path, 'w') as f:
        json.dump(pred, f)
    with open(config_file_path, 'w') as f:
        json.dump(config, f)





if __name__ == "__main__":

    start_time= time.time() ########

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run", type=str, default="1")
    args = parser.parse_args()
    
    predict_dataset(dataset_name = args.dataset, model_path = None, run = args.run)

    end_time= time.time()
    run_time = end_time - start_time
    print(f'runtime: {run_time}')



# source venvs/openflamingo/bin/activate
# cd models/openflamingo
# python inference.py --dataset hateful_memes