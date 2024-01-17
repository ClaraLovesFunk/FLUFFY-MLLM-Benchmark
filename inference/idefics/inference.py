import os
import torch
from PIL import Image
import argparse
import json
from transformers import IdeficsForVisionText2Text, AutoProcessor

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import sys
root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_directory)
print(sys.path)
import utils  
import prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name_formal = "HuggingFaceM4/idefics-9b-instruct"
model_name_informal = "idefics"

def get_model(device):
    model = IdeficsForVisionText2Text.from_pretrained(
        pretrained_model_name_or_path= model_name_formal,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_name_formal, cache_dir=CACHE_DIR)
    return model, processor


def gen_idefics_prompts(sample, task, img_path):

    prompt_generic = prompts.zeroshot(test_sample=sample, task=task)
    prompt_idefics = [
        [
            "User: " + prompt_generic,
            img_path,
            "<end_of_utterance>",

            "\nAssistant:",
        ],
    ]

    return prompt_idefics


def gen_output(device, data_text, model, processor, image_dir_path, tasks):
    
    pred = []

    for sample in data_text:   
        output_sample = {'text_input_id': sample['text_input_id']}

        for task in tasks:

            image_file_path = os.path.join(image_dir_path, sample['image_id'])
            prompt_idefics = gen_idefics_prompts(sample, task, image_file_path)

            inputs = processor(prompt_idefics, add_end_of_utterance_token=False, return_tensors="pt").to(device)
            exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

            generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=250, temperature = 0) #check if temperature arg works
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            output_name = "output_" + task
            prompt_name = 'prompt_' + task
            
            output_sample.update({prompt_name: prompt_idefics})
            output_sample.update({output_name: output[0]})

        pred.append(output_sample)

    return pred

def main(dataset_name, run):
    
    model, processor = get_model(device)
  
    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = utils.get_info(dataset_name=dataset_name, model_name=model_name_informal, run=run)

    with open(ds_file_path, 'r') as f:
        data = json.load(f)
    data_text = data['data']

    pred = gen_output(device, data_text, model, processor, image_dir_path, tasks)

    config = {
        'model': model_name_informal,
        'dataset': dataset_name,
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