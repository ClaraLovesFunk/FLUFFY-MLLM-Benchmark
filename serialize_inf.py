import json
import pandas as pd
from PIL import Image
from PIL import ImageOps
import json
import pandas as pd
from lavis.models import load_model_and_preprocess
import torch
import os
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"





def get_model(model_name, device):


    model_info = ModelInfo(model_name)
    lavis_model_type = model_info.get_lavis_model_type()
    lavis_name = model_info.get_lavis_name()

    model, vis_processors, _ = load_model_and_preprocess(name = lavis_name, model_type = lavis_model_type, is_eval = True, device = device)
    print('model loaded')

    return model, vis_processors




def get_dataset_text(dataset_name):


    dataset_info = DatasetInfo(dataset_name)

    text_dataset_split = dataset_info.get_text_dataset_split()

    dataset_file_path = os.path.join('datasets', dataset_name, text_dataset_split + '.json')
    
    data_text = dataset(dataset_name, dataset_file_path).load()

    print('dataset loaded')

    return data_text




def gen_output(device, dataset_name, data_text, model, vis_processors, prep_image, prompt_construct):



    dataset_info = DatasetInfo(dataset_name)

    img_dataset_split = dataset_info.get_img_dataset_split() # find split on which we can eval model (ideally test, otherwise val)
    image_dataset_name = dataset_info.get_img_dataset() # find corresponding image dataset
    tasks = dataset_info.get_tasks() # find tasks associated with current dataset

    

    pred = [] 

    for t in tasks:

        output_task = 'output_' + t

        for sample in data_text:

            #image = prep_image(device, images_dir_path, sample, vis_processors)
            # prep image
            images_dir_path = os.path.join('datasets', image_dataset_name, img_dataset_split)

            image_file_path =  os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")
            image_raw = Image.open(image_file_path) 
            
            if image_raw.mode != 'RGB': 
                image_raw = ImageOps.colorize(image_raw, 'black', 'white')

            image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)

            # make prompt
            prompt = prompt_construct(test_sample = sample,task = t)

            # generate output
            output = model.generate({"image": image, "prompt": prompt})

            sample.update({output_task: output})
            pred.append(sample)

        print(f'output generated for task "{t}"')

    return pred



def save_output(pred, model_name, dataset_name, run, check_create_experiment_dir):

    experiment_dir_path = os.path.join('experiments', model_name, dataset_name, 'run' + str(run))
    experiment_output_file_path = os.path.join(experiment_dir_path, 'output.json')

    check_create_experiment_dir(experiment_dir_path) 

    with open(experiment_output_file_path, 'w') as f: 
        json.dump(pred,f)

    print('save output')






##################################################################

model_name = ['blip2']
dataset_name = ['okvqa', 'aokvqa']
run = [1]



for m in model_name:

    model, vis_processors = get_model(model_name = m, device = device)

    for ds in dataset_name:

        data_text = get_dataset_text(ds)

        for r in run:

            pred = gen_output(device = device, dataset_name = ds, data_text = data_text, model = model, vis_processors = vis_processors, prep_image = prep_image, prompt_construct = prompt_construct)
            save_output(pred = pred, model_name = m, dataset_name = ds, run = r, check_create_experiment_dir = check_create_experiment_dir)
            
            print('\n')
            print('\n')
            print('\n')
            print('YEAAAAAAHHHHHHHHHHHH')
       