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





def get_model(model_name):


    model_info = ModelInfo(model_name)
    lavis_model_type = model_info.get_lavis_model_type()
    lavis_name = model_info.get_lavis_name()

    model, vis_processors, _ = load_model_and_preprocess(name = lavis_name, model_type = lavis_model_type, is_eval = True, device = device)
    print('model loaded')

    return model, vis_processors




def get_dataset_text(dataset_name):


    dataset_info = DatasetInfo(dataset_name)

    split = dataset_info.get_split()

    dataset_file_path = os.path.join('datasets', dataset_name, split + '.json')
    
    data_text = dataset(dataset_name, dataset_file_path).load()

    print('dataset loaded')

    return data_text




def gen_output(device, dataset_name, model, vis_processors, prep_image, prompt_construct):


    dataset_info = DatasetInfo(dataset_name)

    split = dataset_info.get_split() # find split on which we can eval model (ideally test, otherwise val)
    images = dataset_info.get_img_dataset() # find corresponding image dataset
    tasks = dataset_info.get_tasks() # find tasks associated with current dataset

    images_dir_path = os.path.join('datasets', images, split)

    pred = [] 

    for sample in data_text:

        image = prep_image(device, images_dir_path, sample, vis_processors)

        for t in tasks:

            prompt = prompt_construct(test_sample = sample,task = t)
            output = model.generate({"image": image, "prompt": prompt})

            output_task = 'output_' + t
            sample.update({output_task: output})
            pred.append(sample)

    print('output generated')

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
dataset_name = ['aokvqa', 'okvqa']
run = [1]



for m in model_name:

    model, vis_processors = get_model(m)

    for ds in dataset_name:

        data_text = get_dataset_text(ds)

        for r in run:

            print('about to run pred')
            pred = gen_output(device = device, dataset_name = ds, model = model, vis_processors = vis_processors, prep_image = prep_image, prompt_construct = prompt_construct)
            print('just ran pred')
            save_output(pred = pred, model_name = m, dataset_name = ds, run = r, check_create_experiment_dir = check_create_experiment_dir)
            
            print('\n')
            print('\n')
            print('\n')
            print('YEAAAAAAHHHHHHHHHHHH')
       