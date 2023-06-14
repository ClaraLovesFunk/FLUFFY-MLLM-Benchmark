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

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"



model_name = 'blip2'
dataset_name = 'aokvqa'
run = 1






# get model & model properties

model_info = ModelInfo(model_name)
lavis_model_type = model_info.get_lavis_model_type()
lavis_name = model_info.get_lavis_name()

model, vis_processors, _ = load_model_and_preprocess(name = lavis_name, model_type = lavis_model_type, is_eval = True, device = device)
print('model loaded')


# get dataset properties

dataset_info = DatasetInfo(dataset_name)

split = dataset_info.get_split()
images = dataset_info.get_img_dataset()
tasks = dataset_info.get_tasks()
print('dataset loaded')



# get paths

dataset_file_path = os.path.join('datasets', dataset_name, split + '.json')
images_dir_path = os.path.join('datasets', images, split) # split directory gets specified in "get_coco_funciton" #+ '/' + dataset_info.get_split()
experiment_dir_path = os.path.join('experiments', model_name, dataset_name, 'run' + str(run))
experiment_output_file_path = os.path.join(experiment_dir_path, 'output.json')



# check for/make experiment dir

check_create_experiment_dir(experiment_dir_path)



# get dataset

data_text = dataset(dataset_name, dataset_file_path).load()



# generate output 

pred = [] # store pred with all other infos

for sample in data_text:

    image = prep_image(device, images_dir_path, sample, vis_processors)

    for t in tasks:

        prompt = prompt_construct(test_sample = sample,task = t)
        output = model.generate({"image": image, "prompt": prompt})

        output_task = 'output_' + t
        sample.update({output_task: output})
        pred.append(sample)


print('input processed')
with open(experiment_output_file_path, 'w') as f: # save preds along with all infos
    json.dump(pred,f)