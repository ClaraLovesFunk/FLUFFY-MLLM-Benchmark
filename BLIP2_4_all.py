import json
import pandas as pd
from PIL import Image
from PIL import ImageOps
import json
import pandas as pd
from lavis.models import load_model_and_preprocess
import torch
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"



model_name = 'blip2'
dataset_name = 'aokvqa'
run = 1



# get model & model properties

model_info = ModelInfo(model_name)
lavis_model_type = model_info.get_lavis_model_type()
lavis_name = model_info.get_lavis_name()

model, vis_processors, _ = load_model_and_preprocess(name = lavis_name, model_type = lavis_model_type, is_eval = True, device = device)



# get dataset properties

dataset_info = DatasetInfo(dataset_name)

split = dataset_info.get_split()
images = dataset_info.get_img_dataset()
tasks = dataset_info.get_tasks()



# get paths

dataset_path = os.path.join('datasets', dataset_name, split + '.json')
images_path = os.path.join('datasets', images, split) # split directory gets specified in "get_coco_funciton" #+ '/' + dataset_info.get_split()
experiment_path = os.path.join('experiments', model_name, dataset_name, 'run' + str(run),'output.json')



# get dataset

data_text = dataset(dataset_name, dataset_path).load()



# generate output 

pred = [] # store pred with all other infos

for sample in data_text:

    image = prep_image(images_path, sample, vis_processors)

    for t in dataset_info.task():

        output_task = 'output_' + t
        prompt = prompt_construct(test_sample = sample,task = t)
        output = model.generate({"image": image, "prompt": prompt})
        sample.update({output_task: output})
        pred.append(sample)



with open(experiment_path, 'w') as f: # save preds along with all infos
    json.dump(pred,f)