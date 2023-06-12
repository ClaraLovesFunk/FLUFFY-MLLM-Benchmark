#%% 
'''
import json
import pandas as pd
from PIL import Image
from PIL import ImageOps
import json
import pandas as pd
from lavis.models import load_model_and_preprocess'''
import torch
from little_helpers import *

device = "cuda" if torch.cuda.is_available() else "cpu"



model_name = 'blip2'
dataset_name = 'okvqa'
run = 1



# get model properties

model_info = ModelInfo(model_name)
lavis_model_type = model_info.get_lavis_model_type()
lavis_name = model_info.get_lavis_name()


# get dataset properties

dataset_info = DatasetInfo(dataset_name)
split = dataset_info.get_split()
dataset_images = dataset_info.get_img()


# get paths

'''dataset_path = 'datasets/' + dataset_name +'/' + split + '.json'
images_path = 'datasets/' + dataset_images # split directory gets specified in "get_coco_funciton" #+ '/' + dataset_info.get_split()
results_path = 'experiments/' + model_name + '/' + dataset_name + '/run' + str(run) + '/' + 'output.json'
'''

dataset_path = os.path.join('datasets', dataset_name, split + '.json')
images_path = os.path.join('datasets', dataset_images) # split directory gets specified in "get_coco_funciton" #+ '/' + dataset_info.get_split()
results_path = os.path.join('experiments', model_name, dataset_name, 'run' + str(run),'output.json')

print(dataset_path)
print(images_path)
print(results_path)

# load textual data

with open(dataset_path, 'r') as f:
    X_text = json.load(f)

X_text = X_text['questions']



# load model & its processor

model, vis_processors, _ = load_model_and_preprocess(name = lavis_name, model_type = lavis_model_type, is_eval = True, device = device)



# generate output 

pred = [] # store pred with all other infos
for i in X_text:


    # get & prepare test sample

    image_path =  os.path.join(images_path, f"{split}", f"{i['image_id']:012}.jpg")

    image_raw = Image.open(image_path) 

    if image_raw.mode != 'RGB': 
        image_raw = ImageOps.colorize(image_raw, 'black', 'white')

    image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)
    
    # make prompt

    prompt_da = prompt_construct(test_sample = i,task = 'direct_answer')
    #prompt_MC = prompt_construct(test_sample = i,task = 'MC_answer')
    
    
    # generate text with model

    generated_da = model.generate({"image": image, "prompt": prompt_da})
    #generated_MC = model.generate({"image": image, "prompt": prompt_MC})

    # store output

    i.update({'output_da': generated_da})
    #i.update({'output_MC': generated_MC})
    pred.append(i)



# save 

with open(results_path, 'w') as f: # save preds along with all infos
    json.dump(pred,f)
    


# %%
