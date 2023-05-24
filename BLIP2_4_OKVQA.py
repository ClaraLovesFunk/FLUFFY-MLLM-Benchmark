#%% 

import json
import pandas as pd
from PIL import Image
from PIL import ImageOps
import torch
import json
import pandas as pd
from lavis.models import load_model_and_preprocess

from little_helpers import *


split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'okvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/'

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'




device = "cuda" if torch.cuda.is_available() else "cpu"




# load textual data

with open(text_input_file, 'r') as f:
    X_text = json.load(f)

X_text = X_text['questions']



# load model & its processor

#model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)



# generate output 

pred = [] # store pred with all other infos
for i in X_text:


    # get & prepare test sample

    image_path = get_coco_path('train', i['image_id'], images_input_dir)
    print(image_path)
    image_raw = Image.open(image_path) 
    print('found')

    if image_raw.mode != 'RGB': 
        image_raw = ImageOps.colorize(image_raw, 'black', 'white')

    #image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)
    
    # make prompt

    prompt_da = prompt_construct(test_sample = i,task = 'direct_answer')
    #prompt_MC = prompt_construct(test_sample = i,task = 'MC_answer')
    
    
    # generate text with model

    generated_da ='test'
    #generated_da = model.generate({"image": image, "prompt": prompt_da})
    #generated_MC = model.generate({"image": image, "prompt": prompt_MC})

    # store output

    i.update({'output_da': generated_da})
    #i.update({'output_MC': generated_MC})
    pred.append(i)



# save 

with open(text_output_file, 'w') as f: # save preds along with all infos
    json.dump(pred,f)
    


# %%
