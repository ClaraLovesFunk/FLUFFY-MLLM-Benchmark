
#%%
from PIL import Image
#from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json
import pandas as pd
from lavis.models import load_model_and_preprocess
from little_helpers import *

split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'aokvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/'

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'



device = "cuda" if torch.cuda.is_available() else "cpu"




# get data

with open(text_input_file, 'r') as f:
    X_text = json.load(f)

X_y_text = [] # store textual input and output



# load model & its processor

model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)



# generate output 

for i in X_text:


    # get & prepare test sample

    image_path = get_coco_path('val', i['image_id'], images_input_dir)
    image_raw = Image.open(image_path) 
    image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)
    
    # make prompt

    prompt = prompt_construct(test_sample = i,task = 'direct_answer')
    
    # generate text with model

    generated_text = model.generate({"image": image, "prompt": prompt})

    # store output

    i.update({'output': generated_text})
    X_y_text.append(i)

    # viz

    image_raw.show() 
    print(prompt) 
    print(f'generated_text: {generated_text}')
    


# save & reload

with open(text_output_file, 'w') as f:
    json.dump(X_y_text, f)

with open(text_output_file, 'r') as f:
    X_y_text = json.load(f)

X_y_text = pd.DataFrame(X_y_text)
display(X_y_text)
# %%
