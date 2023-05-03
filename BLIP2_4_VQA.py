#%%
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json
import pandas as pd
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



#'''
# load model & its processor

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16
    )
model.to(device)#'''



# generate output 

for i in X_text[:5]:

    image_path = get_coco_path('val', i['image_id'], images_input_dir)
    image = Image.open(image_path) 
    image.show() 

    prompt = "Question: " + i['question'] + " Answer:"
    print(prompt) ##### MORE EXPLICIT Q?   
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f'generated_text: {generated_text}')
    
    
    # store output


    i.update({'output': generated_text})
    X_y_text.append(i)


with open(text_output_file, 'w') as f:
    json.dump(X_y_text, f)

with open(text_output_file, 'r') as f:
    X_y_text = json.load(f)

X_y_text = pd.DataFrame(X_y_text)
display(X_y_text)
# %%
