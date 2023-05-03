#%%
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json
import pandas as pd
from little_helpers import *

images_dir = 'datasets/coco2017'
labels_file = "datasets/aokvqa/val.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
# load model & its processor

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16
    )
model.to(device)'''


# get data

with open(labels_file, 'r') as f:
    X_text = json.load(f)


# store textual input and output

'''X_y_text = pd.DataFrame(X_text)
X_y_text = X_y_text.drop(index=X_y_text.index)
X_y_text['output'] = []'''

X_y_text = []


# generate output 

for i in X_text[:5]:

    image_path = get_coco_path('val', i['image_id'], images_dir)
    
    image = Image.open(image_path) 
    image.show() 

    prompt = i['question']
    print(prompt) ##### MORE EXPLICIT Q?   
    '''
    inputs = processor(images=image_path, text=prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    '''
    # store output
    generated_text = 'test'
    i.update({'output': generated_text})
    #X_y_text = pd.concat([X_y_text, i], ignore_index=True)
    X_y_text.append(i)


#X_y_text = pd.DataFrame(X_y_text)
#X_y_text = X_y_text.to_dict()
#X_y_text = dict(X_y_text)
#print(type(X_y_text))

with open('my_file.json', 'w') as f:
    json.dump(X_y_text, f)

with open('my_file.json', 'r') as f:
    X_y_text = json.load(f)

X_y_text = pd.DataFrame(X_y_text)
display(X_y_text)
# %%
