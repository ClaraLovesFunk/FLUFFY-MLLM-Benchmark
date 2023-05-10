
#%%
from PIL import Image
from PIL import ImageOps
import torch
import json
import pandas as pd
from lavis.models import load_model_and_preprocess
from little_helpers import *



# FLAGS

viz_flag = False


split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'aokvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/'

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'
text_output_file_aokvqa_offcl = results_dir + 'predictions_val.json'



device = "cuda" if torch.cuda.is_available() else "cpu"



with open(text_input_file, 'r') as f:
    X_text = json.load(f)

pred = [] # store pred with all other infos
pred_aokvqa_format = {} # store just question id and generated text



# load model & its processor

model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)



# generate output 
j=0
for i in X_text:
    j +=1 
    print(j)

    # get & prepare test sample

    image_path = get_coco_path('val', i['image_id'], images_input_dir)
    image_raw = Image.open(image_path) 

    if image_raw.mode != 'RGB': # Convert the image to RGB if it's not already
        image_raw = ImageOps.colorize(image_raw, 'black', 'white')

    image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)
    
    # make prompt

    prompt = prompt_construct(test_sample = i,task = 'direct_answer')
    
    # generate text with model

    generated_text = model.generate({"image": image, "prompt": prompt})

    # store output

    i.update({'output': generated_text})
    pred.append(i)

    pred_aokvqa_format = {
        i['question_id']: {
            #'multiple_choice': '<prediction>',
            'direct_answer': generated_text
        }
    }

    # viz

    if viz_flag == True:

        image_raw.show() 
        print(prompt) 
        print(f'generated_text: {generated_text}')
    


# save & reload

with open(text_output_file, 'w') as f:
    json.dump(pred,f)
    
with open(text_output_file_aokvqa_offcl, 'w') as f:
    json.dump(pred_aokvqa_format, f)

with open(text_output_file_aokvqa_offcl, 'r') as f:
    pred_aokvqa_format = json.load(f)

pred_aokvqa_format = pd.DataFrame(pred_aokvqa_format)
display(pred_aokvqa_format)
# %%
