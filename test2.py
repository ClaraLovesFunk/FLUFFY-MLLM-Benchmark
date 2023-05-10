#%% 

import os
import json
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64
from little_helpers import *



images_dir = 'datasets/coco2017'
flag_show_outputs = True

if flag_show_outputs:
    labels_file = 'experiments/blip2/aokvqa/output.json' 
else:
    labels_file = 'datasets/aokvqa/val.json'





with open(labels_file, 'r') as f:
    labels_data = json.load(f)

first_five_labels = labels_data[:10]

data = []
desired_row_indices = [3, 4, 15]
undesired_row_indices = [0, 1, 2]


# GOOD EXAMPLES

for index in desired_row_indices:

    label_info = labels_data[index]
    image_path = get_coco_path('val', label_info['image_id'], images_dir)
    img = Image.open(image_path)
    img.thumbnail((100, 100))
    dict_fulldata = {'image': image_to_html(img)}
    dict_fulldata.update(label_info)
    dict_fulldata.update({'img_path': image_path})
    data.append(dict_fulldata)
    
pred_good = pd.DataFrame(data)

# drop irrelevant info
pred_good = pred_good.drop(['split', 'image_id', 'question_id', 'rationales', 'img_path'],axis=1)

print('Good Examples:')
pd.set_option('display.max_colwidth', None)
display(HTML(pred_good.to_html(escape=False)))



# BAD EXAMPLES
data = []
for index in undesired_row_indices:

    label_info = labels_data[index]
    image_path = get_coco_path('val', label_info['image_id'], images_dir)
    img = Image.open(image_path)
    img.thumbnail((100, 100))
    dict_fulldata = {'image': image_to_html(img)}
    dict_fulldata.update(label_info)
    dict_fulldata.update({'img_path': image_path})
    data.append(dict_fulldata)
    
pred_bad = pd.DataFrame(data)

# drop irrelevant info
pred_bad = pred_bad.drop(['split', 'image_id', 'question_id', 'rationales', 'img_path'],axis=1)

print('Bad Examples:')
pd.set_option('display.max_colwidth', None)
display(HTML(pred_bad.to_html(escape=False)))
# %%
