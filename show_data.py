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

first_five_labels = labels_data[:5]

data = []

for label_info in first_five_labels:

    image_path = get_coco_path('val', label_info['image_id'], images_dir)

    img = Image.open(image_path)
    img.thumbnail((100, 100))
    
    # create df with all relevant info
    dict_fulldata = {'image': image_to_html(img)}
    dict_fulldata.update(label_info)
    dict_fulldata.update({'img_path': image_path})

    data.append(dict_fulldata)
    
df = pd.DataFrame(data)
pd.set_option('display.max_colwidth', None)
display(HTML(df.to_html(escape=False)))

# %%
