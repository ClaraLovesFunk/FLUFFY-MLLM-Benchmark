#%% 

import json
import pandas as pd

from little_helpers import *



images_dir = 'datasets/coco2017'
input_file = 'datasets/okvqa/mscoco_val2014_annotations.json'
q_file = 'datasets/okvqa/OpenEnded_mscoco_val2014_questions.json'
output_file = 'experiments/blip2/aokvqa/output.json' 
example_file = 'experiments/blip2/aokvqa/examples.json' 
split_sec = 'val'

flag_show_inputs = True
flag_show_outputs = False
flag_show_outputs_examples = False

n_inputs = 3
n_outputs = 3
n_good_exampes = 3
n_bad_exampes = 3




# INPUT DATA

with open(q_file, 'r') as f:
    input = json.load(f)

input = input['questions']

input = pd.DataFrame(input)

display(input[:20])



# HUMAN ANNOTATIONS

with open(input_file, 'r') as f:
    labels = json.load(f)

labels = labels['annotations']

flat_data = []

for l in labels:

    for answer in l['answers']:

        entry = l.copy()  
        entry.pop('answers')  
        entry.update(answer)  
        flat_data.append(entry)

labels = pd.DataFrame(flat_data)
display(labels[:20])
# %%
