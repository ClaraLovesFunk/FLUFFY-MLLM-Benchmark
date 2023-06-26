#%% 

import json
import pandas as pd
from PIL import Image
from IPython.display import display, HTML

from utils import *



images_dir = 'datasets/coco2017'
input_file = 'datasets/aokvqa/val.json'
output_file = 'experiments/blip2/aokvqa/run1/output.json' 
example_file = 'experiments/blip2/aokvqa/run1/examples.json' 
split_sec = 'all'

flag_show_inputs = False
flag_show_outputs = True
flag_show_outputs_examples = True

n_inputs = 3
n_outputs = 25
n_good_examples = 3
n_bad_examples = 3


with open(input_file, 'r') as f:
    data_input = json.load(f)

with open(output_file, 'r') as f:
    data_output = json.load(f)

with open(example_file, 'r') as f:
    data_output_examples = json.load(f)




if flag_show_inputs:

    print('Inputs')
    data_samples_input = data_input[:n_inputs]
    data_samples_output = data_samples_input # dummy variable to fill in nec function argument
    data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output, split_sec,images_dir, tasks = ['direct answer', 'multiple choice'])


if flag_show_outputs:

    print('Outputs')
    data_samples_input = data_input[:n_outputs]
    data_samples_output = data_output[:n_outputs]
    data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output, split_sec,images_dir, tasks = ['direct answer', 'multiple choice'])


if flag_show_outputs_examples:

    print('Correct Output :)))')

    # load all example indice
    examples = data_output_examples['direct answer'] #examples only for easy direct answers

    id_good_examples = []
    id_bad_examples = []

    for key, value in examples.items():
        if value == 1:
            id_good_examples.append(key)
        if value == 0:
            id_bad_examples.append(key)

    # limit number of examples
    id_good_examples = id_good_examples[:n_good_examples]
    id_bad_examples = id_bad_examples[:n_bad_examples]

    # good examples

    data_samples_input = [d for d in data_input if d['question_id'] in id_good_examples]
    data_samples_output = [d for d in data_output if d['text_input_id'] in id_good_examples]

    data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output, split_sec,images_dir, tasks = ['direct answer', 'multiple choice'])

    
    # bad examples
    
    data_samples_input = [d for d in data_input if d['question_id'] in id_bad_examples]
    data_samples_output = [d for d in data_output if d['text_input_id'] in id_bad_examples]

    data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output, split_sec,images_dir, tasks = ['direct answer', 'multiple choice'])


# %%
