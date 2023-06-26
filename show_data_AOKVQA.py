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
flag_show_outputs_examples = False

n_inputs = 3
n_outputs = 3
n_good_exampes = 3
n_bad_exampes = 3



# get textual data

if flag_show_inputs or flag_show_outputs:
    with open(input_file, 'r') as f:
        data_input = json.load(f)

if flag_show_outputs or flag_show_outputs_examples:
    with open(output_file, 'r') as f:
        data_output = json.load(f)

if flag_show_outputs_examples:
    with open(example_file, 'r') as f:
        data_output_examples = json.load(f)



# show textual and image data jointly

if flag_show_inputs:

    print('Inputs')
    data_samples_input = data_input[:n_inputs]
    data_samples_output = data_samples_input # dummy variable to fill in nec function argument
    data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output, split_sec,images_dir)


if flag_show_outputs:

    print('Outputs')
    data_samples_input = data_input[:n_inputs]
    data_samples_output = data_output[:n_outputs]
    data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output, split_sec,images_dir, tasks = ['direct answer', 'multiple choice'])









if flag_show_outputs_examples:

    print(':))) Good Examples')
    good_example_index = data_output_examples["example_indice_da"]['good pred'][:n_good_exampes]
    data_samples = [data_output[i] for i in good_example_index]
    data_incl_image = add_imgs_text_data(data_samples, split_sec,images_dir)

    print(':_((( Bad Examples')
    good_example_index = data_output_examples["example_indice_da"]['bad pred'][:n_good_exampes]
    data_samples = [data_output[i] for i in good_example_index]
    data_incl_image = add_imgs_text_data(data_samples, split_sec,images_dir)




# %%
