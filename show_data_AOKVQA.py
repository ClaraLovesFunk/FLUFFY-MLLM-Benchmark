#%% 

import json
import pandas as pd
from PIL import Image
from IPython.display import display, HTML

from little_helpers import *



images_dir = 'datasets/coco2017'
input_file = 'datasets/aokvqa/val.json'
output_file = 'experiments/blip2/aokvqa/output.json' 
example_file = 'experiments/blip2/aokvqa/examples.json' 
split_sec = 'all'

flag_show_inputs = True
flag_show_outputs = True
flag_show_outputs_examples = True

n_inputs = 3
n_outputs = 3
n_good_exampes = 3
n_bad_exampes = 3



# get textual data

if flag_show_inputs:
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
    data_samples = data_input[:n_inputs]
    data_incl_image = add_imgs_text_data(data_samples, split_sec,images_dir)


if flag_show_outputs:

    print('Outputs')
    data_samples = data_output[:n_outputs]
    data_incl_image = add_imgs_text_data(data_samples, split_sec,images_dir)


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
