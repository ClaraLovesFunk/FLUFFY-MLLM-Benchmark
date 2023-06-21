#%% 

import json
import pandas as pd

from utils import *



# VARIABLES

model_sec = 'blip2'

dataset_sec = 'aokvqa'

run = 'run1'

images_sec = 'coco2017'
split_images_sec = 'all'
split_text_sec = 'val'

dataset_root_dir = 'datasets'

n_inputs = 3
n_outputs = 3
n_good_exampes = 3
n_bad_exampes = 3

FLAG_show_input = True
FLAG_show_output = False
FLAG_show_labels = True



# PATHS

dir_input_images = dataset_root_dir + '/'+ images_sec 
dir_input_text = dataset_root_dir + '/'+ dataset_sec +'/'
dir_results = 'experiments/' + model_sec + '/' + dataset_sec + '/'    # add run

file_input_text = dir_input_text + '/'+ split_text_sec + '.json'
file_output_text = dir_results + 'output.json'
file_labels_text = dataset_root_dir + '/' + dataset_sec + '/' + split_text_sec + '_labels.json'




print(f'Peep into {dataset_sec}')


if FLAG_show_input:

    with open(file_input_text, 'r') as f:
        input = json.load(f)

    input = input['questions']
    
    print('Inputs')
    data_samples = input[:n_inputs]
    data_incl_image = add_imgs_text_data(data_samples, split_sec = 'all',images_dir=dir_input_images)




if FLAG_show_output:

    # get model output

    with open(file_output_text, 'r') as f:
        output = json.load(f)

    print('Outputs')

    # get annotation data

    with open(file_labels_text, 'r') as f:
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


    # extract labels by 1st annotator

    for item in output:
        question_id = item['question_id']
        
        filtered_labels = labels[labels['question_id'] == question_id]
        
        filtered_rows = filtered_labels[filtered_labels['answer_id'] == 1]
        
        if len(filtered_rows) > 0:

            answer_value = filtered_rows.iloc[0]['answer']
            item['answer'] = answer_value
        else:
            item['answer'] = None


    # show outputs with labels

    data_samples = output[:n_outputs]
    data_incl_image = add_imgs_text_data(data_samples, split_sec = 'all',images_dir=dir_input_images)
   



if FLAG_show_labels:

    with open(file_labels_text, 'r') as f:
        labels = json.load(f)

    labels = labels['annotations']

    flat_data = []

    for l in labels:

        for answer in l['answers']:

            entry = l.copy()  
            entry.pop('answers')  
            entry.update(answer)  
            flat_data.append(entry)

    print('Labels')
    labels = pd.DataFrame(flat_data)
    display(labels[:10])




# %%
