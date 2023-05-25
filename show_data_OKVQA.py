#%% 

import json
import pandas as pd

from little_helpers import *



images_dir = 'datasets/coco2017'
labels_file = 'datasets/okvqa/val_labels.json'
text_input_file = 'datasets/okvqa/val.json'
output_file = 'experiments/blip2/okvqa/output.json' 
example_file = 'experiments/blip2/okvqa/examples.json' 
split_sec = 'val'

n_inputs = 3
n_outputs = 3
n_good_exampes = 3
n_bad_exampes = 3

FLAG_show_input = False
FLAG_show_output = True
FLAG_show_labels = False




if FLAG_show_input:

    with open(text_input_file, 'r') as f:
        input = json.load(f)

    input = input['questions']

    print('Inputs')
    data_samples = input[:n_inputs]
    data_incl_image = add_imgs_text_data(data_samples, split_sec = 'all',images_dir=images_dir)




if FLAG_show_output:

    # get model output

    with open(output_file, 'r') as f:
        output = json.load(f)

    print('Outputs')

    # get annotation data

    with open(labels_file, 'r') as f:
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
    data_incl_image = add_imgs_text_data(data_samples, split_sec = 'all',images_dir=images_dir)
   



if FLAG_show_labels:

    with open(labels_file, 'r') as f:
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
    display(labels[:10])




# %%
