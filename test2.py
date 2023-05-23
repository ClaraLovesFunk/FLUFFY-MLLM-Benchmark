import json
import pandas as pd
from PIL import Image
from IPython.display import display, HTML

from little_helpers import *



images_dir = 'datasets/coco2017'
input_file = 'datasets/aokvqa/val.json'
output_file = 'experiments/blip2/aokvqa/output.json' 
example_file = 'experiments/blip2/aokvqa/examples.json' 
split_sec = 'val'

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




def eval_aokvqa(dataset, preds, multiple_choice=False, strict=True):

    if isinstance(dataset, list):  # checks if dataset is of type list; if yes, it transforms it into a dict with question id as key
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    # If the preds is a list, transform it into a dictionary with question id as key
    if isinstance(preds, list):  
        preds = { preds[i]['question_id'] : preds[i] for i in range(len(preds)) }
       
    if multiple_choice is False: # if we look at direct answer task, we only look at instances with easy direct answers (or not difficult_direct_answers)
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    if strict: #dataset_qids is a subset of preds_qids ???
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys(): # for each question id q
        if q not in preds.keys(): #if we didnt generate a pred for a q in the dataset, we append 0.0 to the acc array
            acc.append(0.0)
            continue

        pred = preds[q]['output'][0]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            print(pred)
            print(direct_answers)
            print(num_match)

            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    return acc


acc = eval_aokvqa(dataset = data_input, preds=data_output, multiple_choice=False, strict=True)

print(acc)