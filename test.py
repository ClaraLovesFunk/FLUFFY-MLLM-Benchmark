#python evaluation/eval_predictions.py --aokvqa-dir ${AOKVQA_DIR} --split val --preds ./predictions_val.json



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
    # If the dataset is a list, transform it into a dictionary with question id as key
    if isinstance(dataset, list):  
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    # If the preds is a list, transform it into a dictionary with question id as key
    if isinstance(preds, list):  
        preds = { preds[i]['question_id'] : preds[i] for i in range(len(preds)) }

    # If we look at direct answer task, filter out instances with difficult direct answers
    if multiple_choice is False: 
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    # If in strict mode, assert that all question ids in the dataset must also be present in the predictions
    if strict: 
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # Prepare an empty list to hold accuracy values for each question
    acc = []

    # For each question in the dataset
    for q in dataset.keys(): 
        # If there's no prediction for a question in the dataset, append 0.0 to the accuracy list
        if q not in preds.keys(): 
            acc.append(0.0)
            continue

        # For each question that does have a prediction, extract the prediction, the choices and the direct answers
        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        # If in multiple choice setting
        if multiple_choice:
            # In strict mode, assert that the prediction must be a valid choice
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            # Get the index of the correct choice
            correct_choice_idx = dataset[q]['correct_choice_idx']
            # Add to accuracy list whether the prediction matches the correct choice (1.0 for correct, 0.0 for incorrect)
            acc.append( float(pred == choices[correct_choice_idx]) )
        # If in direct answer setting
        else:
            # Count the number of direct answers that match the prediction
            num_match = sum([pred == da for da in direct_answers])
            # Calculate accuracy score for this question (maximum 1.0, minimum 0.0)
            vqa_acc = min(1.0, num_match / 3.0)
            # Add the accuracy score to the list
            acc.append(vqa_acc)

    # Calculate the average accuracy across all questions, and multiply by 100 to convert to percentage
    acc = sum(acc) / len(acc) * 100

    # Return the average accuracy
    print('acc')
    return acc

acc = eval_aokvqa(dataset = data_input, preds=data_output, multiple_choice=False, strict=True)

print(acc)