import pandas as pd
from evaluations.aokvqa.eval_vqa.vqa import VQA 
from evaluations.aokvqa.eval_vqa.vqaEval import VQAEval
import json
import evaluations.utils_eval as utils_eval

VALID_ANS_VALUES = ['0', '1', '2', '3']
TASK_NAME = "multiple choice (aokvqa)"
#POS_LABEL = ""
label_name = "correct_choice"
output_name = "output_multiple choice (aokvqa)"
dataset_name = "aokvqa"



def eval_aokvqa(input, output, task, strict=True): # MESSING WITH SOURCE CODE: replaced the variable "multiple_choice" because ultimately it means the same as task type (direct answer/MC)

    '''
    aokvqa's method of computing accuracy by regarding how many of their proposed direct_answers the predictions matches
    (the answer that the aokvqa authors want the most are written the most often in a list of multiple possible direct answers,
    while a just satisfactory answer is written less)
    
    accuracy per instance can be values between 0-1, not just 0 and 1'''

    if task == 'direct answer': # MESSING WITH SOURCE CODE

        multiple_choice = False # MESSING WITH SOURCE CODE

    else:

        multiple_choice = True # MESSING WITH SOURCE CODE

    if isinstance(input, list):  # checks if dataset is of type list; if yes, it transforms it into a dict with question id as key
        input = { input[i]['question_id'] : input[i] for i in range(len(input)) }
        
    # If the preds is a list, transform it into a dictionary with question id as key
    if isinstance(output, list):  
        output = { output[i]['text_input_id'] : output[i] for i in range(len(output)) }
       
    if multiple_choice is False: # if we look at direct answer task, we only look at instances with easy direct answers (or not difficult_direct_answers)
        input = {k:v for k,v in input.items() if v['difficult_direct_answer'] is False}

    if strict: #dataset_qids is a subset of preds_qids ???
        dataset_qids = set(input.keys())
        preds_qids = set(output.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []
    examples = {}

    for q in input.keys(): # for each question id q
        if q not in output.keys(): #if we didnt generate a pred for a q in the dataset, we append 0.0 to the acc array
            acc.append(0.0)
            continue
        if multiple_choice:
            pred = output[q]['output_multiple choice (aokvqa)']#[0]
        else: 
            pred = output[q]['output_direct answer (aokvqa)']#[0]
        
        choices = input[q]['choices']
        direct_answers = input[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:

            '''
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
                '''
            correct_choice_idx = input[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
            
            # save (in)correct examples
            if pred == choices[correct_choice_idx]:
                examples.update({q:1})
            else:
                examples.update({q:0})  
                        
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])

            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

            # save (in)correct examples
            if num_match >= 1:
                examples.update({q:1})
            else:
                examples.update({q:0})

    acc = sum(acc) / len(acc) #* 100

    return acc, examples




def evaluate_aokvqa(ds_text_file_path, experiment_output_file_path, model):

    with open('datasets/aokvqa/ds_original.json', 'r') as f: # load original file (not restructured one for our benchmark)
        data_text = json.load(f)
    output = utils_eval.load_data(experiment_output_file_path)
    with open('datasets/aokvqa/ds_benchmark.json', 'r') as f: # to get labels use the file that was reformatted for the benchmark
        data_text_labels = json.load(f)
    labels = utils_eval.get_id_2_label_dict(data_text_labels, label_name, dataset_name)

    scores = {}
    examples = {}
    
    # direct answer
    acc_da, ex = eval_aokvqa(input = data_text, output=output, task = 'direct answer', strict=True)
    scores['direct answer'] = {'accuracy': acc_da}
    examples['direct answer'] = ex

    # multiple choice
    acc_MC, ex = eval_aokvqa(input = data_text, output=output, task = 'multiple choice', strict=True)
    scores['multiple choice'] = {'accuracy': acc_MC}
    examples['multiple choice'] = ex
    valid_ans_ratio, _, _ = utils_eval.get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset_name, data_text)
    valid_ans_ratio = {'multiple choice': valid_ans_ratio}


    return scores, examples, valid_ans_ratio