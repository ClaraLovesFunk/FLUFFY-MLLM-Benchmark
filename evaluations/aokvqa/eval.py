import pandas as pd
from evaluations.aokvqa.eval_vqa.vqa import VQA 
from evaluations.aokvqa.eval_vqa.vqaEval import VQAEval
import json
import evaluations.utils_eval as utils_eval

VALID_ANS_VALUES = "sample-dependent"
TASK_NAME = "multiple choice (aokvqa)"
POS_LABEL = ""
label_name = "correct_choice"
output_name = "output_multiple choice (aokvqa)"
dataset_name = "aokvqa"



def eval_aokvqa(input, output, task, strict=True): # MESSING WITH SOURCE CODE: replaced the variable "multiple_choice" because ultimately it means the same as task type (direct answer/MC)

    '''
    aokvqa's method of computing accuracy by regarding how many of their proposed direct_answers the predictions matches
    (the answer that the aokvqa authors want the most are written the most often in a list of multiple possible direct answers,
    while a just satisfactory answer is written less)
    
    accuracy per instance can be values between 0-1, not just 0 and 1'''

    if task == 'direct answer (aokvqa)': # MESSING WITH SOURCE CODE

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


#output = utils_eval.load_data('experiments/blip2/aokvqa/run1/output_aux_hard.json')
#output = utils_eval.load_data(soft.json')

    


def evaluate_aokvqa(ds_text_file_path, experiment_output_file_path, model, mode):

    modes = ['soft', 'hard']
    for mode in modes:
        
        print(mode)

        # load and preprocess data and get valid answer scores
        input_original = utils_eval.load_data('datasets/aokvqa/ds_original.json')
        input_benchmark = utils_eval.load_data('datasets/aokvqa/ds_benchmark.json')
        output_original = utils_eval.load_data('experiments/blip2/aokvqa/run1/output.json')
        output_transformed_4_eval_mode = utils_eval.load_data('experiments/blip2/aokvqa/run1/output_aux_' + mode + '.json')

        valid_ans_ratio_dict = {} 
        scores_dict = {}
        examples_dict = {}
        
        y_pred_dict_all_tasks = {}
        tasks = ["direct answer (aokvqa)", "multiple choice (aokvqa)"] # labels for da: 'nZANMFWTuwNWznuT9RBNXr': ['parking', 'watch', 'pedestrians', 'people', 'pedestrian', 'pedestrians', 'pedstrains', 'signal board', 'pedestrians', 'warning']
        task2label_name = {
            "direct answer (aokvqa)": "correct_direct_answer_short",
            "multiple choice (aokvqa)": "correct_multiple_choice_answer"
        }
        for task in tasks:
            label_name = task2label_name[task]
            labels = utils_eval.get_id_2_label_dict(input_benchmark, label_name, dataset_name) 
            
            valid_ans_ratio, y_pred, y_true, y_pred_dict, y_true_dict = utils_eval.get_clean_valid_preds_trues(
                output = output_original, 
                output_name = "output_"+ task, 
                VALID_ANS_VALUES = VALID_ANS_VALUES, 
                labels = labels, 
                model = model, 
                dataset_name = dataset_name, 
                data_text = input_benchmark, 
                mode = mode, 
                task = task)
            y_pred_dict_all_tasks[task] = y_pred_dict
            valid_ans_ratio_dict[task] = valid_ans_ratio
        
        # transform output according to evaluation modus
        utils_eval.make_output_aux_eval(
            output_original_path = 'experiments/blip2/aokvqa/run1/output.json',
            y_pred_dict_all_tasks = y_pred_dict_all_tasks,
            mode = mode, 
            tasks = tasks)

        # get scores
        for task in tasks:
            acc, ex = eval_aokvqa(input = input_original, output=output_transformed_4_eval_mode, task = task, strict=True)
            scores_dict[task] = {'accuracy': acc}
            examples_dict[task] = ex


    return scores_dict, examples_dict, valid_ans_ratio_dict



