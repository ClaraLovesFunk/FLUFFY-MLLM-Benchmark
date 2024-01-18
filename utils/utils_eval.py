import re
import string
import json
from sklearn import metrics
import os
import .utils as utils
import math
import shutil
import csv

from .data_loading import load_data

from .path_config import get_paths

from .data_transformation import get_id_2_label_dict
from .data_transformation import extract_answer
from .data_transformation import get_clean_valid_preds_trues























def compute_standard_metrics(y_true, y_pred, pos_label, average='binary', zero_division=0, flag_only_acc = False, dataset_name = None, task = None):
    
    '''
    compute metrics
    '''
    if y_pred == []: 
        '''
        if no valid predictions were made, model cannot be evaluated
        '''
        invalid_ans = float('nan')
        scores = {
            'accuracy': invalid_ans, 
            'precision': invalid_ans, 
            'recall': invalid_ans, 
            'f1': invalid_ans}

    elif y_pred != []:
        '''
        compute metrics, if valid predictions were made
        '''
        if flag_only_acc == True:
            if dataset_name == 'aokvqa':
                if task == 'direct answer (aokvqa)':
                    pred_corr = 0
                    for i in range(len(y_true)):
                        if y_pred[i] in y_true[i]:
                            pred_corr += 1
                    acc = pred_corr/len(y_pred)      
                    scores = {'accuracy': acc}
                elif task == 'multiple choice (aokvqa)':
                    pred_corr = 0
                    for i in range(len(y_true)):
                        if y_pred[i] == y_true[i]:
                            pred_corr += 1
                    acc = pred_corr/len(y_pred)      
                    scores = {'accuracy': acc}
            
            elif dataset_name != 'aokvqa':
                scores = {
                            'accuracy': metrics.accuracy_score(y_true, y_pred),
                            }
            
        
        elif flag_only_acc == False:
            if average=='binary':
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),                  
                }

            else:
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, zero_division=zero_division),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, zero_division=zero_division),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, zero_division=zero_division),                  
                }
    return scores




def get_examples(ds, task, y_pred_dict, y_true_dict):
    '''
    creates dictionary {text_input_id: index_corr}
    index_corr indicates whether a sample was predicted correctly
    given the evaluation modus, dataset, task, ...
    '''
    
    examples = {}
    all_text_input_id = list(y_pred_dict.keys())

    for text_input_id in all_text_input_id:
        if ds == 'aokvqa' and task == 'direct answer (aokvqa)':
            if y_pred_dict[text_input_id] in y_true_dict[text_input_id]:
                correct = 1
            else:
                correct = 0
        else:
            if y_pred_dict[text_input_id] == y_true_dict[text_input_id]:
                correct = 1
            else:
                correct = 0
        examples[text_input_id] = correct
    #average = sum(examples.values())/len(examples)
    #print(average)

    return examples



def make_output_aux_eval(CONFIG_PATH, dataset_name, model_name, run, tasks, mode, y_pred_dict_all_tasks):
    """
    - load the original outputfile
    - replace all predictions with our new cleaned predictions (cleaned, based on model requiriements, dataset and evaluation modus) we got
        in pipeline_preprocess()
    - save the transformed output with a new name
    """
    output_original_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_original_path')
    output_original = load_data(output_original_path)

    for task in tasks:
        y_pred_dict = y_pred_dict_all_tasks[task]
        for item in output_original:
            text_input_id = item.get("text_input_id")
            y_pred_value = y_pred_dict.get(text_input_id)
            if y_pred_value:
                item["output_" + task] = y_pred_value
    output_transformed = output_original # you shall now be called output_transformed

    output_transformed_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_transformed_path')
    directory = os.path.dirname(output_transformed_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_data(output_transformed_path, output_transformed)




def pipeline_preprocess(CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode):
    '''
    preprocess output depending on the model, dataset and evaluation modus
    save preprocessed output in new file
    return predictions once as array [prediction, prediction, ...] for further evaluation with scikitlearn
    return predictions as dictionary {text_input_id: prediction} for other shannanagans
    '''
    dataset_benchmark_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'dataset_benchmark_path')
    output_original_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_original_path')
    
    dataset_benchmark = load_data(dataset_benchmark_path)["data"]
    output_original = load_data(output_original_path)

    valid_ans_ratio_dict = {} 
    label2_y_pred_dict = {}
    y_pred_dict = {}
    y_true_dict = {}

    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()

    for task in tasks:
        label_name = utils.get_task2label_name(task)
        labels = get_id_2_label_dict(dataset_benchmark, label_name, dataset_name) 
        
        valid_ans_ratio, y_pred, y_true, y_pred_dict, y_true_dict = get_clean_valid_preds_trues(
            output = output_original, 
            output_name = "output_"+ task, 
            VALID_ANS_VALUES = VALID_ANS_VALUES, 
            labels = labels, 
            model_name = model_name, 
            dataset_name = dataset_name, 
            data_text = dataset_benchmark, 
            mode = mode, 
            task = task)
        
        
        label2_y_pred_dict[task] = y_pred_dict
        valid_ans_ratio_dict[task] = valid_ans_ratio
        y_pred_dict[task] = y_pred
        y_true_dict[task] = y_true
    
    make_output_aux_eval(CONFIG_PATH, dataset_name, model_name, run, tasks, mode, label2_y_pred_dict)

    return  y_pred_dict, y_true_dict, label2_y_pred_dict, valid_ans_ratio_dict




def calculate_average_accuracy_over_all_ds(CONFIG_PATH, model_name, mode_name):
    
    config = load_data(CONFIG_PATH)
    dataset_names = config['dataset_names']

    def average(values):
        valid_values = [v for v in values if not math.isnan(v)] 
        return sum(valid_values) / len(valid_values) if valid_values else 0

    model_scores = {}
    for dataset_name in dataset_names:
        dataset_scores = {}
        scores_file_path = f"experiments/{model_name}/{dataset_name}/run1/scores_{mode_name}.json"
        file_data = load_data(scores_file_path)
        DatasetInfo = utils.DatasetInfo(dataset_name)
        tasks = DatasetInfo.get_tasks()
        for task in tasks:
            task_score = file_data[task]['accuracy']
            dataset_scores[task] = task_score 

        dataset_scores_average = average(dataset_scores.values())
        model_scores[dataset_name] = dataset_scores_average
    model_scores_average = {"average accuracy": average(model_scores.values())}
    print(f"Average Accuracy: {model_scores_average}")

    scores_average_file_path = f"experiments/{model_name}/scores_average_{mode_name}.json" # needs to be edited when running more runs
    save_data(scores_average_file_path, model_scores_average)




