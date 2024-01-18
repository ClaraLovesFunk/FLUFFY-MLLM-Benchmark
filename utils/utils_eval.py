import re
import string
import json
from sklearn import metrics
import os
import .utils as utils
import math
import shutil
import csv

from .file_operations import load_data

from .path_config import get_paths

from .data_transformation import get_id_2_label_dict
from .data_transformation import extract_answer
from .data_transformation import get_clean_valid_preds_trues






























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







