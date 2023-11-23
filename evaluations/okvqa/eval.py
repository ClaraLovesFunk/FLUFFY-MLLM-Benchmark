import pandas as pd
import json
import os
import evaluations.utils_eval as utils_eval
import utils

from evaluations.okvqa.eval_vqa.vqa import VQA 
from evaluations.okvqa.eval_vqa.vqaEval import VQAEval

VALID_ANS_VALUES = "no-ans-validity"
TASK_NAME = "direct answer (okvqa)"
POS_LABEL = ""
label_name = "correct_direct_answer_short"
output_name = "output_direct answer (okvqa)"
dataset_name = "okvqa"




def transform_output_4_okvqa(resFile_original, resFile):

    with open(resFile_original, 'r') as f:
        data = json.load(f)
    
    transformed_data = []
    for item in data:
        question_id = item['text_input_id']
        answer = item['output_direct answer (okvqa)']
        transformed_data.append({'question_id': question_id, 'answer': answer})

    with open(resFile, 'w') as f:
        json.dump(transformed_data, f)


def acc_okvqa(annFile, quesFile, resFile, resFile_original, transform_output_4_okvqa):

	transform_output_4_okvqa(resFile_original, resFile) # get output in the format needed for okvqa's original eval
        
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)
	vqaEval = VQAEval(vqa, vqaRes, n=2)   # n is precision of accuracy (number of places after decimal), default is 2
	vqaEval.evaluate() 

	acc = vqaEval.accuracy['overall']

	return acc



def evaluate_okvqa(CONFIG_PATH, dataset_name, model_name, mode, run):

    dataset_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'dataset_path')
    output_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_path')

    #def pipeline_transform_output_get_valscore(dataset_path, output_path, model, mode):
        
    #input_benchmark_path = dataset_path 
    output_original_path = output_path 
    output_transformed_4_eval_mode_path = output_original_path.rsplit('.json', 1)[0] + '_aux_' + mode + '.json' 

    input_benchmark = utils_eval.load_data(dataset_path)
    input_benchmark = input_benchmark["data"]
    output_original = utils_eval.load_data(output_original_path)
    
    valid_ans_ratio_dict = {} 
    
    # transform output according to evaluation modus
    y_pred_dict_all_tasks = {}
    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()
    task2label_name = {                                                 
        "direct answer (okvqa)": "correct_direct_answer_short"
    }
    for task in tasks:
        label_name = task2label_name[task]
        labels = utils_eval.get_id_2_label_dict(input_benchmark, label_name, dataset_name) 
        
        valid_ans_ratio, y_pred, y_true, y_pred_dict, y_true_dict = utils_eval.get_clean_valid_preds_trues(
            output = output_original, 
            output_name = "output_"+ task, 
            VALID_ANS_VALUES = VALID_ANS_VALUES, 
            labels = labels, 
            model = model_name, 
            dataset_name = dataset_name, 
            data_text = input_benchmark, 
            mode = mode, 
            task = task)
        y_pred_dict_all_tasks[task] = y_pred_dict
        valid_ans_ratio_dict[task] = valid_ans_ratio
    
    utils_eval.make_output_aux_eval(
        output_original_path = output_original_path,
        y_pred_dict_all_tasks = y_pred_dict_all_tasks,
        mode = mode, 
        tasks = tasks)

    # load transformed data & get scores
    output_transformed_4_eval_mode = utils_eval.load_data(output_transformed_4_eval_mode_path)




    scores_dict = {}
    examples_dict = {}
    
    ds_dir = os.path.dirname(dataset_path)
    ds_text_annotations_file_path = os.path.join(ds_dir, 'ds_original_labels.json') # determine where original datasetfile lies (there are two files that hold the dataset, one with questions  one with labels)
    ds_text_questions_file_path = os.path.join(ds_dir, 'ds_original.json') # determine where original datasetfile lies (there are two files that hold the dataset, one with questions  one with labels)
    experiment_output_okvqa_format_file_path = os.path.join(ds_dir, 'output_okvqa_format.json') # determine the path where to store and later delete reformatted outputfile

    acc = acc_okvqa(ds_text_annotations_file_path, 
                    ds_text_questions_file_path, 
                    experiment_output_okvqa_format_file_path, 
                    output_transformed_4_eval_mode_path,  # use output cleaned and preprocessing depending on eval mode
                    transform_output_4_okvqa)
    acc = acc/100 # we do not want percent
    scores_dict[task] = {'accuracy': acc}
        
    os.remove(experiment_output_okvqa_format_file_path) # deletes output file in okvqa format, after it has been used for evalation

    #data_text = utils_eval.load_data(ds_text_file_path)
    #data_text = data_text["data"]
    #output = utils_eval.load_data(experiment_output_file_path)  ########### change to new
    #labels = utils_eval.get_id_2_label_dict(data_text, label_name, dataset_name) 
    #examples = utils_eval.get_examples(dataset_name, output, output_name, labels)
    #examples_dict[task] = examples
    
    return scores_dict, examples_dict, valid_ans_ratio_dict