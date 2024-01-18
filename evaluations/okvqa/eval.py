import json
import os
from utils.info import DatasetInfo
from utils.file_and_path_utils import get_paths
from utils.evaluation_metrics import pipeline_preprocess
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

	transform_output_4_okvqa(resFile_original, resFile)
        
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)
	vqaEval = VQAEval(vqa, vqaRes, n=2)   
	vqaEval.evaluate() 

	acc = vqaEval.accuracy['overall']

	return acc


def evaluate_okvqa(CONFIG_PATH, dataset_name, model_name, mode, run):

    dataset_benchmark_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'dataset_benchmark_path')
    output_transformed_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_transformed_path')
    
    _, _, _, valid_ans_ratio_dict = pipeline_preprocess(
         CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode)
    
    scores_dict = {}
    examples_dict = {}
    
    DatasetInfo_instance = DatasetInfo(dataset_name)
    tasks = DatasetInfo_instance.get_tasks()
    for task in tasks:
     
        ds_dir = os.path.dirname(dataset_benchmark_path)
        ds_text_annotations_file_path = os.path.join(ds_dir, 'ds_original_labels.json') # determine where original datasetfile lies (there are two files that hold the dataset, one with questions  one with labels)
        ds_text_questions_file_path = os.path.join(ds_dir, 'ds_original.json') # determine where original datasetfile lies (there are two files that hold the dataset, one with questions  one with labels)
        experiment_output_okvqa_format_file_path = os.path.join(ds_dir, 'output_okvqa_format.json') # determine the path where to store and later delete reformatted outputfile

        acc = acc_okvqa(ds_text_annotations_file_path, 
                        ds_text_questions_file_path, 
                        experiment_output_okvqa_format_file_path, 
                        output_transformed_path,  # our transformed output file
                        transform_output_4_okvqa)
        acc = acc/100
        scores_dict[task] = {'accuracy': acc}
            
        os.remove(experiment_output_okvqa_format_file_path) # deletes output file in okvqa format, after it has been used for evalation
    
    return scores_dict, examples_dict, valid_ans_ratio_dict