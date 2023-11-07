import pandas as pd
import json
import os
import evaluations.utils_eval as utils_eval

from evaluations.okvqa.eval_vqa.vqa import VQA 
from evaluations.okvqa.eval_vqa.vqaEval import VQAEval

VALID_ANS_VALUES = ['']
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
        print(question_id)
        answer = item['output_direct answer (okvqa)']#[0]
        print(f'asnswer: {answer}')
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



def evaluate_okvqa(ds_text_file_path, experiment_output_file_path, model):

    ds_dir = os.path.dirname(ds_text_file_path)
    ds_text_annotations_file_path = os.path.join(ds_dir, 'ds_original_labels.json')
    ds_text_questions_file_path = os.path.join(ds_dir, 'ds_original.json')
    experiment_output_okvqa_format_file_path = os.path.join(ds_dir, 'output_okvqa_format.json')

    scores = {}
    examples = {}

    acc = acc_okvqa(ds_text_annotations_file_path, 
                    ds_text_questions_file_path, 
                    experiment_output_okvqa_format_file_path, 
                    experiment_output_file_path, 
                    transform_output_4_okvqa)
    acc = acc/100 # we do not want percent
    scores = {'accuracy': acc} 

    os.remove(experiment_output_okvqa_format_file_path) # deletes output file in okvqa format, after it has been used for evalation

    data_text = utils_eval.load_data(ds_text_file_path)
    output = utils_eval.load_data(experiment_output_file_path)
    labels = utils_eval.get_id_2_label_dict(data_text, label_name, dataset_name) 

    examples = utils_eval.get_examples(dataset_name, output, output_name, labels)

    scores = {TASK_NAME: scores}
    examples = {TASK_NAME: examples}

    return scores, examples