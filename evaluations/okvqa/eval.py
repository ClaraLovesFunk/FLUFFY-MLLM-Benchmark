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

    transformed_data = [{'question_id': item['text_input_id'], 'answer': item['output_direct answer (okvqa)'][0]} for item in data]

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

    print(experiment_output_file_path)
    ds_dir = os.path.dirname(ds_text_file_path)
    ds_text_annotations_file_path = os.path.join(ds_dir, 'ds_original_labels.json')
    ds_text_questions_file_path = os.path.join(ds_dir, 'ds_original.json')
    experiment_output_okvqa_format_file_path = os.path.join(ds_dir, 'output_okvqa_format.json')
    #experiment_scores_file_path = experiment_output_file_path

    scores = {}
    examples = {}

    acc = acc_okvqa(ds_text_annotations_file_path, 
                    ds_text_questions_file_path, 
                    experiment_output_okvqa_format_file_path, 
                    experiment_output_file_path, 
                    transform_output_4_okvqa)

    scores['direct answer'] = {'accuracy': acc} 

    # delete output file in okvqa format, after it has been used for evalation
    os.remove(experiment_output_okvqa_format_file_path)

    examples['direct answer'] = 'test' ##################################################################### IDENTIFY EXAMPLES!





    # data_text = utils_eval.load_data(ds_text_file_path)
    # output = utils_eval.load_data(experiment_output_file_path)
    # labels = utils_eval.get_id_2_label_dict(data_text, label_name, dataset_name)

    # #valid_ans_ratio, y_pred, y_true = utils_eval.get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset_name, data_text)
    # scores = utils_eval.compute_standard_metrics(y_true, y_pred, pos_label = POS_LABEL, average='binary', flag_only_acc = False)
    # examples = utils_eval.get_examples(output, output_name, labels)

    #valid_ans_ratio = {TASK_NAME: valid_ans_ratio}
    scores = {TASK_NAME: scores}
    examples = {TASK_NAME: examples}

    return scores, examples