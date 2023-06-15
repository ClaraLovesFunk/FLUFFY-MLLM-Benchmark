from vqa import VQA #*
from vqaEval import VQAEval
import json




experiment_scores_file_path = 'experiments/blip2/okvqa/run1/scores.json'
dataset_annotations_file_path     ='datasets/okvqa/val_labels.json' 
dataset_questions_file_path    ='datasets/okvqa/val.json'
experiment_output_okvqa_format_file_path = 'experiments/blip2/okvqa/run1/output_okvqaformat.json'
experiment_output_file_path = 'experiments/blip2/okvqa/run1/output.json'

# add those files to big script


'''
images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)
dataset_file_path = os.path.join(datasets_dir, dataset_name, text_dataset_split + '.json') # RENAME
dataset_annotations_file_path = os.path.join(datasets_dir, dataset_name, text_dataset_split + '_labels.json') # ADD
dataset_questions_file_path = os.path.join(datasets_dir, dataset_name, text_dataset_split + '.json') # ADD
experiment_dir_path = os.path.join(experiments_dir, m, dataset_name, 'run' + str(run))
experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
experiment_output_okvqa_format_file_path = os.path.join(experiment_dir_path, 'output_okvqa_format.json') # ADD
experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)'''


def transform_output_4_okvqa(resFile_original, resFile):

    with open(resFile_original, 'r') as f:
        data = json.load(f)

    transformed_data = [{'question_id': item['question_id'], 'answer': item['output_direct_answer'][0]} for item in data]

    with open(resFile, 'w') as f:
        json.dump(transformed_data, f)




def acc_okvqa(eval_file, annFile, quesFile, resFile, resFile_original, transform_output_4_okvqa):

	transform_output_4_okvqa(resFile_original, resFile) # get output in the format needed for okvqa's original eval
        
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)
	vqaEval = VQAEval(vqa, vqaRes, n=2)   # n is precision of accuracy (number of places after decimal), default is 2
	vqaEval.evaluate() 

	acc = vqaEval.accuracy['overall']

	return acc




acc = acc_okvqa(experiment_scores_file_path, dataset_annotations_file_path, dataset_questions_file_path, experiment_output_okvqa_format_file_path, experiment_output_file_path, transform_output_4_okvqa)

with open(experiment_scores_file_path, 'w') as f: 
	json.dump(acc,f)
