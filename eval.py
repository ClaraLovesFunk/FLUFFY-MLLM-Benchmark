import json
from eval_modules import *
from utils_general.utils import *



# directory variables

datasets_dir = 'datasets'
experiments_dir = 'experiments'
eval_file_name = 'scores.json'
output_file_name = 'output.json'
examples_file_name = 'examples.json' # file indicating which sample was predicted correctly/incorrectly




# experiment variables

model_name = ['blip2']
dataset_name = ['okvqa','aokvqa']  
run = [1]



for m in model_name:


    for ds in dataset_name:

        dataset_info = DatasetInfo(ds)
        text_dataset_split = dataset_info.get_text_dataset_split()
        img_dataset_name = dataset_info.get_img_dataset_name()
        img_dataset_split = dataset_info.get_img_dataset_split() 
        image_dataset_name = dataset_info.get_img_dataset_name()
        tasks = dataset_info.get_tasks()
        
        # paths relevant for all dataset

        images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)
        dataset_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '.json') # RENAME

        
        for r in run:

            experiment_dir_path = os.path.join(experiments_dir, m, ds, 'run' + str(r))
            experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
            experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
            experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)

            
            if ds == 'okvqa':

                dataset_annotations_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '_labels.json') # ADD
                dataset_questions_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '.json') # ADD
                experiment_output_okvqa_format_file_path = os.path.join(experiment_dir_path, 'output_okvqa_format.json') # ADD

                scores = {}
                
                acc = acc_okvqa(experiment_scores_file_path, 
                                dataset_annotations_file_path, 
                                dataset_questions_file_path, 
                                experiment_output_okvqa_format_file_path, 
                                experiment_output_file_path, 
                                transform_output_4_okvqa)
                
                scores['direct answer'] = {'accuracy': acc}
            
                with open(experiment_scores_file_path, 'w') as f: 
                    json.dump(scores,f) 


            if ds == 'aokvqa':

                scores = {}

                data_text = dataset(ds, dataset_file_path).load()
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)

                acc_da = eval_aokvqa(input = data_text, output=output, task = 'direct_answer', strict=True)
                scores['direct answer'] = {'accuracy': acc_da}

                acc_MC = eval_aokvqa(input = data_text, output=output, task = 'multiple_choice', strict=True)
                scores['multiple choice'] = {'accuracy': acc_MC}