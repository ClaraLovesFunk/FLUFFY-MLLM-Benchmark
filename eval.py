import json
from sklearn import metrics
import numpy as np
from eval_modules import *
from utils import *



# directory variables

datasets_dir = 'datasets'
experiments_dir = 'experiments'
eval_file_name = 'scores.json'
output_file_name = 'output.json'
examples_file_name = 'examples.json' # file indicating which sample was predicted correctly/incorrectly




# experiment variables

model_name = ['blip2']
dataset_name = ['okvqa','aokvqa', 'mvsa', 'mami', 'hateful_memes']  #hateful_memes', 'mami', 'mvsa', 'okvqa'
run = [1]



for m in model_name:


    for ds in dataset_name:

        # get all ds info

        dataset_info = DatasetInfo(ds)
        text_dataset_split = dataset_info.get_text_dataset_split()
        img_dataset_name = dataset_info.get_img_dataset_name()
        img_dataset_split = dataset_info.get_img_dataset_split() 
        image_dataset_name = dataset_info.get_img_dataset_name()
        tasks = dataset_info.get_tasks()
        
        # paths relevant for all dataset

        ds_images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)
        ds_text_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '.json') 

        
        for r in run:

            experiment_dir_path = os.path.join(experiments_dir, m, ds, 'run' + str(r))
            experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
            experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
            experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)

            
            if ds == 'okvqa':

                ds_text_annotations_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '_labels.json')
                ds_text_questions_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '.json')
                experiment_output_okvqa_format_file_path = os.path.join(experiment_dir_path, 'output_okvqa_format.json')

                scores = {}
                
                acc = acc_okvqa(experiment_scores_file_path, 
                                ds_text_annotations_file_path, 
                                ds_text_questions_file_path, 
                                experiment_output_okvqa_format_file_path, 
                                experiment_output_file_path, 
                                transform_output_4_okvqa)
                
                scores['direct answer'] = {'accuracy': acc} 

                # delete output file in okvqa format, after it has been used for evalation
                os.remove(experiment_output_okvqa_format_file_path)


            if ds == 'aokvqa':

                scores = {}

                data_text = dataset(ds, ds_text_file_path).load()
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)

                acc_da = eval_aokvqa(input = data_text, output=output, task = 'direct answer', strict=True)
                scores['direct answer'] = {'accuracy': acc_da}

                acc_MC = eval_aokvqa(input = data_text, output=output, task = 'multiple choice', strict=True)
                scores['multiple choice'] = {'accuracy': acc_MC}


            if ds in ['hateful_memes', 'mami', 'mvsa']:

                scores = {}

                # load input
                data_text = dataset(ds, ds_text_file_path).load()
                y_true = [item["label"] for item in data_text if "label" in item]
                
                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                

                output_column_header = 'output_' + tasks[0]

                y_pred = [item[output_column_header] for item in output if output_column_header in item]
                   
                if ds in ['hateful_memes']: 

                    y_pred = [int(string) for string in y_pred]
                
                    scores[tasks[0]] = {
                        'accuracy': metrics.accuracy_score(y_true, y_pred),
                        'precision': metrics.precision_score(y_true, y_pred),
                        'recall': metrics.recall_score(y_true, y_pred),
                        'f1': metrics.f1_score(y_true, y_pred),
                        #'roc_auc': metrics.roc_auc_score(y_true, y_pred),
                        #'log_loss': metrics.log_loss(y_true, y_pred),
                        #'jaccard_score': metrics.jaccard_score(y_true, y_pred)
                    }

                if ds in ['mvsa', 'mami']:
                    scores[tasks[0]] = {
                        'accuracy': metrics.accuracy_score(y_true, y_pred),
                        'precision (weighted)': metrics.precision_score(y_true, y_pred, average='weighted'),
                        'recall (weighted)': metrics.recall_score(y_true, y_pred, average='weighted'),
                        'f1 (weighted)': metrics.f1_score(y_true, y_pred, average='weighted'), ####### ADD OTHER AVERAGING METHODS
                        #'roc_auc (weighted)': metrics.roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr'),
                        #'log_loss': metrics.log_loss(y_true, y_pred),
                        #'jaccard_score (weighted)': metrics.jaccard_score(y_true, y_pred, average='weighted'),
                        #'zero_one_loss': metrics.zero_one_loss(y_true, y_pred),
                        #'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
                        #'cohen_kappa': metrics.cohen_kappa_score(y_true, y_pred),
                        #'hamming_loss': metrics.hamming_loss(y_true, y_pred),
                        #'classification_report': metrics.classification_report(y_true, y_pred),
                        #'confusion_matrix': metrics.confusion_matrix(y_true, y_pred),
                        #'normalized_mutual_info_score': metrics.normalized_mutual_info_score(y_true, y_pred),
                        #'adjusted_rand_score': metrics.adjusted_rand_score(y_true, y_pred)
                    }



            # save results
            
            with open(experiment_scores_file_path, 'w') as f: 
                json.dump(scores,f)