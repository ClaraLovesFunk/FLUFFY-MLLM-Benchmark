import json
from sklearn import metrics
import numpy as np
import time
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
dataset_name = ['gqa']  # 'okvqa','aokvqa', 'mvsa', 'mami', 'hateful_memes'           #'clevr', 'gqa', 'esnlive'
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
        input_id_name = dataset_info.get_input_id_name()
        
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
                examples = {}

                data_text = dataset(ds, ds_text_file_path).load()
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                    #output = output[11:12] ################ DELETE

                acc_da, ex = eval_aokvqa(input = data_text, output=output, task = 'direct answer', strict=True)
                scores['direct answer'] = {'accuracy': acc_da}
                examples['direct answer'] = ex
                

                acc_MC, ex = eval_aokvqa(input = data_text, output=output, task = 'multiple choice', strict=True)
                scores['multiple choice'] = {'accuracy': acc_MC}
                examples['multiple choice'] = ex


            if ds in ['hateful_memes']:
                scores = {}
                examples = {}
                examples_task = {}

                # load input
                data_text = dataset(ds, ds_text_file_path).load()
                y_true = [item["label"] for item in data_text if "label" in item]
                y_true = ['hateful' if item == 1 else 'not-hateful' for item in y_true]

                
                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                
                output_column_header = 'output_' + tasks[0]

                y_pred = [item[output_column_header] for item in output if output_column_header in item]
                #y_pred = list(map(int, y_pred))

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred),
                    'recall': metrics.recall_score(y_true, y_pred),
                    'f1': metrics.f1_score(y_true, y_pred),                  
                }

                
                for input_i in data_text:
                    input_id = input_i.get('id')

                    y_true = str(input_i.get('label'))

                    # Find the corresponding output dictionary based on 'input_id'
                    output_i = next((item for item in output if item.get('text_input_id') == input_id), None)

                    y_pred = output_i.get('output_hate classification')

                    # Compare y_true with y_pred
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                
                examples['hate classification'] = examples_task



            if ds in ['mami']:


                scores = {}
                examples = {}
                examples_task = {}

                # load input
                data_text = dataset(ds, ds_text_file_path).load()
                y_true = [item["label"] for item in data_text if "label" in item]
                y_true = ['sexist' if item == 1 else 'not-sexist' for item in y_true]
                
                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                
                output_column_header = 'output_' + tasks[0]

                y_pred = [item[output_column_header] for item in output if output_column_header in item]
                
                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision (weighted)': metrics.precision_score(y_true, y_pred, average='weighted'),
                    'recall (weighted)': metrics.recall_score(y_true, y_pred, average='weighted'),
                    'f1 (weighted)': metrics.f1_score(y_true, y_pred, average='weighted')
                }

                for input_i in data_text:
                    input_id = input_i.get('id')

                    y_true = str(input_i.get('label'))

                    # Find the corresponding output dictionary based on 'input_id'
                    output_i = next((item for item in output if item.get('text_input_id') == input_id), None)

                    y_pred = output_i.get('output_sexism classification')

                    # Compare y_true with y_pred
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                
                examples['sexism classification'] = examples_task

            
            if ds in ['mvsa']:


                scores = {}
                examples = {}
                examples_task = {}

                # load input
                data_text = dataset(ds, ds_text_file_path).load()
                y_true = [item["label"] for item in data_text if "label" in item]
                
                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                
                output_column_header = 'output_' + tasks[0]

                y_pred = [item[output_column_header] for item in output if output_column_header in item]
                
                
                #y_pred = [int(string) for string in y_pred]

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision (weighted)': metrics.precision_score(y_true, y_pred, average='weighted'),
                    'recall (weighted)': metrics.recall_score(y_true, y_pred, average='weighted'),
                    'f1 (weighted)': metrics.f1_score(y_true, y_pred, average='weighted')
                }

                
                for input_i in data_text:

                    input_id = input_i.get('id')

                    y_true = str(input_i.get('label'))

                    # Find the corresponding output dictionary based on 'input_id'
                    output_i = next((item for item in output if item.get('text_input_id') == input_id), None)

                    y_pred = output_i.get('output_sentiment analysis')

                    # Compare y_true with y_pred
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                
                examples['sentiment analysis'] = examples_task


            
            
            if ds in ['clevr']:


                scores = {}
                examples = {}
                examples_task = {}
                
                
                # load input
                data_text = dataset(ds, ds_text_file_path).load()
                #data_text = data_text[:1000] ############ DELETE 

                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                #output = output[:1000] ############ DELETE 


                # 1. Get a new list "input_ids" that contains all input_ids from the list "input"
                input_ids = [item['input_id'] for item in data_text]

                # 2. Get a list "y_true", in which the answer from the list "input" for each input_id is listed
                input_dict = {item['input_id']: item['answer'] for item in data_text}
                y_true = [input_dict[id] for id in input_ids]

                # 3. Get a list "y_pred", in which the "output_direct answer" from the list "output" for each input_id is listed
                output_dict = {item['text_input_id']: item['output_direct answer'] for item in output}
                y_pred = [output_dict[id] for id in input_ids if id in output_dict]

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred)
                }

                output_column_header = 'output_' + tasks[0]

                for input_i in data_text:
                    input_id = input_i.get(input_id_name)

                    y_true = str(input_i.get('answer'))

                    # Find the corresponding output dictionary based on 'input_id'
                    output_i = next((item for item in output if item.get('text_input_id') == input_id), None)

                    y_pred = output_i.get(output_column_header)

                    # Compare y_true with y_pred
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                
                examples[tasks[0]] = examples_task


            

            if ds in ['gqa']:


                scores = {}
                examples = {}
                examples_task = {}
                
                
                # load input
                data_text = dataset(ds, ds_text_file_path).load()

                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                #output = output[:3] 

                # 1. Get a new list "input_ids" that contains all input_ids from the list "output"
                input_ids = [item['text_input_id'] for item in output]

                # 2. Get a list "y_true", in which the answer from the list "input" for each input_id is listed
                input_dict = {item['input_id']: item['answer'] for item in data_text}
                y_true = [input_dict[id] for id in input_ids]

                # 3. Get a list "y_pred", in which the "output_direct answer" from the list "output" for each input_id is listed
                output_dict = {item['text_input_id']: item['output_direct answer'] for item in output}
                y_pred = [output_dict[id] for id in input_ids if id in output_dict]

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred)
                }

                output_column_header = 'output_' + tasks[0]

                for output_i in output:

                    input_id = output_i.get('text_input_id')
                    print(input_id)

                    y_true = next((item['answer'] for item in data_text if item.get('input_id') == input_id), None)
                    y_pred = next((item['output_direct answer'] for item in output if item.get('text_input_id') == input_id), None)
                    
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                    
                examples[tasks[0]] = examples_task




            # save results
            
            with open(experiment_scores_file_path, 'w') as f: 
                json.dump(scores,f)

            with open(experiment_examples_file_path, 'w') as f: 
                json.dump(examples,f)