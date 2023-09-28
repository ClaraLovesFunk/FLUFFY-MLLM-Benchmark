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
valid_ans_file_name = 'valid_ans.json'
output_file_name = 'output.json'
examples_file_name = 'examples.json' # file indicating which sample was predicted correctly/incorrectly




# experiment variables

model_name = ['blip2']
dataset_name = ['mvsa', 'mami', 'hateful_memes']  # 'okvqa','aokvqa', 'mvsa', 'mami', 'hateful_memes', 'clevr', 'gqa', 'esnlive', 'scienceqa'
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
        ds_text_file_path = os.path.join(datasets_dir, ds, 'ds_benchmark.json') 

        
        for r in run:

            experiment_dir_path = os.path.join(experiments_dir, m, ds, 'run' + str(r))
            experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
            experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
            experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)
            experiment_valid_ans_file_path = os.path.join(experiment_dir_path, valid_ans_file_name)

            
            if ds == 'okvqa':

                ''' valid answers: all answers'''

                ds_text_annotations_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '_labels.json')
                ds_text_questions_file_path = os.path.join(datasets_dir, ds, text_dataset_split + '.json')
                experiment_output_okvqa_format_file_path = os.path.join(experiment_dir_path, 'output_okvqa_format.json')

                scores = {}
                examples = {}
                
                acc = acc_okvqa(experiment_scores_file_path, 
                                ds_text_annotations_file_path, 
                                ds_text_questions_file_path, 
                                experiment_output_okvqa_format_file_path, 
                                experiment_output_file_path, 
                                transform_output_4_okvqa)
                
                scores['direct answer'] = {'accuracy': acc} 

                # delete output file in okvqa format, after it has been used for evalation
                os.remove(experiment_output_okvqa_format_file_path)
                examples['direct answer'] = 'test' ##################################################################### IDENTIFY EXAMPLES!


            if ds == 'aokvqa':

                scores = {}
                examples = {}

                data_text = dataset(ds, ds_text_file_path).load()
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                    #output = output[11:12] ################ DELETE

                ''' valid answers for task "direct asnwers": all answers'''

                acc_da, ex = eval_aokvqa(input = data_text, output=output, task = 'direct answer', strict=True)
                scores['direct answer'] = {'accuracy': acc_da}
                examples['direct answer'] = ex
                
                ''' valid answers for task "multiple choice": one of the choices. the choices that can be found in input[answer_choices]'''

                acc_MC, ex = eval_aokvqa(input = data_text, output=output, task = 'multiple choice', strict=True)
                scores['multiple choice'] = {'accuracy': acc_MC}
                examples['multiple choice'] = ex


            if ds in ['hateful_memes']:

                valid_ans_values = ['hateful', 'not hateful']

                scores = {}
                examples = {}
                examples_task = {}
                valid_ans_ratio = {} 

                with open(ds_text_file_path, 'r') as f:
                    data_text = json.load(f)

                id_to_label = {
                    item["text_input_id"]: item["classification_label"]
                    for item in data_text["data"] 
                    if "text_input_id" in item and "classification_label" in item
                }

                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)

                # Filter valid output and match with true labels
                y_true = []
                y_pred = []
                valid_count = 0

                for item in output:

                    pred_value = item.get(f"output_{tasks[0]}")
                    if pred_value in valid_ans_values and item["text_input_id"] in id_to_label:
                        valid_count += 1
                        y_pred.append(pred_value)
                        y_true.append(id_to_label[item["text_input_id"]])

                valid_ans_ratio[tasks[0]] = valid_count / len(output) if len(output) != 0 else 0
            

                

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, pos_label="hateful"),
                    'recall': metrics.recall_score(y_true, y_pred, pos_label="hateful"),
                    'f1': metrics.f1_score(y_true, y_pred, pos_label="hateful"),                  
                }

                # Build examples dictionary for analysis
                for item in output:
                    input_id = item["text_input_id"]
                    true_label = id_to_label.get(input_id, None)
                    if true_label:
                        pred_label = item.get(f"output_{tasks[0]}")
                        examples_task[input_id] = 1 if str(true_label) == pred_label else 0

                examples[tasks[0]] = examples_task



            if ds in ['mami']:

                valid_ans_values = ['sexist', 'not sexist']

                scores = {}
                examples = {}
                examples_task = {}
                valid_ans_ratio = {} 

                with open(ds_text_file_path, 'r') as f:
                    data_text = json.load(f)

                id_to_label = {
                    item["text_input_id"]: item["classification_label"]
                    for item in data_text["data"] 
                    if "text_input_id" in item and "classification_label" in item
                }


                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)

                # Filter valid output and match with true labels
                y_true = []
                y_pred = []
                valid_count = 0

                
                for item in output:

                    pred_value = item.get(f"output_{tasks[0]}")
                    if pred_value in valid_ans_values and item["text_input_id"] in id_to_label:
                        valid_count += 1
                        y_pred.append(pred_value)
                        y_true.append(id_to_label[item["text_input_id"]])

                valid_ans_ratio[tasks[0]] = valid_count / len(output) if len(output) != 0 else 0
            


                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, pos_label="sexist"),
                    'recall': metrics.recall_score(y_true, y_pred, pos_label="sexist"),
                    'f1': metrics.f1_score(y_true, y_pred, pos_label="sexist"),                  
                }

                # Build examples dictionary for analysis
                for item in output:
                    input_id = item["text_input_id"]
                    true_label = id_to_label.get(input_id, None)
                    if true_label:
                        pred_label = item.get(f"output_{tasks[0]}")
                        examples_task[input_id] = 1 if str(true_label) == pred_label else 0

                examples[tasks[0]] = examples_task


            
            if ds in ['mvsa']:

                valid_ans_values = ['Positive', 'Negative', 'Neutral']

                scores = {}
                examples = {}
                examples_task = {}
                valid_ans_ratio = {} 

                with open(ds_text_file_path, 'r') as f:
                    data_text = json.load(f)

                id_to_label = {
                    item["text_input_id"]: item["classification_label"]
                    for item in data_text["data"] 
                    if "text_input_id" in item and "classification_label" in item
                }

                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)

                # Filter valid output and match with true labels
                y_true = []
                y_pred = []
                valid_count = 0

                for item in output:

                    pred_value = item.get(f"output_{tasks[0]}")
                    if pred_value in valid_ans_values and item["text_input_id"] in id_to_label:
                        valid_count += 1
                        y_pred.append(pred_value)
                        y_true.append(id_to_label[item["text_input_id"]])

                # Calculate scores
                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision (weighted)': metrics.precision_score(y_true, y_pred, average='weighted'),
                    'recall (weighted)': metrics.recall_score(y_true, y_pred, average='weighted'),
                    'f1 (weighted)': metrics.f1_score(y_true, y_pred, average='weighted')
                }

                # Calculate valid answers ratio
                valid_ans_ratio[tasks[0]] = valid_count / len(output) if len(output) != 0 else 0

                # Build examples dictionary for analysis
                for item in output:
                    input_id = item["text_input_id"]
                    true_label = id_to_label.get(input_id, None)
                    if true_label:
                        pred_label = item.get(f"output_{tasks[0]}")
                        examples_task[input_id] = 1 if str(true_label) == pred_label else 0

                examples[tasks[0]] = examples_task


            
            
            if ds in ['clevr']:

                '''valid answers: all answers'''


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

                output_name = 'output_' + tasks[0]

                for input_i in data_text:
                    input_id = input_i.get(input_id_name)

                    y_true = str(input_i.get('answer'))

                    # Find the corresponding output dictionary based on 'input_id'
                    output_i = next((item for item in output if item.get('text_input_id') == input_id), None)

                    y_pred = output_i.get(output_name)

                    # Compare y_true with y_pred
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                
                examples[tasks[0]] = examples_task


            

            if ds in ['gqa']:

                '''valid answers: all answers'''

                scores = {}
                examples = {}
                examples_task = {}

                output_name = 'output_' + tasks[0]
                
                # load data

                data_text = dataset(ds, ds_text_file_path).load()

                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                #output = output[:3] 


                # get y_true and y_pred

                input_ids = [item['text_input_id'] for item in output]

                input_dict = {item[input_id_name]: item['answer'] for item in data_text}
                y_true = [input_dict[id] for id in input_ids]

                output_dict = {item['text_input_id']: item[output_name] for item in output}
                y_pred = [output_dict[id] for id in input_ids if id in output_dict]

                # eval score

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred)
                }

                # example indice

                for output_i in output:

                    input_id = output_i.get('text_input_id')

                    y_true = next((item['answer'] for item in data_text if item.get(input_id_name) == input_id), None)
                    y_pred = next((item[output_name] for item in output if item.get('text_input_id') == input_id), None)
                    
                    examples_task[input_id] = 1 if y_true == y_pred else 0
                    
                examples[tasks[0]] = examples_task






            if ds in ['esnlive']:

                '''valid answers: {entailment, contradiction, neutral}'''

                scores = {}
                examples = {}
                examples_task = {}

                # load inputf
                data_text = dataset(ds, ds_text_file_path).load()
                y_true = [item["label"] for item in data_text if "label" in item]
                
                # load output
                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                
                output_name = 'output_' + tasks[0]

                y_pred = [item[output_name] for item in output if output_name in item]
                
                
                #y_pred = [int(string) for string in y_pred]

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision (weighted)': metrics.precision_score(y_true, y_pred, average='weighted'),
                    'recall (weighted)': metrics.recall_score(y_true, y_pred, average='weighted'),
                    'f1 (weighted)': metrics.f1_score(y_true, y_pred, average='weighted')
                }

                
                for input_i in data_text:

                    input_id = input_i.get('question_id')

                    y_true = str(input_i.get('label'))

                    output_i = next((item for item in output if item.get('text_input_id') == input_id), None)
                    y_pred = output_i.get(output_name)

                    examples_task[input_id] = 1 if y_true == y_pred else 0
                
                examples[tasks[0]] = examples_task



            if ds in ['scienceqa']: ##### :)) account for output not being integers or strings convertable to integers


                '''valid answer: a number representing the right choice. valid answers include the numbers from 0 to n. n is the amount of choices for that instances. the instances have different number of choices.'''

                scores = {}
                examples = {}
                examples_task = {}

                output_name = 'output_' + tasks[0]
                
                # load data

                data_text = dataset(ds, ds_text_file_path).load()

                with open(experiment_output_file_path, 'r') as f:
                    output = json.load(f)
                #output = output[:3] 
                


                # get y_true and y_pred

                input_ids = [item['text_input_id'] for item in output]

                input_dict = {item[input_id_name]: item['answer'] for item in data_text}
                y_true = [input_dict[id] for id in input_ids]
                y_true = list(map(str, y_true)) 
                print(f'---------y_true -------- {y_true[:2]}')

                output_dict = {item['text_input_id']: item[output_name] for item in output}
                y_pred = [output_dict[id] for id in input_ids if id in output_dict]
                y_pred = list(map(str, y_pred))
                print(f'---------y_pred -------- {y_pred[:2]}')

                # eval score

                scores[tasks[0]] = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred)
                }

                # example indice

                for output_i in output:

                    input_id = output_i.get('text_input_id')

                    y_true = next((item['answer'] for item in data_text if item.get(input_id_name) == input_id), None)
                    y_true = str(y_true)

                    y_pred = next((item[output_name] for item in output if item.get('text_input_id') == input_id), None)
                    y_pred = str(y_pred)

                    examples_task[input_id] = 1 if y_true == y_pred else 0
                    
                examples[tasks[0]] = examples_task


            
            


            # save results
            
            with open(experiment_scores_file_path, 'w') as f: 
                json.dump(scores,f, indent=4)

            with open(experiment_valid_ans_file_path, 'w') as f: 
                json.dump(valid_ans_ratio,f, indent=4)

            with open(experiment_examples_file_path, 'w') as f: 
                json.dump(examples,f, indent=4)