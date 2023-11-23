import re
import string
import json
from sklearn import metrics
import os
import utils




def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    



def save_data(filepath, file):
    with open(filepath, 'w') as f: 
        return json.dump(file,f, indent=4)
    



def get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'None'):

    config = load_data(CONFIG_PATH)
    
    dataset_benchmark_path = os.path.join(config['datasets_dir'], dataset_name, config['dataset_file_name'])
    experiment_dir_path = os.path.join(config['experiments_dir'], model_name, dataset_name, 'run' + run)
    output_original_path = os.path.join(experiment_dir_path, config['output_file_name'])
    scores_path = os.path.join(experiment_dir_path, config['eval_file_' + mode])
    examples_path = os.path.join(experiment_dir_path, config['examples_file_' + mode])
    val_ratio_path = os.path.join(experiment_dir_path, config['valid_ans_file_' + mode])

    output_transformed_path = output_original_path.replace("output", "output_aux_" + mode)

    if value_of_interest == 'dataset_benchmark_path':
        return dataset_benchmark_path
    elif value_of_interest == 'output_original_path':
        return output_original_path
    elif value_of_interest == 'output_transformed_path':
        return output_transformed_path
    elif value_of_interest == 'scores_path':
        return scores_path
    elif value_of_interest == 'examples_path':
        return examples_path
    elif value_of_interest == 'val_ratio_path':
        return val_ratio_path
    elif value_of_interest == 'None':
        return dataset_benchmark_path, output_original_path, output_transformed_path, scores_path, examples_path, val_ratio_path




def get_id_2_label_dict(data_text, label_name, dataset_name):
    '''
    {text_input_id: label}
    '''
    labels = {}
    for item in data_text:
        text_input_id = item["text_input_id"]
        label_value = item[label_name]
        labels[text_input_id] = label_value

    return labels




def extract_answer(model, dataset, output_raw):
    '''
    prepocessing depending one the model used, since some models first generate a certain string pattern
    before giving the actual ouput, e.g idefics writes the prompt and "\nAssistant:" before giving 
    the actual output. 
    '''

    if model == 'idefics':
        output_clean = output_raw.split("\nAssistant: ")[-1].strip()
            
    elif model == 'openflamingo':
        pattern = r'\bAnswer:\s*(.+)'
        match = re.search(pattern, output_raw, re.IGNORECASE)
        if match:
            output_clean = match.group(1).strip()
        else:
            output_clean = output_raw

        output_clean = re.sub(".<|endofchunk|>", '', output_clean)
        output_clean = re.sub(r'\|\|', '', output_clean)
        output_clean = re.sub(r'\. ', '', output_clean)

    elif model == 'adept':
        ''' 
        Corrected regular expression pattern to match the text following \u0004
        '''
        pattern = r'\u0004\s(.+)'
        match = re.search(pattern, output_raw)
        if match:
            output_clean = match.group(1).strip()
        else:
            output_clean = output_raw

    else:
       output_clean = output_raw

    output_clean = output_clean.lower()
    output_clean = re.sub(r'\.', '', output_clean)
    #print(f'output_clean: {output_clean}')
    if not output_clean.strip():
        output_clean = "NaN"

    return output_clean




def get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model_name, dataset_name, data_text, mode, task = None): #delete output_name
    '''
    - cleans the output
        - with respect to the model used to generate the output
        - basic preprocessing, s.a. removing punctuation, lower case
        - checks for the validity of the output and only further regards valid output for later computing the metrics
        - distinction between hard and soft evaluation
            - for hard evaluation check whether the output matches a valid label exactly
            - for soft evaluation check whether a valid label occurs somewhere in the output for that sample
    '''
    y_true, y_pred = [], []
    y_true_dict, y_pred_dict = {}, {}
    valid_count = 0

    output_name = "output_" + task

    def add_valid_info(text_input_id, pred_value, label_value):
        '''
        - add predictions and labels to lists y_pred and y_true for further computation of eval metrics with sckitlearn
        - add predictions and labels to dictionary so we can use them later to make a dataframe with all info to show good and 
            bad examples 
        '''
        y_pred.append(pred_value)
        y_true.append(label_value)
        y_pred_dict[text_input_id] = pred_value
        y_true_dict[text_input_id] = label_value
        return y_pred, y_true, y_pred_dict, y_true_dict
    
    for item in output:      
        output_raw = str(item[output_name]) 
        #print(f'output_raw: {output_raw}')
        pred_value = extract_answer(model_name, dataset_name, output_raw)
        #print(f'pred_value: {pred_value}')
        text_input_id = item["text_input_id"]
        label_value = str(labels[text_input_id]).lower()
        sample = next((d for d in data_text if d['text_input_id'] == text_input_id), None)
        
        if VALID_ANS_VALUES == "sample-dependent":
            '''
            for datasets, where valid answers must be determined for each sample
            '''
            if dataset_name in ["scienceqa"]: 
                no_choices = len(sample['answer_choices'])
                VALID_ANS_VALUES_sample_dependent = [str(i) for i in range(no_choices)]
                if mode == 'soft':
                    matches = [val for val in VALID_ANS_VALUES_sample_dependent if val in pred_value] # if exactly one of the multiple choice choices is in the output, replace the model's jibberjabber with only that choice
                    if len(matches) == 1:
                        pred_value = matches[0]
                if pred_value in VALID_ANS_VALUES_sample_dependent:
                    valid_count += 1
                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
            
            elif dataset_name in ["aokvqa"]:
                if task == 'multiple choice (aokvqa)':
                    VALID_ANS_VALUES_sample_dependent = sample['answer_choices']
                    if mode == 'soft':
                        matches = [val for val in VALID_ANS_VALUES_sample_dependent if val in pred_value] # if exactly one of the multiple choice choices is in the output, replace the model's jibberjabber with only that choice
                        if len(matches) == 1:
                            pred_value = matches[0]
                    if pred_value in VALID_ANS_VALUES_sample_dependent:
                        valid_count += 1
                    y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                if task == 'direct answer (aokvqa)':
                    CORR_ANS_VALUES_sample_dependent = sample['correct_direct_answer_short'] # all these answers are regarded as correct
                    if mode == 'soft':
                        matches = [val for val in CORR_ANS_VALUES_sample_dependent if val in pred_value]
                        if matches != []:
                            pred_value = matches[0]
                    valid_count += 1
                    y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
        
            
        elif VALID_ANS_VALUES == "no-ans-validity":
            '''
            for direct answer tasks where no validity can be determined
            '''
            if dataset_name in ["okvqa"]:
                CORR_ANS_VALUES_sample_dependent = sample['correct_direct_answer_short'] # all these answers are regarded as correct
                if mode == 'soft':
                    matches = [val for val in CORR_ANS_VALUES_sample_dependent if val in pred_value]
                    if matches != []:
                        pred_value = matches[0]
                valid_count += 1
                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
            if dataset_name in ["clevr", "gqa"]:
                CORR_ANS_VALUES_sample_dependent = str(sample['correct_direct_answer_short'])
                if mode == 'soft':
                    if CORR_ANS_VALUES_sample_dependent in pred_value:
                        pred_value = CORR_ANS_VALUES_sample_dependent
                valid_count += 1
                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)

        else:
            '''
            for classification or multiple choice task, where valid answers are the same for all instances
            '''
            if mode == 'soft':
                matches = [val for val in VALID_ANS_VALUES if val in pred_value] # if exactly one of the labels is in the output, replace the model's jibberjabber with only that label
                if len(matches) == 1:
                    pred_value = matches[0]
            if pred_value in VALID_ANS_VALUES:
                valid_count += 1
                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
        

    valid_ans_ratio = valid_count / len(output) if output else 0
    #print(f'valid_ans_ratio: {valid_ans_ratio}')

    return valid_ans_ratio, y_pred, y_true, y_pred_dict, y_true_dict




def compute_standard_metrics(y_true, y_pred, pos_label, average='binary', zero_division=0, flag_only_acc = False, dataset_name = None, task = None):
    
    '''
    compute metrics
    '''
    if y_pred == []: 
        '''
        if no valid predictions were made, model cannot be evaluated
        '''
        invalid_ans = float('nan')
        scores = {
            'accuracy': invalid_ans, 
            'precision': invalid_ans, 
            'recall': invalid_ans, 
            'f1': invalid_ans}

    elif y_pred != []:
        '''
        compute metrics, if valid predictions were made
        '''
        if flag_only_acc == True:
            if dataset_name == 'aokvqa':
                if task == 'direct answer (aokvqa)':
                    pred_corr = 0
                    for i in range(len(y_true)):
                        if y_pred[i] in y_true[i]:
                            pred_corr += 1
                    acc = pred_corr/len(y_pred)      
                    scores = {'accuracy': acc}
                elif task == 'multiple choice (aokvqa)':
                    pred_corr = 0
                    for i in range(len(y_true)):
                        if y_pred[i] == y_true[i]:
                            pred_corr += 1
                    acc = pred_corr/len(y_pred)      
                    scores = {'accuracy': acc}
            
            elif dataset_name != 'aokvqa':
                scores = {
                            'accuracy': metrics.accuracy_score(y_true, y_pred),
                            }
            
        
        elif flag_only_acc == False:
            if average=='binary':
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),                  
                }

            else:
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, zero_division=zero_division),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, zero_division=zero_division),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, zero_division=zero_division),                  
                }
    return scores




def get_examples(ds, task, y_pred_dict, y_true_dict):
    '''
    creates dictionary {text_input_id: index_corr}
    index_corr indicates whether a sample was predicted correctly
    given the evaluation modus, dataset, task, ...
    '''
    
    examples = {}
    all_text_input_id = list(y_pred_dict.keys())

    for text_input_id in all_text_input_id:
        if ds == 'aokvqa' and task == 'direct answer (aokvqa)':
            if y_pred_dict[text_input_id] in y_true_dict[text_input_id]:
                correct = 1
            else:
                correct = 0
        else:
            if y_pred_dict[text_input_id] == y_true_dict[text_input_id]:
                correct = 1
            else:
                correct = 0
        examples[text_input_id] = correct
    #average = sum(examples.values())/len(examples)
    #print(average)

    return examples



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
    task2label_name = {                                                 
        "direct answer (okvqa)": "correct_direct_answer_short",
        "direct answer (aokvqa)": "correct_direct_answer_short",
        "multiple choice (aokvqa)": "correct_multiple_choice_answer",
        "multiple choice (sqa)": "correct_choice",
        "direct answer (clevr)": "correct_direct_answer_short",
        "direct answer (gqa)": "correct_direct_answer_short",
        "hate classification": "classification_label",
    }
    for task in tasks:
        label_name = task2label_name[task]
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