import re 

from .path_config import get_paths
from .data_loading import load_data
from .utils import DatasetInfo
from .utils import get_task2label_name
from .data_transformation import get_id_2_label_dict
from .data_transformation import make_output_aux_eval


def extract_answer(model, dataset_name, output_raw):
    '''
    prepocessing depending one the model used, since some models first generate a certain string pattern
    before giving the actual ouput, e.g idefics writes the prompt and "\nAssistant:" before giving 
    the actual output. 
    '''

    if model == 'idefics':
        output_clean = output_raw.split("\nAssistant: ")[-1].strip()
            
    elif model == 'openflamingo':
        '''
        insert code.
        remove the following string whereever it occurs: "\u00a0"
        '''
        if dataset_name == 'mvsa':
            pattern = r'\bSentiment:\s*(.+)'
        elif dataset_name == 'mami':
            pattern = r'\bSexism Label:\s*(.+)' 
        elif dataset_name == 'hateful_memes':
            pattern = r'\bHate Label:\s*(.+)' 
        else:
            pattern = r'\bAnswer:\s*(.+)'
        match = re.search(pattern, output_raw, re.IGNORECASE)
        if match:
            output_clean = match.group(1).strip()
        else:
            output_clean = output_raw

        #print(f'output_raw: {output_raw}')

        output_clean = re.sub("\u00a0", '', output_clean)
        #print(f'output_clean_1: {output_clean}')
        output_clean = re.sub("<|endofchunk|>", '', output_clean)
        output_clean = re.sub(r'\|\|', '', output_clean)
        output_clean = re.sub(r'\. ', '', output_clean)

        
        #print(f'output_clean2: {output_clean}')


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

    def add_processed_valid_info(text_input_id, pred_value, label_value):
        '''
        - add predictions and labels to lists y_pred and y_true for further computation of eval metrics with sckitlearn
        #- add predictions and labels to dictionary so we can use them later to make a dataframe with all info to show good and 
            bad examples 
        '''
        y_pred.append(pred_value)
        y_true.append(label_value)
        #y_pred_dict[text_input_id] = pred_value
        #y_true_dict[text_input_id] = label_value
        return y_pred, y_true#, y_pred_dict, y_true_dict

    def add_processed_info(text_input_id, pred_value, label_value):
        '''
        - add predictions and labels to dictionary so we can use them later to make a dataframe with all info to show good and 
            bad examples 
        '''
        y_pred_dict[text_input_id] = pred_value
        y_true_dict[text_input_id] = label_value

        return y_pred_dict, y_true_dict
    
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
                        valid_count += 1
                        y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value)
                    y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)
                if mode == 'hard':
                    if pred_value in VALID_ANS_VALUES_sample_dependent:
                        count = VALID_ANS_VALUES_sample_dependent.count(pred_value)
                       
                        if count == 1:
                            valid_count += 1
                            y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value)
                    y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)
                
            elif dataset_name in ["aokvqa"]:
                if task == 'multiple choice (aokvqa)':
                    VALID_ANS_VALUES_sample_dependent = sample['answer_choices']
                    if mode == 'soft':
                        matches = [val for val in VALID_ANS_VALUES_sample_dependent if val in pred_value] # if exactly one of the multiple choice choices is in the output, replace the model's jibberjabber with only that choice
                        if len(matches) == 1:
                            pred_value = matches[0]
                    if pred_value in VALID_ANS_VALUES_sample_dependent:
                        valid_count += 1
                    y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value) # because we dont kick out instances because the model did not give a valid answer
                    y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)
                if task == 'direct answer (aokvqa)':
                    CORR_ANS_VALUES_sample_dependent = sample['correct_direct_answer_short'] # all these answers are regarded as correct
                    if mode == 'soft':
                        matches = [val for val in CORR_ANS_VALUES_sample_dependent if val in pred_value]
                        if matches != []:
                            pred_value = matches[0]
                    valid_count += 1
                    y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value)
                    y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)
        
            
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
                y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value)
                y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)
            if dataset_name in ["clevr", "gqa"]:
                CORR_ANS_VALUES_sample_dependent = str(sample['correct_direct_answer_short'])
                if mode == 'soft':
                    if CORR_ANS_VALUES_sample_dependent in pred_value:
                        pred_value = CORR_ANS_VALUES_sample_dependent
                valid_count += 1
                y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value)
                y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)

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
                y_pred, y_true = add_processed_valid_info(text_input_id, pred_value, label_value)
            y_pred_dict, y_true_dict = add_processed_info(text_input_id, pred_value, label_value)
        

    valid_ans_ratio = valid_count / len(output) if output else 0
    #print(f'valid_ans_ratio: {valid_ans_ratio}')

    return valid_ans_ratio, y_pred, y_true, y_pred_dict, y_true_dict


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

    DatasetInfo_instance = DatasetInfo(dataset_name)
    tasks = DatasetInfo_instance.get_tasks()

    for task in tasks:
        label_name = get_task2label_name(task)
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