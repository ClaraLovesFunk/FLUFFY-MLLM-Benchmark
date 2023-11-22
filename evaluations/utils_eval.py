import re
import string
import json
from sklearn import metrics




def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    



def get_id_2_label_dict(data_text, label_name, dataset_name):
    '''
    {text_input_id: label}
    '''
    data_text = data_text["data"] 
    labels = {
        item["text_input_id"]: item[label_name]
        for item in data_text
        if "text_input_id" in item and label_name in item
    }
    return labels




def extract_answer(model, dataset, output_raw):
    '''
    prepocessing depending one the model used, since some models first generate a certain string pattern
    before giving the actual ouput, e.g idefics writes the prompt and "\nAssistant:" before giving 
    the actual output. 
    '''
    if model == 'idefics':
        output_clean = output_raw.split("\nAssistant: ")[-1].strip().lower() #Extracting the answer after "Assistant: "
            
    elif model == 'openflamingo':
        '''
        Using a regular expression to capture the portion after "\nAnswer:" followed by any number of dots
        '''
        if dataset in ['hateful_memes', 'mami', 'esnlive', 'scienceqa', 'aokvqa', 'clevr', 'gqa']:
            match = re.search(r'\nAnswer:\.+(.*)', output_raw)
        if dataset in ['mvsa']:
            match = re.search(r'\nSentiment: \.+(.*)', output_raw)
        if dataset in ['mami']:
            match = re.search(r'\nSexism Label: \.+(.*)', output_raw)
        if dataset in ['hateful_memes']:
            match = re.search(r'\nHate Label: \.+(.*)', output_raw)
        if match:
            output_clean = match.group(1).strip().lower()
        else:
            output_clean = output_raw.lower()

    elif model == 'adept':
        ''' 
        Corrected regular expression pattern to match the text following \u0004
        '''
        pattern = r'\u0004\s(.+)'
        match = re.search(pattern, output_raw)
        if match:
            output_clean = match.group(1).strip().lower()
        else:
            output_clean = output_raw.lower()

    else:
        output_clean = output_raw.lower()

    '''
    Remove any punctuation from the output
    '''
    output_clean = ''.join(ch for ch in output_clean if ch not in string.punctuation)

    return output_clean




def get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset_name, data_text, mode, task = None):
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
    data_text = data_text["data"]
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
        pred_value = extract_answer(model, dataset_name, output_raw)
        text_input_id = item["text_input_id"]
        label_value = str(labels[text_input_id]).lower()
        
        if VALID_ANS_VALUES == "sample-dependent":
            '''
            for dataset, where valid answers must be determined for each sample
            '''
            if dataset_name in ["scienceqa"]:    
                sample = next((d for d in data_text if d['text_input_id'] == text_input_id), None)
                if sample is not None:
                    no_choices = len(sample['answer_choices'])
                    VALID_ANS_VALUES_sample_dependent = [str(i) for i in range(no_choices)]
                    if pred_value in VALID_ANS_VALUES_sample_dependent and text_input_id in labels:   
                        valid_count += 1
                        y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)

            
            elif dataset_name in ["aokvqa"]:
                if task == 'multiple choice (aokvqa)':
                    if mode == 'hard':
                        sample = next((d for d in data_text if d['text_input_id'] == text_input_id), None)
                        if sample is not None:
                            #VALID_ANS_VALUES_sample_dependent = sample['answer_choices'] ####### delete
                            if text_input_id in labels: #and pred_value in VALID_ANS_VALUES_sample_dependent: 
                                valid_count += 1
                                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                    if mode == 'soft':
                        sample = next((d for d in data_text if d['text_input_id'] == text_input_id), None)
                        if sample is not None:
                            VALID_ANS_VALUES_sample_dependent = sample['answer_choices']
                            if text_input_id in labels: #and any(val in pred_value for val in VALID_ANS_VALUES_sample_dependent):
                                matches = [val for val in VALID_ANS_VALUES_sample_dependent if val in pred_value]
                                if len(matches) == 1:
                                    for val in VALID_ANS_VALUES_sample_dependent:
                                        if val in pred_value:
                                            pred_value = val
                                            valid_count += 1
                                            y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                                            break

                                

                if task == 'direct answer (aokvqa)':
                    if mode == 'hard':
                        sample = next((d for d in data_text if d['text_input_id'] == text_input_id), None)
                        if sample is not None:
                            if text_input_id in labels: 
                                valid_count += 1
                                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                    if mode == 'soft':
                        sample = next((d for d in data_text if d['text_input_id'] == text_input_id), None)
                        if sample is not None:
                            CORR_ANS_VALUES_sample_dependent = sample['correct_direct_answer_short'] # all these answers are regarded as correct
                            if text_input_id in labels:
                                matches = [val for val in CORR_ANS_VALUES_sample_dependent if val in pred_value]
                                if matches != []:
                                    pred_value = matches[0]
                                valid_count += 1
                                y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                 
            
        elif VALID_ANS_VALUES == "no-ans-validity":
            '''
            for direct answer tasks where no validity can be determined
            '''
            if mode == 'hard':
                if item["text_input_id"] in labels:
                    y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                valid_ans_ratio = None
            if mode == 'soft':
                if item["text_input_id"] in labels:
                    if label_value in pred_value:
                        pred_value = label_value
                    y_pred, y_true, y_pred_dict, y_true_dict = add_valid_info(text_input_id, pred_value, label_value)
                valid_ans_ratio = None

        else:
            '''
            for classification or multiple choice task, where valid answers are the same for all instances
            '''
            if mode == 'hard' and pred_value in VALID_ANS_VALUES and text_input_id in labels:
                valid_count += 1
                y_pred.append(pred_value)
                y_true.append(label_value)
                y_pred_dict[text_input_id] = pred_value
                y_true_dict[text_input_id] = label_value

            elif mode == 'soft':
                matched_values = [val for val in VALID_ANS_VALUES if val in pred_value]
                if len(matched_values) == 1 and text_input_id in labels:
                    valid_count += 1
                    pred_value = matched_values[0] 
                    y_pred.append(pred_value)
                    y_true.append(label_value)
                    y_pred_dict[text_input_id] = pred_value
                    y_true_dict[text_input_id] = label_value

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



def make_output_aux_eval(output_path, y_pred_dict_all_tasks, mode, tasks):
    """
    Modify the output based on y_pred_dict and save to a new file.
    """
    with open(output_path, 'r') as file:
        output = json.load(file)
    output_modified = output

    for task in tasks:
        y_pred_dict = y_pred_dict_all_tasks[task]

        y_pred_dict1 = y_pred_dict_all_tasks['direct answer (aokvqa)']
        y_pred_dict2 = y_pred_dict_all_tasks['multiple choice (aokvqa)']

        for item in output_modified:
            text_input_id = item.get("text_input_id")
            y_pred_value = y_pred_dict.get(text_input_id)

            # y_pred_value_dict1 = y_pred_dict1.get(text_input_id)
            # y_pred_value_dict2 = y_pred_dict2.get(text_input_id)

            # print(f'y_pred_value_dict1: {y_pred_value_dict1}')
            # print(f'y_pred_value_dict2: {y_pred_value_dict2}')
            if y_pred_value:
                item["output_" + task] = y_pred_value

    output_modified_path = output_path.replace("output", "output_aux_" + mode)

    with open(output_modified_path, 'w') as file:
        json.dump(output_modified, file, indent=4)