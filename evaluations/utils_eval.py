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
            output_clean = output_raw

    elif model == 'adept':
        ''' 
        Corrected regular expression pattern to match the text following \u0004
        '''
        pattern = r'\u0004\s(.+)'
        match = re.search(pattern, output_raw)
        if match:
            output_clean = match.group(1).strip().lower()
        else:
            output_clean = output_raw  

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
    valid_count = 0
    data_text = data_text["data"]
    for item in output:      
        output_raw = str(item[output_name]) 
        pred_value = extract_answer(model, dataset_name, output_raw)
        
        if VALID_ANS_VALUES == "sample-dependent":
            '''
            for dataset, where valid answers must be determined for each sample
            '''
            if dataset_name in ["scienceqa"]:    
                sample = next((d for d in data_text if d['text_input_id'] == item["text_input_id"]), None)
                if sample is not None:
                    no_choices = len(sample['answer_choices'])
                    VALID_ANS_VALUES_sample_dependent = [str(i) for i in range(no_choices)]
                    if pred_value in VALID_ANS_VALUES_sample_dependent and item["text_input_id"] in labels:   
                        valid_count += 1
                        y_pred.append(pred_value)
                        y_true.append(str(labels[item["text_input_id"]]).lower())
            
            elif dataset_name in ["aokvqa"]:
                if mode == 'hard':
                    sample = next((d for d in data_text if d['text_input_id'] == item["text_input_id"]), None)
                    if sample is not None:
                        VALID_ANS_VALUES_sample_dependent = sample['answer_choices']
                        pred_value = pred_value.lower()
                        if item["text_input_id"] in labels and pred_value in VALID_ANS_VALUES_sample_dependent: 
                            valid_count += 1
                            y_pred.append(pred_value)
                            y_true.append(labels[item["text_input_id"]])

                elif mode == 'soft': ###########
                    sample = next((d for d in data_text if d['text_input_id'] == item["text_input_id"]), None)
                    if sample is not None:
                        VALID_ANS_VALUES_sample_dependent = sample['answer_choices']
                        pred_value = pred_value.lower()
                        if item["text_input_id"] in labels and pred_value in VALID_ANS_VALUES_sample_dependent: #mode == 'hard' and pred_value in VALID_ANS_VALUES 
                            valid_count += 1
                            y_pred.append(pred_value)
                            y_true.append(labels[item["text_input_id"]])
                     
            
        elif VALID_ANS_VALUES == "no-ans-validity":
            '''
            for direct answer tasks where no validity can be determined
            '''
            if mode == 'hard':
                if item["text_input_id"] in labels:
                    y_pred.append(pred_value)
                    y_true.append(str(labels[item["text_input_id"]]).lower())
                valid_ans_ratio = None
            if mode == 'soft':
                if item["text_input_id"] in labels:
                    label_str = str(labels[item["text_input_id"]]).lower()
                    if label_str in pred_value:
                        y_pred.append(label_str)
                    else:
                        y_pred.append(pred_value)
                    y_true.append(label_str)
                valid_ans_ratio = None

        else:
            '''
            for classification or multiple choice task, where valid answers are the same for all instances
            '''
            if mode == 'hard' and pred_value in VALID_ANS_VALUES and item["text_input_id"] in labels:
                valid_count += 1
                y_pred.append(pred_value)
                y_true.append(str(labels[item["text_input_id"]]).lower())

            elif mode == 'soft':
                matched_values = [val for val in VALID_ANS_VALUES if val in pred_value]
                if len(matched_values) == 1 and item["text_input_id"] in labels:
                    valid_count += 1
                    y_pred.append(matched_values[0])
                    y_true.append(str(labels[item["text_input_id"]]).lower())

    valid_ans_ratio = valid_count / len(output) if output else 0

    return valid_ans_ratio, y_pred, y_true




def compute_standard_metrics(y_true, y_pred, pos_label, average='binary', zero_division=0, flag_only_acc = False, dataset_name = None):
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
                pred_corr = 0
                for i in range(len(y_true)):
                    if y_pred[i] in y_true[i]:
                        pred_corr += 1
                acc = pred_corr/len(y_pred)      
                scores = {
                    'accuracy': acc,
                }
            
            elif dataset_name != 'aokvqa':
                scores = {
                            'accuracy': metrics.accuracy_score(y_true, y_pred),
                            }
            
        
        if flag_only_acc == False:
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




def get_examples(ds, output, output_name, labels, mode, task):
    '''
    creates dictionary '''
    if mode == 'soft':
        if ds == 'okvqa':   ######## this should be the same as aok
            examples = {
                item["text_input_id"]: 1 if any(str(label).lower() == str(item.get(output_name)).lower() for label in labels[item["text_input_id"]]) else 0
                for item in output
                if "text_input_id" in item and item["text_input_id"] in labels
            }

        elif ds == 'aokvqa':
            if task == 'direct answer (aokvqa)':
                examples = {}
                for item in output:
                    text_input_id = item.get("text_input_id")
                    if text_input_id and text_input_id in labels:
                        label_set = set(str(label).lower() for label in labels[text_input_id])
                        output_value = str(item.get('output_direct answer (aokvqa)', '')).lower()
                        match = any(label in output_value for label in label_set)
                        examples[text_input_id] = 1 if match else 0
                        #print(f"Output: '{output_value}' | Labels: {label_set} | Match: {'Yes' if match else 'No'}")

            if task == 'multiple choice (aokvqa)':
                examples = {}
                for item in output:
                    text_input_id = item.get("text_input_id")
                    if text_input_id and text_input_id in labels:
                        label = labels[text_input_id]
                        output_value = str(item.get('output_multiple choice (aokvqa)', '')).lower()
                        match = label in output_value
                        examples[text_input_id] = 1 if match else 0
                        #print(f"Output: '{output_value}' | Labels: {label} | Match: {'Yes' if match else 'No'}")

        else: ######### 
            examples = {
                item["text_input_id"]: 1 if str(labels[item["text_input_id"]]) == item.get(output_name) else 0
                for item in output if "text_input_id" in item and item["text_input_id"] in labels
            }

    if mode == 'hard':

        if ds == 'okvqa':
            examples = {
                item["text_input_id"]: 1 if any(str(label).lower() == str(item.get(output_name)).lower() for label in labels[item["text_input_id"]]) else 0
                for item in output
                if "text_input_id" in item and item["text_input_id"] in labels
            }

        elif ds == 'aokvqa':
            if task == 'direct answer (aokvqa)':
                examples = {}
                for item in output:
                    text_input_id = item.get("text_input_id")
                    if text_input_id and text_input_id in labels:
                        label_set = set(str(label).lower() for label in labels[text_input_id])
                        output_value = str(item.get('output_direct answer (aokvqa)', '')).lower()
                        match = 1 if output_value in label_set else 0
                        examples[text_input_id] = match

            if task == 'multiple choice (aokvqa)':
                print('blup')

        else:
            examples = {
                item["text_input_id"]: 1 if str(labels[item["text_input_id"]]) == item.get(output_name) else 0
                for item in output if "text_input_id" in item and item["text_input_id"] in labels
            }

    return examples