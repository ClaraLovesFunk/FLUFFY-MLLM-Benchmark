import re
import string
import json
from sklearn import metrics


def extract_answer(model, dataset, output_raw):

    if model == 'idefics':
        output_clean = output_raw.split("\nAssistant: ")[-1].strip().lower() #Extracting the answer after "Assistant: "
            
    elif model == 'openflamingo':
        # Using a regular expression to capture the portion after "\nAnswer:" followed by any number of dots
        if dataset in ['hateful_memes', 'mami', 'esnlive', 'scienceqa']:
            match = re.search(r'\nAnswer:\.+(.*)', output_raw)
        if dataset in ['mvsa']:
            match = re.search(r'\nSentiment: \.+(.*)', output_raw)
        if match:
            output_clean = match.group(1).strip().lower()
        else:
            output_clean = output_raw
    else:
        output_clean = output_raw.lower()

    # Remove any punctuation from the output
    output_clean = ''.join(ch for ch in output_clean if ch not in string.punctuation)

    return output_clean


def compute_standard_metrics(y_true, y_pred, pos_label, average='binary', flag_only_acc = False):

    if y_pred == []: # if no valid predictions were made, model cannot be evaluated
        invalid_ans = float('nan')
        scores = {
            'accuracy': invalid_ans, 
            'precision': invalid_ans, 
            'recall': invalid_ans, 
            'f1': invalid_ans}

    else:
        if flag_only_acc == True:
            scores = {
                        'accuracy': metrics.accuracy_score(y_true, y_pred),
                        }
        
        if flag_only_acc == False:
            if average=='binary':
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, pos_label=pos_label),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, pos_label=pos_label),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, pos_label=pos_label),                  
                }

            else:
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average),
                    'recall': metrics.recall_score(y_true, y_pred, average = average),
                    'f1': metrics.f1_score(y_true, y_pred, average = average),                  
                }
    return scores


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    

def get_id_2_label_dict(data_text, label_name, dataset_name):

    if dataset_name not in ['aokvqa']:
        data_text = data_text["data"]
    
    labels = {
        item["text_input_id"]: item[label_name]
        for item in data_text
        if "text_input_id" in item and label_name in item
    }

    return labels


def get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset_name, data_text):
    
    y_true, y_pred = [], []
    valid_count = 0
    data_text = data_text["data"]
    
    for item in output:      

        output_raw = item[output_name]
        pred_value = extract_answer(model, dataset_name, output_raw)
        
        if VALID_ANS_VALUES == "sample-dependent":
            if dataset_name in ["scienceqa"]:
                sample = next((d for d in data_text if d['text_input_id'] == item["text_input_id"]), None)
                if sample is not None:
                    no_choices = len(sample['answer_choices'])
                    VALID_ANS_VALUES_sample_dependent = [str(i) for i in range(no_choices)]

                    if pred_value in VALID_ANS_VALUES_sample_dependent: # and item["text_input_id"] in labels:   
                        valid_count += 1
                        y_pred.append(pred_value)
                        y_true.append(str(labels[item["text_input_id"]]).lower())
            
        
        if pred_value in VALID_ANS_VALUES and item["text_input_id"] in labels:   
            valid_count += 1
            y_pred.append(pred_value)
            y_true.append(str(labels[item["text_input_id"]]).lower())

    valid_ans_ratio = valid_count / len(output) if output else 0

    return valid_ans_ratio, y_pred, y_true


def get_examples(output, output_name, labels):
    
    examples = {
        item["text_input_id"]: 1 if str(labels[item["text_input_id"]]) == item.get(output_name) else 0
        for item in output if "text_input_id" in item and item["text_input_id"] in labels
    }

    return examples