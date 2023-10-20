import re
import string
import json
from sklearn import metrics


def extract_answer(model, dataset, output_raw):

    if model == 'idefics':
        output_clean = output_raw.split("\nAssistant: ")[-1].strip().lower() #Extracting the answer after "Assistant: "
            
    elif model == 'openflamingo':
        # Using a regular expression to capture the portion after "\nAnswer:" followed by any number of dots
        if dataset in ['hateful_memes', 'mami']:
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


def compute_standard_metrics(y_true, y_pred, pos_label, average='binary'):

    if y_pred == []: # if no valid predictions were made, model cannot be evaluated
        print('invalid output')
        invalid_ans = float('nan')
        scores = {
            'accuracy': invalid_ans, 
            'precision': invalid_ans, 
            'recall': invalid_ans, 
            'f1': invalid_ans}
        print(scores)

    else:
        print('valid output')
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
    

def get_id_2_label_dict(data_text, label_name):

    labels = {
        item["text_input_id"]: item[label_name]
        for item in data_text["data"] 
        if "text_input_id" in item and label_name in item
    }

    return labels


def get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset):
    
    y_true, y_pred = [], []
    valid_count = 0
    
    for item in output:     #######################
        output_raw = item[output_name]
        #print(output_raw)
        pred_value = extract_answer(model, dataset, output_raw)
        #print(pred_value)

        #print(VALID_ANS_VALUES) 
        if pred_value in VALID_ANS_VALUES and item["text_input_id"] in labels:   #if pred_value in VALID_ANS_VALUES and item["text_input_id"] in labels:
            #print('test')
            valid_count += 1
            y_pred.append(pred_value)
            y_true.append(labels[item["text_input_id"]].lower())

    valid_ans_ratio = valid_count / len(output) if output else 0

    return valid_ans_ratio, y_pred, y_true


def get_examples(output, output_name, labels):

    examples = {
        item["text_input_id"]: 1 if labels[item["text_input_id"]] == item.get(output_name) else 0
        for item in output if "text_input_id" in item and item["text_input_id"] in labels
    }

    return examples