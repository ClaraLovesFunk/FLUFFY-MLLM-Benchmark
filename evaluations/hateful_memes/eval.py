import json
from sklearn import metrics

VALID_ANS_VALUES = ['hateful', 'not hateful']
TASK_NAME = "hate classification"
POS_LABEL = "hateful"

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, pos_label=POS_LABEL),
        'recall': metrics.recall_score(y_true, y_pred, pos_label=POS_LABEL),
        'f1': metrics.f1_score(y_true, y_pred, pos_label=POS_LABEL),                  
    }

def evaluate_hateful_memes(ds_text_file_path, experiment_output_file_path):
    data_text = load_json(ds_text_file_path)
    output = load_json(experiment_output_file_path)

    id_to_label = {
        item["text_input_id"]: item["classification_label"]
        for item in data_text["data"] 
        if "text_input_id" in item and "classification_label" in item
    }

    y_true, y_pred = [], []
    valid_count = 0

    for item in output:
        pred_value = item.get("output_hate classification")
        if pred_value in VALID_ANS_VALUES and item["text_input_id"] in id_to_label:
            valid_count += 1
            y_pred.append(pred_value)
            y_true.append(id_to_label[item["text_input_id"]])

    valid_ans_ratio = {TASK_NAME: valid_count / len(output) if output else 0}

    scores = {TASK_NAME: compute_metrics(y_true, y_pred)}

    examples = {
        TASK_NAME: {
            item["text_input_id"]: 1 if id_to_label[item["text_input_id"]] == item.get("output_hate classification") else 0
            for item in output if "text_input_id" in item and item["text_input_id"] in id_to_label
        }
    }

    return scores, examples, valid_ans_ratio
