import json
from sklearn import metrics
import numpy as np


def evaluate_hateful_memes(ds_text_file_path, experiment_output_file_path):

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

        pred_value = item.get("output_hate classification")
        if pred_value in valid_ans_values and item["text_input_id"] in id_to_label:
            valid_count += 1
            y_pred.append(pred_value)
            y_true.append(id_to_label[item["text_input_id"]])

    valid_ans_ratio["hate classification"] = valid_count / len(output) if len(output) != 0 else 0

    # metrics
    scores["hate classification"] = {
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
            pred_label = item.get(f"output_hate classification")
            examples_task[input_id] = 1 if str(true_label) == pred_label else 0

    examples["hate classification"] = examples_task


    return scores, examples, valid_ans_ratio