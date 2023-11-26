import json
import os
import evaluations.utils_eval as utils_eval
import utils
import math  # Importing math to check for NaN

CONFIG_PATH = 'config.json'

config = utils_eval.load_data(CONFIG_PATH)
model_names = ['idefics']  # config['model_names']
dataset_names = config['dataset_names']
mode_names = ['soft', 'hard']


def average(values):
    valid_values = [v for v in values if not math.isnan(v)]  # Filter out NaN values
    return sum(valid_values) / len(valid_values) if valid_values else 0

for mode_name in mode_names:
    print(f"Mode: {mode_name}")
    for model in model_names:
        model_scores = {}
        for dataset_name in dataset_names:
            scores_file_path = f"experiments/{model}/{dataset_name}/run1/scores_{mode_name}.json"
            file_data = utils_eval.load_data(scores_file_path)
            DatasetInfo = utils.DatasetInfo(dataset_name)
            tasks = DatasetInfo.get_tasks()
            dataset_scores = {}
            for task in tasks:
                task_score = file_data[task]['accuracy']
                dataset_scores[task] = task_score if not math.isnan(task_score) else 0  # Handle NaN here if needed
            #print(dataset_scores)
            dataset_scores_average = average(dataset_scores.values())
            #print(dataset_scores_average)
            model_scores[dataset_name] = dataset_scores_average
        model_scores_average = {"average accuracy": average(model_scores.values())}
        print(f"{model} Average Accuracy: {model_scores_average}")

        scores_average_file_path = f"experiments/{model}/scores_average_{mode_name}.json" # needs to be edited when running more runs
        utils_eval.save_data(scores_average_file_path, model_scores_average)