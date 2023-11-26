import json
import os
import evaluations.utils_eval as utils_eval
import utils



CONFIG_PATH = 'config.json'

config = utils_eval.load_data(CONFIG_PATH)
model_names = ['idefics'] #config['model_names']
dataset_names = config['dataset_names']
mode_names = ['soft', 'hard']


def average(values):
    return sum(values) / len(values) if values else 0

for mode_name in mode_names:
    print(f"Mode: {mode_name}")
    for model in model_names:
        model_scores = {}
        for dataset_name in dataset_names:
            scores_file_path = f"experiments/{model}/{dataset_name}/run1/scores_{mode_name}.json"
            file_data = utils_eval.load_data(scores_file_path)
            DatasetInfo = utils.DatasetInfo(dataset_name)
            tasks = DatasetInfo.get_tasks()
            dataset_scores = {task: file_data[task]['accuracy'] for task in tasks}
            dataset_scores_average = average(dataset_scores.values())
            model_scores[dataset_name] = dataset_scores_average
        model_scores_average = average(model_scores.values())
        print(f"{model} Average Accuracy: {model_scores_average}")
