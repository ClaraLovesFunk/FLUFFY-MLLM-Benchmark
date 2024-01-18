import json
import csv
import os
import shutil

from .utils import DatasetInfo
from .utils import get_task2label_name
from .data_loading import load_data


    
    
def save_data(filepath, file):
    with open(filepath, 'w') as f: 
        return json.dump(file,f, indent=4)
    


def make_examples_file(model_name, dataset_name, run):
    run = "run"+run
    output_path = os.path.join('experiments', model_name, dataset_name, run, 'output_aux_hard.json')
    examples_path = os.path.join('examples', model_name, dataset_name, run, 'examples.json')
    input_data_path = os.path.join('datasets', dataset_name, 'ds_benchmark.json')
    destination_dir = os.path.dirname(examples_path)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    shutil.copyfile(output_path, examples_path)

    with open(examples_path, 'r') as file:
        output = json.load(file)

    with open(input_data_path, 'r') as file:
        data = json.load(file)
        input_data = data["data"]

    dataset_info_instance = DatasetInfo(dataset_name)
    tasks = dataset_info_instance.get_tasks()
    
    examples = []
    
    for output_i in output:
        example_i = {}
        text_input_id = output_i.get("text_input_id")
        for item in input_data:
            if item.get("text_input_id") == text_input_id:
                example_i["image_id"] = item.get("image_id")
                example_i["text_input_id"]= text_input_id
                for task in tasks:
                    label_name = get_task2label_name(task)
                    example_i["prompt_" + task] = output_i.get("prompt_" + task, "")
                    example_i["output_" + task] = output_i.get("output_" + task, "")
                    example_i[label_name] = item.get(label_name)
                break
        
        examples.append(example_i)
    
    # Write the examples to a CSV file
    headers = examples[0].keys()
    with open(examples_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for example in examples:
            writer.writerow(example)

    # Save the updated examples back to examples.json
    with open(examples_path, 'w') as file:
        json.dump(examples, file, indent=4)

    return None


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}", f"{image_id:012}.jpg")


def check_create_experiment_dir(experiment_dir_path):

    if not os.path.exists(experiment_dir_path):

        os.makedirs(experiment_dir_path)


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