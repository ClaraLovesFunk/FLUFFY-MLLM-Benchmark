import json
import csv
import os
import shutil

from .utils import DatasetInfo
from .utils import get_task2label_name


    
    
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
