import json
import shutil
import os
import utils

model_name = 'blip2'
dataset_name = 'hateful_memes'
run = 'run1'
task = 'hate classification'
label_name = 'classification_label'

def make_examples_file(model_name, dataset_name, run, task, label_name):
    output_path = os.path.join('experiments', model_name, dataset_name, run, 'output_aux_hard.json')
    examples_path = os.path.join('examples', model_name, dataset_name, run, 'examples.json')
    input_data_path = os.path.join('datasets', dataset_name, 'ds_benchmark.json')
    destination_dir = os.path.dirname(examples_path)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    shutil.copyfile(output_path, examples_path)

    with open(examples_path, 'r') as file:
        examples = json.load(file)

    with open(input_data_path, 'r') as file:
        data = json.load(file)
        input_data = data["data"]

    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()

    for example in examples:
        text_input_id = example.get("text_input_id")
        for item in input_data:
            if item.get("text_input_id") == text_input_id:
                example["image_id"] = item.get("image_id")
                for task in tasks:
                    label_name = utils.get_task2label_name(task)
                    example["prompt_" + task] = item.get("prompt_" + task, "")
                    example["output_" + task] = item.get("output_" + task, "")
                    example["label"] = item.get(label_name)
                break

    # Save the updated examples back to examples.json
    with open(examples_path, 'w') as file:
        json.dump(examples, file, indent=4)

    return None

make_examples_file(model_name, dataset_name, run, task, label_name)
