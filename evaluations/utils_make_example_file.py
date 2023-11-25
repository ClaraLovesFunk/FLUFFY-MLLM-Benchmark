import json
import shutil
import os
import utils



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

    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()
    
    examples = []
    
    for output_i in output:
        example_i = {}
        text_input_id = output_i.get("text_input_id")
        for item in input_data:
            if item.get("text_input_id") == text_input_id:
                example_i["image_id"] = item.get("image_id")
                example_i["text_input_id"]= text_input_id
                for task in tasks:
                    label_name = utils.get_task2label_name(task)
                    example_i["prompt_" + task] = output_i.get("prompt_" + task, "")
                    example_i["output_" + task] = output_i.get("output_" + task, "")
                    example_i[label_name] = item.get(label_name)
                break
        
        examples.append(example_i)

    # Save the updated examples back to examples.json
    with open(examples_path, 'w') as file:
        json.dump(examples, file, indent=4)

    return None