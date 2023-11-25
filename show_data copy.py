#%% 

import os
import json
import utils
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64




FLAG_SHOW_OUTPUT = True
FLAG_SHOW_GOOD_EXAMPLES = False
FLAG_SHOW_BAD_EXAMPLES = False

n_examples = 2
n_good_examples = 2
n_bad_examples = 2


with open('config.json', 'r') as file:
    config = json.load(file)

datasets_dir = config['datasets_dir']
experiments_dir = config['experiments_dir']
eval_file_name = config['eval_file_name']
output_file_name = config['output_file_name']
examples_file_name = config['examples_file_name'] 
dataset_name_all = ['aokvqa'] #config['dataset_names'] 
model_name = 'llava'
run = '1'






def image_to_html(image):
    # Save the image to a temporary location
    image.save('temp.png')
    # Encode the image to base64 string
    with open('temp.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    # Remove the temporary file
    os.remove('temp.png')
    # Return the HTML img tag with the encoded image and explicit width and height
    return f'<img src="data:image/png;base64,{encoded_image}" width="{image.width}px" height="{image.height}px"/>'


def show_imgs_text_output(dataset_name, data_samples_input, data_samples_output, images_dir, tasks):
    
    data_incl_image = []

    for i, sample in enumerate(data_samples_input): 

        image_file_name = sample['image_id']
        image_path = os.path.join(images_dir, image_file_name)
        img = Image.open(image_path)

        # Calculate new size, maintaining aspect ratio
        scaling_factor = 2  # for example, to double the size
        new_size = (int(img.size[0] * scaling_factor), int(img.size[1] * scaling_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # add images to textual data
        dict_fulldata = {'image': image_to_html(img)}
        # dict_fulldata.update(sample)  # This line is commented, consider if you need to add it.

        # add output
        dict_fulldata.update(data_samples_output[i])

        data_incl_image.append(dict_fulldata)

    # turn into dataframe
    data_incl_image = pd.DataFrame(data_incl_image)

    # set pandas display option
    pd.set_option('display.max_colwidth', None)
    display(HTML(data_incl_image.to_html(escape=False)))
    print('\n')
    print('\n')

    return data_incl_image



def get_examples(data_output_examples, n_good_examples, n_bad_examples):

    examples = data_output_examples[t] # these examples are only for easy direct answers, since aokvqa's evaluation also only regards those
        
    id_good_examples = []
    id_bad_examples = []

    for key, value in examples.items():
        if value == 1:
            id_good_examples.append(key)
        if value == 0:
            id_bad_examples.append(key)

    id_good_examples = id_good_examples[:n_good_examples]
    id_bad_examples = id_bad_examples[:n_bad_examples]
    
    return id_good_examples, id_bad_examples


def load_all_data(ds_file_path, output_file_path, examples_file_path):

    with open(ds_file_path, 'r') as f:
        data_text = json.load(f)
        data_text = data_text['data']
        
    with open(output_file_path, 'r') as f:
        output = json.load(f)

    with open(examples_file_path, 'r') as f:
        example = json.load(f)

    return data_text, output, example






for dataset_name in dataset_name_all:

    tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split = utils.get_info(dataset_name, model_name, run)
    scores_file_path = os.path.join(output_dir_path, eval_file_name)
    examples_file_path = os.path.join(output_dir_path, examples_file_name)

    data_text, output, example = load_all_data(ds_file_path, output_file_path, examples_file_path)

    for t in tasks: 

        #id_good_examples, id_bad_examples = get_examples(example, n_good_examples, n_bad_examples)
    
        print(f'Dataset: {dataset_name}, Task: {t}')

        if FLAG_SHOW_OUTPUT:
            print('Output')
            data_samples_input = data_text[:n_examples]

            print(data_samples_input)

            data_samples_output = output[:n_examples]
            data_incl_image = show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,image_dir_path, tasks = tasks)

        if FLAG_SHOW_GOOD_EXAMPLES:
            print('Correct Output')
            input_id_name = 'text_input_id'
            
            data_samples_input = [d for d in data_text if str(d['text_input_id']) in id_good_examples]
            data_samples_output = [d for d in output if str(d['text_input_id']) in id_good_examples]
            
            data_incl_image = show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,image_dir_path, tasks = tasks)

        if FLAG_SHOW_BAD_EXAMPLES:
            print('Incorrect Output')
            input_id_name = utils.dataset_info.get_input_id_name()

            data_samples_input = [d for d in data_text if str(d['text_input_id']) in id_bad_examples]
            data_samples_output = [d for d in output if str(d['text_input_id']) in id_bad_examples]
            
            data_incl_image = show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,image_dir_path, tasks = tasks)


# %%
