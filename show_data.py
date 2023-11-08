#%% 

import os
import json
import utils
import show_data_modules
import os
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64
import pandas as pd
from PIL import Image
import prompts



FLAG_SHOW_OUTPUT = True
FLAG_SHOW_GOOD_EXAMPLES = True
FLAG_SHOW_BAD_EXAMPLES = True

dataset_name_all = ['mvsa'] 
model = 'llava'
run = 1
n_examples = 4
n_good_examples = 2
n_bad_examples = 2


datasets_dir = 'datasets'
experiments_dir = 'experiments'
eval_file_name = 'scores.json'
output_file_name = 'output.json'
examples_file_name = 'examples.json' 



def image_to_html(image):
    image.save('temp.png')
    with open('temp.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    os.remove('temp.png')
    return f'<img src="data:image/png;base64,{encoded_image}"/>'



def show_imgs_text_output(dataset_name, data_samples_input, data_samples_output, images_dir, tasks):
    data_incl_image = []

    for i, sample in enumerate(data_samples_input): # Notice the use of enumerate


        image_file_name = sample['image_id']
        image_path = os.path.join(images_dir, image_file_name)
        img = Image.open(image_path)
        img.thumbnail((100, 100))

        # add images to textual data
        dict_fulldata = {'image': image_to_html(img)}
        dict_fulldata.update(sample) # Use 'sample' instead of 'data_samples_input[i]'

        # add prompt
        for t in tasks:
            prompt = prompts.zeroshot(sample, t) # Use 'sample' instead of 'data_samples_input[i]'
            prompt_header = 'prompt ' + t
            dict_fulldata.update({prompt_header: prompt})

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


for dataset_name in dataset_name_all:

    dataset_info = utils.DatasetInfo(dataset_name)
    text_dataset_split = dataset_info.get_text_dataset_split()
    img_dataset_name = dataset_info.get_img_dataset_name()
    img_dataset_split = dataset_info.get_img_dataset_split() 
    image_dataset_name = dataset_info.get_img_dataset_name()
    tasks = dataset_info.get_tasks()

    ds_text_file_path = os.path.join(datasets_dir, dataset_name, 'ds_benchmark.json')

    ds_images_dir_path = os.path.join(datasets_dir, image_dataset_name)

    experiment_dir_path = os.path.join(experiments_dir, model, dataset_name, 'run' + str(run))
    experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
    experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
    experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)



    # load data

    with open(ds_text_file_path, 'r') as f:
        data_input = json.load(f)
        data_input = data_input['data']
        
    with open(experiment_output_file_path, 'r') as f:
        data_output = json.load(f)

    with open(experiment_examples_file_path, 'r') as f:
        data_output_examples = json.load(f)

    

    # get indice of n good/bad examples

    tasks = dataset_info.get_tasks()

    for t in tasks: 

        examples = data_output_examples[t] # these examples are only for easy direct answers, since aokvqa's evaluation also only regards those
        

        id_good_examples = []
        id_bad_examples = []

        for key, value in examples.items():
            if value == 1:
                id_good_examples.append(key)
            if value == 0:
                id_bad_examples.append(key)

        id_good_examples = id_good_examples[:n_good_examples]
        print(id_good_examples)
        
        id_bad_examples = id_bad_examples[:n_bad_examples]



        # show examples

        print(f'Dataset: {dataset_name}, Task: {t}')

        if FLAG_SHOW_OUTPUT:
            print('Output')
            data_samples_input = data_input[:n_examples]

            print(data_samples_input)

            data_samples_output = data_output[:n_examples]
            data_incl_image = show_data_modules.show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,ds_images_dir_path, tasks = tasks)

        if FLAG_SHOW_GOOD_EXAMPLES:
            print('Correct Output')
            input_id_name = dataset_info.get_input_id_name()
            
            data_samples_input = [d for d in data_input if str(d['text_input_id']) in id_good_examples]
            data_samples_output = [d for d in data_output if str(d['text_input_id']) in id_good_examples]
            
            data_incl_image = show_data_modules.show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,ds_images_dir_path, tasks = tasks)

        if FLAG_SHOW_BAD_EXAMPLES:
            print('Incorrect Output')
            input_id_name = dataset_info.get_input_id_name()

            data_samples_input = [d for d in data_input if str(d['text_input_id']) in id_bad_examples]
            data_samples_output = [d for d in data_output if str(d['text_input_id']) in id_bad_examples]
            
            data_incl_image = show_data_modules.show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,ds_images_dir_path, tasks = tasks)


# %%
