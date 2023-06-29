#%% 

import json
from utils import *
from show_data_modules import *




# experiment variables

dataset = 'aokvqa'
model = 'blip2'
run = 1
n_good_examples = 3
n_bad_examples = 3



# directory variables

datasets_dir = 'datasets'
experiments_dir = 'experiments'
eval_file_name = 'scores.json'
output_file_name = 'output.json'
examples_file_name = 'examples.json' 



# paths

dataset_info = DatasetInfo(dataset)
text_dataset_split = dataset_info.get_text_dataset_split()
img_dataset_name = dataset_info.get_img_dataset_name()
img_dataset_split = dataset_info.get_img_dataset_split() 
image_dataset_name = dataset_info.get_img_dataset_name()
tasks = dataset_info.get_tasks()

ds_text_file_path = os.path.join(datasets_dir, dataset, text_dataset_split + '.json')

ds_images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)

experiment_dir_path = os.path.join(experiments_dir, model, dataset, 'run' + str(run))
experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)





# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------




# load data

with open(ds_text_file_path, 'r') as f:
    data_input = json.load(f)

with open(experiment_output_file_path, 'r') as f:
    data_output = json.load(f)

with open(experiment_examples_file_path, 'r') as f:
    data_output_examples = json.load(f)



# get indice of n good/bad examples

examples = data_output_examples['direct answer'] # these examples are only for easy direct answers, since aokvqa's evaluation also only regards those

id_good_examples = []
id_bad_examples = []

for key, value in examples.items():
    if value == 1:
        id_good_examples.append(key)
    if value == 0:
        id_bad_examples.append(key)

id_good_examples = id_good_examples[:n_good_examples]
id_bad_examples = id_bad_examples[:n_bad_examples]



# show examples

print('Correct Output')
data_samples_input = [d for d in data_input if d['question_id'] in id_good_examples]
data_samples_output = [d for d in data_output if d['text_input_id'] in id_good_examples]
data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output,ds_images_dir_path, tasks = ['direct answer', 'multiple choice'])


print('Incorrect Output')
data_samples_input = [d for d in data_input if d['question_id'] in id_bad_examples]
data_samples_output = [d for d in data_output if d['text_input_id'] in id_bad_examples]
data_incl_image = add_imgs_text_data(data_samples_input, data_samples_output,ds_images_dir_path, tasks = ['direct answer', 'multiple choice'])


# %%
