import json
from eval_modules import *
from utils import *



# directory variables

datasets_dir = 'datasets'
experiments_dir = 'experiments'
eval_file_name = 'scores.json'
output_file_name = 'output.json'
examples_file_name = 'examples.json' # FIND BETTER NAME!!!!




# experiment variables

model_name = ['blip2']
dataset_name = ['okvqa', 'aokvqa']
run = [1]



for m in model_name:

    for ds in dataset_name:

        dataset_info = DatasetInfo(ds)
        text_dataset_split = dataset_info.get_text_dataset_split()
        img_dataset_name = dataset_info.get_img_dataset_name()
        img_dataset_split = dataset_info.get_img_dataset_split() 
        image_dataset_name = dataset_info.get_img_dataset_name()
        tasks = dataset_info.get_tasks()

        images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)
        dataset_file_path = os.path.join(datasets_dir, dataset_name, text_dataset_split + '.json')
        experiment_dir_path = os.path.join(experiments_dir, m, dataset_name, 'run' + str(run))
        experiment_scores_file_path = os.path.join(experiment_dir_path, eval_file_name)
        experiment_output_file_path = os.path.join(experiment_dir_path, output_file_name)
        experiment_examples_file_path = os.path.join(experiment_dir_path, examples_file_name)



        # load dataset

        data_text = dataset(ds, dataset_file_path).load()



        # load outputs

        with open(experiment_output_file_path, 'r') as f:
            output = json.load(f)



        # compute scores

        scores_alltasks = {}

        for t in tasks:

            scores_task = {}

            # compute all evaluation metrics
    
            for e in eval_metrics:

                # fill out the code

            # store all evaluation metrics associated with the task here: scores_task

            # fill out the code

        # update scores_alltasks, so it holds all tasks and the respective associated evaluation scores

        # save the scores

        with open(experiment_scores_file_path, 'w') as f: 
            json.dump(scores_alltasks,f)