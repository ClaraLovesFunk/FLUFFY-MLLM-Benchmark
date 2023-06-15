import json
from PIL import Image
from PIL import ImageOps
import json
from lavis.models import load_model_and_preprocess
import torch
import os
import time
from utils import *




time_run_script_start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"




# directory variables

datasets_dir = 'datasets'
experiments_dir = 'experiments'




def get_model(model_name, device):


    time_load_model_start = time.time()

    model_info = ModelInfo(model_name)
    lavis_model_type = model_info.get_lavis_model_type()
    lavis_name = model_info.get_lavis_name()

    model, vis_processors, _ = load_model_and_preprocess(name = lavis_name, model_type = lavis_model_type, is_eval = True, device = device)
    
    time_load_model_end = time.time()
    time_loading_model = (time_load_model_end - time_load_model_start)/60
    print(f'Time to load model "{model_name}": {time_loading_model:.2f} min')


    return model, vis_processors, time_loading_model




def get_dataset_text(dataset_name):


    time_load_data_text_start = time.time()

    dataset_info = DatasetInfo(dataset_name)
    text_dataset_split = dataset_info.get_text_dataset_split()
    dataset_file_path = os.path.join(datasets_dir, dataset_name, text_dataset_split + '.json')
    
    data_text = dataset(dataset_name, dataset_file_path).load()

    time_load_data_text_end = time.time()
    time_loading_data_text = (time_load_data_text_end - time_load_data_text_start)/60


    return data_text, time_loading_data_text




def gen_output(device, dataset_name, data_text, model, vis_processors, prompt_construct):


    dataset_info = DatasetInfo(dataset_name)
    img_dataset_split = dataset_info.get_img_dataset_split() 
    image_dataset_name = dataset_info.get_img_dataset_name() 
    tasks = dataset_info.get_tasks() 

    images_dir_path = os.path.join(datasets_dir, image_dataset_name, img_dataset_split)
 
    pred = [] 
    time_inference = {}

    for t in tasks:
        
        time_task_inference_start = time.time()
        
        output_task = 'output_' + t

        for sample in data_text:

            # prep image

            image_file_path =  os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")
            
            image_raw = Image.open(image_file_path) 
            
            if image_raw.mode != 'RGB': 
                image_raw = ImageOps.colorize(image_raw, 'black', 'white')

            image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)

            # make prompt

            prompt = prompt_construct(test_sample = sample,task = t)

            # generate output

            output = model.generate({"image": image, "prompt": prompt})

            sample.update({output_task: output})
            pred.append(sample)

        time_task_inference_end = time.time()
        time_task_inference = (time_task_inference_end - time_task_inference_start)/60
        print(f'Time to perform inference for task "{t}": {time_task_inference:.2f} min')
        time_inference[t] = time_task_inference


    return pred, time_inference



def save_output(pred, model_name, dataset_name, run, check_create_experiment_dir, time_loading_model, time_loading_data_text, time_inference):


    # save output

    experiment_dir_path = os.path.join(experiments_dir, model_name, dataset_name, 'run' + str(run))
    experiment_output_file_path = os.path.join(experiment_dir_path, 'output.json')

    check_create_experiment_dir(experiment_dir_path) 

    with open(experiment_output_file_path, 'w') as f: 
        json.dump(pred,f)


    # save config info

    experiment_config_file_path = os.path.join(experiment_dir_path, 'config.json')

    config = {}

    run_times = {}
    run_times['load model'] = time_loading_model
    run_times['load text data'] = time_loading_data_text
    run_times['inference'] = time_inference

    config['model'] = model_name
    config['dataset'] = dataset_name
    config['run'] = run
    config['run times'] = run_times

    with open(experiment_config_file_path, 'w') as f: 
        json.dump(config,f)








###########################################################################################
###########################################################################################
################################   RUN THE DAMN THING   ###################################
###########################################################################################
###########################################################################################




model_name = ['blip2']
dataset_name = ['okvqa', 'aokvqa']
run = [1]




for m in model_name:

    model, vis_processors, time_loading_model = get_model(model_name = m, device = device)

    for ds in dataset_name:

        data_text, time_loading_data_text = get_dataset_text(ds)

        for r in run:
            print('\n')
            print('-------------------------------------------------------------------')
            print(f'EXPERIMENT "{m} x {ds} x run {str(r)}"')
            print('\n')
        
            pred, time_inference = gen_output(device = device, dataset_name = ds, data_text = data_text, model = model, vis_processors = vis_processors, prompt_construct = prompt_construct)
            save_output(pred = pred, model_name = m, dataset_name = ds, run = r, check_create_experiment_dir = check_create_experiment_dir, time_loading_model = time_loading_model, time_loading_data_text = time_loading_data_text, time_inference = time_inference)
            
            print('-------------------------------------------------------------------')




time_run_script_end = time.time()
time_run_script = (time_run_script_end - time_run_script_start)/60/60
print('\n')
print('\n')
print('\n')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('---------------------------- THE END ------------------------------')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('\n')
print('\n')
print('\n')
print(f"Total run time: {time_run_script:.2f} h")