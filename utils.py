import os
import json
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64
import random
import pandas as pd
from PIL import Image
from PIL import ImageOps
import requests
from io import BytesIO


# from AOKVQA git (https://github.com/allenai/aokvqa#downloading-the-dataset)

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset


# from AOKVQA git (https://github.com/allenai/aokvqa#downloading-the-dataset)

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}", f"{image_id:012}.jpg")







    



class DatasetInfo():

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

        self.text_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'val',
            'mvsa': 'test',
            'mami': 'test',
            'hateful_memes': 'dev', 
            'clevr': 'val_sampled',
            'esnlive': 'test',
            'gqa': 'val_sampled',
            'scienceqa': 'test'
        }
        
        self.img_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'all',
            'mvsa': 'all',
            'mami': 'all',
            'hateful_memes': 'all',
            'clevr': 'val',
            'esnlive': 'all',
            'gqa': 'all',
            'scienceqa': 'test'
        }

        self.img_dataset_name = {
            'aokvqa': 'coco2017/val',
            'okvqa': 'coco2017/all', 
            'mvsa': 'mvsa/images/all',
            'mami': 'mami/images/all',
            'hateful_memes': 'hateful_memes/images/all',
            'clevr': 'clevr/images/val',
            'esnlive': 'flickr30k_images/all',
            'gqa': 'gqa/images/all',
            'scienceqa': 'scienceqa/images/test'
        }

        self.tasks = {
            'aokvqa': ['direct answer (aokvqa)', 'multiple choice (aokvqa)'], 
            'okvqa': ['direct answer (okvqa)'],
            'mvsa': ['sentiment analysis'],
            'mami': ['sexism classification'],
            'hateful_memes': ['hate classification'],
            'clevr': ['direct answer (clevr)'],
            'esnlive': ['entailment prediction'],
            'gqa': ['direct answer (gqa)'],
            'scienceqa': ['multiple choice (sqa)']                                        
        }

        self.input_id_name = {
            'aokvqa': 'question_id',
            'okvqa': 'question_id', 
            'mvsa': 'id',
            'mami': 'id',
            'hateful_memes': 'id',
            'clevr': 'input_id',
            'esnlive': 'question_id',
            'gqa': 'input_id',         #'question_id', #changed to input_id!!!!,
            'scienceqa': 'input_id'
        } 
        

    def get_text_dataset_split(self):
        return self.text_dataset_split[self.dataset_name]
    
    def get_img_dataset_split(self):
        return self.img_dataset_split[self.dataset_name]
    
    def get_img_dataset_name(self):
        return self.img_dataset_name[self.dataset_name]

    def get_tasks(self):
        return self.tasks[self.dataset_name]
    
    def get_input_id_name(self):
        return self.input_id_name[self.dataset_name]



class ModelInfo():

    def __init__(self, model_name):

        self.model_name = model_name
        self.lavis_model_type = {
            'blip2': 'pretrain_flant5xxl',
        }
        self.lavis_name = {
            'blip2': 'blip2_t5'
        }
        
        
    def get_lavis_model_type(self):
        return self.lavis_model_type[self.model_name]
    
    def get_lavis_name(self):
        return self.lavis_name[self.model_name]
    


# class dataset():

#     def __init__(self, dataset_name, dataset_path):

#         self.dataset_name = dataset_name
#         self.dataset_path = dataset_path

#     def load(self):

#         with open(self.dataset_path, 'r') as f:
#             X_text = json.load(f)

#         if self.dataset_name == 'okvqa':
#             X_text = X_text['questions']
#         elif self.dataset_name == 'clevr':
#             X_text = X_text['questions']      

#         return X_text
    





def check_create_experiment_dir(experiment_dir_path):

    if not os.path.exists(experiment_dir_path):

        os.makedirs(experiment_dir_path)




# def get_img_path(dataset_name, images_dir_path, sample):

#     if dataset_name =='okvqa': 
#         img_path = os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")

#     if dataset_name =='aokvqa': 
#         img_path = os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")
    
#     if dataset_name =='mvsa': 
#         img_path = os.path.join(images_dir_path, sample['id'])

#     if dataset_name =='mami': 
#         img_path = os.path.join(images_dir_path, sample['id'])

#     if dataset_name =='hateful_memes': 
#         img_path = os.path.join(images_dir_path, sample['image_path'])

#     if dataset_name =='clevr': 
#         img_path = os.path.join(images_dir_path, sample['image_filename'])

#     if dataset_name =='esnlive': 

#         # get sample
#         filename = sample['img_id']

#         # Extract the numeric part of the filename
#         numeric_part = filename.split("_")[1].split(".")[0]

#         # Convert the numeric part to integer to remove leading zeros, and then back to string
#         numeric_part = str(int(numeric_part))

#         # Append ".jpg" to the numeric part
#         new_filename = f"{numeric_part}.jpg"

#         img_path = os.path.join(images_dir_path, new_filename)

#     if dataset_name =='gqa': 
#         img_path = os.path.join(images_dir_path, sample['imageId'] + '.jpg')

#     if dataset_name =='scienceqa': 
#         img_path = os.path.join(images_dir_path, sample['input_id'],'image.png')

    
#     return img_path


# def get_text_input_id(dataset_name, sample):
    
#     if dataset_name =='okvqa': 
#         text_input_id = sample['question_id']

#     if dataset_name =='aokvqa': 
#         text_input_id = sample['question_id']
    
#     if dataset_name =='mvsa': 
#         text_input_id = sample['id'] 

#     if dataset_name =='mami': 
#         text_input_id = sample['id'] 

#     if dataset_name =='hateful_memes': 
#         text_input_id = sample['id']

#     if dataset_name =='clevr': 
#         text_input_id = sample['input_id']

#     if dataset_name =='esnlive': 
#         text_input_id = sample['question_id']

#     if dataset_name =='gqa': 
#         text_input_id = sample['input_id']

#     if dataset_name =='scienceqa': 
#         text_input_id = sample['input_id']
    
#     return text_input_id






def get_info(dataset_name, model_name, run):

    dataset_info = DatasetInfo(dataset_name)
    img_dataset_name = dataset_info.get_img_dataset_name()
    tasks = dataset_info.get_tasks()
    split = dataset_info.get_text_dataset_split()

    base_path = '/home/users/cwicharz/project/Testing-Multimodal-LLMs'
    ds_file_path = os.path.join(base_path, 'datasets', dataset_name, 'ds_benchmark.json')
    image_dir_path = os.path.join(base_path, 'datasets', img_dataset_name) 
    output_dir_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run)
    output_file_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run, 'output.json' )
    config_file_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run, 'config.json' )

    return tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split



def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image