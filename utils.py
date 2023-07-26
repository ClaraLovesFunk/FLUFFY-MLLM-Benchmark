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



def prompt_construct(test_sample, task):

    question_formal = 'Questions: '
    choices_formal = 'Choices: '
    choices_end_formal = '.'
    answer_formal = 'Answer: '
    tweet_formal = 'Tweet: '
    meme_text_formal = 'Meme Text: '
    sentiment_formal = 'Sentiment: '
    sexist_label_formal = 'Sexism Label: '
    hate_label_formal = 'Hate Label: '
    hypothesis_text_formal = 'Hypothesis: '
    entailment_label = 'Answer: '

     

    if task == 'direct answer': 
        text_input = test_sample['question']
        instruction = 'Answer the following question! '
        prompt =  instruction +  '\n' + question_formal +  text_input +  '\n' + answer_formal

    if task == 'multiple choice': 
        text_input = test_sample['question']
        instruction = 'Answer the following question by selecting from the choices below! '
        choices_content = test_sample['choices']
        choices_content = ', '.join(choices_content)
        prompt =  instruction +  '\n' + question_formal +  text_input + '\n' + choices_formal  + choices_content + choices_end_formal +  '\n' + answer_formal
    
    if task == 'multiple choice (sqa)': 
        text_input = test_sample['question']
        instruction = 'Answer the following question by giving the zero-based index of the selected answer. If you want to select the first answer, write "0", if you want to select the second answer, write "1" and so on.'
        choices_content = test_sample['choices']
        choices_content = ', '.join(choices_content)
        prompt =  instruction +  '\n' + question_formal +  text_input + '\n' + choices_formal  + choices_content + choices_end_formal +  '\n' + answer_formal
    
    if task == 'sentiment analysis':
        text_input = test_sample['text']
        instruction = 'Predict the sentiment of the tweet in combination with the image! The sentiment can be either "Positive", "Negative" or "Neutral".'
        prompt =  instruction +  '\n' +  tweet_formal + text_input + '\n' + sentiment_formal

    if task == 'sexism classification':
        text_input = test_sample['text']
        instruction = "Classify the following meme as 'sexist' or 'not-sexist'."
        prompt =  instruction +  '\n' +  meme_text_formal + text_input + '\n' + sexist_label_formal

    if task == 'hate classification':
        text_input = test_sample['text']
        instruction = "Classify the following meme as 'hateful' or 'not-hateful'."
        prompt =  instruction +  '\n' +  meme_text_formal + text_input + '\n' + hate_label_formal

    if task == 'entailment prediction':
        text_input = test_sample['sent']
        instruction = '''
                        Classify the following image as 'entailment', if there is enough evidence in the image to conclude that the following hypothesis is true. 
                        Classify the following image as 'contradiction', if there is enough evidence in the image to conclude that the following hypothesis is false.
                        Classify the following image as 'neutral', if neither of the earlier two are true.'''
        prompt =  instruction +  '\n' +  hypothesis_text_formal + text_input + '\n' + entailment_label


    return prompt




class DatasetInfo():

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

        self.text_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'val',
            'mvsa': 'test',
            'mami': 'test',
            'hateful_memes': 'dev', 
            'clevr': 'val',
            'esnlive': 'test',
            'gqa': 'val',
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
            'aokvqa': 'coco2017',
            'okvqa': 'coco2017', 
            'mvsa': 'mvsa/images',
            'mami': 'mami/images',
            'hateful_memes': 'hateful_memes/images',
            'clevr': 'clevr/images',
            'esnlive': 'flickr30k_images',
            'gqa': 'gqa/images',
            'scienceqa': 'scienceqa/images'
        }

        self.tasks = {
            'aokvqa': ['direct answer', 'multiple choice'],  #### 
            'okvqa': ['direct answer'],
            'mvsa': ['sentiment analysis'],
            'mami': ['sexism classification'],
            'hateful_memes': ['hate classification'],
            'clevr': ['direct answer'],
            'esnlive': ['entailment prediction'],
            'gqa': ['direct answer'],
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
            'blip2': 'pretrain_flant5xxl'
        }
        self.lavis_name = {
            'blip2': 'blip2_t5'
        }
        
    def get_lavis_model_type(self):
        return self.lavis_model_type[self.model_name]
    
    def get_lavis_name(self):
        return self.lavis_name[self.model_name]
    


class dataset():

    def __init__(self, dataset_name, dataset_path):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def load(self):

        with open(self.dataset_path, 'r') as f:
            X_text = json.load(f)

        if self.dataset_name == 'okvqa':
            X_text = X_text['questions']
        elif self.dataset_name == 'clevr':
            X_text = X_text['questions']      

        return X_text
    





def check_create_experiment_dir(experiment_dir_path):

    if not os.path.exists(experiment_dir_path):

        os.makedirs(experiment_dir_path)




def get_img_path(dataset_name, images_dir_path, sample):

    if dataset_name =='okvqa': 
        img_path = os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")

    if dataset_name =='aokvqa': 
        img_path = os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")
    
    if dataset_name =='mvsa': 
        img_path = os.path.join(images_dir_path, sample['id'])

    if dataset_name =='mami': 
        img_path = os.path.join(images_dir_path, sample['id'])

    if dataset_name =='hateful_memes': 
        img_path = os.path.join(images_dir_path, sample['image_path'])

    if dataset_name =='clevr': 
        img_path = os.path.join(images_dir_path, sample['image_filename'])

    if dataset_name =='esnlive': 

        # get sample
        filename = sample['img_id']

        # Extract the numeric part of the filename
        numeric_part = filename.split("_")[1].split(".")[0]

        # Convert the numeric part to integer to remove leading zeros, and then back to string
        numeric_part = str(int(numeric_part))

        # Append ".jpg" to the numeric part
        new_filename = f"{numeric_part}.jpg"

        img_path = os.path.join(images_dir_path, new_filename)

    if dataset_name =='gqa': 
        img_path = os.path.join(images_dir_path, sample['imageId'] + '.jpg')

    if dataset_name =='scienceqa': 
        img_path = os.path.join(images_dir_path, sample['input_id'],'image.png')

    
    return img_path


def get_text_input_id(dataset_name, sample):
    
    if dataset_name =='okvqa': 
        text_input_id = sample['question_id']

    if dataset_name =='aokvqa': 
        text_input_id = sample['question_id']
    
    if dataset_name =='mvsa': 
        text_input_id = sample['id'] 

    if dataset_name =='mami': 
        text_input_id = sample['id'] 

    if dataset_name =='hateful_memes': 
        text_input_id = sample['id']

    if dataset_name =='clevr': 
        text_input_id = sample['input_id']

    if dataset_name =='esnlive': 
        text_input_id = sample['question_id']

    if dataset_name =='gqa': 
        text_input_id = sample['input_id']

    if dataset_name =='scienceqa': 
        text_input_id = sample['input_id']
    
    return text_input_id

