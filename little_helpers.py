import os
import json
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64
import random


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



def image_to_html(image):
    image.save('temp.png')
    with open('temp.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    os.remove('temp.png')
    return f'<img src="data:image/png;base64,{encoded_image}"/>'



def demo_sample_select(demo_strategy = 'random', train_data):

    if demo_strategy == 'random':
        

        # choose random sample

        random_i = random.randint(0, len(train_data)) 


        # get visual and textual data
        
        image_path = get_coco_path('val', train_data[random_i]['image_id'], images_input_dir)
        image = Image.open(image_path) 
        question = train_data[random_i]['question'] 

    


    


    return None



def prompt_construct(task = 'direct_answer', ): #task = [direct_answer, MC_answer]
    
    question_formal = 'Questions: '
    choices_formal = 'Choices: '
    answer_formal = 'Answer: '

    question = 'What is on the image?' #####

    if task == 'direct_answer':
        instruction = 'Answer the following question! '
    if task == 'MC_answer':
        instruction = 'Answer the following question by selecting from the choices below! '
        choices = 'ans1, ans2, ans3, ans4 '

    