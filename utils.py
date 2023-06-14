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



def image_to_html(image):
    image.save('temp.png')
    with open('temp.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    os.remove('temp.png')
    return f'<img src="data:image/png;base64,{encoded_image}"/>'







'''
def sample_select(demo_strategy = 'random', train_data):

    if demo_strategy == 'random':
        

        # choose random sample

        random_i = random.randint(0, len(train_data)) 


        # get visual and textual data
        
        image_path = get_coco_path('val', train_data[random_i]['image_id'], images_input_dir)
        image = Image.open(image_path) 
        question = train_data[random_i]['question'] 

    return None

    '''


def prompt_construct(test_sample, task):

    '''
    task = {direct_answer, MC_answer}
    ''' 

    question_formal = 'Questions: '
    choices_formal = 'Choices: '
    choices_end_formal = '.'
    answer_formal = 'Answer: '

    question_content = test_sample['question'] 

    if task == 'direct_answer':
        instruction = 'Answer the following question! '
        prompt =  instruction +  '\n' + question_formal +  question_content +  '\n' + answer_formal

    if task == 'multiple_choice':
        instruction = 'Answer the following question by selecting from the choices below! '
        choices_content = test_sample['choices']
        choices_content = ', '.join(choices_content)
        prompt =  instruction +  '\n' + question_formal +  question_content + '\n' + choices_formal  + choices_content + choices_end_formal +  '\n' + answer_formal
    
    
    return prompt


def add_imgs_text_data(data_samples, split_sec,images_dir):

    data_incl_image = []
    
    for data_sample_i in data_samples:

        # get images
        image_path = get_coco_path(split_sec, data_sample_i['image_id'], images_dir)
        img = Image.open(image_path)
        img.thumbnail((100, 100))
        
        # add images to textual data
        dict_fulldata = {'image': image_to_html(img)}
        dict_fulldata.update(data_sample_i)
        #dict_fulldata.update({'img_path': image_path})

        data_incl_image.append(dict_fulldata)

    # turn into dataframe  
    data_incl_image = pd.DataFrame(data_incl_image)

    # drop irrelevant info
    #data_incl_image = data_incl_image.drop(['split', 'image_id', 'question_id', 'rationales', 'img_path'],axis=1)

    pd.set_option('display.max_colwidth', None)
    display(HTML(data_incl_image.to_html(escape=False)))
    print('\n')
    print('\n')

    return data_incl_image



class DatasetInfo():

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name
        self.split = {
            'aokvqa': 'val',
            'okvqa': 'val'
        }
        self.img_dataset = {
            'aokvqa': 'coco2017',
            'okvqa': 'coco2014'
        }

        self.tasks = {
            'aokvqa': ['direct_answer', 'multiple_choice'],
            'okvqa': ['direct_answer']
        }
        

    def get_split(self):
        return self.split[self.dataset_name]
    
    def get_img_dataset(self):
        return self.img_dataset[self.dataset_name]

    def get_tasks(self):
        return self.tasks[self.dataset_name]



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

        return X_text
    






def prep_image(device, images_path, sample, vis_processors):

    image_path =  os.path.join(images_path, f"{sample['image_id']:012}.jpg")
    image_raw = Image.open(image_path) 

    if image_raw.mode != 'RGB': 
        image_raw = ImageOps.colorize(image_raw, 'black', 'white')

    image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)

    return image



# check for/create experiment directory

def check_create_experiment_dir(experiment_dir_path):

    if not os.path.exists(experiment_dir_path):

        os.makedirs(experiment_dir_path)