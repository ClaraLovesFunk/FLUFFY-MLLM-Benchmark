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

    question_formal = 'Questions: '
    choices_formal = 'Choices: '
    choices_end_formal = '.'
    answer_formal = 'Answer: '
    tweet_formal = 'Tweet: '
    meme_text_formal = 'Meme Text: '
    sentiment_formal = 'Sentiment: '
    sexist_label_formal = 'Sexism Label: '

     

    if task == 'direct answer': ###  underscore deleted
        text_input = test_sample['question']
        instruction = 'Answer the following question! '
        prompt =  instruction +  '\n' + question_formal +  text_input +  '\n' + answer_formal

    if task == 'multiple choice': ###  underscore deleted
        text_input = test_sample['question']
        instruction = 'Answer the following question by selecting from the choices below! '
        choices_content = test_sample['choices']
        choices_content = ', '.join(choices_content)
        prompt =  instruction +  '\n' + question_formal +  text_input + '\n' + choices_formal  + choices_content + choices_end_formal +  '\n' + answer_formal
    
    if task == 'sentiment analysis':
        text_input = test_sample['text']
        instruction = 'Predict the sentiment of the tweet in combination with the image! The sentiment can be either "Positive", "Negative" or "Neutral".'
        prompt =  instruction +  '\n' +  tweet_formal + text_input + '\n' + sentiment_formal

    if task == 'sexism classification':
        text_input = test_sample['text']
        instruction = 'Classify the following meme as sexist or not sexist. If it is sexist, give it the Label "1", if it is not sexist "0"!'
        prompt =  instruction +  '\n' +  meme_text_formal + text_input + '\n' + sexist_label_formal
    
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

        self.text_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'val',
            'mvsa': 'test',
            'mami': 'test'
        }
        
        self.img_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'all',
            'mvsa': 'all',
            'mami': 'all'
        }

        self.img_dataset_name = {
            'aokvqa': 'coco2017',
            'okvqa': 'coco2017', 
            'mvsa': 'mvsa/images',
            'mami': 'mami/images'
        }

        self.tasks = {
            'aokvqa': ['direct answer', 'multiple choice'],  #### 
            'okvqa': ['direct answer'],
            'mvsa': ['sentiment analysis'],
            'mami': ['sexism classification']                                               
        }
        

    def get_text_dataset_split(self):
        return self.text_dataset_split[self.dataset_name]
    
    def get_img_dataset_split(self):
        return self.img_dataset_split[self.dataset_name]
    
    def get_img_dataset_name(self):
        return self.img_dataset_name[self.dataset_name]

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
    





'''
def prep_image(device, images_path, sample, vis_processors):

    image_path =  os.path.join(images_path, f"{sample['image_id']:012}.jpg")
    image_raw = Image.open(image_path) 
    image_raw = image_raw.to(device)

    if image_raw.mode != 'RGB': 
        image_raw = ImageOps.colorize(image_raw, 'black', 'white')

    image = vis_processors["eval"](image_raw).unsqueeze(0).to(device)

    return image'''





def check_create_experiment_dir(experiment_dir_path):
    '''
    check for/create experiment directory
    '''

    if not os.path.exists(experiment_dir_path):

        os.makedirs(experiment_dir_path)


'''class Dataset_util():

    def __init__(self, dataset_name, images_dir_path, sample):
        self.dataset_name = dataset_name
        self.images_dir_path = images_dir_path
        self.sample = sample
        self.img_path = {
            'okvqa': os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg"),
            'aokvqa': os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg"),
            'mvsa': os.path.join(images_dir_path, f"{sample['image_id']}.jpg")
        }
    
    def get_img_path(self):
        return self.img_path[self.dataset_name]

# Create an instance of the Dataset_util class
dataset_util = Dataset_util(dataset_name='mvsa', images_dir_path='path/to/images', sample={'image_id': 'example'})

# Call the get_img_path method
img_path = dataset_util.get_img_path()

print(img_path)
'''


def get_img_path(dataset_name, images_dir_path, sample):

    if dataset_name =='okvqa': 
        img_path = os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")

    if dataset_name =='aokvqa': 
        img_path = os.path.join(images_dir_path, f"{sample['image_id']:012}.jpg")
    
    if dataset_name =='mvsa': 
        img_path = os.path.join(images_dir_path, sample['id'])

    if dataset_name =='mami': 
        img_path = os.path.join(images_dir_path, sample['id'])
    
    return img_path


def get_text_input_id(dataset_name, sample):
    
    if dataset_name =='okvqa': 
        text_input_id = sample['question_id']

    if dataset_name =='aokvqa': 
        text_input_id = sample['question_id']
    
    if dataset_name =='mvsa': 
        text_input_id = sample['id'] # image id is the only identifier

    if dataset_name =='mami': 
        text_input_id = sample['id'] # image id is the only identifier
    
    return text_input_id

