import os

import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64

import pandas as pd
from PIL import Image


from utils import *




def image_to_html(image):
    image.save('temp.png')
    with open('temp.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    os.remove('temp.png')
    return f'<img src="data:image/png;base64,{encoded_image}"/>'



def show_imgs_text_output(dataset_name, data_samples_input, data_samples_output,images_dir, tasks):



    data_incl_image = []
    

    for i in range(len(data_samples_input)):

        # get images
        
        if dataset_name == 'aokvqa':
            image_path = os.path.join(images_dir, f"{data_samples_input[i]['image_id']:012}.jpg") 
        if dataset_name == 'hateful_memes':
            image_path = os.path.join(images_dir, f"{int(data_samples_input[i]['id']):05}.png")
        if dataset_name == 'mvsa':
            image_path = os.path.join(images_dir, data_samples_input[i]['id'])
        if dataset_name == 'mami':
            image_path = os.path.join(images_dir, data_samples_input[i]['id'])
        
        img = Image.open(image_path)
        img.thumbnail((100, 100))
        
        # add images to textual data
        dict_fulldata = {'image': image_to_html(img)}
        dict_fulldata.update(data_samples_input[i])

        
        # add prompt
        for t in tasks:
            prompt = prompt_construct(data_samples_input[i], t)
            prompt_header = 'prompt ' + t
            dict_fulldata.update({prompt_header: prompt})
            
            
        


        # add output
        dict_fulldata.update(data_samples_output[i]) 

        data_incl_image.append(dict_fulldata)

    # turn into dataframe  
    data_incl_image = pd.DataFrame(data_incl_image)

    # drop irrelevant info
    if dataset_name == 'aokvqa':
        data_incl_image = data_incl_image.drop(['split', 'image_id', 'question_id', 'rationales', 'text_input_id'],axis=1) #
    if dataset_name == 'mami':
        data_incl_image = data_incl_image.drop(['image_ofa_caption', 'image_vit_caption', 'image_path', 'image_lavis_caption', 'text_input_id'],axis=1)
    if dataset_name == 'mvsa':
        data_incl_image = data_incl_image.drop(['image_ofa_caption', 'image_vit_caption', 'image_path', 'image_lavis_caption', 'text_input_id'],axis=1)
    if dataset_name == 'hateful_memes':
        data_incl_image = data_incl_image.drop(['image_ofa_caption', 'image_vit_caption', 'image_path', 'image_lavis_caption', 'text_input_id'],axis=1)


    pd.set_option('display.max_colwidth', None)
    display(HTML(data_incl_image.to_html(escape=False)))
    print('\n')
    print('\n')

    return data_incl_image
