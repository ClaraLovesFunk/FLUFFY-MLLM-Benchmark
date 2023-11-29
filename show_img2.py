#%%

import json
import os
from PIL import Image
import matplotlib.pyplot as plt

model_name = "openflamingo"
dataset_name = "mami"
n_examples=15




img_dir = {
    "mami": "datasets/mami/images/all",
    "hateful_memes": "datasets/hateful_memes/images/all",
    "aokvqa": "",
    "okvqa":"",
    "esnlive": "",
    "mvsa":"datasets/mvsa/images/all",
    "gqa": "",
    "clevr": "",
    "scienceqa":"",
}

def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

def display_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')  
    plt.show()

def show_images_from_json(model_name, dataset_name, n_examples):

    examples_file_path = os.path.join('examples', model_name, dataset_name, 'run1/examples.json')
 
    # Load the JSON data
    data = load_json_data(examples_file_path)

    # Loop over the first n dictionaries
    for entry in data[:n_examples]:
        image_id = entry['image_id']
        image_path = os.path.join(img_dir[dataset_name], image_id)
        
        print(f"Showing image: {image_id}")
        display_image(image_path)
        print(f'prompt: {entry["prompt_sexism classification"]}')
        print(f'output: {entry["output_sexism classification"]}')
        print(f'label: {entry["classification_label"]}')


# Call the function to show the first n images
show_images_from_json(model_name = "openflamingo", dataset_name = "mami", n_examples=n_examples)



               


# %%
