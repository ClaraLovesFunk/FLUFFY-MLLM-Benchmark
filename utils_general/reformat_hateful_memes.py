'''
manual steps:
1. create a new folder in datasets/hateful_memes called "imagess"
2. create a new fubfolder in datasets/hateful_memes/imagess called "all"
3. move all images from datasets/hateful_memes/images to datasets/hateful_memes/imagess/all
4. delete folder images (datasets/hateful_memes/images)
5. transforming jsonl files in datasets/hateful_memes/splits to json files
6. moving them to datasets/hateful_memes
7. delete folder "splits"

'''

import os
import shutil
import json
import jsonlines

'''
# 1. Create a new folder in datasets/hateful_memes called "imagess"
new_folder = 'datasets/hateful_memes/imagess'
os.makedirs(new_folder, exist_ok=True)

# 2. Create a new subfolder in datasets/hateful_memes/imagess called "all"
new_subfolder = 'datasets/hateful_memes/imagess/all'
os.makedirs(new_subfolder, exist_ok=True)

# 3. Move all images from datasets/hateful_memes/images to datasets/hateful_memes/imagess/all
source_folder = 'datasets/hateful_memes/images'
for file_name in os.listdir(source_folder):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        shutil.move(os.path.join(source_folder, file_name), new_subfolder)

# 4. Delete folder images (datasets/hateful_memes/images)
shutil.rmtree(source_folder)

# 5. Transforming jsonl files in datasets/hateful_memes/splits to json files
jsonl_folder = 'datasets/hateful_memes/splits'
for file_name in os.listdir(jsonl_folder):
    if file_name.endswith('.jsonl'):
        with jsonlines.open(os.path.join(jsonl_folder, file_name)) as reader:
            json_data = list(reader)
            with open(os.path.join(jsonl_folder, file_name.replace('.jsonl', '.json')), 'w') as json_file:
                json.dump(json_data, json_file)

# 6. Move json files to datasets/hateful_memes
for file_name in os.listdir(jsonl_folder):
    if file_name.endswith('.json'):
        shutil.move(os.path.join(jsonl_folder, file_name), 'datasets/hateful_memes')

# 7. Delete folder "splits"
shutil.rmtree(jsonl_folder)


old_directory = "datasets/hateful_memes/imagess"
new_directory = "datasets/hateful_memes/images"

# Rename the directory
os.rename(old_directory, new_directory)

'''







# REFORMATTING IMG PATHS INSIDE EACH JSON FILE

import os
import json

# Path to the directory
directory_path = 'datasets/hateful_memes'

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        filepath = os.path.join(directory_path, filename)

        # Load the data from the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Iterate over the list of dictionaries and update the 'img' field
        for item in data:
            # Check if the 'img' key is in the dictionary
            if 'img' in item:
                item['img'] = item['img'].replace('img/', '')

        # Save the updated data back to the JSON file
        with open(filepath, 'w') as f:
            json.dump(data, f)
