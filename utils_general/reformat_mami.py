'''
manual steps:
1. create a new folder in mami called "images"
2. create a new fubfolder in mami/images called "all"
3. move all images in mami/images/all

automated steps:
1. renaming of files e.g. from train_1.jsonl to train.jsonl
2. transforming jsonl files to json files
3. moving them to datasets/mami
4. delete folder "splits"

'''




import json
import glob
import os

# Define the directory containing your .jsonl files
directory = "datasets/mami/splits"
output_directory = "datasets/mami"
splits_folder_path = 'datasets/mami/splits'


# Define mapping of old filename to new filename
filename_mapping = {
    #"test_1.jsonl": "test.jsonl",
    "validation_1.jsonl": "val.jsonl",
    "train_1.jsonl": "train.jsonl",
}

# Loop through each jsonl file in the directory
for old_filename, new_filename in filename_mapping.items():
    old_filepath = os.path.join(directory, old_filename)
    new_filepath_jsonl = os.path.join(directory, new_filename)
    new_filepath_json = os.path.join(output_directory, new_filename.replace('.jsonl', '.json'))
    
    # Rename the file
    os.rename(old_filepath, new_filepath_jsonl)

    # Initialize an empty list to store all json objects
    all_json_objects = []

    # Load json objects from each line in the renamed jsonl file and extend the list
    with open(new_filepath_jsonl) as f:
        all_json_objects.extend(json.loads(line) for line in f)

    # Write the combined json objects into the output file
    with open(new_filepath_json, 'w') as f:
        json.dump(all_json_objects, f)

    # Delete the jsonl file after converting to json
    os.remove(new_filepath_jsonl)




# Delete the folder
os.rmdir(splits_folder_path)