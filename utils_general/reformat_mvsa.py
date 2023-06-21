#!/usr/bin/env python3

'''
first, the mvsa directory needs to be reorganized:

- creation of subfolder "train", "val", "test"
- jsonl files were moved to the respective directories
- creation of new subfolder mvsa/images
- creation of new subfolder mvsa/images/all
- all images were moved to mvsa/images/all
'''

import json
import glob

# Define the directory containing your .jsonl files
directory = "datasets/mvsa/test"

# Define output filename
outfile = "datasets/mvsa/test.json"

# Initialize an empty list to store all json objects
all_json_objects = []

# Loop through each jsonl file in the directory
for jsonlfile in glob.glob(directory + "/*.jsonl"):
    with open(jsonlfile) as f:
        # Load json objects from each line in the current jsonl file and extend the list
        all_json_objects.extend(json.loads(line) for line in f)

# Write the combined json objects into the output file
with open(outfile, 'w') as f:
    json.dump(all_json_objects, f)
