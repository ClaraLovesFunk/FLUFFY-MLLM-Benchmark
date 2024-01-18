import json
import random


# Set a seed for reproducibility
random.seed(0)

# Load original data
with open('datasets/gqa/val.json', 'r') as f:
    data = json.load(f)

# Shuffle data to ensure random selection
random.shuffle(data)

# Initialize a list to store selected instances and a set to store imageIds
selected_data = []
image_ids = set()

for instance in data:
    if instance['imageId'] not in image_ids:
        # Add instance to selected data and its imageId to the set
        selected_data.append(instance)
        image_ids.add(instance['imageId'])

        # Stop when we have 1000 unique instances
        if len(selected_data) == 1000:
            break

# Store sampled data
with open('datasets/gqa/val_sampled.json', 'w') as f:
    json.dump(selected_data, f, indent=4)