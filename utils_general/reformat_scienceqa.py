import json
from collections import OrderedDict

# Load original dataset
with open("datasets/scienceqa/problems.json", 'r') as file:
    data = json.load(file)

# Initialize empty lists for train, val and test
train_data = []
val_data = []
test_data = []

# Iterate through data and separate into train, val and test
for key, value in data.items():
    if value['image'] is not None:
        ordered_value = OrderedDict()
        ordered_value['input_id'] = key  # Add the key as 'input_id' field
        for sub_key, sub_value in value.items():  # iterate over original dictionary
            ordered_value[sub_key] = sub_value  # creating new ordered dictionary with 'input_id' as the first key
        if ordered_value['split'] == 'train':
            train_data.append(ordered_value)
        elif ordered_value['split'] == 'val':
            val_data.append(ordered_value)
        elif ordered_value['split'] == 'test':
            test_data.append(ordered_value)

# Save separated datasets into their respective json files
with open("datasets/scienceqa/train.json", 'w') as file:
    json.dump(train_data, file, indent=2)

with open("datasets/scienceqa/val.json", 'w') as file:
    json.dump(val_data, file, indent=2)

with open("datasets/scienceqa/test.json", 'w') as file:
    json.dump(test_data, file, indent=2)
