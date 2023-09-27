import json

# File path
dataset_path = "datasets/hateful_memes/ds_benchmark.json"

# Read the dataset
with open(dataset_path, 'r') as f:
    data = json.load(f)

# Iterate through the data and replace classification_label value as needed
for item in data["data"]:
    if item["classification_label"] == "not-hateful":
        item["classification_label"] = "not hateful"

# Write the modified data back to the file
with open(dataset_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Dataset updated successfully!")
