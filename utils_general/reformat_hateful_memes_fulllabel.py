import json

# Define the mapping of labels
label_mapping = {
    1: "hateful",
    0: "not-hateful"
}

# Read the original JSON file
with open("datasets/hateful_memes/ds_benchmark.json", "r") as file:
    dataset = json.load(file)

# Modify the labels
for entry in dataset['data']:
    entry['classification_label'] = label_mapping[entry['classification_label']]

# Write to the new JSON file
with open("datasets/hateful_memes/ds_benchmark_fulllabel.json", "w") as file:
    json.dump(dataset, file, indent=4)

print("Conversion completed successfully!")
