import json

# Load the dataset
with open('datasets/mami/ds_benchmark.json', 'r') as file:
    dataset = json.load(file)

# Update classification_label values
for entry in dataset['data']:
    if entry['classification_label'] == "0":
        entry['classification_label'] = 'not sexist'
    elif entry['classification_label'] == "1":
        entry['classification_label'] = 'sexist'

# Save the updated dataset
with open('datasets/mami/ds_benchmark_updated.json', 'w') as file:
    json.dump(dataset, file, indent=4)

print("Dataset updated and saved as ds_benchmark_updated.json")
