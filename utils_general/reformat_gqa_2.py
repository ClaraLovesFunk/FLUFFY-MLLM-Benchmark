import json

# Load the existing json file
with open('datasets/gqa/val.json', 'r') as file:
    data = json.load(file)

# Create a new list of dictionaries
new_data = []
for input_id, contents in data.items():
    new_entry = {'input_id': input_id}
    new_entry.update(contents)
    new_data.append(new_entry)

# Save the new data structure to a new file
with open('datasets/gqa/new_val.json', 'w') as file:
    json.dump(new_data, file, indent=4)
