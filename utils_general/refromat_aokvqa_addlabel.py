import json

# Path to the JSON file
file_path = 'datasets/aokvqa/ds_benchmark.json'

# Load the data from the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Process each dictionary in the 'data' list
for entry in data["data"]:
    entry["correct_multiple_choice_answer"] = entry["answer_choices"][entry["correct_choice"]]

# Save the modified data back to the JSON file
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

# Confirm that the file has been updated
"Data file updated successfully."
