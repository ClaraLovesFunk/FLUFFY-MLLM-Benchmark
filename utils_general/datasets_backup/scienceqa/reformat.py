import json

# Define the absolute paths for the original and new data files
path_prefix = 'datasets/scienceqa/'
original_file_path = path_prefix + 'ds_original.json'
benchmark_file_path = path_prefix + 'ds_benchmark.json'

# Load the original data
with open(original_file_path, 'r') as f:
    original_data = json.load(f)

# Process and reformat the data
reformatted_data = []
for entry in original_data:
    # Create the new data entry format
    new_entry = {
        "text_input_id": entry["input_id"],
        "image_id": f"{entry['input_id']}/{entry['image']}", # Updating the image_id format
        "text_input": entry["question"],
        "answer_choices": entry["choices"],
        "correct_choice": entry["answer"]
    }
    reformatted_data.append(new_entry)

# Create the final structure for the new data file
final_structure = {
    "split": "test",
    "data": reformatted_data
}

# Write the reformatted data to ds_benchmark.json
with open(benchmark_file_path, 'w') as f:
    json.dump(final_structure, f, indent=2)

print("Data has been successfully reformatted and saved in ds_benchmark.json.")
