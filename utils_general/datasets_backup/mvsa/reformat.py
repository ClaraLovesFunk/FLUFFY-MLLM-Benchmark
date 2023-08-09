import json

# Define the absolute paths for the original and new data files
path_prefix = 'datasets/mvsa/'
original_file_path = path_prefix + 'ds_original.json'
benchmark_file_path = path_prefix + 'ds_benchmark.json'

# Load the original data
with open(original_file_path, 'r') as f:
    original_data = json.load(f)

# Process and reformat the data
reformatted_data = []
for entry in original_data:
    # Extract just the filename from the image_path
    image_filename = entry["image_path"].split("/")[-1]

    # Create the new data entry format
    new_entry = {
        "text_input_id": entry["id"],
        "image_id": image_filename,
        "text_input": entry["text"],
        "classification_label": entry["label"]
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
