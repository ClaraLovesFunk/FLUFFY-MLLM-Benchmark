import json
import os

# Define the absolute paths for the original and new data files
path_prefix = 'datasets/esnlive/'
original_file_path = os.path.join(path_prefix, 'ds_original.json')
benchmark_file_path = os.path.join(path_prefix, 'ds_benchmark.json')

# Load the original data from ds_original.json
with open(original_file_path, 'r') as f:
    original_data = json.load(f)

# Reformat the data
reformatted_data = []
for sample in original_data:
    # Process the filename for image_id
    filename = sample['img_id']
    numeric_part = filename.split("_")[1].split(".")[0]
    numeric_part = str(int(numeric_part))
    new_filename = f"{numeric_part}.jpg"
    
    new_item = {
        "text_input_id": sample["question_id"],
        "image_id": new_filename,
        "text_input": sample["sent"],
        "classification_label": sample["label"]
    }
    reformatted_data.append(new_item)

# Structure the data for the ds_benchmark.json file
structured_data = {
    "split": "test",
    "data": reformatted_data
}

# Write the structured data to ds_benchmark.json
with open(benchmark_file_path, 'w') as f:
    json.dump(structured_data, f, indent=2)

print("Data has been successfully reformatted and saved in ds_benchmark.json")
