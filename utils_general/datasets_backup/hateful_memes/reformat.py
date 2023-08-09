import json

# Define the absolute paths for the original and new data files
path_prefix = 'datasets/hateful_memes/'
original_file_path = path_prefix + 'ds_original.json'
benchmark_file_path = path_prefix + 'ds_benchmark.json'

# Load the original data
with open(original_file_path, 'r') as f:
    original_data = json.load(f)

# Reformat the data
reformatted_data = []
for item in original_data:
    new_item = {
        "text_input_id": item["id"],
        "image_id": item["image_path"],
        "text_input": item["text"],
        "classification_label": item["label"]
    }
    reformatted_data.append(new_item)

# Structure the data for ds_benchmark.json
structured_data = {
    "split": "dev",
    "data": reformatted_data
}

# Write the structured data to ds_benchmark.json
with open(benchmark_file_path, 'w') as f:
    json.dump(structured_data, f, indent=2)

print("Data has been successfully reformatted and saved in ds_benchmark.json")
