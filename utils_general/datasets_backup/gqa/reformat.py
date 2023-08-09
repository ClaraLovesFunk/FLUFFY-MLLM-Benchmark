import json

# Define paths for the original and new data files
path_prefix = 'datasets/gqa/'
original_file_path = path_prefix + 'ds_original.json'
benchmark_file_path = path_prefix + 'ds_benchmark.json'

# Load the original data
with open(original_file_path, 'r') as file:
    data = json.load(file)

# Reformat the data
new_data = []
for item in data:
    new_item = {
        "text_input_id": item["input_id"],
        "image_id": item["imageId"] + '.jpg',
        "text_input": item["question"],
        "correct_direct_answer_short": item["answer"],
        "correct_direct_answer_long": item["fullAnswer"]
    }
    new_data.append(new_item)

# Structure the data for the ds_benchmark.json file
structured_data = {
    "split": "val_sampled (sampled instances with unique images)",
    "data": new_data
}

# Write the structured data to ds_benchmark.json
with open(benchmark_file_path, 'w') as file:
    json.dump(structured_data, file, indent=2)

print("Data has been successfully reformatted and saved in ds_benchmark.json")
