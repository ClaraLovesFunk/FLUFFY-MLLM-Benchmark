import json

# Define the absolute paths for the original and new data files
path_prefix = 'datasets/clevr/'
original_file_path = path_prefix + 'ds_original.json'
benchmark_file_path = path_prefix + 'ds_benchmark.json'

# Load the original data from ds_original.json
with open(original_file_path, 'r') as f:
    original_data = json.load(f)

# Extract the questions list from the original data
questions = original_data["questions"]

# Reformat the data
reformatted_data = []
for item in questions:
    new_item = {
        "text_input_id": item["input_id"],
        "image_id": item["image_filename"],
        "text_input": item["question"],
        "correct_direct_answer_short": item["answer"]
    }
    reformatted_data.append(new_item)

# Structure the reformatted data as required
structured_data = {
    "split": "val_sampled (sampled instances with unique images)", 
    "data": reformatted_data
}

# Write the structured data to ds_benchmark.json
with open(benchmark_file_path, 'w') as f:
    json.dump(structured_data, f, indent=2)

print("Data has been successfully reformatted and saved in ds_benchmark.json")
