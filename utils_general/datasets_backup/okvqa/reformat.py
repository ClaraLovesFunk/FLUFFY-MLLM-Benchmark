import json

# Define the absolute paths for the original and new data files
path_prefix = 'datasets/okvqa/'
original_file_path = path_prefix + 'ds_original.json'
original_labels_file_path = path_prefix + 'ds_original_labels.json'
benchmark_file_path = path_prefix + 'ds_benchmark.json'

# Load the original data and labels
with open(original_file_path, 'r') as f:
    original_data = json.load(f)

with open(original_labels_file_path, 'r') as f:
    original_labels_data = json.load(f)

# Create a mapping of question_id to answers for easy lookup
question_to_answer_map = {annotation['question_id']: annotation['answers'] for annotation in original_labels_data['annotations']}

# Process and reformat the data
reformatted_data = []
for question in original_data['questions']:
    # Extract the answers for the current question_id
    answers = question_to_answer_map[question['question_id']]
    
    # Simplify the answers to just a list of answer strings
    simplified_answers = [ans['answer'] for ans in answers]

    # Create the new data entry format
    new_entry = {
        "text_input_id": question["question_id"],
        "image_id": f"{question['image_id']:012}.jpg",
        "text_input": question["question"],
        "correct_direct_answer_short": answers
    }
    reformatted_data.append(new_entry)

# Create the final structure for the new data file
final_structure = {
    "split": "val",
    "data": reformatted_data
}

# Write the reformatted data to ds_benchmark.json
with open(benchmark_file_path, 'w') as f:
    json.dump(final_structure, f, indent=2)

print("Data has been successfully reformatted and saved in ds_benchmark.json.")
