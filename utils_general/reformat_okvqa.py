import json

# Load the data from the old file
with open('datasets/okvqa/ds_benchmark_old.json', 'r') as file:
    data = json.load(file)

# Process the data to remove duplicate answers
for entry in data['data']:
    unique_answers = set()
    # Iterate through each answer and add it to the set if not already present
    for ans in entry['correct_direct_answer_short']:
        unique_answers.add(ans['answer'])

    # Convert the set back to a list and assign it back to the entry
    entry['correct_direct_answer_short'] = list(unique_answers)

# Now write the processed data to the new file
with open('datasets/okvqa/ds_benchmark.json', 'w') as file:
    json.dump(data, file, indent=2)
