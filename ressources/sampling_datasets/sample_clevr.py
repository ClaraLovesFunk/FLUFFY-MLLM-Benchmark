import json
import random


# Set a seed for reproducibility
random.seed(42)

# Load original data
with open('datasets/clevr/val.json', 'r') as f:
    data = json.load(f)

# Extract questions
original_questions = data["questions"]

# Shuffle questions to ensure random selection
random.shuffle(original_questions)

# Initialize a list to store selected instances and a set to store image filenames
selected_questions = []
image_filenames = set()

for question in original_questions:
    if question['image_filename'] not in image_filenames:
        # Add question to selected data and its image_filename to the set
        selected_questions.append(question)
        image_filenames.add(question['image_filename'])

        # Stop when we have 1000 unique instances
        if len(selected_questions) == 1000:
            break

# Create a new dictionary to store the selected questions and the original info
selected_data = {
    "info": data["info"],
    "questions": selected_questions
}

# Store sampled data
with open('datasets/clevr/val_sampled.json', 'w') as f:
    json.dump(selected_data, f, indent=4)