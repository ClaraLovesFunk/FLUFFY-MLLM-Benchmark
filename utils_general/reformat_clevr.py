import json

# clevr has multiple questions to one image. to create a unique identifier for each input, merge question_id and imagefilename to one "input_id"
# Load your JSON data
with open('datasets/clevr/val.json', 'r') as f:
    data = json.load(f)

# Update the data
for question in data['questions']:
    question['input_id'] = str(question['question_index']) + '_' + question['image_filename']

# Save the updated data back to the file
with open('datasets/clevr/val.json', 'w') as f:
    json.dump(data, f)
