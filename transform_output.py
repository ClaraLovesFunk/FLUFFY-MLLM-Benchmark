import json

# Load the data from output.json
with open('experiments/blip2/okvqa/output.json', 'r') as f:
    data = json.load(f)

# Transform the data
transformed_data = [{'question_id': item['question_id'], 'answer': item['output_da'][0]} for item in data]

# Save the transformed data to output_fakeform.json
with open('experiments/blip2/okvqa/output_fakeform.json', 'w') as f:
    json.dump(transformed_data, f)
