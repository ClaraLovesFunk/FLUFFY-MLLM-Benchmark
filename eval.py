#%%

import pandas as pd
import json
from IPython.display import display, HTML

split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'aokvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/'

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'
text_output_file_aokvqa_offcl = results_dir + 'predictions_val.json'
eval_file = results_dir + f'eval_{split_sec}.json'
example_file = results_dir + f'examples_{split_sec}.json'


# load model predictions

with open(text_output_file, 'r') as f:
    pred = json.load(f)

pred = pd.DataFrame(pred)


# Get the ground truth labels and model predictions

y_true = pred["direct_answers"].tolist()
y_pred = pred["output"].tolist()


# Initialize a counter for correct predictions

correct_predictions = 0
incorrect_predictions = 0


# Create a list to store the indices of correct samples
correct_indices = []
incorrect_indices = []


# Compare the model's answer with the list of potential correct answers
for i in range(len(y_true)):
    if y_pred[i][0] in y_true[i]:
        correct_indices.append(i)
    else:
        incorrect_indices.append(i)


# Calculate the accuracy metric

accuracy = len(correct_indices) / len(y_true)



# save and reload eval metric

accuracy_dict = {"accuracy": accuracy}

with open(eval_file, 'w') as f:
    json.dump(accuracy_dict,f)

with open(eval_file, 'r') as f:
    accuracy = json.load(f)

print(f'Accuracy: {accuracy["accuracy"]:.4f}')



# save and reload indice of good/bad examples

example_indice = {"good pred": correct_indices,
                  "bad pred": incorrect_indices}

with open(example_file, 'w') as f:
    json.dump(example_indice,f)

with open(example_file, 'r') as f:
    examples = json.load(f)

print(f'Indice of good examples: {examples["good pred"][:3]}')
print(f'Indice of bad examples: {examples["bad pred"][:3]}')

# %%
