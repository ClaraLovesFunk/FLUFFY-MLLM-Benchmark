#%%

import pandas as pd
import json
from IPython.display import display, HTML
from sklearn.metrics import accuracy_score

split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'aokvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/'

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'
text_output_file_aokvqa_offcl = results_dir + 'output_aokvqa.json'
eval_file = results_dir + f'scores.json'
example_file = results_dir + f'examples.json'


# load model predictions

with open(text_output_file, 'r') as f:
    pred = json.load(f)

pred = pd.DataFrame(pred)


# Get the ground truth labels and model predictions

y_true = pred["direct_answers"].tolist()
y_pred = pred["output"].tolist()
print(y_true)



# Create a list to store the indices of correct samples
correct_indices = []
correct_n = 0

incorrect_indices = []
incorrect_n = 0


# Compare the model's answer with the list of potential correct answers
for i in range(len(y_true)):
    if y_pred[i][0] in y_true[i]:
        correct_indices.append(i)
        correct_n += 1
    else:
        incorrect_indices.append(i)
        incorrect_n += 1



# Calculate the accuracy metric

accuracy =  correct_n / len(y_true)



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
