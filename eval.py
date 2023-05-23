#%%


import json
from IPython.display import display, HTML
from sklearn.metrics import accuracy_score

from eval_modules import *

split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'aokvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/'

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'
eval_file = results_dir + f'scores.json'
example_file = results_dir + f'examples.json'



with open(text_output_file, 'r') as f:
    output = json.load(f)


with open(text_input_file, 'r') as f:
    input = json.load(f)



# compute metrics & example indice

acc_strict_standard, example_indice = acc_strict_standard(input = input, output = output, multiple_choice=False, strict=True)
acc_aokvqa = eval_aokvqa(input = input, output=output, multiple_choice=False, strict=True)

scores = {"acc_strict_standard": acc_strict_standard,
          "acc_aokvqa": acc_aokvqa}

with open(eval_file, 'w') as f: 
    json.dump(scores,f)

with open(example_file, 'w') as f:
    json.dump(example_indice,f)







# %%