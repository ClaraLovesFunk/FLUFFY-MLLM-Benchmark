import json
from eval_modules import *

split_sec = 'val'
images_sec = 'coco2017'
dataset_sec = 'aokvqa'
model_sec = 'blip2'

images_input_dir = 'datasets/' + images_sec
text_input_dir = 'datasets/' + dataset_sec +'/'
results_dir = 'experiments/' + model_sec + '/' + dataset_sec + '/' + 'run1/' 

text_input_file = text_input_dir + split_sec + '.json'
text_output_file = results_dir + 'output.json'
eval_file = results_dir + f'scores.json'
example_file = results_dir + f'examples.json'



with open(text_output_file, 'r') as f:
    output = json.load(f)


with open(text_input_file, 'r') as f:
    input = json.load(f)


acc_aokvqa_da = eval_aokvqa(input = input, output=output, task = 'direct_answer', strict=True)
acc_aokvqa_MC = eval_aokvqa(input = input, output=output, task = 'multiple_choice', strict=True)

