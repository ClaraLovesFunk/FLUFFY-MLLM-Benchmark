import json
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



# task "direct answer": eval + get example indice

acc_strict_standard_da, example_indice_da = acc_strict_standard(input = input, output = output, multiple_choice=False, strict=True)
acc_aokvqa_da = eval_aokvqa(input = input, output=output, multiple_choice=False, strict=True)

scores_da = {"acc": acc_aokvqa_da} #"acc_strict_standard": acc_strict_standard_da,}



# task "Multiple Choice": eval + get example indice

acc_strict_standard_MC, example_indice_MC = acc_strict_standard(input = input, output = output, multiple_choice=True, strict=True)
acc_aokvqa_MC = eval_aokvqa(input = input, output=output, multiple_choice=True, strict=True)

scores_MC = {"acc": acc_aokvqa_MC} #{"acc_strict_standard": acc_strict_standard_MC,
          



# store metrics + example indice

scores_alltasks = {"direct answer": scores_da,
                   "multiple choice": scores_MC}

example_indice_alltasks = {"direct answer": example_indice_da,
                           "multiple choice": example_indice_MC}


print(scores_alltasks)

with open(eval_file, 'w') as f: 
    json.dump(scores_alltasks,f)

with open(example_file, 'w') as f:
    json.dump(example_indice_alltasks,f)







# %%