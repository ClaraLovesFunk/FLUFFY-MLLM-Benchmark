from utils.info import get_paths
from utils.file_and_path_utils import load_data
from utils.answer_processing import pipeline_preprocess
from utils.info import DatasetInfo


VALID_ANS_VALUES = "sample-dependent"
TASK_NAME = "multiple choice (aokvqa)"
POS_LABEL = ""
label_name = "correct_choice"
output_name = "output_multiple choice (aokvqa)"
dataset_name = "aokvqa"


def eval_aokvqa(input, output, task, strict=True): 
    if task == 'direct answer (aokvqa)': 
        multiple_choice = False 
    else:
        multiple_choice = True 

    if isinstance(input, list):
        input = { input[i]['question_id'] : input[i] for i in range(len(input)) }
        
    if isinstance(output, list):  
        output = { output[i]['text_input_id'] : output[i] for i in range(len(output)) }
       
    if multiple_choice is False: 
        input = {k:v for k,v in input.items() if v['difficult_direct_answer'] is False}

    if strict: 
        dataset_qids = set(input.keys())
        preds_qids = set(output.keys())
        assert dataset_qids.issubset(preds_qids)

    acc = []
    examples = {}

    for q in input.keys(): 
        if q not in output.keys(): 
            acc.append(0.0)
            continue
        if multiple_choice:
            pred = output[q]['output_multiple choice (aokvqa)']#[0]
        else: 
            pred = output[q]['output_direct answer (aokvqa)']#[0]
        
        choices = input[q]['choices']
        direct_answers = input[q]['direct_answers']

        if multiple_choice:
            '''
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
                '''
            correct_choice_idx = input[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
            
            if pred == choices[correct_choice_idx]:
                examples.update({q:1})
            else:
                examples.update({q:0})  
                        
        else:
            num_match = sum([pred == da for da in direct_answers])

            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

            if num_match >= 1:
                examples.update({q:1})
            else:
                examples.update({q:0})

    acc = sum(acc) / len(acc) 

    return acc, examples


def evaluate_aokvqa(CONFIG_PATH, dataset_name, model_name, mode, run):

    input_original_path = 'datasets/aokvqa/ds_original.json'
    dataset_benchmark_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'dataset_benchmark_path')
    output_transformed_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_transformed_path')
    
    input_original = load_data(input_original_path)
    input_benchmark = load_data(dataset_benchmark_path)
    input_benchmark = input_benchmark["data"]
    
    _, _, _, valid_ans_ratio_dict = pipeline_preprocess(
         CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode)
    output_transformed = load_data(output_transformed_path)
    
    scores_dict = {}
    examples_dict = {}
    
    DatasetInfo_instance = DatasetInfo(dataset_name)
    tasks = DatasetInfo_instance.get_tasks()
    for task in tasks:
            
        acc, ex = eval_aokvqa(input = input_original, output=output_transformed, task = task, strict=True)
        scores_dict[task] = {'accuracy': acc}
        examples_dict[task] = ex

    return scores_dict, examples_dict, valid_ans_ratio_dict