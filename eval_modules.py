import pandas as pd
from utils_eval.vqa import VQA 
from utils_eval.vqaEval import VQAEval
import json



def acc_strict_standard (input, output, multiple_choice=False, strict=True):

    output = pd.DataFrame(output)
    input = pd.DataFrame(input)

    
    # Create a list to store the indices of correct/incorrect samples and a variable to count the occurances of correct/incorrect predicitions

    correct_indices = []
    correct_n = 0

    incorrect_indices = []
    incorrect_n = 0


    if multiple_choice == True:
      
        # Get the ground truth labels and model predictions

        y_true = []

        choices = input['choices']
        ix_correct_choice = input['correct_choice_idx']

        for i, i_correct_idx in enumerate(ix_correct_choice):
            correct_choice = choices[i][i_correct_idx]
            y_true.append([correct_choice])

        y_pred = output["output_multiple_choice"].tolist()

        # Compare the model's answer with the list of potential correct answers

        for i in range(len(y_true)):
            if y_pred[i][0] == y_true[i][0]:

                correct_indices.append(i)
                correct_n += 1

            else:
                incorrect_indices.append(i)
                incorrect_n += 1


    if multiple_choice == False:

        # Get the ground truth labels and model predictions

        y_true = input["direct_answers"].tolist()
        y_pred = output["output_direct_answer"].tolist()

        # Compare the model's answer with the list of potential correct answers

        for i in range(len(y_true)):
            if y_pred[i][0] in y_true[i]:

                correct_indices.append(i)
                correct_n += 1
            else:
                incorrect_indices.append(i)
                incorrect_n += 1


    # Calculate the accuracy metric

    acc =  correct_n / len(y_true)

    # store example indice
    example_indice = {"good pred": correct_indices, 
                    "bad pred": incorrect_indices}

    return acc, example_indice






def eval_aokvqa(input, output, task, strict=True): # MESSING WITH SOURCE CODE: replaced the variable "multiple_choice" because ultimately it means the same as task type (direct answer/MC)

    '''
    aokvqa's method of computing accuracy by regarding how many of their proposed direct_answers the predictions matches
    (the answer that the aokvqa authors want the most are written the most often in a list of multiple possible direct answers,
    while a just satisfactory answer is written less)
    
    accuracy per instance can be values between 0-1, not just 0 and 1'''

    if task == 'direct answer': # MESSING WITH SOURCE CODE

        multiple_choice = False # MESSING WITH SOURCE CODE

    else:

        multiple_choice = True # MESSING WITH SOURCE CODE

    if isinstance(input, list):  # checks if dataset is of type list; if yes, it transforms it into a dict with question id as key
        input = { input[i]['question_id'] : input[i] for i in range(len(input)) }
        

    # If the preds is a list, transform it into a dictionary with question id as key
    if isinstance(output, list):  
        output = { output[i]['text_input_id'] : output[i] for i in range(len(output)) }
       
    if multiple_choice is False: # if we look at direct answer task, we only look at instances with easy direct answers (or not difficult_direct_answers)
        input = {k:v for k,v in input.items() if v['difficult_direct_answer'] is False}

    if strict: #dataset_qids is a subset of preds_qids ???
        dataset_qids = set(input.keys())
        preds_qids = set(output.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in input.keys(): # for each question id q
        if q not in output.keys(): #if we didnt generate a pred for a q in the dataset, we append 0.0 to the acc array
            acc.append(0.0)
            continue
        if multiple_choice:
            pred = output[q]['output_multiple choice'][0]
        else: 
            pred = output[q]['output_direct answer'][0]
        
        choices = input[q]['choices']
        direct_answers = input[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:

            '''
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
                '''
            correct_choice_idx = input[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )

        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])

            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) #* 100

    return acc



def transform_output_4_okvqa(resFile_original, resFile):

    with open(resFile_original, 'r') as f:
        data = json.load(f)

    transformed_data = [{'question_id': item['text_input_id'], 'answer': item['output_direct answer'][0]} for item in data]

    with open(resFile, 'w') as f:
        json.dump(transformed_data, f)




def acc_okvqa(eval_file, annFile, quesFile, resFile, resFile_original, transform_output_4_okvqa):

	transform_output_4_okvqa(resFile_original, resFile) # get output in the format needed for okvqa's original eval
        
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)
	vqaEval = VQAEval(vqa, vqaRes, n=2)   # n is precision of accuracy (number of places after decimal), default is 2
	vqaEval.evaluate() 

	acc = vqaEval.accuracy['overall']

	return acc