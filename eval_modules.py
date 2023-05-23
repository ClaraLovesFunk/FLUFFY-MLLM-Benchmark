import pandas as pd



def acc_strict_standard (input, output, multiple_choice=False, strict=True):

    output = pd.DataFrame(output)
    input = pd.DataFrame(input)

    # Get the ground truth labels and model predictions

    y_true = input["direct_answers"].tolist()
    y_pred = output["output"].tolist()

    # Create a list to store the indices of correct/incorrect samples and a variable to count the occurances of correct/incorrect predicitions

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

    acc =  correct_n / len(y_true)

    # store example indice
    example_indice = {"good pred": correct_indices, 
                      "bad pred": incorrect_indices}

    return acc, example_indice






def eval_aokvqa(input, output, multiple_choice=False, strict=True):

    '''
    aokvqa's method of computing accuracy by regarding how many of their proposed direct_answers the predictions matches
    (the answer that the aokvqa authors want the most are written the most often in a list of multiple possible direct answers,
    while a just satisfactory answer is written less)
    
    accuracy per instance can be values between 0-1, not just 0 and 1'''


    if isinstance(input, list):  # checks if dataset is of type list; if yes, it transforms it into a dict with question id as key
        input = { input[i]['question_id'] : input[i] for i in range(len(input)) }
        

    # If the preds is a list, transform it into a dictionary with question id as key
    if isinstance(output, list):  
        output = { output[i]['question_id'] : output[i] for i in range(len(output)) }
       
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

        pred = output[q]['output'][0]
        choices = input[q]['choices']
        direct_answers = input[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = input[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])

            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) #* 100

    return acc