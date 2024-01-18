def get_examples(ds, task, y_pred_dict, y_true_dict):
    '''
    creates dictionary {text_input_id: index_corr}
    index_corr indicates whether a sample was predicted correctly
    given the evaluation modus, dataset, task, ...
    '''
    examples = {}
    all_text_input_id = list(y_pred_dict.keys())
    for text_input_id in all_text_input_id:
        if ds == 'aokvqa' and task == 'direct answer (aokvqa)':
            if y_pred_dict[text_input_id] in y_true_dict[text_input_id]:
                correct = 1
            else:
                correct = 0
        else:
            if y_pred_dict[text_input_id] == y_true_dict[text_input_id]:
                correct = 1
            else:
                correct = 0
        examples[text_input_id] = correct
    return examples