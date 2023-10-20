import evaluations.utils_eval as utils_eval

VALID_ANS_VALUES = ['sexist', 'not sexist']
TASK_NAME = "sexism classification"
POS_LABEL = "sexist"
label_name = "classification_label"
output_name = "output_sexism classification"
dataset_name = "mami"



def evaluate_mami(ds_text_file_path, experiment_output_file_path, model):

    data_text = utils_eval.load_data(ds_text_file_path)
    output = utils_eval.load_data(experiment_output_file_path)
    labels = utils_eval.get_id_2_label_dict(data_text, label_name)

    valid_ans_ratio, y_pred, y_true = utils_eval.get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset_name)
    scores = utils_eval.compute_standard_metrics(y_true, y_pred, pos_label = POS_LABEL, average='binary')
    print(scores)
    examples = utils_eval.get_examples(output, output_name, labels)

    valid_ans_ratio = {TASK_NAME: valid_ans_ratio}
    scores = {TASK_NAME: scores}
    examples = {TASK_NAME: examples}

    return scores, examples, valid_ans_ratio