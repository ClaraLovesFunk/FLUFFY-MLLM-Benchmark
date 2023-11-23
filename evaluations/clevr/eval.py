import evaluations.utils_eval as utils_eval
import utils

VALID_ANS_VALUES = "no-ans-validity"
TASK_NAME = "direct answer (clevr)"
POS_LABEL = ""
label_name = "correct_direct_answer_short"
output_name = "output_direct answer (clevr)"
dataset_name = "clevr"



def evaluate_clevr(CONFIG_PATH, dataset_name, model_name, mode, run):
    
    # input_original_path = 'datasets/aokvqa/ds_original.json'
    # dataset_benchmark_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'dataset_benchmark_path')
    #output_original_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_original_path')
    # output_transformed_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_transformed_path')
    
    # input_original = utils_eval.load_data(input_original_path)
    # input_benchmark = utils_eval.load_data(dataset_benchmark_path)
    # input_benchmark = input_benchmark["data"]
    #output_original = utils_eval.load_data(output_original_path)
    #output_transformed = utils_eval.load_data(output_transformed_path)

    # preprocess output & get valid answer ratio 
    y_pred_dict, y_true_dict, label2_y_pred_dict, valid_ans_ratio_dict = utils_eval.pipeline_preprocess(
         CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode)
    
    # do the official evaluation, but with output data transformed according to evaluation modus
    scores_dict = {}
    examples_dict = {}
    
    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()
    for task in tasks:


                
        # data_text = utils_eval.load_data(ds_text_file_path)
        # output = utils_eval.load_data(experiment_output_file_path)
        # labels = utils_eval.get_id_2_label_dict(data_text, label_name, dataset_name)

        # _, y_pred, y_true = utils_eval.get_clean_valid_preds_trues(output, output_name, VALID_ANS_VALUES, labels, model, dataset_name, data_text, mode)
        y_pred = y_pred_dict[task]
        y_true = y_true_dict[task]
        scores = utils_eval.compute_standard_metrics(y_true, y_pred, pos_label = POS_LABEL, average='binary', zero_division=0, flag_only_acc = True)
        scores_dict[task] = scores
        # examples = utils_eval.get_examples(dataset_name, output, output_name, labels)

        # scores = {TASK_NAME: scores}
        # examples = {TASK_NAME: examples}

    return scores_dict, examples_dict, valid_ans_ratio_dict