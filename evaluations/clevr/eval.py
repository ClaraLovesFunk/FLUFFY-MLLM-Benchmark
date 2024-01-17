import evaluations.utils_eval as utils_eval
import utils_general.utils as utils

VALID_ANS_VALUES = "no-ans-validity"
TASK_NAME = "direct answer (clevr)"
POS_LABEL = ""
label_name = "correct_direct_answer_short"
output_name = "output_direct answer (clevr)"
dataset_name = "clevr"



def evaluate_clevr(CONFIG_PATH, dataset_name, model_name, mode, run):
    
    # preprocess output & get valid answer ratio 
    y_pred_dict, y_true_dict, label2_y_pred_dict, valid_ans_ratio_dict = utils_eval.pipeline_preprocess(
         CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode)
    
    # do the evaluation, but with output data transformed according to evaluation modus
    scores_dict = {}
    examples_dict = {}
    
    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()
    for task in tasks:
        y_pred = y_pred_dict[task]
        y_true = y_true_dict[task]
        scores = utils_eval.compute_standard_metrics(y_true, y_pred, pos_label = POS_LABEL, average='binary', zero_division=0, flag_only_acc = True)
        scores_dict[task] = scores

    return scores_dict, examples_dict, valid_ans_ratio_dict