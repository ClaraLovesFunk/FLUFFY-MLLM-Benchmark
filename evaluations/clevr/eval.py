from utils.info import DatasetInfo
from utils.evaluation_metrics import compute_standard_metrics
from utils.evaluation_metrics import pipeline_preprocess

VALID_ANS_VALUES = "no-ans-validity"
TASK_NAME = "direct answer (clevr)"
POS_LABEL = ""
label_name = "correct_direct_answer_short"
output_name = "output_direct answer (clevr)"
dataset_name = "clevr"


def evaluate_clevr(CONFIG_PATH, dataset_name, model_name, mode, run):
    
    # preprocess output & get valid answer ratio 
    y_pred_dict, y_true_dict, _, valid_ans_ratio_dict = pipeline_preprocess(
         CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode)
    
    # do the evaluation, but with output data transformed according to evaluation modus
    scores_dict = {}
    examples_dict = {}
    
    DatasetInfo_instance = DatasetInfo(dataset_name)
    tasks = DatasetInfo_instance.get_tasks()
    for task in tasks:
        y_pred = y_pred_dict[task]
        y_true = y_true_dict[task]
        scores = compute_standard_metrics(
            y_true, 
            y_pred, 
            pos_label=POS_LABEL, 
            average='binary', 
            zero_division=0, 
            flag_only_acc=True)
        scores_dict[task] = scores

    return scores_dict, examples_dict, valid_ans_ratio_dict