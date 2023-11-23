import evaluations.utils_eval as utils_eval
import utils

VALID_ANS_VALUES = ['positive', 'negative', 'neutral']
TASK_NAME = "sentiment analysis"
POS_LABEL = ""
label_name = "classification_label"
output_name = "output_sentiment analysis"
dataset_name = "mvsa"



def evaluate_mvsa(CONFIG_PATH, dataset_name, model_name, mode, run):

    # preprocess output & get valid answer ratio 
    y_pred_dict, y_true_dict, label2_y_pred_dict, valid_ans_ratio_dict = utils_eval.pipeline_preprocess(
         CONFIG_PATH, VALID_ANS_VALUES, dataset_name, model_name, run, mode)
    
    # do the official evaluation, but with output data transformed according to evaluation modus
    scores_dict = {}
    examples_dict = {}
    
    DatasetInfo = utils.DatasetInfo(dataset_name)
    tasks = DatasetInfo.get_tasks()
    for task in tasks:
        y_pred = y_pred_dict[task]
        y_true = y_true_dict[task]

        #print(label2_y_pred_dict[task]["1.jpg"])

        scores = utils_eval.compute_standard_metrics(y_true, y_pred, pos_label = POS_LABEL, average='weighted', zero_division=0, flag_only_acc = False)
        scores_dict[task] = scores

    return scores_dict, examples_dict, valid_ans_ratio_dict