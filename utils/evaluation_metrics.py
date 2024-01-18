from sklearn import metrics
from .data_loading import load_data
from .file_and_path_utils import save_data
from .info import DatasetInfo

import math

def compute_standard_metrics(y_true, y_pred, pos_label, average='binary', zero_division=0, flag_only_acc = False, dataset_name = None, task = None):
    
    '''
    compute metrics
    '''
    if y_pred == []: 
        '''
        if no valid predictions were made, model cannot be evaluated
        '''
        invalid_ans = float('nan')
        scores = {
            'accuracy': invalid_ans, 
            'precision': invalid_ans, 
            'recall': invalid_ans, 
            'f1': invalid_ans}

    elif y_pred != []:
        '''
        compute metrics, if valid predictions were made
        '''
        if flag_only_acc == True:
            if dataset_name == 'aokvqa':
                if task == 'direct answer (aokvqa)':
                    pred_corr = 0
                    for i in range(len(y_true)):
                        if y_pred[i] in y_true[i]:
                            pred_corr += 1
                    acc = pred_corr/len(y_pred)      
                    scores = {'accuracy': acc}
                elif task == 'multiple choice (aokvqa)':
                    pred_corr = 0
                    for i in range(len(y_true)):
                        if y_pred[i] == y_true[i]:
                            pred_corr += 1
                    acc = pred_corr/len(y_pred)      
                    scores = {'accuracy': acc}
            
            elif dataset_name != 'aokvqa':
                scores = {
                            'accuracy': metrics.accuracy_score(y_true, y_pred),
                            }
            
        
        elif flag_only_acc == False:
            if average=='binary':
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, pos_label=pos_label, zero_division=zero_division),                  
                }

            else:
                scores = {
                    'accuracy': metrics.accuracy_score(y_true, y_pred),
                    'precision': metrics.precision_score(y_true, y_pred, average = average, zero_division=zero_division),
                    'recall': metrics.recall_score(y_true, y_pred, average = average, zero_division=zero_division),
                    'f1': metrics.f1_score(y_true, y_pred, average = average, zero_division=zero_division),                  
                }
    return scores



def calculate_average_accuracy_over_all_ds(CONFIG_PATH, model_name, mode_name):
    
    config = load_data(CONFIG_PATH)
    dataset_names = config['dataset_names']

    def average(values):
        valid_values = [v for v in values if not math.isnan(v)] 
        return sum(valid_values) / len(valid_values) if valid_values else 0

    model_scores = {}
    for dataset_name in dataset_names:
        dataset_scores = {}
        scores_file_path = f"experiments/{model_name}/{dataset_name}/run1/scores_{mode_name}.json"
        file_data = load_data(scores_file_path)
        DatasetInfo_instance = .DatasetInfo(dataset_name)
        tasks = DatasetInfo_instance.get_tasks()
        for task in tasks:
            task_score = file_data[task]['accuracy']
            dataset_scores[task] = task_score 

        dataset_scores_average = average(dataset_scores.values())
        model_scores[dataset_name] = dataset_scores_average
    model_scores_average = {"average accuracy": average(model_scores.values())}
    print(f"Average Accuracy: {model_scores_average}")

    scores_average_file_path = f"experiments/{model_name}/scores_average_{mode_name}.json" # needs to be edited when running more runs
    save_data(scores_average_file_path, model_scores_average)
