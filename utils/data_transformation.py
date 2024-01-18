import os
from .data_loading import load_data
from .file_and_path_utils import get_paths
from .file_and_path_utils import save_data


def get_id_2_label_dict(data_text, label_name, dataset_name):
    '''
    {text_input_id: label}
    '''
    labels = {}
    for item in data_text:
        text_input_id = item["text_input_id"]
        label_value = item[label_name]
        labels[text_input_id] = label_value

    return labels


def make_output_aux_eval(CONFIG_PATH, dataset_name, model_name, run, tasks, mode, y_pred_dict_all_tasks):
    """
    - load the original outputfile
    - replace all predictions with our new cleaned predictions (cleaned, based on model requiriements, dataset and evaluation modus) we got
        in pipeline_preprocess()
    - save the transformed output with a new name
    """
    output_original_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_original_path')
    output_original = load_data(output_original_path)
    for task in tasks:
        y_pred_dict = y_pred_dict_all_tasks[task]
        for item in output_original:
            text_input_id = item.get("text_input_id")
            y_pred_value = y_pred_dict.get(text_input_id)
            if y_pred_value:
                item["output_" + task] = y_pred_value
    output_transformed = output_original # you shall now be called output_transformed
    output_transformed_path = get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'output_transformed_path')
    directory = os.path.dirname(output_transformed_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_data(output_transformed_path, output_transformed)