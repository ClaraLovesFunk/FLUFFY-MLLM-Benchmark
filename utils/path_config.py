import os
from .data_loading import load_data




def get_paths(CONFIG_PATH, dataset_name, model_name, run, mode, value_of_interest = 'None'):

    config = load_data(CONFIG_PATH)
    
    dataset_benchmark_path = os.path.join(config['datasets_dir'], dataset_name, config['dataset_file_name'])
    experiment_dir_path = os.path.join(config['experiments_dir'], model_name, dataset_name, 'run' + run)
    output_original_path = os.path.join(experiment_dir_path, config['output_file_name'])
    scores_path = os.path.join(experiment_dir_path, config['eval_file_' + mode])
    examples_path = os.path.join(experiment_dir_path, config['examples_file_' + mode])
    val_ratio_path = os.path.join(experiment_dir_path, config['valid_ans_file_' + mode])

    output_transformed_path = output_original_path.replace("output", "output_aux_" + mode)

    if value_of_interest == 'dataset_benchmark_path':
        return dataset_benchmark_path
    elif value_of_interest == 'output_original_path':
        return output_original_path
    elif value_of_interest == 'output_transformed_path':
        return output_transformed_path
    elif value_of_interest == 'scores_path':
        return scores_path
    elif value_of_interest == 'examples_path':
        return examples_path
    elif value_of_interest == 'val_ratio_path':
        return val_ratio_path
    elif value_of_interest == 'None':
        return dataset_benchmark_path, output_original_path, output_transformed_path, scores_path, examples_path, val_ratio_path