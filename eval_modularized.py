import argparse
import json
import os
from itertools import product

from evaluations.hateful_memes.eval import evaluate_hateful_memes
from evaluations.mami.eval import evaluate_mami
from evaluations.mvsa.eval import evaluate_mvsa


CONFIG_PATH = 'config.json'
ALL_KEYWORD = 'all'
DS_WITH_VAL_ANS = ['mami', 'hateful_memes', 'mvsa']

def get_paths(config, dataset, model, run):
    ds_text_file_path = os.path.join(config['datasets_dir'], dataset, config['dataset_file_name'])
    experiment_dir_path = os.path.join(config['experiments_dir'], model, dataset, 'run' + run)
    experiment_scores_file_path = os.path.join(experiment_dir_path, config['eval_file_name'])
    experiment_examples_file_path = os.path.join(experiment_dir_path, config['examples_file_name'])
    experiment_output_file_path = os.path.join(experiment_dir_path, config['output_file_name'])
    experiment_valid_ans_file_path = os.path.join(experiment_dir_path, config['valid_ans_file_name'])

    return ds_text_file_path, experiment_output_file_path, experiment_scores_file_path, experiment_examples_file_path, experiment_valid_ans_file_path




def main(args):
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    selected_models = config['model_names'] if args.models == ALL_KEYWORD else [args.models]
    selected_datasets = config['dataset_names'] if args.datasets == ALL_KEYWORD else [args.datasets]
    run = args.run

    for model, dataset in product(selected_models, selected_datasets):

        print(f' {model} on {dataset}:')

        (
            ds_text_file_path, 
            experiment_output_file_path, 
            experiment_scores_file_path, 
            experiment_examples_file_path, 
            experiment_valid_ans_file_path
        ) = get_paths(config, dataset, model, run)

        if dataset == 'hateful_memes':
            scores, examples, valid_ans_ratio = evaluate_hateful_memes(ds_text_file_path, experiment_output_file_path, model)

        if dataset == 'mami':
            scores, examples, valid_ans_ratio = evaluate_mami(ds_text_file_path, experiment_output_file_path, model)

        if dataset == 'mvsa':
            scores, examples, valid_ans_ratio = evaluate_mvsa(ds_text_file_path, experiment_output_file_path, model)



        with open(experiment_scores_file_path, 'w') as f: 
            json.dump(scores,f, indent=4)
        with open(experiment_examples_file_path, 'w') as f: 
            json.dump(examples,f, indent=4)
        if dataset in DS_WITH_VAL_ANS:
            with open(experiment_valid_ans_file_path, 'w') as f: 
                json.dump(valid_ans_ratio,f, indent=4)
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on datasets.')
    parser.add_argument('--models', type=str, required=True, help='Name of the model.')
    parser.add_argument('--datasets', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--run', type=str, default="1", help='Index of run.')
    args = parser.parse_args()

    main(args)

    print("VOILA!")



'''


python3 eval_modularized.py --models all --datasets all
python3 eval_modularized.py --models blip2 --datasets all
python3 eval_modularized.py --models blip2 --datasets mvsa

'''