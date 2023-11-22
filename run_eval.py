import argparse
import json
import os
from itertools import product

from evaluations.hateful_memes.eval import evaluate_hateful_memes
from evaluations.mami.eval import evaluate_mami
from evaluations.mvsa.eval import evaluate_mvsa
from evaluations.esnlive.eval import evaluate_esnlive
from evaluations.scienceqa.eval import evaluate_scienceqa
from evaluations.aokvqa.eval import evaluate_aokvqa
from evaluations.okvqa.eval import evaluate_okvqa
from evaluations.gqa.eval import evaluate_gqa
from evaluations.clevr.eval import evaluate_clevr

import evaluations.utils_eval as utils_eval


CONFIG_PATH = 'config.json'
ALL_KEYWORD = 'all'
DS_WITH_VAL_ANS = ['mami', 'hateful_memes', 'mvsa', 'esnlive', 'aokvqa']

def get_paths(config, dataset, model, run, mode):
    
    ds_text_file_path = os.path.join(config['datasets_dir'], dataset, config['dataset_file_name'])
    experiment_dir_path = os.path.join(config['experiments_dir'], model, dataset, 'run' + run)
    experiment_output_file_path = os.path.join(experiment_dir_path, config['output_file_name'])
    experiment_scores_file_path = os.path.join(experiment_dir_path, config['eval_file_' + mode])
    experiment_examples_file_path = os.path.join(experiment_dir_path, config['examples_file_' + mode])
    experiment_valid_ans_file_path = os.path.join(experiment_dir_path, config['valid_ans_file_' + mode])

    return ds_text_file_path, experiment_output_file_path, experiment_scores_file_path, experiment_examples_file_path, experiment_valid_ans_file_path




def main(args):
    
    config = utils_eval.load_data(CONFIG_PATH)
    
    selected_models = config['model_names'] if args.models == ALL_KEYWORD else [args.models]
    selected_datasets = config['dataset_names'] if args.datasets == ALL_KEYWORD else [args.datasets]
    
    mode = args.mode
    run = args.run

    for model, dataset in product(selected_models, selected_datasets):

        print(f' {model} on {dataset}:')

        (
            ds_text_file_path, 
            experiment_output_file_path, 
            experiment_scores_file_path, 
            experiment_examples_file_path, 
            experiment_valid_ans_file_path
        ) = get_paths(config, dataset, model, run, mode = args.mode)

        if dataset == 'hateful_memes':
            scores, examples, valid_ans_ratio = evaluate_hateful_memes(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == 'mami':
            scores, examples, valid_ans_ratio = evaluate_mami(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == 'mvsa':
            scores, examples, valid_ans_ratio = evaluate_mvsa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "esnlive":
            scores, examples, valid_ans_ratio = evaluate_esnlive(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "scienceqa":
            scores, examples, valid_ans_ratio = evaluate_scienceqa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "aokvqa":
            scores, examples, valid_ans_ratio = evaluate_aokvqa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "okvqa":
            scores, examples = evaluate_okvqa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "gqa":
            scores, examples = evaluate_gqa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "clevr":
            scores, examples = evaluate_clevr(ds_text_file_path, experiment_output_file_path, model, mode)


        print(scores)
        
        utils_eval.save_data( , scores)
        utils_eval.save_data(experiment_examples_file_path, examples)
        if dataset in DS_WITH_VAL_ANS:
            utils_eval.save_data(experiment_valid_ans_file_path, valid_ans_ratio)
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on datasets.')
    parser.add_argument('--models', type=str, required=True, help='Name of the model.')
    parser.add_argument('--datasets', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--run', type=str, default="1", help='Index of run.')
    parser.add_argument('--mode', type=str, default="hard", choices=["hard", "soft"], help='Evaluation mode: hard or soft.')
    args = parser.parse_args()

    main(args)

    print("VOILA!")



'''


python3 run_eval.py --models all --datasets all
python3 run_eval.py --models openflamingo --datasets aokvqa --mode soft
python3 run_eval.py --models blip2 --datasets aokvqa --mode hard
python3 run_eval.py --models instructblip --datasets aokvqa --mode hard
python3 run_eval.py --models adept --datasets aokvqa --mode hard



'''