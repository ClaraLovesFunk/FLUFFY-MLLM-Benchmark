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





def main(args):
    
    config = utils_eval.load_data(CONFIG_PATH)
    
    selected_models = config['model_names'] if args.models == ALL_KEYWORD else [args.models]
    selected_datasets = config['dataset_names'] if args.datasets == ALL_KEYWORD else [args.datasets]
    
    modes = [args.mode] if args.mode else ["hard", "soft"]
    run = args.run

    for model, dataset, mode in product(selected_models, selected_datasets, modes):

        print(f'Evaluating {model} on {dataset} in {mode} mode:')


        (
            ds_text_file_path, 
            experiment_output_file_path, 
            experiment_scores_file_path, 
            experiment_examples_file_path, 
            experiment_valid_ans_file_path
        ) = utils_eval.get_paths(config, dataset, model, run, mode = mode)

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
            scores, examples, valid_ans_ratio = evaluate_okvqa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "gqa":
            scores, examples, valid_ans_ratio = evaluate_gqa(ds_text_file_path, experiment_output_file_path, model, mode)

        if dataset == "clevr":
            scores, examples = evaluate_clevr(ds_text_file_path, experiment_output_file_path, model, mode)


        print(scores)
        
        utils_eval.save_data(experiment_scores_file_path, scores)
        utils_eval.save_data(experiment_examples_file_path, examples)
        if dataset in DS_WITH_VAL_ANS:
            utils_eval.save_data(experiment_valid_ans_file_path, valid_ans_ratio)
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on datasets.')
    parser.add_argument('--models', type=str, required=True, help='Name of the model.')
    parser.add_argument('--datasets', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--run', type=str, default="1", help='Index of run.')
    parser.add_argument('--mode', type=str, help='Evaluation mode: hard or soft.')
    args = parser.parse_args()

    main(args)

    print("VOILA!")



'''


python3 run_eval.py --models blip2 --datasets aokvqa

python3 run_eval.py --models blip2 --datasets okvqa




'''