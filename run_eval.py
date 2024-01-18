import argparse
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

import utils.utils_eval as utils_eval


CONFIG_PATH = 'config.json'
ALL_KEYWORD = 'all'



def main(args):
    
    config = utils_eval.load_data(CONFIG_PATH)
    selected_models = config['model_names'] if args.models == ALL_KEYWORD else [args.models]
    selected_datasets = config['dataset_names'] if args.datasets == ALL_KEYWORD else [args.datasets]
    selected_modes = [args.mode] if args.mode else ["hard", "soft"]
    run = args.run
    calc_avrg = args.calcavrg

    for model_name, dataset_name, mode_name in product(selected_models, selected_datasets, selected_modes):

        print(f'Evaluating {model_name} on {dataset_name} in {mode_name} mode:')

        if dataset_name == 'hateful_memes':
            scores, examples, valid_ans_ratio = evaluate_hateful_memes(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == 'mami':
            scores, examples, valid_ans_ratio = evaluate_mami(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == 'mvsa':
            scores, examples, valid_ans_ratio = evaluate_mvsa(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == "esnlive":
            scores, examples, valid_ans_ratio = evaluate_esnlive(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == "scienceqa":
            scores, examples, valid_ans_ratio = evaluate_scienceqa(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == "aokvqa":
            scores, examples, valid_ans_ratio = evaluate_aokvqa(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == "okvqa":
            scores, examples, valid_ans_ratio = evaluate_okvqa(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == "gqa":
            scores, examples, valid_ans_ratio = evaluate_gqa(CONFIG_PATH, dataset_name, model_name, mode_name, run)

        if dataset_name == "clevr":
            scores, examples, valid_ans_ratio = evaluate_clevr(CONFIG_PATH, dataset_name, model_name, mode_name, run)


        print(scores)
        
        scores_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode_name, value_of_interest = 'scores_path')
        examples_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode_name, value_of_interest = 'examples_path')
        val_ratio_path = utils_eval.get_paths(CONFIG_PATH, dataset_name, model_name, run, mode_name, value_of_interest = 'val_ratio_path')

        utils_eval.save_data(scores_path, scores)
        utils_eval.save_data(examples_path, examples)
        utils_eval.save_data(val_ratio_path, valid_ans_ratio)

        utils_eval.make_examples_file(model_name, dataset_name, run)

    # calculate overall accuracy average

    for model_name, mode_name in product(selected_models, selected_modes):
        if calc_avrg == "yes":
            utils_eval.calculate_average_accuracy_over_all_ds(CONFIG_PATH, model_name, mode_name)


        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on datasets.')
    parser.add_argument('--models', type=str, required=True, help='Name of the model.')
    parser.add_argument('--datasets', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--run', type=str, default="1", help='Index of run.')
    parser.add_argument('--mode', type=str, help='Evaluation mode: hard or soft.')
    parser.add_argument('--calcavrg', type=str, default="no", help='Should the average per model over all datasets be calculated? Answer with "yes" or "no"')   ####
    args = parser.parse_args()

    main(args)

    print("VOILA!")



'''
python3 run_eval.py --models all --datasets all
python3 run_eval.py --models blip2 --datasets aokvqa
python3 run_eval.py --models all --datasets all
'''