![Benchmark Image](ressources/readme/benchmark_img.png)

# Multimodal Large Language Model (MLLM) Benchmark

This repository houses an evaluation of seven leading MLLMs across nine datasets testing various cognitive capabilities underpinning language generation tasks.

## Benchmark Overview

FLUFFY stands as a comprehensive benchmark encompassing over 30,000 instances across nine datasets in visual question answering & reasoning, knowledgeable visual question answering, and classification. It distinguishes itself by offering in-depth insights into the architecture and training data of MLLMs, coupled with both quantitative performance assessments and qualitative analyses. Our evaluation also introduces post-processing tolerance to address observed limitations in instruction-following capabilities.

## Key Findings

- **Top Performers:** BLIP-2, InstructBlip, and LLaVA emerged as the leading models in our benchmark.
- **Variability in Instruction-Following:** Models showed considerable variation in their ability to follow instructions, with all facing challenges to some extent. BLIP models could follow instructions best.
- **Logical Reasoning:** LLaVA distinguished itself with the production of advanced chains of thought (even though being prompted to not do so), which coincided with advanced reasoning capabilities.
- **Visual Feature Extraction:** BLIP models use a light-weight transformer with cross-attention to bridge the vision and language module. We hypothesize that the cross-attention allows for goal-oriented visual feature extraction, which boosts the overall MLLM performance.
- **Language Encoder:** LLaVA's success is further hypothesized to be supported by the advanced language model LLaMA-2 Chat.

## Future Directions

- Additionally adopting few-shot prompting.
- Integrating tasks that necessitate longer responses and add automatic scoring systems for the evaluation of those tasks.
- Prompting models to generate intuitive textual responses instead of indices in tasks such as multiple-choice tasks.
- Adopting experimental setups to isolate performance drivers.

## Results

The table below shows the results when applying post-processing tolerance. Further performance assessment and discussion can be found in `report.pdf`.

TABLE_PLACEHOLDER

## Replicating/Extending Benchmark

Welcome to our project! To get started with using our repository follow these steps:

### Get Started

1. Clone the repository to your local machine: 
```bash
git clone https://github.com/ClaraLovesFunk/Testing-Multimodal-LLMs
```
2. Navigate to the repository directory: 
```bash
cd Testing-Multimodal-LLMs
```
3. Choose model(s) and dataset(s) to run them on.

### Prepare Datasets

1. Go to the directory `datasets`. Create a new directory with the name of the dataset of interest (if you use a dataset that was already implemented check `config.json` and use the same name as we did previously.)

2. Obtain the dataset and further proceed with the split of interest.

3. Turn the textual data with the split of interest into a json file and call it ds_benchmark.json.

4. Transform the structure of the textual data as shown in the example dataset in `datasets/example_dataset`. In case you want to use a dataset that has its own evaluation protocol such as A-OK-VQA or OKVQA additionally keep a copy of the original file(s) and leave their name as is (e.g., for A-OK-VQA the dataset is spread over two files one containing image ids one containing labels.)

5. Create a subdirectory in your directory `datasets/dataset_of_interest` called images and store the images.

6. If you added a new dataset to the benchmark follow these additional instructions:
    - Go to `utils/answer_processing.py` and adapt the function `get_clean_valid_preds_trues()`` for your new dataset. Read the function description and follow the example of one of the already implemented datasets.
    - Go to `utils/info.py` and extend the class DatasetInfo() and the function ​`​get_task2label_name()`` for your dataset.

### Prepare Inference

1. If you want to implement a model that was already implemented by us go to `inference` and to the directory with the name of the model you want to run. If you want to run a new model create a new directory with the name of the model you want to run within the directory `inference`. All the following instructions are assuming you want to run a model we have implemented. If you want to implement your own model follow the structure we have chosen and will describe now.

2. For first orientation go to the `inference/` and to the model of interest and read the specific README.md.

3. Create a virtual environment for each model you want to run. Store the virtual environment in the directory `venvs`. Install the dependencies via the requirements.txt file in each model directory. Repeat the following steps for each model.

  ```bash
  python3 -m venv venvs/<model_name>
  source venvs/<model_name>/bin/activate
  pip install -r inference/<model_directory>/requirements.txt
  ```

4. For further information about the implementation read the respective README.md. 

5. If you added a new model to the benchmark do the following before running inference:
    - add the model name in `config.json`.
    - Go to `utils/answer_processing.py` and add another condition for your model to the function `extract_answer()`. Follow the example of one of the models that were already implemented that you can see in that function.
    - Go to `inference` and create a new directory with the name of your model. Create the files README.md, inference.py. In `infernce.py` you need to actually write the implementation for your model. Ultimately this script needs to be run by `run_inference.py. You can use other models such as idefics as examples.

### Run Inference

To run a model on a dataset and save the output you can do: 
```bash
python3 run_inference.py -models model_of_interest -datasets dataset_of_interest
```

If you want to run all models or all datasets replace the model of dataset of interest with the string ‘all’ such as:
```bash
python3 run_inference.py -models all -datasets dataset_of_interest
```

You can also run multiple models on multiple datasets e.g.:
```bash
python3 run_inference.py -models model_of_interest1 model_of_interest2 -datasets dataset_of_interest1 dataset_of_interest1
``````



### Run Evaluation

If you want to evaluate a model on a dataset and save the metrics indicate the correctly/incorrectly predicted samples as you can see in `evaluations` do:
```bash
python run_eval.py --models model_of_interest --datasets dataset_of_interest --run run_of_interest --mode hard --calcavrg no
```
`run_of_interest` is an integer that represents the run of your experiment. `Mode` can be either the string `hard` or `soft` representing whether you want to evaluate without or with post-processing tolerance. `Calcavrg` determines whether the average per model over all datasets should be calculated. This argument can take the values `yes` or `no`. You can also use the `all` keyword or a list of `models/datasets` as when running inference.

## Repository Structure

This section outlines the organization of the repository detailing the directories and their contents to facilitate navigation and understanding of where key files and resources are located.

- `/datasets`: Contains directories for each dataset used in the benchmark.
  - `/example_dataset`: An example dataset directory.
    - `ds_benchmark.json`: Demonstrates the format for dataset split of interest.
- `/evaluation/`: Holds evaluation scripts and resources for each dataset.
  - `eval.py`: the script holding the dataset-specific evaluation function(s) that are called from `run_eval.py`
  - `/eval_dataset_of_interest`: Optional subdirectory for additional resources.
  - `/README.md`: Provides details on dataset-specific evaluations.
- `/experiments`: Documents the experiments conducted including configurations and outputs.
  - `/model/dataset/run`: Contains files documenting individual experiment runs
    - `/config.json`: Information on the experiment's setup.
    - `/output.json`: Raw output from the model.
    - `/output_aux_hard.json`: Cleaned output without post-processing tolerance (for further explanation see `report.pdf`).
    - `/output_aux_soft.json`: Cleaned output with post-processing tolerance (for further explanation see `report.pdf`).
    - `/scores_hard.json`: Metrics when processing output without post-processing tolerance.
    - `/scores_soft.json`: Metrics when processing output with post-processing tolerance
    - `/examples_hard.json`: Indicates which dataset samples were predicted correctly when not applying post-processing tolerance.
    - `/examples_soft.json`: Indicates which dataset samples were predicted correctly when applying post-processing tolerance.
    - `/valid_ans_hard.json`: Ratio of valid answers when not applying post-processing tolerance.
    - `/valid_ans_soft.json`: Ratio of valid answers applying post-processing tolerance.
- `/inference`: Includes inference scripts and model-specific resources.
  - `/model_of_interest`: Includes inference scripts and model-specific resources.
    - `inference.py`: inference script for running a model on a specified dataset given a model configuration. It loads a pre-trained model, processes input data, and generates predictions. Additionally, it saves the model's outputs and related information to the specified directory for evaluation purposes when called from another script.
    - `/subdirectory_of_interest`: optional subdirectory that can contain additional resources for the model to run such as a cloned git repository
    - `README.md`: further information on the model of interest & its implementation
    - `Requirements.txt`: Stores dependencies that need to be installed in the respective environment.
- `/resources`: Auxiliary resources like images or additional scripts related to the project.
- `/utils`: Utility scripts for common functions across the project.
  - `config.json`: Configuration settings for the benchmarking process.
  - `prompts.py`: Generates prompts for the models based on the benchmark datasets.
- `README.md`: The main documentation providing detailed information about the project setup and usage.
- `report.pdf`: A comprehensive report of the benchmark findings.
- `run_eval.py` and `run_inference.py`: Main scripts for running evaluations and inferences.
