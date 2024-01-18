import os


class ModelInfo():

    def __init__(self, model_name):
        self.model_name = model_name
        self.lavis_model_type = {
            'blip2': 'pretrain_flant5xxl',
            'instructblip': 'vicuna7b',
        }
        self.lavis_name = {
            'blip2': 'blip2_t5',
            'instructblip': 'blip2_vicuna_instruct',
        }
        
    def get_lavis_model_type(self):
        return self.lavis_model_type[self.model_name]
    
    def get_lavis_name(self):
        return self.lavis_name[self.model_name]


class DatasetInfo():

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.text_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'val',
            'mvsa': 'test',
            'mami': 'test',
            'hateful_memes': 'dev', 
            'clevr': 'val_sampled',
            'esnlive': 'test',
            'gqa': 'val_sampled',
            'scienceqa': 'test'
        }
        self.img_dataset_split = {
            'aokvqa': 'val',
            'okvqa': 'all',
            'mvsa': 'all',
            'mami': 'all',
            'hateful_memes': 'all',
            'clevr': 'val',
            'esnlive': 'all',
            'gqa': 'all',
            'scienceqa': 'test'
        }
        self.img_dataset_name = {
            'aokvqa': 'coco2017/val',
            'okvqa': 'coco2017/all', 
            'mvsa': 'mvsa/images/all',
            'mami': 'mami/images/all',
            'hateful_memes': 'hateful_memes/images/all',
            'clevr': 'clevr/images/val',
            'esnlive': 'flickr30k_images/all',
            'gqa': 'gqa/images/all',
            'scienceqa': 'scienceqa/images/test'
        }
        self.tasks = {
            'aokvqa': ['direct answer (aokvqa)', 'multiple choice (aokvqa)'], 
            'okvqa': ['direct answer (okvqa)'],
            'mvsa': ['sentiment analysis'],
            'mami': ['sexism classification'],
            'hateful_memes': ['hate classification'],
            'clevr': ['direct answer (clevr)'],
            'esnlive': ['entailment prediction'],
            'gqa': ['direct answer (gqa)'],
            'scienceqa': ['multiple choice (sqa)']                                        
        }
        self.input_id_name = {
            'aokvqa': 'question_id',
            'okvqa': 'question_id', 
            'mvsa': 'id',
            'mami': 'id',
            'hateful_memes': 'id',
            'clevr': 'input_id',
            'esnlive': 'question_id',
            'gqa': 'input_id',         
            'scienceqa': 'input_id'
        } 
        
    def get_text_dataset_split(self):
        return self.text_dataset_split[self.dataset_name]
    
    def get_img_dataset_split(self):
        return self.img_dataset_split[self.dataset_name]
    
    def get_img_dataset_name(self):
        return self.img_dataset_name[self.dataset_name]

    def get_tasks(self):
        return self.tasks[self.dataset_name]
    
    def get_input_id_name(self):
        return self.input_id_name[self.dataset_name]


def get_info(dataset_name, model_name, run):

    dataset_info = DatasetInfo(dataset_name)
    img_dataset_name = dataset_info.get_img_dataset_name()
    tasks = dataset_info.get_tasks()
    split = dataset_info.get_text_dataset_split()

    base_path = '/home/users/cwicharz/project/Testing-Multimodal-LLMs'
    ds_file_path = os.path.join(base_path, 'datasets', dataset_name, 'ds_benchmark.json')
    image_dir_path = os.path.join(base_path, 'datasets', img_dataset_name) 
    output_dir_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run)
    output_file_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run, 'output.json' )
    config_file_path = os.path.join(base_path, 'experiments', model_name, dataset_name, 'run' + run, 'config.json' )

    return tasks, ds_file_path, image_dir_path, output_dir_path, output_file_path, config_file_path, split


def get_task2label_name(task):
    '''
    get the name of the key, the label is stored in in the input data
    '''
    task2label_name = {
        "direct answer (okvqa)": "correct_direct_answer_short",
        "direct answer (aokvqa)": "correct_direct_answer_short",
        "multiple choice (aokvqa)": "correct_multiple_choice_answer",
        "multiple choice (sqa)": "correct_choice",
        "direct answer (clevr)": "correct_direct_answer_short",
        "direct answer (gqa)": "correct_direct_answer_short",
        "hate classification": "classification_label",
        "sexism classification": "classification_label",
        "sentiment analysis": "classification_label",
        "entailment prediction": "classification_label"
    }
    label_name = task2label_name.get(task, "Unknown Task")

    return label_name