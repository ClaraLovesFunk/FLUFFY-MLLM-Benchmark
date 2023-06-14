from utils import *

dataset_name = 'aokvqa'

dataset_info = DatasetInfo(dataset_name)

split = dataset_info.get_split()
images = dataset_info.get_img_dataset()
tasks = dataset_info.get_tasks()

dataset_file_path = os.path.join('datasets', dataset_name, split + '.json')

data_text = dataset(dataset_name, dataset_file_path).load()

# get dataset properties

dataset_info = DatasetInfo(dataset_name)


for sample in data_text:

    #image = prep_image(device, images_dir_path, sample, vis_processors)

    for t in tasks:

        prompt = prompt_construct(test_sample = sample,task = t)
        #output = model.generate({"image": image, "prompt": prompt})

        output_task = 'output_' + t