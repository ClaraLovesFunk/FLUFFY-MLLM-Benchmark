import os
import pandas as pd
from PIL import Image
from IPython.display import display, HTML
import base64
import pandas as pd
from PIL import Image

import prompts




def image_to_html(image):
    image.save('temp.png')
    with open('temp.png', 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    os.remove('temp.png')
    return f'<img src="data:image/png;base64,{encoded_image}"/>'



def show_imgs_text_output(dataset_name, data_samples_input, data_samples_output, images_dir, tasks):
    data_incl_image = []

    for i, sample in enumerate(data_samples_input): # Notice the use of enumerate


        image_file_name = sample['image_id']
        image_path = os.path.join(images_dir, image_file_name)
        img = Image.open(image_path)
        img.thumbnail((100, 100))

        # add images to textual data
        dict_fulldata = {'image': image_to_html(img)}
        dict_fulldata.update(sample) # Use 'sample' instead of 'data_samples_input[i]'

        # add prompt
        for t in tasks:
            prompt = prompts.zeroshot(sample, t) # Use 'sample' instead of 'data_samples_input[i]'
            prompt_header = 'prompt ' + t
            dict_fulldata.update({prompt_header: prompt})

        # add output
        dict_fulldata.update(data_samples_output[i])

        data_incl_image.append(dict_fulldata)

    # turn into dataframe
    data_incl_image = pd.DataFrame(data_incl_image)

    # set pandas display option
    pd.set_option('display.max_colwidth', None)
    display(HTML(data_incl_image.to_html(escape=False)))
    print('\n')
    print('\n')

    return data_incl_image

