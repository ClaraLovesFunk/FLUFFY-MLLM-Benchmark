import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from tqdm import notebook
from gill import models
from gill import utils


model_dir = 'checkpoints/gill_opt/'
model = models.load_gill(model_dir)

pancakes_img = utils.get_image_from_url('https://images.pexels.com/photos/376464/pexels-photo-376464.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1')
prompts = [
    pancakes_img,
    'A picture of'
]

plt.imshow(pancakes_img)
plt.show()

return_outputs = model.generate_for_images_and_texts(prompts, num_words=16, min_word_tokens=16)
print(return_outputs[0])