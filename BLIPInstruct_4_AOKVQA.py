#%%
from PIL import Image

import torch

from lavis.models import load_model_and_preprocess

#import Display
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
print('test')
raw_image = Image.open("/home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets/coco2017/test/000000000001.jpg").convert("RGB")
#display(raw_image.resize((596, 437)))



# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)


llm_model_path = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/LAVIS/lavis/models/blip2_models/blip2_vicuna_instruct"

model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=llm_model_path, is_eval=True, device=device)




# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
model.generate({"image": image, "prompt": "What is unusual about this image?"})



# %%
