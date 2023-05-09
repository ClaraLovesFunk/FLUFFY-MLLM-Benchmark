#%%

from lavis.models import load_model_and_preprocess
import torch
from PIL import Image


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

raw_image = Image.open("datasets/coco2017/test/000000000001.jpg")
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print(model.generate({"image": image, "prompt": "Question: What is in the picture? Answer:"}))


# %%
