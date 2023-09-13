import os

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import torch
from open_flamingo import create_model_and_transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time= time.time() ########

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)

demo_image_one = Image.open('datasets/hateful_memes/images/all/98764.png')
demo_image_two = Image.open('datasets/hateful_memes/images/all/98762.png')
query_image = Image.open('datasets/hateful_memes/images/all/98756.png')


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""

vision_x = [image_processor(demo_image_one).unsqueeze(0).to(device), image_processor(demo_image_two).unsqueeze(0).to(device), image_processor(query_image).unsqueeze(0).to(device)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)



"""
In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""

tokenizer.padding_side = "left" 

lang_x = tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)
lang_x.to(device)


generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))






end_time= time.time()
run_time = end_time - start_time
print(f'runtime: {run_time}')

# source venvs/openflamingo/bin/activate
# cd models/openflamingo
# python inference.py --dataset hateful_memes