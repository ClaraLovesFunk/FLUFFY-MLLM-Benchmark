from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import torch

device = torch.device("cpu") # "cuda" if torch.cuda.is_available() else "cpu")


model_id = "adept/fuyu-8b"
model = FuyuForCausalLM.from_pretrained(model_id)
model.to(device)
processor = FuyuProcessor.from_pretrained(model_id)

# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
image_path = "/home/users/cwicharz/project/Testing-Multimodal-LLMs/datasets/mami/images/all/1.jpg"  
image = Image.open(image_path)

inputs = processor(text=text_prompt, images=image, return_tensors="pt")
for k, v in inputs.items():
    inputs[k] = v.to(device)

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=7) 
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
print(generation_text)