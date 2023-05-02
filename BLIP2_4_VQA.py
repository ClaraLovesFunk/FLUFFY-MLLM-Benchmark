# nvidia-smi
#%%

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

'''device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
'''







from PIL import Image
from transformers import pipeline
import requests

vqa_pipeline = pipeline("visual-question-answering")

#image =  Image.open("elephant.jpeg")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

question = "Is there an elephant?"

print(vqa_pipeline(image, question, top_k=1))


'''

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load BLIP-2 model and tokenizer
model_name = "Salesforce/blip2-opt-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Prepare image and question inputs
image_feature = torch.randn(1, 2048)  # replace with your actual image feature
question = "What color is the car?"

# Tokenize inputs and convert to tensors
inputs = tokenizer(question, return_tensors="pt")
inputs["pixel_values"] = image_feature.unsqueeze(0)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Pass inputs through model and get answer
outputs = model(**inputs)
start_logits, end_logits = outputs.start_logits, outputs.end_logits
start_index = torch.argmax(start_logits, dim=-1)
end_index = torch.argmax(end_logits, dim=-1)
answer_tokens = inputs["input_ids"][0][start_index:end_index+1]
answer = tokenizer.decode(answer_tokens)

print(answer)
'''