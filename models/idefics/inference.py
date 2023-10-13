import torch
import os
from transformers import IdeficsForVisionText2Text, AutoProcessor

CACHE_DIR = '/home/users/cwicharz/project/Testing-Multimodal-LLMs/data/huggingface_cache'
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def load_idefics_model(checkpoint):
    
    model = IdeficsForVisionText2Text.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
    
    return model, processor

checkpoint = "HuggingFaceM4/idefics-9b-instruct" 
model, processor = load_idefics_model(checkpoint)


# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
prompts = [
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

        "\nUser:",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "And who is that?<end_of_utterance>",

        "\nAssistant:",
    ],
]




inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)

exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(output)


print('DOOOOOONNNE')







# prompt_generic = prompts.zeroshot(test_sample=sample, task=t)
# prompts = [
#     [
#         "User: " + prompt_generic,
#         img_path,
#         "<end_of_utterance>",

#         "\nAssistant:",
#     ],
# ]
