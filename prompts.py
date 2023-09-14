import pandas as pd
import random

ds_file_path = "datasets/hateful_memes/ds_benchmark_fulllabel.json"
dataset = pd.read_json(ds_file_path) 
data_list = dataset['data'].tolist()

def prompt_construct_fewshot_openflamingo(samples_IC, sample_query):
    
    prompt = ""

    # add in-context samples
    for s in samples_IC:
        print(str(s['classification_label']))
        prompt += "<image>This meme is " + str(s['classification_label']) + ".<|endofchunk|>"  

    # add query sample
    prompt += "<image>This meme is" + str(sample_query['classification_label'])  

    return prompt


def sample_ICsamples_random(sample_query, data_list, n):
    # Remove the sample_query from the data_list if present
    data_list = [sample for sample in data_list if sample != sample_query]
    
    # Randomly select n samples
    random_samples = random.sample(data_list, n)

    return random_samples



sample_query = data_list[2]
n = 3

random_samples = sample_ICsamples_random(sample_query, data_list, n)
prompt = prompt_construct_fewshot_openflamingo(samples_IC = random_samples, sample_query = sample_query)
print(f'prompt: {prompt}')
