import pandas as pd
import random



def prompt_construct_zeroshot(test_sample, task): ##### rename to prompt_construct_zeroshot

    question_formal = 'Questions: '
    choices_formal = 'Choices: '
    choices_end_formal = '.'
    answer_formal = 'Answer: '
    tweet_formal = 'Tweet: '
    meme_text_formal = 'Meme Text: '
    sentiment_formal = 'Sentiment: '
    sexist_label_formal = 'Sexism Label: '
    hate_label_formal = 'Hate Label: '
    hypothesis_text_formal = 'Hypothesis: '
    entailment_label = 'Answer: '

    if task == 'direct answer (aokvqa)': 
        text_input = test_sample['text_input']
        instruction = '''
                        Answer the following question adhering to these guidelines:
                        1. Omit articles (like 'a', 'an', 'the') before nouns.
                        2. Represent all numbers in word form, not as digits.
                        '''
        prompt =  instruction +  '\n' + question_formal +  text_input +  '\n' + answer_formal

    if task == 'direct answer (okvqa)': 
        text_input = test_sample['text_input']
        instruction = '''
                        Answer the following question adhering to these guidelines:
                        1. Omit articles (like 'a', 'an', 'the') before nouns.
                        2. Represent all numbers as digits, not word form.
                        '''
        prompt =  instruction +  '\n' + question_formal +  text_input +  '\n' + answer_formal

    if task == 'direct answer (clevr)': 
        text_input = test_sample['text_input']
        instruction = '''
                        Answer the following question adhering to these guidelines:
                        1. Omit articles (like 'a', 'an', 'the') before nouns.
                        2. Represent all numbers as digits, not word form.
                        '''
        prompt =  instruction +  '\n' + question_formal +  text_input +  '\n' + answer_formal


    if task == 'direct answer (gqa)': 
        text_input = test_sample['text_input']
        instruction = '''
                        Answer the following question adhering to these guidelines:
                        1. Omit articles (like 'a', 'an', 'the') before nouns.
                        2. Represent all numbers as digits, not word form.
                        '''
        prompt =  instruction +  '\n' + question_formal +  text_input +  '\n' + answer_formal



    if task == 'multiple choice (aokvqa)': 
        text_input = test_sample['text_input']
        instruction = '''
                        Answer the question by choosing the correct index from the options below. 
                        For the first answer, write '1'; for the second, write '2', and so on.
                        '''
        choices_content = test_sample['answer_choices']
        choices_content = ', '.join(choices_content)
        prompt =  instruction +  '\n' + question_formal +  text_input + '\n' + choices_formal  + choices_content + choices_end_formal +  '\n' + answer_formal
    
    if task == 'multiple choice (sqa)': 
        text_input = test_sample['text_input']
        instruction = '''
                        Answer the question by choosing the correct index from the options below. 
                        For the first answer, write '0'; for the second, write '1', and so on.
                        '''
        choices_content = test_sample['answer_choices']
        choices_content = ', '.join(choices_content)
        prompt =  instruction +  '\n' + question_formal +  text_input + '\n' + choices_formal  + choices_content + choices_end_formal +  '\n' + answer_formal
    
    if task == 'sentiment analysis':
        text_input = test_sample['text_input']
        instruction = '''
                        Predict the sentiment of the tweet in combination with the image! 
                        The sentiment can be either "Positive", "Negative" or "Neutral". 
                        Respond only with one of these three options.
                        '''
        prompt =  instruction +  '\n' +  tweet_formal + text_input + '\n' + sentiment_formal

    if task == 'sexism classification':
        text_input = test_sample['text_input']
        instruction = '''
                        Classify the following meme as 'sexist' or 'not sexist'. 
                        Respond only with one of these two options.
                        '''
        prompt =  instruction +  '\n' +  meme_text_formal + text_input + '\n' + sexist_label_formal

    if task == 'hate classification':
        
        text_input = test_sample['text_input']
        instruction = '''
                        Classify the meme as either 'hateful' or 'not hateful'. 
                        Respond only with one of these two options.
                        '''
        prompt =  instruction +  '\n' +  meme_text_formal + text_input + '\n' + hate_label_formal

    if task == 'entailment prediction':
        text_input = test_sample['text_input']
        instruction = '''
                        Classify the following image as 'entailment', if there is enough evidence in the image to conclude that the following hypothesis is true. 
                        Classify the following image as 'contradiction', if there is enough evidence in the image to conclude that the following hypothesis is false.
                        Classify the following image as 'neutral', if neither of the earlier two are true.
                        Respond only with one of these three options.
                        '''
        prompt =  instruction +  '\n' +  hypothesis_text_formal + text_input + '\n' + entailment_label


    return prompt





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
