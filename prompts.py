def zeroshot(test_sample, task):
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

    instructions = {
        'direct answer (aokvqa)': 'Answer the following question adhering to these guidelines:\n1. Omit articles (like \'a\', \'an\', \'the\') before nouns.\n2. Represent all numbers in word form, not as digits.',
        'direct answer (okvqa)': 'Answer the following question adhering to these guidelines:\n1. Omit articles (like \'a\', \'an\', \'the\') before nouns.\n2. Represent all numbers as digits, not word form.',
        'direct answer (clevr)': 'Answer the following question adhering to these guidelines:\n1. Omit articles (like \'a\', \'an\', \'the\') before nouns.\n2. Represent all numbers as digits, not word form.',
        'direct answer (gqa)': 'Answer the following question adhering to these guidelines:\n1. Omit articles (like \'a\', \'an\', \'the\') before nouns.\n2. Represent all numbers as digits, not word form.',
        'multiple choice (aokvqa)': 'Answer the question by choosing from the options below.',
        'multiple choice (sqa)': 'Answer the question by choosing the correct index from the options below.For the first answer, write \'0\'; for the second, write \'1\', and so on.',
        'sentiment analysis': 'Predict the sentiment of the tweet in combination with the image!The sentiment can be either "Positive", "Negative" or "Neutral".Respond only with one of these three options.',
        'sexism classification': 'Classify the following meme as \'sexist\' or \'not sexist\'. Respond only with one of these two options.',
        'hate classification': 'Classify the meme as either \'hateful\' or \'not hateful\'. Respond only with one of these two options.',
        'entailment prediction': 'Classify the following image as \'entailment\', if there is enough evidence in the image to conclude that the following hypothesis is true. Classify the following image as \'contradiction\', if there is enough evidence in the image to conclude that the following hypothesis is false. Classify the following image as \'neutral\', if neither of the earlier two are true. Respond only with one of these three options.'
    }

    # Base prompt
    prompt = instructions[task]

    # Additional task-specific prompt adjustments
    text_input = test_sample['text_input']
    if 'answer (aokvqa)' in task or 'answer (okvqa)' in task or 'answer (clevr)' in task or 'answer (gqa)' in task:
        prompt += f'\n{question_formal}{text_input}\n{answer_formal}'
    elif 'multiple choice' in task:
        choices_content = ', '.join(test_sample['answer_choices'])
        prompt += f'\n{question_formal}{text_input}\n{choices_formal}{choices_content}{choices_end_formal}\n{answer_formal}'
    elif task == 'sentiment analysis':
        prompt += f'\n{tweet_formal}{text_input}\n{sentiment_formal}'
    elif task == 'sexism classification':
        prompt += f'\n{meme_text_formal}{text_input}\n{sexist_label_formal}'
    elif task == 'hate classification':
        prompt += f'\n{meme_text_formal}{text_input}\n{hate_label_formal}'
    elif task == 'entailment prediction':
        prompt += f'\n{hypothesis_text_formal}{text_input}\n{entailment_label}'

    return prompt