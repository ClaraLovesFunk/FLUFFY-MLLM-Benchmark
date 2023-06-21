#%%

import pandas as pd

path_mvsa = 'datasets/mvsa/test.json'
path_mami = 'datasets/mami/test.json'
path_mami = 'datasets/hateful_memes/test_unseen.json'

# Read the JSON file into a DataFrame
df = pd.read_json(path_mami)

# Display the DataFrame
df.head(10)


# %%
