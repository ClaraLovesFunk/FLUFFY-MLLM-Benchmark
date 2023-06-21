#%%

import json
import pandas as pd
from utils import *



# Open the JSONL file and collect objects in a list
data = []
with open('datasets/MVSA/splits/test_1_with_lavis.jsonl', 'r') as file:
    for line in file:
        obj = json.loads(line)
        data.append(obj)

# Create a DataFrame from the collected data
df = pd.DataFrame(data)



prompt_construct(df.iloc[0], 'sentiment analysis')


# Display the DataFrame
df.head(5)

# %%
