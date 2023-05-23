#%%

import os
import json
import pandas as pd
from IPython.display import display, HTML

root_dir = "experiments"

data = []

# Go through each directory and subdirectory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "scores.json":
            with open(os.path.join(dirpath, filename), 'r') as f:
                scores = json.load(f)

            # Extract model and dataset names from the directory path
            path_parts = dirpath.split(os.sep)
            model = path_parts[-2]
            dataset = path_parts[-1]

            # Include model and dataset information in the dictionary
            scores['Model'] = model
            scores['Dataset'] = dataset

            # Add the scores dictionary to our data list
            data.append(scores)

df = pd.DataFrame(data)

# first, we ensure that we have a multi-index in both columns and rows before pivot
df = pd.DataFrame(data)



# reset the index
df.reset_index(inplace=True)

# melt the DataFrame so that the column names become a 'Metric' column
df_melted = df.melt(id_vars=['Model', 'Dataset'], var_name='Metric', value_name='Value')

# now we pivot again, this time with 'Model' and 'Metric' as index, 'Dataset' as columns and 'Value' as values
df_pivot = df_melted.pivot(index=['Model', 'Metric'], columns='Dataset', values='Value')

# sort the index for better visualization
df_pivot.sort_index(axis=0, level=0, inplace=True)

print(df_pivot)
df_pivot.head()


# %%
