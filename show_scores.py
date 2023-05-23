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

# Melt the DataFrame so that the column names become a 'Metric' column
df_melted = df.melt(id_vars=['Model', 'Dataset'], var_name='Metric', value_name='Value')

# Pivot the DataFrame, this time with 'Model' as index, and 'Dataset' and 'Metric' as columns.
df_pivot = df_melted.pivot_table(index='Model', columns=['Dataset', 'Metric'], values='Value')

# This will sort the DataFrame for better visualization.
df_pivot.sort_index(axis=1, level=0, inplace=True)

# Hide index name
df_pivot.index.name = ""

# Display the DataFrame as HTML with Model names, Dataset names and Metric names bold
html = (df_pivot.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold')]}  # make all headers bold
])
    .format("{:.2f}")  # adjust number formatting if necessary
    .render())

display(HTML(html))

# %%
