import os
import json
import pandas as pd

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

            # For each task in the JSON file
            for task, metrics in scores.items():
                for metric, value in metrics.items():
                    # Create a dictionary for each metric, model, dataset, and task combination
                    data.append({'Model': model, 'Dataset': dataset, 'Task': task, 'Metric': metric, 'Value': value})

df = pd.DataFrame(data)

# Pivot the DataFrame, this time with 'Model' as index, and 'Dataset', 'Task', and 'Metric' as columns.
df_pivot = df.pivot_table(index='Model', columns=['Dataset', 'Task', 'Metric'], values='Value')

# This will sort the DataFrame for better visualization.
df_pivot.sort_index(axis=1, level=0, inplace=True)

# Creating markdown table
markdown_table = df_pivot.to_markdown()

# We'll add an extra line after the topmost header line to simulate the divider line
lines = markdown_table.split("\n")
first_header_line_index = [i for i, line in enumerate(lines) if "|---" in line][0]
lines.insert(first_header_line_index, "| --- | --- | --- |")

# Joining the lines back together
markdown_table = "\n".join(lines)

# Writing to README.md
with open("README.md", "w") as f:
    f.write(markdown_table)
