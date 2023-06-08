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

# Hide index name
df_pivot.index.name = ""

markdown = df_pivot.to_markdown()

heading = "# Testing-Multimodal-LLMs\n\n"
subheading1 = "## Benchmark "
subheading2 = "## Checklist "
table = """


| Models              | A-OKVQA | OKVQA | VQA-v2 | EMU | E-SNLI-VE | VCR |
|---------------------|---------|-------|--------|-----|-----------|-----|
| BLIP-2              |&#x2714; &#x2714;|&#x2714; &#x2714;|        |     |           |     |
| BLIP-vicuna         |         |       |        |     |           |     |
| Prismer             |         |       |        |     |           |     |
| OpenFlamingo*       |         |       |        |     |           |     |
| MiniGPT             |         |       |        |     |           |     |
| Llava               |         |       |        |     |           |     |
| Otter*              |         |       |        |     |           |     |
| Fromage             |         |       |        |     |           |     |
| MAGMA (no hf)       |         |       |        |     |           |     |
| Limber (no hf)      |         |       |        |     |           |     |
| MAPL (no hf)        |         |       |        |     |           |     |
| FLAN-T5 (text)      |         |       |        |     |           |     |
| GPT4AIl (text)      |         |       |        |     |           |     |
| OpenAssistant (text)|         |       |        |     |           |     |

* Paper Coming Soon
"""

with open("README.md", "w") as f:
    f.write(heading + subheading1 + markdown + subheading2 + table)
