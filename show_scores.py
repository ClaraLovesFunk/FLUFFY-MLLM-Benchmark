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

# ... Rest of your code ...

html = (df_pivot.style.set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold'), ('border-bottom', '1px solid black')]}  # make all headers bold and add bottom border
])
    .format("{:.2f}")  # adjust number formatting if necessary
    .render())



def add_top_header_border(html_str):
    # Split the HTML string into lines
    lines = html_str.split("\n")

    # Identify the topmost headers
    topmost_headers = [line for line in lines if "level0 col" in line and "col_heading" in line]

    # For each topmost header, add a style attribute for the bottom border
    for i, line in enumerate(lines):
        if line in topmost_headers:
            # Insert the style attribute after the opening tag in the line
            first_tag_end = line.find('>') 
            style_str = ' style="border-bottom: 1px solid black;"'
            lines[i] = line[:first_tag_end] + style_str + line[first_tag_end:]

    # Join the lines back together into a single HTML string
    html_str = "\n".join(lines)

    return html_str

# Adjust the rendered HTML to add a bottom border to the topmost headers
html = add_top_header_border(html)

heading = "# Testing-Multimodal-LLMs\n\n"
table = """
* Paper Coming Soon

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
"""

with open("README.md", "w") as f:
    f.write(heading + html + table)
