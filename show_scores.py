import os
import json
import pandas as pd
from IPython.display import display, HTML
from bs4 import BeautifulSoup

root_dir = "experiments"

table_title = {
    "soft": "Evaluation with Post-Processing Tolerance",
    "hard": "Evaluation without Post-Processing Tolerance"
}

def process_scores(file_name, mode):
    data = []

    # Go through each directory and subdirectory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == file_name:
                with open(os.path.join(dirpath, filename), 'r') as f:
                    scores = json.load(f)

                # Extract model and dataset names from the directory path
                path_parts = dirpath.split(os.sep)
                model = path_parts[-3]
                dataset = path_parts[-2]
                run = path_parts[-1]

                # For each task in the JSON file
                for task, metrics in scores.items():
                    for metric, value in metrics.items():
                        # Create a dictionary for each metric, model, dataset, and task combination
                        data.append({'Model': model, 'Dataset': dataset, 'Task': task, 'Metric': metric, 'Value': value})

    df = pd.DataFrame(data)

    # Pivot the DataFrame
    df_pivot = df.pivot_table(index='Model', columns=['Dataset', 'Task', 'Metric'], values='Value')
    df_pivot.sort_index(axis=1, level=0, inplace=True)
    df_pivot.index.name = ""

    html = (df_pivot.style.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('border-bottom', '1px solid black')]}
    ]).format("{:.2f}").render())

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find and remove the style tags
    for style in soup.find_all('style'):
        style.decompose()

    html = str(soup)
    html = add_top_header_border(html)


    return f"### {table_title[mode]}\n\n" + html



def add_top_header_border(html_str):
    # Split the HTML string into lines
    lines = html_str.split("\n")

    # Remove the CSS style block
    lines = [line for line in lines if not line.startswith('<style type="text/css">')]

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


# Process both score files
html_soft = process_scores("scores_soft.json", "soft")
html_hard = process_scores("scores_hard.json", "hard")

# Add the image at the top of the README
image = '<img src="utils_general/fluffy.png" width="100%" />\n\n'
benchmark_subheader = "## Benchmark\n\n"
space = '\n\n'

# Add the Model Implementation section
model_implementation_section = """
## Model Implementation

For some models, the respective GitHub repository needed to be cloned and files tweaked. 
The cloned and modified repositories are collected in `models`. 
To implement the models yourself, follow the instructions in `MAKE_ME_RUN.md`, 
that can be found in `models/model_of_interest`.

"""

# Write the content to the README file
with open("README.md", "w") as f:
    f.write(image + benchmark_subheader + space)
    f.write(html_soft + space + html_hard)
    f.write(model_implementation_section)