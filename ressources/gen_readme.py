import os
import json
import pandas as pd
from IPython.display import display, HTML
from bs4 import BeautifulSoup






root_dir = "experiments"

table_title = {
    "soft": "Metrics - Evaluation with Post-Processing Tolerance",
    "hard": "Metrics - Evaluation without Post-Processing Tolerance",
    "valid_ans_soft": "Valid Answer Ratio - Evaluation with Post-Processing Tolerance",
    "valid_ans_hard": "Valid Answer Ratio - Evaluation without Post-Processing Tolerance"
}


def process_scores(file_name, mode):
    data = []

    # Dictionary to hold average accuracies
    average_accuracies = {}

    # Go through each model directory
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if os.path.isdir(model_path):
            avg_accuracy_file = os.path.join(model_path, f"scores_average_{mode}.json")

            # Check if the average accuracy file exists and read it
            if os.path.exists(avg_accuracy_file):
                with open(avg_accuracy_file, 'r') as f:
                    avg_accuracy_data = json.load(f)
                    average_accuracies[model_dir] = avg_accuracy_data["average accuracy"]

    # Go through each directory and subdirectory for the scores
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == file_name:
                with open(os.path.join(dirpath, filename), 'r') as f:
                    scores = json.load(f)

                # Extract model name from the directory path
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

    # Add the average accuracy column to the pivot table
    df_pivot['Average Accuracy'] = df_pivot.index.map(average_accuracies.get)
    df_pivot = df_pivot.sort_values(by='Average Accuracy', ascending=False)


    html = (df_pivot.style.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('border-bottom', '1px solid black')]}
    ]).format("{:.2f}").to_html())

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find and remove the style tags
    for style in soup.find_all('style'):
        style.decompose()

    html = str(soup)
    html = add_top_header_border(html)

    return f"### {table_title[mode]}\n\n" + html


def process_valid_answer_scores(mode):
    data = []

    # Go through each model and dataset directory for the valid answer scores
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if os.path.isdir(model_path):
            for dataset_dir in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset_dir, "run1")
                score_file = os.path.join(dataset_path, f"valid_ans_{mode}.json")

                # Check if the score file exists and read it
                if os.path.exists(score_file):
                    with open(score_file, 'r') as f:
                        scores = json.load(f)

                    # For each task in the JSON file
                    for task, score in scores.items():
                        # Append data for model, dataset, and task
                        data.append({'Model': model_dir, 'Dataset': dataset_dir, 'Task': task, 'Score': score})

    df = pd.DataFrame(data)

    # Pivot the DataFrame
    df_pivot = df.pivot_table(index='Model', columns=['Dataset', 'Task'], values='Score')
    df_pivot.sort_index(axis=1, level=0, inplace=True)
    df_pivot.index.name = ""

    # Calculate average ratios
    df_pivot['Average Ratio'] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values(by='Average Ratio', ascending=False)


    html = (df_pivot.style.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('border-bottom', '1px solid black')]}
    ]).format("{:.2f}").to_html())

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find and remove the style tags
    for style in soup.find_all('style'):
        style.decompose()

    html = str(soup)
    html = add_top_header_border(html)

    title = table_title['valid_ans_'+ mode]
    print(title)
    return f"### {title}\n\n" + html


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

html_valid_ans_soft = process_valid_answer_scores("soft")
html_valid_ans_hard = process_valid_answer_scores("hard")

html_soft = process_scores("scores_soft.json", "soft")
html_hard = process_scores("scores_hard.json", "hard")

# Add the image at the top of the README
image = '<img src="ressources/img_for_readme/fluffy.png" width="100%" />\n\n'
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
    f.write(html_soft + space) # Only include the first and second tables
    f.write(html_valid_ans_soft + space)
    f.write(model_implementation_section)