# gen_readme.py

# Define the paths to the files within the 'ressources' directory
text_readme_path = 'ressources/readme/README_TEXT.md'
tables_readme_path = 'ressources/readme/README_TABLES.md'
output_readme_path = 'README.md'  # This will be in the current working directory

# Read the content of README_TEXT.md
with open(text_readme_path, 'r') as file:
    readme_text = file.read()

# Read the content of README_TABLES.md
with open(tables_readme_path, 'r') as file:
    readme_tables = file.read()

# Replace the placeholder in README_TEXT.md with the content of README_TABLES.md
readme_text = readme_text.replace('TABLE_PLACEHOLDER', readme_tables)

# Write the combined content to README.md
with open(output_readme_path, 'w') as file:
    file.write(readme_text)

print("README.md has been generated with tables inserted.")
