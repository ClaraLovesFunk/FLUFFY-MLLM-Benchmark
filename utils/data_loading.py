import json

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    