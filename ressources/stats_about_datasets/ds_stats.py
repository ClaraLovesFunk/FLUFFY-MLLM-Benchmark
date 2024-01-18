import os
import json


# Directory paths
datasets_dir = 'datasets'
output_file_path = 'stats/ds_stats.json'

# Ensure the stats directory exists
if not os.path.exists('stats'):
    os.makedirs('stats')

# Count instances for each dataset
dataset_counts = {}
total_instances = 0

# For each dataset directory
for dataset_name in os.listdir(datasets_dir):
    dataset_dir = os.path.join(datasets_dir, dataset_name)
    
    # Ensure it's a directory
    if os.path.isdir(dataset_dir):
        ds_file_path = os.path.join(dataset_dir, 'ds_benchmark.json')
        
        # If the benchmark file exists
        if os.path.exists(ds_file_path):
            with open(ds_file_path, 'r') as file:
                data = json.load(file)
                # Count instances with text_input_id key
                instance_count = sum(1 for item in data['data'] if 'text_input_id' in item)
                dataset_counts[dataset_name] = instance_count
                total_instances += instance_count

# Save the results
results = {
    'datasets': dataset_counts,
    'total_instances': total_instances
}

with open(output_file_path, 'w') as out_file:
    json.dump(results, out_file, indent=4)

print(f"Saved dataset statistics to {output_file_path}")