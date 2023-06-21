#!/bin/bash

# Define the path of the main directory.
main_dir="datasets/coco2014"

# Navigate to the main directory.
cd $main_dir

# Create a new directory named "all" inside the main directory.
mkdir -p all

# Find and copy all the files from the "test", "train", and "val" directories to the "all" directory.
find test train val -type f -exec cp -n {} all \;
