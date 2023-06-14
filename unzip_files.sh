#!/bin/bash

# Set the directory path and zip file names
directory="datasets"
zip_files=("FB_Hateful_Meme.zip" "mami.zip" "mvsa.zip")

# Change to the directory
cd "$directory"

# Loop through the zip files
for zip_file in "${zip_files[@]}"; do
  # Unzip the file
  unzip "$zip_file"

  # Delete the unpacked zip file
  rm "$zip_file"
done
