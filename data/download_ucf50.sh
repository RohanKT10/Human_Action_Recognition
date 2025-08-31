#!/bin/bash

# Move into data folder
cd "$(dirname "$0")"

# Download UCF50 dataset (RAR format)
echo "Downloading UCF50 dataset..."
wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF50.rar" -O UCF50.rar

# Extract RAR preserving folder structure into current folder
echo "Extracting dataset..."
unrar x UCF50.rar

# Remove the archive
rm UCF50.rar

echo "UCF50 dataset ready at $(pwd)/UCF50"