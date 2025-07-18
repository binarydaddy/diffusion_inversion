#!/bin/bash

# Remove existing symlinks/directories if they exist to prevent infinite nesting
rm -rf ./validation_result ./validation_data

# Create fresh symlinks
ln -s /data/inversion_data/validation_result ./validation_result
ln -s /data/inversion_data/validation_data ./validation_data

echo "Symlinks created successfully:"
echo "  validation_result -> /data/inversion_data/validation_result"
echo "  validation_data -> /data/inversion_data/validation_data"