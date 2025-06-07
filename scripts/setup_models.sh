#!/bin/bash

# Create base models directory
mkdir -p models/finetuned/t_lite

# Move existing models if they exist
if [ -d "src/ml/models/finetuned/t_lite" ]; then
    echo "Moving existing models to new location..."
    mv src/ml/models/finetuned/t_lite/* models/finetuned/t_lite/
fi

echo "Models directory structure created at ./models" 