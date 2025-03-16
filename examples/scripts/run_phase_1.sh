#!/bin/bash

CONFIG_DIR="examples/config/phase_1"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    ./predictor.sh --load_config "$file"
done

echo "All configurations processed."
