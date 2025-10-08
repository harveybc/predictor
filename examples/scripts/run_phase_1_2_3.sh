#!/bin/bash

CONFIG_DIR="examples/config/phase_1"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_2_1"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_2_2"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_2_3"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_2_4"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_3"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

echo "All configurations processed."
