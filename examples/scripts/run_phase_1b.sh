#!/bin/bash

CONFIG_DIR="examples/config/phase_1b"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_1b_dec"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_1b_candlestick"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

CONFIG_DIR="examples/config/phase_1b_candlestick_dec"

for file in "$CONFIG_DIR"/*.json; do
    echo "Running preprocessor with configuration: $(basename "$file")"
    sh ./predictor.sh --load_config "$file"
done

echo "All configurations processed."
