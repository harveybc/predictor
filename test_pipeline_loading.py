#!/usr/bin/env python3
"""Quick test to verify which pipeline is being loaded with the config"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

from app.plugin_loader import load_plugin
import json

print("Testing pipeline plugin loading...")

# Load the config file
config_file = "examples/config/phase_2_daily/phase_2_6_cnn_1d_config.json"
try:
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"Config loaded. pipeline_plugin: {config.get('pipeline_plugin', 'NOT FOUND')}")
except Exception as e:
    print(f"Failed to load config: {e}")
    sys.exit(1)

# Try to load the pipeline plugin
pipeline_name = config.get('pipeline_plugin', 'default_pipeline')
print(f"Attempting to load pipeline: {pipeline_name}")

try:
    pipeline_class, _ = load_plugin('pipeline.plugins', pipeline_name)
    print(f"✓ Successfully loaded pipeline class: {pipeline_class}")
    print(f"✓ Pipeline class file: {pipeline_class.__module__}")
    
    # Instantiate it
    pipeline_instance = pipeline_class()
    print(f"✓ Pipeline instance created: {type(pipeline_instance)}")
    
except Exception as e:
    print(f"✗ Failed to load pipeline '{pipeline_name}': {e}")
    import traceback
    traceback.print_exc()
