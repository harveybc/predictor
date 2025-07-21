#!/usr/bin/env python3
"""Test script for the new phase2_6 plugins"""

import sys
import os

print("Starting plugin test...")

try:
    print("Testing pipeline plugin import...")
    from pipeline_plugins.phase2_6_pipeline import Phase26PipelinePlugin
    print("✓ Pipeline plugin imported")
    
    pipeline = Phase26PipelinePlugin()
    print("✓ Pipeline plugin instantiated")
    
except Exception as e:
    print(f"✗ Pipeline plugin failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Testing preprocessor plugin import...")
    from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin
    print("✓ Preprocessor plugin imported")
    
    preprocessor = PreprocessorPlugin()
    print("✓ Preprocessor plugin instantiated")
    
except Exception as e:
    print(f"✗ Preprocessor plugin failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Testing plugin loader...")
    from app.plugin_loader import load_plugin
    
    pipeline_class, _ = load_plugin('pipeline.plugins', 'phase2_6_pipeline')
    print("✓ Pipeline plugin loaded via plugin loader")
    
    preprocessor_class, _ = load_plugin('preprocessor.plugins', 'phase2_6_preprocessor')
    print("✓ Preprocessor plugin loaded via plugin loader")
    
except Exception as e:
    print(f"✗ Plugin loader failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")
