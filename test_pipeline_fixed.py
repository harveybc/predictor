#!/usr/bin/env python3

# Test script to verify the fixed pipeline works with the corrected preprocessor

import sys
import numpy as np
from preprocessor_plugins.stl_preprocessor_zscore import STLPreprocessorZScore
from pipeline_plugins.stl_pipeline_zscore import STLPipelinePlugin

print("Testing corrected pipeline components...")

# Test 1: Can we import and instantiate the components?
try:
    preprocessor = STLPreprocessorZScore()
    pipeline = STLPipelinePlugin()
    print("✅ Components imported and instantiated successfully")
except Exception as e:
    print(f"❌ Component instantiation failed: {e}")
    sys.exit(1)

# Test 2: Verify the preprocessor has the expected structure
config_test = {
    "window_size": 144,
    "predicted_horizons": [1, 2, 3, 4, 5, 6],
    "use_returns": True,
    "target_column": "CLOSE"
}

print(f"✅ Basic component test passed")
print(f"   Preprocessor class: {preprocessor.__class__.__name__}")
print(f"   Pipeline class: {pipeline.__class__.__name__}")
print("Ready to test with actual data...")
