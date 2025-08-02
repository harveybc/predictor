#!/usr/bin/env python3
"""
Quick test to verify data alignment after the cardinality fix.
"""

import sys
import os
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

from app.plugin_loader import load_plugin

def test_data_alignment():
    """Test that X and Y data have matching cardinality."""
    print("TESTING DATA ALIGNMENT")
    print("=" * 50)
    
    # Load configuration
    config = {
        "x_train_file": "examples/data/phase_6/normalized_d4.csv",
        "y_train_file": "examples/data/phase_6/normalized_d4.csv",
        "x_validation_file": "examples/data/phase_6/normalized_d5.csv",
        "y_validation_file": "examples/data/phase_6/normalized_d5.csv",
        "x_test_file": "examples/data/phase_6/normalized_d6.csv",
        "y_test_file": "examples/data/phase_6/normalized_d6.csv",
        "use_normalization_json": "examples/data/phase_6/normalization_config_b.json",
        "target_column": "CLOSE",
        "window_size": 144,
        "predicted_horizons": [1, 2, 3, 4, 5, 6],
        "use_returns": True,
        "normalize_features": True,
        "max_steps_train": 25200,
        "max_steps_test": 6300
    }
    
    # Load preprocessor
    try:
        preprocessor_class, _ = load_plugin("preprocessor.plugins", "stl_preprocessor_zscore")
        preprocessor_plugin = preprocessor_class()
        print("✅ Preprocessor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load preprocessor: {e}")
        return False
    
    # Run preprocessing
    try:
        datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
        print("✅ Preprocessing completed successfully")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False
    
    # Check data shapes
    print("\nDATA SHAPES:")
    print(f"X_train: {datasets['x_train'].shape}")
    print(f"X_val: {datasets['x_val'].shape}")
    print(f"X_test: {datasets['x_test'].shape}")
    
    # Check Y shapes for all horizons
    horizons = config['predicted_horizons']
    print(f"\nY SHAPES for horizons {horizons}:")
    
    y_train_shapes = []
    y_val_shapes = []
    y_test_shapes = []
    
    for h in horizons:
        key = f"output_horizon_{h}"
        train_shape = datasets['y_train'][key].shape if key in datasets['y_train'] else "MISSING"
        val_shape = datasets['y_val'][key].shape if key in datasets['y_val'] else "MISSING"
        test_shape = datasets['y_test'][key].shape if key in datasets['y_test'] else "MISSING"
        
        y_train_shapes.append(train_shape[0] if train_shape != "MISSING" else 0)
        y_val_shapes.append(val_shape[0] if val_shape != "MISSING" else 0)
        y_test_shapes.append(test_shape[0] if test_shape != "MISSING" else 0)
        
        print(f"  H{h}: Train={train_shape}, Val={val_shape}, Test={test_shape}")
    
    # Check for alignment
    print("\nALIGNMENT CHECK:")
    x_train_samples = datasets['x_train'].shape[0]
    x_val_samples = datasets['x_val'].shape[0]
    x_test_samples = datasets['x_test'].shape[0]
    
    # Check if all Y shapes are consistent
    train_consistent = len(set(y_train_shapes)) == 1
    val_consistent = len(set(y_val_shapes)) == 1
    test_consistent = len(set(y_test_shapes)) == 1
    
    print(f"Train - X: {x_train_samples}, Y: {y_train_shapes}, Consistent: {train_consistent}")
    print(f"Val   - X: {x_val_samples}, Y: {y_val_shapes}, Consistent: {val_consistent}")
    print(f"Test  - X: {x_test_samples}, Y: {y_test_shapes}, Consistent: {test_consistent}")
    
    # Check X-Y alignment
    train_aligned = x_train_samples == y_train_shapes[0] if y_train_shapes else False
    val_aligned = x_val_samples == y_val_shapes[0] if y_val_shapes else False
    test_aligned = x_test_samples == y_test_shapes[0] if y_test_shapes else False
    
    print(f"\nX-Y ALIGNMENT:")
    print(f"Train aligned: {train_aligned}")
    print(f"Val aligned: {val_aligned}")
    print(f"Test aligned: {test_aligned}")
    
    all_good = (train_consistent and val_consistent and test_consistent and 
                train_aligned and val_aligned and test_aligned)
    
    if all_good:
        print("\n✅ ALL DATA PROPERLY ALIGNED - CARDINALITY ISSUE FIXED!")
        return True
    else:
        print("\n❌ DATA ALIGNMENT ISSUES REMAIN")
        return False

if __name__ == "__main__":
    test_data_alignment()
