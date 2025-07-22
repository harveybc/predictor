#!/usr/bin/env python3
"""
Test the data leakage fix by running the corrected target calculation.
"""

import sys
import json
import numpy as np

# Add project path
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

def test_leakage_fix():
    """Test that the data leakage fix works correctly"""
    print("="*60)
    print("TESTING DATA LEAKAGE FIX")
    print("="*60)
    
    # Load hourly config
    config_path = 'examples/config/phase_2/phase_2_6_cnn_1h_config.json'
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Hourly config not found, using minimal config")
        config = {
            "x_train_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
            "x_validation_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv", 
            "x_test_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
            "y_train_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
            "y_validation_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
            "y_test_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
            "target_column": "CLOSE",
            "window_size": 144,
            "predicted_horizons": [1, 2, 3, 4, 5, 6],
            "use_returns": True,
            "normalize_features": True,
            "use_normalization_json": "/home/harveybc/Documents/GitHub/feature-eng/output_config.json",
            "expected_feature_count": 55,
            "use_stl": True,
            "use_multi_tapper": True,
            "use_wavelets": False
        }
    
    # Use smaller dataset for testing
    config.update({
        "max_steps_train": 1000,
        "max_steps_val": 1000, 
        "max_steps_test": 1000,
    })
    
    print(f"Testing horizons: {config['predicted_horizons']}")
    print(f"Window size: {config['window_size']}")
    
    # Import and run preprocessor
    from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin as Phase26Preprocessor
    
    print("\n--- Running Phase2_6 Preprocessor (With Leakage Fix) ---")
    preprocessor = Phase26Preprocessor()
    result = preprocessor.process_data(config)
    
    # Extract first few samples to verify no leakage
    X_train = result["x_train"]
    y_train_list = result["y_train"]
    
    print(f"\n--- Verifying No Data Leakage ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"Number of horizons: {len(y_train_list)}")
    
    # For a simple test, let's check that targets are reasonable
    for i, horizon in enumerate(config['predicted_horizons']):
        targets = y_train_list[i]
        print(f"\nHorizon {horizon}:")
        print(f"  Target shape: {targets.shape}")
        print(f"  Target mean: {np.mean(targets):.6f}")
        print(f"  Target std: {np.std(targets):.6f}")
        print(f"  Target range: [{np.min(targets):.6f}, {np.max(targets):.6f}]")
        
        # Check variance - should be reasonable for financial returns
        if np.std(targets) < 1e-6:
            print(f"  🚨 WARNING: Very low variance suggests possible issues")
        else:
            print(f"  ✅ Good: Reasonable target variance")
    
    # Test with a simple linear regression to see if performance is realistic
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    print(f"\n--- Testing Model Performance (Should be Realistic Now) ---")
    
    # Use first horizon for testing
    X_test_flat = X_train.reshape(X_train.shape[0], -1)
    y_test = y_train_list[0]  # First horizon
    
    # Train simple model
    model = LinearRegression()
    model.fit(X_test_flat, y_test)
    y_pred = model.predict(X_test_flat)
    
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression R²: {r2:.6f}")
    
    if r2 > 0.99:
        print("🚨 WARNING: R² still suspiciously high (>0.99) - possible remaining leakage")
    elif r2 > 0.1:
        print("⚠️  CAUTION: R² moderately high - check for subtle leakage")
    else:
        print("✅ GOOD: R² looks realistic for financial prediction")
    
    return r2

if __name__ == "__main__":
    r2 = test_leakage_fix()
    print(f"\n" + "="*60)
    print(f"FINAL R² SCORE: {r2:.6f}")
    print("="*60)
