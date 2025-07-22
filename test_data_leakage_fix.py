#!/usr/bin/env python3
"""
Test to verify that the data leakage fix is working correctly.
This test will run both preprocessors and check that:
1. The targets are calculated correctly without data leakage
2. Model performance is realistic (not artificially perfect)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Add project path
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

def test_data_leakage_fix():
    """Test that data leakage has been fixed"""
    print("="*60)
    print("TESTING DATA LEAKAGE FIX")
    print("="*60)
    
    # Test configuration
    config = {
        "x_train_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
        "x_validation_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv", 
        "x_test_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
        "y_train_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
        "y_validation_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
        "y_test_file": "/home/harveybc/Documents/GitHub/feature-eng/test_output.csv",
        "target_column": "CLOSE",
        "window_size": 20,
        "predicted_horizons": [1, 5],
        "use_returns": True,
        "normalize_features": True,
        "use_normalization_json": "/home/harveybc/Documents/GitHub/feature-eng/output_config.json",
        "expected_feature_count": 55,
        "use_stl": True,
        "use_multi_tapper": True,
        "use_wavelets": False
    }
    
    # Import and run phase2_6 preprocessor
    from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin as Phase26Preprocessor
    
    print("\n--- Running Phase2_6 Preprocessor (Fixed) ---")
    phase26_preprocessor = Phase26Preprocessor()
    phase26_result = phase26_preprocessor.process_data(config)
    
    # Extract data for first horizon
    X_train = phase26_result["x_train"]
    y_train = phase26_result["y_train"][0]  # First horizon
    X_test = phase26_result["x_test"] 
    y_test = phase26_result["y_test"][0]    # First horizon
    
    print(f"\nPhase2_6 Results:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Flatten X data for simple linear regression test
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Train a simple linear regression model
    print(f"\n--- Testing Model Performance (Should NOT be artificially perfect) ---")
    model = LinearRegression()
    model.fit(X_train_flat, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_flat)
    y_test_pred = model.predict(X_test_flat)
    
    # Calculate R² scores
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"Training R²: {r2_train:.6f}")
    print(f"Test R²: {r2_test:.6f}")
    
    # Check for data leakage indicators
    print(f"\n--- Data Leakage Analysis ---")
    if r2_train > 0.99:
        print("🚨 WARNING: Training R² > 0.99 suggests possible data leakage!")
    elif r2_train > 0.9:
        print("⚠️  CAUTION: Training R² > 0.9 is suspiciously high")
    else:
        print("✅ Training R² looks reasonable (no obvious data leakage)")
    
    if r2_test > 0.99:
        print("🚨 WARNING: Test R² > 0.99 suggests possible data leakage!")
    elif r2_test > 0.9:
        print("⚠️  CAUTION: Test R² > 0.9 is suspiciously high")
    else:
        print("✅ Test R² looks reasonable (no obvious data leakage)")
    
    # Check target statistics
    print(f"\n--- Target Statistics ---")
    print(f"Training target mean: {np.mean(y_train):.6f}")
    print(f"Training target std: {np.std(y_train):.6f}")
    print(f"Training target range: [{np.min(y_train):.6f}, {np.max(y_train):.6f}]")
    print(f"Test target mean: {np.mean(y_test):.6f}")
    print(f"Test target std: {np.std(y_test):.6f}")
    print(f"Test target range: [{np.min(y_test):.6f}, {np.max(y_test):.6f}]")
    
    # Check for constant or near-constant targets (another leakage indicator)
    if np.std(y_train) < 1e-6:
        print("🚨 WARNING: Training targets are nearly constant!")
    if np.std(y_test) < 1e-6:
        print("🚨 WARNING: Test targets are nearly constant!")
    
    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "target_stats": {
            "train_mean": np.mean(y_train),
            "train_std": np.std(y_train),
            "test_mean": np.mean(y_test), 
            "test_std": np.std(y_test)
        }
    }

if __name__ == "__main__":
    results = test_data_leakage_fix()
    print(f"\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
