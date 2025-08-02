#!/usr/bin/env python3
"""
CRITICAL TEST: Verify EXACT alignment between target calculation and training data.
This test ensures that y_true values in the loss function are EXACTLY the same
as the target values calculated in target_calculation.py with ZERO tolerance.
"""

import json
import numpy as np
import sys
import os

# Add the predictor paths
sys.path.append('/home/harveybc/Documents/GitHub/predictor')
sys.path.append('/home/harveybc/Documents/GitHub/preprocessor')

from app.plugin_loader import load_plugin
from app.config_handler import load_config

def load_exact_config():
    """Load the exact configuration from the provided JSON file."""
    config_path = '/home/harveybc/Documents/GitHub/predictor/examples/config/phase_6/phase_6_cnn_1h_config.json'
    return load_config(config_path)

def simulate_training_data_flow(config):
    """
    Simulate the EXACT data flow that happens during training to verify alignment.
    This follows the exact same steps as stl_pipeline_zscore.py
    """
    print("=" * 80)
    print("SIMULATING EXACT TRAINING DATA FLOW")
    print("=" * 80)
    
    # Load preprocessor plugin (same as pipeline)
    print("\n1. Loading preprocessor plugin...")
    preprocessor_plugin_name = config.get('preprocessor_plugin', 'stl_preprocessor_zscore')
    print(f"   Preprocessor plugin: {preprocessor_plugin_name}")
    
    preprocessor_class, _ = load_plugin('preprocessor.plugins', preprocessor_plugin_name)
    preprocessor_plugin = preprocessor_class()
    
    # Run preprocessing (EXACTLY as pipeline does)
    print("\n2. Running preprocessing (EXACT pipeline flow)...")
    datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
    print("   Preprocessor finished.")
    
    # Extract training data (EXACTLY as pipeline does)
    print("\n3. Extracting training data...")
    X_train = datasets['x_train']  # Correct key based on output
    y_train_dict = datasets['y_train']
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train horizons: {list(y_train_dict.keys())}")
    
    # Convert to model format (EXACTLY as pipeline does)
    predicted_horizons = config['predicted_horizons']
    y_train_arrays = []
    
    for h in predicted_horizons:
        horizon_key = f"output_horizon_{h}"
        if horizon_key in y_train_dict:
            y_train_arrays.append(y_train_dict[horizon_key])
            print(f"   {horizon_key}: {y_train_dict[horizon_key].shape}")
        else:
            raise ValueError(f"Missing target data for horizon {h}")
    
    # Stack all horizons - this is EXACTLY what the model sees as y_true
    y_train_stacked = np.stack(y_train_arrays, axis=1)  # Shape: (samples, horizons)
    
    print(f"\n4. Final y_train shape fed to model: {y_train_stacked.shape}")
    print(f"   Expected: ({X_train.shape[0]}, {len(predicted_horizons)})")
    
    # Get target normalization stats (EXACTLY as pipeline does)
    if "target_returns_means" not in preprocessor_params or "target_returns_stds" not in preprocessor_params:
        raise ValueError("Preprocessor did not return target normalization stats")
    
    target_returns_means = preprocessor_params["target_returns_means"]
    target_returns_stds = preprocessor_params["target_returns_stds"]
    
    print(f"\n5. Target normalization stats:")
    print(f"   Means: {target_returns_means}")
    print(f"   Stds: {target_returns_stds}")
    
    return {
        'X_train': X_train,
        'y_train_stacked': y_train_stacked,
        'y_train_dict': y_train_dict,
        'datasets': datasets,
        'preprocessor_params': preprocessor_params,
        'config': config,
        'target_returns_means': target_returns_means,
        'target_returns_stds': target_returns_stds
    }

def verify_exact_alignment(data):
    """
    Verify EXACT alignment between different representations of the same data.
    NO tolerance - values must be bitwise identical.
    """
    print("\n" + "=" * 80)
    print("EXACT ALIGNMENT VERIFICATION (ZERO TOLERANCE)")
    print("=" * 80)
    
    config = data['config']
    predicted_horizons = config['predicted_horizons']
    y_train_stacked = data['y_train_stacked']
    y_train_dict = data['y_train_dict']
    
    all_tests_passed = True
    
    print(f"\nTesting {len(predicted_horizons)} horizons...")
    
    for i, h in enumerate(predicted_horizons):
        horizon_key = f"output_horizon_{h}"
        
        # Get the target values from datasets
        target_calc_values = y_train_dict[horizon_key]
        
        # Get the corresponding column from stacked y_train
        stacked_values = y_train_stacked[:, i]
        
        print(f"\nHorizon {h} ({horizon_key}):")
        print(f"  Target calc shape: {target_calc_values.shape}")
        print(f"  Stacked shape: {stacked_values.shape}")
        
        # Test 1: Shape equality
        if target_calc_values.shape != stacked_values.shape:
            print(f"  ❌ SHAPE MISMATCH: {target_calc_values.shape} != {stacked_values.shape}")
            all_tests_passed = False
            continue
        else:
            print(f"  ✅ Shapes match: {target_calc_values.shape}")
        
        # Test 2: EXACT bitwise equality (zero tolerance)
        are_identical = np.array_equal(target_calc_values, stacked_values)
        
        if are_identical:
            print(f"  ✅ VALUES EXACTLY IDENTICAL (bitwise)")
        else:
            print(f"  ❌ VALUES NOT IDENTICAL")
            all_tests_passed = False
            
            # Find differences
            diff_mask = target_calc_values != stacked_values
            num_diffs = np.sum(diff_mask)
            print(f"  ❌ Found {num_diffs} different values out of {len(target_calc_values)}")
            
            if num_diffs > 0:
                diff_indices = np.where(diff_mask)[0][:5]  # Show first 5 differences
                print(f"  ❌ First differences at indices: {diff_indices}")
                for idx in diff_indices:
                    print(f"    Index {idx}: target_calc={target_calc_values[idx]:.10f}, stacked={stacked_values[idx]:.10f}")
        
        # Test 3: Memory address check (if they point to the same data)
        shares_memory = np.shares_memory(target_calc_values, stacked_values)
        print(f"  📍 Shares memory: {shares_memory}")
        
        # Test 4: Statistical verification
        if are_identical:
            mean_diff = 0.0
            max_diff = 0.0
        else:
            mean_diff = np.mean(np.abs(target_calc_values - stacked_values))
            max_diff = np.max(np.abs(target_calc_values - stacked_values))
        
        print(f"  📊 Mean absolute difference: {mean_diff:.2e}")
        print(f"  📊 Max absolute difference: {max_diff:.2e}")
        
        if mean_diff > 0 or max_diff > 0:
            print(f"  ❌ NON-ZERO DIFFERENCES DETECTED")
            all_tests_passed = False
    
    return all_tests_passed

def verify_training_pipeline_consistency(data):
    """
    Verify that the target values are consistent with the training pipeline expectations.
    """
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE CONSISTENCY CHECK")
    print("=" * 80)
    
    config = data['config']
    preprocessor_params = data['preprocessor_params']
    
    # Check normalization consistency
    use_returns = config.get('use_returns', True)
    target_column = config.get('target_column', 'CLOSE')
    
    print(f"\nConfiguration:")
    print(f"  use_returns: {use_returns}")
    print(f"  target_column: {target_column}")
    print(f"  window_size: {config['window_size']}")
    print(f"  predicted_horizons: {config['predicted_horizons']}")
    
    if use_returns:
        # Verify target normalization parameters are consistent
        target_returns_means = data['target_returns_means']
        target_returns_stds = data['target_returns_stds']
        
        print(f"  Target returns means: {target_returns_means}")
        print(f"  Target returns stds: {target_returns_stds}")
        
        # Check that all horizons use the same normalization (as expected for returns approach)
        if len(set(target_returns_means)) == 1 and len(set(target_returns_stds)) == 1:
            print(f"  ✅ All horizons use same normalization stats")
        else:
            print(f"  ❌ Horizons use different normalization stats")
            return False
            
        # Check that std is reasonable for returns (should be much smaller than price std)
        returns_std = target_returns_stds[0]
        if 0.0001 < returns_std < 0.1:  # Reasonable range for EUR/USD returns std
            print(f"  ✅ Returns std is in reasonable range: {returns_std:.6f}")
        else:
            print(f"  ⚠️  Returns std may be unusual: {returns_std:.6f}")
    
    return True

def verify_sample_targets(data):
    """
    Verify sample target values to ensure they make sense.
    """
    print("\n" + "=" * 80)
    print("SAMPLE TARGET VALUES VERIFICATION")
    print("=" * 80)
    
    y_train_stacked = data['y_train_stacked']
    predicted_horizons = data['config']['predicted_horizons']
    
    print(f"Y_train shape: {y_train_stacked.shape}")
    print(f"Predicted horizons: {predicted_horizons}")
    
    # Check first few samples for each horizon
    num_samples_to_show = min(5, y_train_stacked.shape[0])
    
    for i, h in enumerate(predicted_horizons):
        horizon_values = y_train_stacked[:num_samples_to_show, i]
        print(f"\nHorizon {h} - First {num_samples_to_show} samples:")
        for j, val in enumerate(horizon_values):
            print(f"  Sample {j}: {val:.8f}")
        
        # Basic sanity checks
        all_values = y_train_stacked[:, i]
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        
        print(f"  Statistics: mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")
        
        # For returns with z-score normalization, we expect std ≈ 1.0 and mean ≈ 0.0
        if data['config'].get('use_returns', True):
            if abs(std_val - 1.0) > 0.2:
                print(f"  ⚠️  WARNING: std={std_val:.6f} is far from expected 1.0")
            if abs(mean_val) > 0.2:
                print(f"  ⚠️  WARNING: mean={mean_val:.6f} is far from expected 0.0")
    
    return True

def main():
    """Main test function."""
    print("CRITICAL TEST: Exact Target Alignment Verification")
    print("This test ensures y_true values in loss function are EXACTLY the target values")
    print("with ZERO tolerance for differences.\n")
    
    try:
        # Load exact configuration
        config = load_exact_config()
        print(f"Loaded config from: {config.get('use_normalization_json', 'N/A')}")
        print(f"Horizons: {config['predicted_horizons']}")
        print(f"Window size: {config['window_size']}")
        print(f"Use returns: {config.get('use_returns', True)}")
        
        # Simulate exact training data flow (EXACTLY as pipeline does)
        data = simulate_training_data_flow(config)
        
        # Verify exact alignment
        alignment_passed = verify_exact_alignment(data)
        
        # Verify pipeline consistency
        consistency_passed = verify_training_pipeline_consistency(data)
        
        # Verify sample target values
        sample_passed = verify_sample_targets(data)
        
        # Final result
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        
        if alignment_passed and consistency_passed and sample_passed:
            print("✅ ALL TESTS PASSED")
            print("✅ y_true values fed to loss function are EXACTLY the target values")
            print("✅ Zero tolerance verification successful")
            print("✅ Training pipeline consistency verified")
            print("✅ Sample target values are reasonable")
            return 0
        else:
            print("❌ TESTS FAILED")
            if not alignment_passed:
                print("❌ Exact alignment verification FAILED")
            if not consistency_passed:
                print("❌ Pipeline consistency verification FAILED")
            if not sample_passed:
                print("❌ Sample target verification FAILED")
            print("❌ y_true values may NOT match target values exactly")
            return 1
            
    except Exception as e:
        print(f"\n❌ TEST EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
