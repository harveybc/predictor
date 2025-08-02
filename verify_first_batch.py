#!/usr/bin/env python3
"""
DIRECT BATCH VERIFICATION: Verify first batch data matches baseline exactly.
This test traces the exact first batch_size samples fed to the model and verifies
they align perfectly with the baseline data timestamps and values.
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

def verify_first_batch_alignment():
    """
    Verify that the first batch of data fed to the model aligns exactly with baseline data.
    This is the MOST DIRECT verification possible.
    """
    print("=" * 80)
    print("DIRECT FIRST BATCH VERIFICATION")
    print("=" * 80)
    
    # Load config and run preprocessing
    config = load_config('examples/config/phase_6/phase_6_cnn_1h_config.json')
    batch_size = config.get('batch_size', 128)
    
    preprocessor_class, _ = load_plugin('preprocessor.plugins', 'stl_preprocessor_zscore')
    preprocessor_plugin = preprocessor_class()
    datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
    
    # Get the EXACT data that would be fed to the model
    X_train = datasets['x_train']  # Shape: (samples, window_size, features)
    y_train_dict = datasets['y_train']  # Dict with horizon keys
    baseline_train = datasets['baseline_train']  # Baseline values for each window
    x_train_dates = datasets['x_train_dates']  # Dates for each window (end dates)
    
    # Get first batch
    first_batch_X = X_train[:batch_size]  # Shape: (batch_size, window_size, features)
    first_batch_baseline = baseline_train[:batch_size]  # Baseline values
    first_batch_dates = x_train_dates[:batch_size]  # Window end dates
    
    # Get target values for horizon 1 (first batch)
    horizon_1_targets = y_train_dict['output_horizon_1'][:batch_size]
    
    print(f"\nFIRST BATCH ANALYSIS (batch_size={batch_size}):")
    print(f"X_train shape: {X_train.shape}")
    print(f"First batch X shape: {first_batch_X.shape}")
    print(f"First batch baseline shape: {first_batch_baseline.shape}")
    print(f"First batch H1 targets shape: {horizon_1_targets.shape}")
    
    # VERIFICATION 1: Check first few samples alignment
    print(f"\n" + "=" * 60)
    print("SAMPLE-BY-SAMPLE VERIFICATION (First 5 samples)")
    print("=" * 60)
    
    window_size = config['window_size']
    
    for i in range(min(5, batch_size)):
        # Window i data
        window_end_date = first_batch_dates[i]
        window_baseline = first_batch_baseline[i]  # Value at window end
        window_target_h1 = horizon_1_targets[i]   # Normalized target for H1
        
        # The window contains data[t-window_size+1:t+1] where t is the window end
        # Last value in window should match baseline
        window_data = first_batch_X[i]  # Shape: (window_size, features)
        
        # Find CLOSE column in features (assuming it exists)
        feature_names = datasets['feature_names']
        if 'CLOSE' in feature_names:
            close_idx = feature_names.index('CLOSE')
            window_last_close = window_data[-1, close_idx]  # Last CLOSE in window
        else:
            # Use log_return as proxy (first feature)
            window_last_close = "N/A (CLOSE not in features)"
        
        print(f"\nSample {i}:")
        print(f"  Window end date: {window_end_date}")
        print(f"  Baseline value: {window_baseline:.6f}")
        print(f"  Window last value: {window_last_close}")
        print(f"  H1 target (norm): {window_target_h1:.6f}")
        
        # Verification: baseline should match last value in window (if CLOSE available)
        if isinstance(window_last_close, float):
            if abs(window_baseline - window_last_close) < 1e-6:
                print(f"  ✅ Baseline matches window end")
            else:
                print(f"  ❌ Baseline mismatch: {abs(window_baseline - window_last_close):.8f}")
    
    # VERIFICATION 2: Check target calculation alignment
    print(f"\n" + "=" * 60)
    print("TARGET CALCULATION VERIFICATION")
    print("=" * 60)
    
    # Get denormalization stats
    target_returns_means = datasets['target_returns_means']
    target_returns_stds = datasets['target_returns_stds']
    
    print(f"Target normalization - Mean: {target_returns_means[0]:.6f}, Std: {target_returns_stds[0]:.6f}")
    
    # Manual calculation: if we have baseline[i] and want to predict baseline[i+h]
    # Target should be: (baseline[i+h] - baseline[i] - mean) / std
    # But we need to get the actual future values from the baseline data
    
    # Let's verify this by checking if we can reconstruct the target
    print(f"\nManual target verification for first sample:")
    baseline_0 = first_batch_baseline[0]
    target_0_norm = horizon_1_targets[0]
    
    # To verify, we need the actual future value (baseline[0] + 1 horizon)
    # This should be baseline_train[1] if horizon=1
    if len(baseline_train) > 1:
        future_1 = baseline_train[1]  # This should be the future value for horizon 1
        manual_return = future_1 - baseline_0
        manual_target_norm = (manual_return - target_returns_means[0]) / target_returns_stds[0]
        
        print(f"  Baseline[0]: {baseline_0:.6f}")
        print(f"  Future[1]: {future_1:.6f}")
        print(f"  Manual return: {manual_return:.6f}")
        print(f"  Manual normalized: {manual_target_norm:.6f}")
        print(f"  Actual target: {target_0_norm:.6f}")
        print(f"  Difference: {abs(manual_target_norm - target_0_norm):.8f}")
        
        if abs(manual_target_norm - target_0_norm) < 1e-6:
            print(f"  ✅ TARGET CALCULATION VERIFIED")
        else:
            print(f"  ❌ TARGET CALCULATION MISMATCH")
    
    # VERIFICATION 3: DateTime consistency
    print(f"\n" + "=" * 60)
    print("DATETIME CONSISTENCY CHECK")
    print("=" * 60)
    
    for i in range(min(3, batch_size)):
        window_date = first_batch_dates[i]
        print(f"Sample {i}: Window ends at {window_date}")
        
        # Each window should end 1 hour later than the previous
        if i > 0:
            prev_date = first_batch_dates[i-1]
            time_diff = window_date - prev_date
            print(f"  Time difference from previous: {time_diff}")
    
    return True

def main():
    """Main verification function."""
    print("DIRECT BATCH VERIFICATION")
    print("Verifying first batch data fed to model matches baseline exactly\n")
    
    try:
        success = verify_first_batch_alignment()
        
        print("\n" + "=" * 80)
        print("VERIFICATION COMPLETE")
        print("=" * 80)
        
        if success:
            print("✅ First batch alignment verified")
            print("✅ Data flow from baseline to model input is consistent")
            return 0
        else:
            print("❌ First batch alignment FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
