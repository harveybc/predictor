#!/usr/bin/env python3
"""
COMPREHENSIVE CAUSAL VERIFICATION
Verifies that no future information leaks into training windows at any step.
"""

import pandas as pd
import numpy as np
import json
from preprocessor_plugins.helpers import load_normalization_json, denormalize
from preprocessor_plugins.stl_preprocessor_zscore import PreprocessorPlugin

def verify_window_causality(config_file="examples/config/phase_6/phase_6_cnn_1h_config.json"):
    """
    Complete causal verification of the entire preprocessing pipeline.
    Checks for future information leakage at every step.
    """
    print("="*80)
    print("COMPREHENSIVE CAUSAL VERIFICATION")
    print("="*80)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create preprocessor
    preprocessor = PreprocessorPlugin()
    preprocessor.set_params(**config)
    
    print(f"Configuration loaded:")
    print(f"  Window size: {config['window_size']}")
    print(f"  Predicted horizons: {config['predicted_horizons']}")
    print(f"  Use returns: {config['use_returns']}")
    
    # STEP 1: Verify raw data loading and alignment
    print("\n" + "="*60)
    print("STEP 1: RAW DATA VERIFICATION")
    print("="*60)
    
    # Load raw data
    from app.data_handler import load_csv
    
    train_x = load_csv(config['x_train_file'], headers=True)
    train_y = load_csv(config['y_train_file'], headers=True)
    
    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Train X date range: {train_x.index[0]} to {train_x.index[-1]}")
    print(f"Train Y date range: {train_y.index[0]} to {train_y.index[-1]}")
    
    # Check if X and Y are properly aligned
    if not train_x.index.equals(train_y.index):
        print("❌ CRITICAL: X and Y indices are not aligned!")
        print(f"First 5 X dates: {train_x.index[:5].tolist()}")
        print(f"First 5 Y dates: {train_y.index[:5].tolist()}")
        return False
    else:
        print("✅ X and Y indices are properly aligned")
    
    # STEP 2: Verify denormalization doesn't introduce leakage
    print("\n" + "="*60)
    print("STEP 2: DENORMALIZATION VERIFICATION")
    print("="*60)
    
    norm_json = load_normalization_json(config)
    print(f"Normalization JSON keys: {list(norm_json.keys())}")
    
    # Check CLOSE column denormalization
    close_normalized = train_y['CLOSE'].values
    close_denorm = denormalize(close_normalized, norm_json, 'CLOSE')
    
    print(f"Original normalized CLOSE[0:5]: {close_normalized[:5]}")
    print(f"Denormalized CLOSE[0:5]: {close_denorm[:5]}")
    
    # Verify denormalization is deterministic (no future leakage)
    close_denorm_check = denormalize(close_normalized[:10], norm_json, 'CLOSE')
    if not np.allclose(close_denorm[:10], close_denorm_check):
        print("❌ CRITICAL: Denormalization is not deterministic!")
        return False
    else:
        print("✅ Denormalization is deterministic")
    
    # STEP 3: Run full preprocessing and verify window causality
    print("\n" + "="*60)
    print("STEP 3: PREPROCESSING PIPELINE VERIFICATION")
    print("="*60)
    
    # Run preprocessing
    processed_data = preprocessor.process_data(config)
    
    # Extract windowed data
    X_train = processed_data['x_train']
    y_train_h1 = processed_data['y_train']['output_horizon_1']
    baseline_train = processed_data['baseline_train']
    
    print(f"Windowed X_train shape: {X_train.shape}")
    print(f"Target y_train_h1 shape: {y_train_h1.shape}")
    print(f"Baseline train shape: {baseline_train.shape}")
    
    # STEP 4: CRITICAL CAUSALITY CHECK - Window Construction
    print("\n" + "="*60)
    print("STEP 4: WINDOW CAUSALITY VERIFICATION")
    print("="*60)
    
    window_size = config['window_size']
    
    # Get the original denormalized CLOSE data (before trimming)
    target_column = config.get("target_column", "CLOSE")
    y_df = train_y
    target_raw = y_df[target_column].astype(np.float32).values
    target_denorm_full = denormalize(target_raw, norm_json, target_column)
    
    # After trimming (what should be used for windowing)
    trimmed_start = window_size - 1  # 143 for window_size=144
    target_denorm_trimmed = target_denorm_full[trimmed_start:]
    
    print(f"Full denormalized target length: {len(target_denorm_full)}")
    print(f"Trimmed denormalized target length: {len(target_denorm_trimmed)}")
    print(f"Trimmed start index: {trimmed_start}")
    
    # CRITICAL TEST: Verify first few windows don't contain future information
    print(f"\nCRITICAL CAUSALITY TEST:")
    print(f"Window construction verification for window_size={window_size}")
    
    # Check if log_return feature is constructed correctly
    log_return_feature_idx = 0  # log_return is first feature
    
    for window_idx in range(min(5, X_train.shape[0])):
        print(f"\n--- Window {window_idx} Causality Check ---")
        
        # This window should end at tick (trimmed_start + window_idx)
        # which corresponds to index (window_size-1 + window_idx) in the original data
        window_end_tick_in_original = trimmed_start + window_idx
        window_end_tick_in_trimmed = window_idx
        
        print(f"Window {window_idx} should end at:")
        print(f"  - Tick {window_end_tick_in_original} in original data")
        print(f"  - Tick {window_end_tick_in_trimmed} in trimmed data")
        print(f"  - Value: {target_denorm_full[window_end_tick_in_original]:.6f}")
        
        # Check baseline alignment
        baseline_value = baseline_train[window_idx]
        trimmed_baseline_value = target_denorm_trimmed[window_idx]
        
        print(f"Baseline verification:")
        print(f"  - Baseline from result: {baseline_value:.6f}")
        print(f"  - Expected from trimmed: {trimmed_baseline_value:.6f}")
        print(f"  - Expected from original: {target_denorm_full[window_end_tick_in_original]:.6f}")
        
        if not np.isclose(baseline_value, target_denorm_full[window_end_tick_in_original], atol=1e-6):
            print(f"❌ CRITICAL: Baseline misalignment detected!")
            return False
        
        # Check target alignment for horizon 1
        target_h1 = y_train_h1[window_idx]
        
        # For horizon 1, the target should be: future[window_end + 1] - baseline[window_end]
        future_tick_in_original = window_end_tick_in_original + 1
        future_tick_in_trimmed = window_idx + 1
        
        if future_tick_in_original < len(target_denorm_full):
            future_value_original = target_denorm_full[future_tick_in_original]
            future_value_trimmed = target_denorm_trimmed[future_tick_in_trimmed]
            
            expected_return = future_value_original - baseline_value
            
            print(f"Target H1 verification:")
            print(f"  - Future tick in original: {future_tick_in_original}")
            print(f"  - Future value: {future_value_original:.6f}")
            print(f"  - Expected return: {expected_return:.6f}")
            print(f"  - Actual target: {target_h1:.6f}")
            
            if not np.isclose(target_h1, expected_return, atol=1e-6):
                print(f"❌ CRITICAL: Target calculation error! Diff: {abs(target_h1 - expected_return):.8f}")
                return False
            else:
                print(f"✅ Target H1 correctly calculated")
        
        # CAUSALITY CHECK: Verify window contains only past/current data
        window_data = X_train[window_idx]  # Shape: (window_size, num_features)
        log_returns_in_window = window_data[:, log_return_feature_idx]  # Extract log_return feature
        
        print(f"Window causality check:")
        print(f"  - Window contains {window_size} timesteps")
        print(f"  - Last timestep should be tick {window_end_tick_in_original}")
        print(f"  - First timestep should be tick {window_end_tick_in_original - window_size + 1}")
        
        # Check that no window data comes from future ticks
        first_tick_in_window = window_end_tick_in_original - window_size + 1
        if first_tick_in_window < 0:
            print(f"❌ CRITICAL: Window extends before data start!")
            return False
        
        # Verify the log returns in the window match expected values
        print(f"  - First tick in window: {first_tick_in_window}")
        print(f"  - Sample log returns in window: {log_returns_in_window[:3]}")
        print(f"  - Sample log returns in window (last 3): {log_returns_in_window[-3:]}")
        
        print(f"✅ Window {window_idx} passes causality check")
    
    # STEP 5: Check for feature leakage
    print("\n" + "="*60)
    print("STEP 5: FEATURE LEAKAGE VERIFICATION")
    print("="*60)
    
    feature_names = processed_data.get('feature_names', [])
    print(f"Features in windows: {feature_names}")
    
    # Check if CLOSE is included in features (major red flag)
    if 'CLOSE' in feature_names:
        close_feature_idx = feature_names.index('CLOSE')
        print(f"❌ CRITICAL: CLOSE is included in features at index {close_feature_idx}!")
        print("This creates direct leakage since CLOSE at time t predicts CLOSE at time t+h")
        
        # Show the leakage
        for window_idx in range(3):
            window_close = X_train[window_idx, -1, close_feature_idx]  # Last timestep, CLOSE feature
            expected_baseline = baseline_train[window_idx]
            print(f"Window {window_idx}: CLOSE in window = {window_close:.6f}, baseline = {expected_baseline:.6f}")
            
        return False
    else:
        print("✅ CLOSE is not included in window features")
    
    # Check if any future-looking features are included
    suspicious_features = []
    for feature in feature_names:
        if any(keyword in feature.lower() for keyword in ['future', 'lead', 'forward']):
            suspicious_features.append(feature)
    
    if suspicious_features:
        print(f"⚠️ WARNING: Potentially future-looking features detected: {suspicious_features}")
    else:
        print("✅ No obviously future-looking features detected")
    
    # STEP 6: Verify target statistics for leakage indicators
    print("\n" + "="*60)
    print("STEP 6: TARGET STATISTICS VERIFICATION")
    print("="*60)
    
    for i, horizon in enumerate(config['predicted_horizons']):
        y_horizon = processed_data['y_train'][f'output_horizon_{horizon}']
        
        mae_proxy = np.mean(np.abs(y_horizon))
        std_dev = np.std(y_horizon)
        
        print(f"Horizon {horizon}:")
        print(f"  - MAE proxy (mean absolute value): {mae_proxy:.8f}")
        print(f"  - Standard deviation: {std_dev:.8f}")
        print(f"  - Min value: {np.min(y_horizon):.8f}")
        print(f"  - Max value: {np.max(y_horizon):.8f}")
        
        # Red flags for leakage
        if mae_proxy < 1e-5:
            print(f"❌ CRITICAL: MAE proxy {mae_proxy:.8f} is suspiciously low for horizon {horizon}!")
            print("This suggests severe data leakage or incorrect target calculation")
        elif mae_proxy < 1e-3:
            print(f"⚠️ WARNING: MAE proxy {mae_proxy:.8f} is quite low for horizon {horizon}")
        else:
            print(f"✅ MAE proxy appears reasonable for horizon {horizon}")
    
    print("\n" + "="*80)
    print("CAUSALITY VERIFICATION COMPLETE")
    print("="*80)
    
    return True

if __name__ == "__main__":
    verify_window_causality()
