#!/usr/bin/env python3
"""
VERIFY ACTUAL TARGET CALCULATION WITH REAL DATA
Check if the target calculation is using wrong indices
"""
import json
import sys
import os

# Add paths for imports
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

from app.data_handler import load_csv
from preprocessor_plugins.helpers import load_normalization_json, denormalize

def verify_target_calculation_with_real_data():
    """
    Use actual data from the config to verify target calculation logic.
    """
    print("🔍 VERIFYING TARGET CALCULATION WITH REAL DATA")
    print("=" * 80)
    
    # Load configuration
    config_file = "examples/config/phase_6/phase_6_cnn_1h_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded config: {config_file}")
    print(f"Window size: {config['window_size']}")
    print(f"Horizons: {config['predicted_horizons']}")
    
    # Load a small sample of training data
    print("\nLoading sample training data...")
    train_file = config['x_train_file']
    y_train_file = config['y_train_file']
    
    # Load first 200 rows for analysis
    train_df = load_csv(train_file, max_rows=200, headers=True)
    y_train_df = load_csv(y_train_file, max_rows=200, headers=True)
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Train data index range: {train_df.index[0]} to {train_df.index[-1]}")
    
    # Load normalization config and denormalize close prices
    with open(config['use_normalization_json'], 'r') as f:
        norm_json = json.load(f)
    target_column = config['target_column']
    
    close_raw = y_train_df[target_column].astype('float32').values
    close_denorm = denormalize(close_raw, norm_json, target_column)
    
    print(f"\nClose prices (first 10): {close_denorm[:10]}")
    
    # Apply target trimming (same as target_calculation.py)
    window_size = config['window_size']
    trimmed_start = window_size - 1
    target_trimmed = close_denorm[trimmed_start:]
    
    print(f"\nTarget trimming:")
    print(f"  Original length: {len(close_denorm)}")
    print(f"  Trimmed start: {trimmed_start}")
    print(f"  Trimmed length: {len(target_trimmed)}")
    print(f"  Trimmed data (first 10): {target_trimmed[:10]}")
    
    # Check what the code THINKS the alignment should be
    print(f"\n🔍 ALIGNMENT ANALYSIS:")
    print(f"According to comments in target_calculation.py:")
    print(f"  'Window 0: ends at tick {window_size-1}'")
    print(f"  This means Window 0 baseline should be close_denorm[{window_size-1}] = {close_denorm[window_size-1]:.6f}")
    print(f"  target_trimmed[0] = {target_trimmed[0]:.6f}")
    
    if abs(close_denorm[window_size-1] - target_trimmed[0]) < 1e-10:
        print(f"  ✅ MATCH: target_trimmed[0] matches close_denorm[{window_size-1}]")
    else:
        print(f"  ❌ MISMATCH: target_trimmed[0] does NOT match close_denorm[{window_size-1}]")
        print(f"  🚨 POTENTIAL INDEXING ERROR!")
    
    # Check horizon predictions
    horizons = config['predicted_horizons'][:3]  # Check first 3 horizons
    
    for h in horizons:
        print(f"\n📋 HORIZON {h} CHECK:")
        
        # What should Window 0 predict for horizon h?
        expected_future_tick = (window_size - 1) + h
        if expected_future_tick < len(close_denorm):
            expected_future_value = close_denorm[expected_future_tick]
            print(f"  Window 0 should predict close_denorm[{expected_future_tick}] = {expected_future_value:.6f}")
        else:
            print(f"  Window 0 should predict close_denorm[{expected_future_tick}] = OUT OF BOUNDS")
            continue
        
        # What does the target calculation actually use?
        if h < len(target_trimmed):
            actual_future_value = target_trimmed[h]
            print(f"  target_trimmed[{h}] = {actual_future_value:.6f}")
            
            if abs(expected_future_value - actual_future_value) < 1e-10:
                print(f"  ✅ CORRECT: Horizon {h} target matches expected future value")
            else:
                print(f"  ❌ WRONG: Horizon {h} target does NOT match expected future value")
                print(f"  🚨 CRITICAL ERROR IN TARGET CALCULATION!")
        else:
            print(f"  target_trimmed[{h}] = OUT OF BOUNDS")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    verify_target_calculation_with_real_data()
