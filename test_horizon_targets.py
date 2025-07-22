#!/usr/bin/env python3
"""
Test script to verify that different horizons produce different target values
"""

import sys
import json
import numpy as np

# Add project path
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

def test_horizon_targets():
    """Test that different horizons produce meaningfully different targets"""
    print("="*60)
    print("TESTING HORIZON TARGET CALCULATION")
    print("="*60)
    
    # Load phase2_6 config
    config_path = 'examples/config/phase_2_daily/phase_2_6_cnn_1d_config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Use smaller dataset for testing
    config.update({
        "max_steps_train": 1000,
        "max_steps_val": 1000, 
        "max_steps_test": 1000,
        "window_size": 144,
        "predicted_horizons": [24, 48, 72, 96, 120, 144],  # Same as your config
        "use_returns": True
    })
    
    print(f"Testing horizons: {config['predicted_horizons']}")
    print(f"Window size: {config['window_size']}")
    print(f"Use returns: {config['use_returns']}")
    
    # Import and run preprocessor
    from preprocessor_plugins.phase2_6_preprocessor import PreprocessorPlugin as Phase26Preprocessor
    
    print("\n--- Running Phase2_6 Preprocessor ---")
    preprocessor = Phase26Preprocessor()
    result = preprocessor.process_data(config)
    
    # Extract targets
    y_train_list = result["y_train"]
    
    print(f"\n--- Analyzing Target Differences Between Horizons ---")
    
    # Get first 100 samples for analysis
    sample_size = min(100, len(y_train_list[0]))
    
    print(f"Analyzing first {sample_size} samples...")
    
    for i, horizon in enumerate(config['predicted_horizons']):
        targets = y_train_list[i][:sample_size]
        
        print(f"\nHorizon {horizon}:")
        print(f"  Shape: {targets.shape}")
        print(f"  Mean: {np.mean(targets):.6f}")
        print(f"  Std:  {np.std(targets):.6f}")
        print(f"  Min:  {np.min(targets):.6f}")
        print(f"  Max:  {np.max(targets):.6f}")
        print(f"  Sample values: {targets[:5]}")
    
    # Test correlation between horizons (should be high but not perfect)
    print(f"\n--- Correlation Analysis Between Horizons ---")
    correlations = {}
    
    for i in range(len(config['predicted_horizons'])):
        for j in range(i+1, len(config['predicted_horizons'])):
            h1, h2 = config['predicted_horizons'][i], config['predicted_horizons'][j]
            targets1 = y_train_list[i][:sample_size]
            targets2 = y_train_list[j][:sample_size]
            
            correlation = np.corrcoef(targets1, targets2)[0,1]
            correlations[f"H{h1}_vs_H{h2}"] = correlation
            print(f"  H{h1} vs H{h2}: {correlation:.4f}")
    
    # Check if targets are all identical (bug indicator)
    print(f"\n--- Checking for Identical Targets (Bug Detection) ---")
    all_identical = True
    reference_targets = y_train_list[0][:sample_size]
    
    for i in range(1, len(y_train_list)):
        horizon = config['predicted_horizons'][i]
        targets = y_train_list[i][:sample_size]
        
        # Check if targets are exactly the same
        are_identical = np.allclose(reference_targets, targets, atol=1e-8)
        print(f"  H{config['predicted_horizons'][0]} vs H{horizon}: {'IDENTICAL' if are_identical else 'DIFFERENT'}")
        
        if not are_identical:
            all_identical = False
            # Show difference statistics
            diff = np.abs(reference_targets - targets)
            print(f"    Max difference: {np.max(diff):.8f}")
            print(f"    Mean difference: {np.mean(diff):.8f}")
    
    # Final assessment
    print(f"\n--- ASSESSMENT ---")
    if all_identical:
        print("🚨 CRITICAL BUG: All horizons have IDENTICAL targets!")
        print("   This explains why all horizons have similar training errors.")
    else:
        print("✅ Good: Horizons have different targets.")
        
        # Check if differences are reasonable
        avg_correlation = np.mean(list(correlations.values()))
        print(f"   Average correlation between horizons: {avg_correlation:.4f}")
        
        if avg_correlation > 0.99:
            print("⚠️  WARNING: Correlations very high (>0.99) - targets may be too similar")
        elif avg_correlation > 0.8:
            print("✅ Normal: High but reasonable correlation between horizons")
        else:
            print("✅ Good: Reasonable correlation between horizons")
    
    return {
        "all_identical": all_identical,
        "correlations": correlations,
        "target_stats": {
            f"H{h}": {
                "mean": np.mean(y_train_list[i][:sample_size]),
                "std": np.std(y_train_list[i][:sample_size])
            } for i, h in enumerate(config['predicted_horizons'])
        }
    }

if __name__ == "__main__":
    results = test_horizon_targets()
    print(f"\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
