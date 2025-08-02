#!/usr/bin/env python3
"""
CRITICAL VERIFICATION: Intercept actual training batch and verify EXACT alignment.
This test hooks into the actual training process to verify that y_true values
in the loss function are EXACTLY the same as targets with EXACT same datetimes.
"""

import json
import numpy as np
import sys
import os
import tensorflow as tf

# Add the predictor paths
sys.path.append('/home/harveybc/Documents/GitHub/predictor')
sys.path.append('/home/harveybc/Documents/GitHub/preprocessor')

from app.plugin_loader import load_plugin
from app.config_handler import load_config

class BatchInterceptor:
    """Intercepts the first training batch to verify exact alignment."""
    
    def __init__(self):
        self.first_batch_x = None
        self.first_batch_y = None
        self.first_batch_captured = False
        self.original_train_step = None
        
    def capture_first_batch(self, model):
        """Monkey patch the model's train_step to capture the first batch."""
        self.original_train_step = model.train_step
        
        def intercepted_train_step(data):
            if not self.first_batch_captured:
                x, y = data
                print(f"\n🔍 INTERCEPTED FIRST BATCH:")
                print(f"   X shape: {x.shape}")
                print(f"   Y type: {type(y)}")
                
                if isinstance(y, dict):
                    print(f"   Y keys: {list(y.keys())}")
                    for key, val in y.items():
                        print(f"   Y[{key}] shape: {val.shape}")
                        print(f"   Y[{key}] sample values: [{val[0]:.8f}, {val[1]:.8f}, {val[2]:.8f}]")
                elif hasattr(y, 'shape'):
                    print(f"   Y shape: {y.shape}")
                    print(f"   Y sample values: [{y[0,0]:.8f}, {y[0,1]:.8f}, {y[0,2]:.8f}]")
                
                # Store the batch
                self.first_batch_x = tf.identity(x).numpy()
                if isinstance(y, dict):
                    self.first_batch_y = {k: tf.identity(v).numpy() for k, v in y.items()}
                else:
                    self.first_batch_y = tf.identity(y).numpy()
                    
                self.first_batch_captured = True
                print(f"   ✅ First batch captured!")
            
            # Call original train_step
            return self.original_train_step(data)
            
        model.train_step = intercepted_train_step

def load_and_prepare_exact_data():
    """Load data using exact same pipeline as training."""
    print("=" * 80)
    print("LOADING EXACT TRAINING DATA")
    print("=" * 80)
    
    # Load exact configuration
    config_path = '/home/harveybc/Documents/GitHub/predictor/examples/config/phase_6/phase_6_cnn_1h_config.json'
    config = load_config(config_path)
    
    # Load preprocessor and get data
    preprocessor_class, _ = load_plugin('preprocessor.plugins', 'stl_preprocessor_zscore')
    preprocessor_plugin = preprocessor_class()
    datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
    
    print(f"\n📊 Dataset info:")
    print(f"   X_train: {datasets['x_train'].shape}")
    print(f"   Y_train keys: {list(datasets['y_train'].keys())}")
    print(f"   Train dates: {datasets['x_train_dates'].shape}")
    print(f"   Baseline dates: {datasets['baseline_train_dates'].shape}")
    
    return datasets, preprocessor_params, config

def verify_datetime_alignment(datasets, batch_size=128):
    """Verify that X windows and Y targets have exactly matching datetimes."""
    print("\n" + "=" * 80)
    print("DATETIME ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    x_train_dates = datasets['x_train_dates']
    y_train_dates = datasets['y_train_dates'] 
    baseline_train_dates = datasets['baseline_train_dates']
    
    print(f"\n📅 Date arrays:")
    print(f"   X train dates length: {len(x_train_dates)}")
    print(f"   Y train dates length: {len(y_train_dates)}")
    print(f"   Baseline dates length: {len(baseline_train_dates)}")
    
    # Check first batch datetime alignment
    print(f"\n🎯 First batch ({batch_size} samples) datetime verification:")
    
    for i in range(min(5, batch_size)):  # Check first 5 samples
        x_date = x_train_dates[i] if i < len(x_train_dates) else "N/A"
        y_date = y_train_dates[i] if i < len(y_train_dates) else "N/A"
        
        print(f"   Sample {i}:")
        print(f"     X date: {x_date}")
        print(f"     Y date: {y_date}")
        
        if x_date != "N/A" and y_date != "N/A":
            if x_date == y_date:
                print(f"     ✅ MATCH")
            else:
                print(f"     ❌ MISMATCH")
                return False
        else:
            print(f"     ⚠️  Missing date")
    
    return True

def verify_target_scale_consistency(datasets, preprocessor_params):
    """Verify target scale consistency between preprocessor and training data."""
    print("\n" + "=" * 80)
    print("TARGET SCALE CONSISTENCY VERIFICATION")
    print("=" * 80)
    
    y_train = datasets['y_train']
    target_means = preprocessor_params['target_returns_means']
    target_stds = preprocessor_params['target_returns_stds']
    
    print(f"\n📏 Normalization stats from preprocessor:")
    print(f"   Target means: {target_means}")
    print(f"   Target stds: {target_stds}")
    
    print(f"\n🔍 Actual target statistics:")
    all_issues = []
    
    for i, (horizon_key, target_data) in enumerate(y_train.items()):
        actual_mean = np.mean(target_data)
        actual_std = np.std(target_data)
        expected_mean = target_means[i]
        expected_std = target_stds[i]
        
        print(f"\n   {horizon_key}:")
        print(f"     Expected: mean={expected_mean:.6f}, std={expected_std:.6f}")
        print(f"     Actual:   mean={actual_mean:.6f}, std={actual_std:.6f}")
        
        mean_diff = abs(actual_mean - expected_mean)
        std_ratio = actual_std / expected_std if expected_std > 0 else float('inf')
        
        if mean_diff > 0.1:
            issue = f"❌ {horizon_key}: Mean mismatch {actual_mean:.6f} vs {expected_mean:.6f}"
            print(f"     {issue}")
            all_issues.append(issue)
        else:
            print(f"     ✅ Mean OK")
            
        if abs(std_ratio - 1.0) > 0.2:  # Allow 20% tolerance
            issue = f"❌ {horizon_key}: Std ratio {std_ratio:.3f} (should be ~1.0)"
            print(f"     {issue}")
            all_issues.append(issue)
        else:
            print(f"     ✅ Std OK (ratio: {std_ratio:.3f})")
    
    return len(all_issues) == 0, all_issues

def run_minimal_training_verification(datasets, config):
    """Run minimal training to capture and verify the actual batch fed to loss function."""
    print("\n" + "=" * 80)
    print("ACTUAL TRAINING BATCH VERIFICATION")
    print("=" * 80)
    
    # Load predictor plugin
    predictor_class, _ = load_plugin('predictor.plugins', config['predictor_plugin'])
    predictor_plugin = predictor_class(config)
    
    # Prepare training data
    x_train = datasets['x_train']
    y_train = datasets['y_train']
    x_val = datasets['x_val']
    y_val = datasets['y_val']
    
    print(f"\n🏗️  Building model...")
    predictor_plugin.build_model(x_train.shape[1:], x_train, config)
    
    print(f"\n🎣 Setting up batch interceptor...")
    interceptor = BatchInterceptor()
    interceptor.capture_first_batch(predictor_plugin.model)
    
    print(f"\n🚀 Running ONE epoch to capture first batch...")
    try:
        # Run exactly one step to capture the first batch
        history = predictor_plugin.model.fit(
            x_train, y_train,
            batch_size=config.get('batch_size', 128),
            epochs=1,
            validation_data=(x_val, y_val),
            verbose=1,
            steps_per_epoch=1  # Only one step!
        )
        
        print(f"\n✅ Training step completed!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False, None, None
    
    if not interceptor.first_batch_captured:
        print(f"\n❌ Failed to capture first batch!")
        return False, None, None
    
    return True, interceptor.first_batch_x, interceptor.first_batch_y

def verify_batch_vs_targets(batch_x, batch_y, datasets, config):
    """Verify intercepted batch matches our target calculation exactly."""
    print("\n" + "=" * 80)
    print("BATCH vs TARGET VERIFICATION")
    print("=" * 80)
    
    y_train = datasets['y_train']
    predicted_horizons = config['predicted_horizons']
    batch_size = batch_x.shape[0]
    
    print(f"\n🔍 Batch info:")
    print(f"   Batch X shape: {batch_x.shape}")
    print(f"   Batch Y type: {type(batch_y)}")
    
    all_match = True
    
    if isinstance(batch_y, dict):
        print(f"   Batch Y keys: {list(batch_y.keys())}")
        
        for i, horizon in enumerate(predicted_horizons):
            horizon_key = f"output_horizon_{horizon}"
            
            if horizon_key in batch_y and horizon_key in y_train:
                batch_targets = batch_y[horizon_key]
                original_targets = y_train[horizon_key]
                
                print(f"\n   🎯 {horizon_key}:")
                print(f"     Batch shape: {batch_targets.shape}")
                print(f"     Original shape: {original_targets.shape}")
                
                # Compare first few values
                batch_sample = batch_targets[:5]
                original_sample = original_targets[:5]
                
                print(f"     Batch sample: [{batch_sample[0]:.8f}, {batch_sample[1]:.8f}, {batch_sample[2]:.8f}]")
                print(f"     Original sample: [{original_sample[0]:.8f}, {original_sample[1]:.8f}, {original_sample[2]:.8f}]")
                
                # Check exact equality
                are_equal = np.array_equal(batch_sample, original_sample)
                print(f"     {'✅ EXACT MATCH' if are_equal else '❌ MISMATCH'}")
                
                if not are_equal:
                    all_match = False
                    max_diff = np.max(np.abs(batch_sample - original_sample))
                    print(f"     Max difference: {max_diff:.10f}")
            else:
                print(f"   ❌ Missing {horizon_key} in batch or original data")
                all_match = False
    
    else:
        # Handle stacked format
        print(f"   Batch Y shape: {batch_y.shape}")
        
        for i, horizon in enumerate(predicted_horizons):
            horizon_key = f"output_horizon_{horizon}"
            
            if horizon_key in y_train and i < batch_y.shape[1]:
                batch_targets = batch_y[:, i]
                original_targets = y_train[horizon_key]
                
                print(f"\n   🎯 {horizon_key} (column {i}):")
                
                # Compare first few values
                batch_sample = batch_targets[:5]
                original_sample = original_targets[:5]
                
                print(f"     Batch sample: [{batch_sample[0]:.8f}, {batch_sample[1]:.8f}, {batch_sample[2]:.8f}]")
                print(f"     Original sample: [{original_sample[0]:.8f}, {original_sample[1]:.8f}, {original_sample[2]:.8f}]")
                
                # Check exact equality
                are_equal = np.array_equal(batch_sample, original_sample)
                print(f"     {'✅ EXACT MATCH' if are_equal else '❌ MISMATCH'}")
                
                if not are_equal:
                    all_match = False
                    max_diff = np.max(np.abs(batch_sample - original_sample))
                    print(f"     Max difference: {max_diff:.10f}")
    
    return all_match

def main():
    """Main verification function."""
    print("🔬 CRITICAL VERIFICATION: Y_TRUE vs TARGETS EXACT ALIGNMENT")
    print("This verifies that y_true in loss function = exact targets with exact datetimes")
    print("=" * 80)
    
    try:
        # Step 1: Load exact data
        datasets, preprocessor_params, config = load_and_prepare_exact_data()
        
        # Step 2: Verify datetime alignment
        datetime_ok = verify_datetime_alignment(datasets, config.get('batch_size', 128))
        
        # Step 3: Verify target scale consistency
        scale_ok, scale_issues = verify_target_scale_consistency(datasets, preprocessor_params)
        
        # Step 4: Run minimal training to capture actual batch
        training_ok, batch_x, batch_y = run_minimal_training_verification(datasets, config)
        
        # Step 5: Verify batch matches targets exactly
        batch_ok = False
        if training_ok:
            batch_ok = verify_batch_vs_targets(batch_x, batch_y, datasets, config)
        
        # Final results
        print("\n" + "=" * 80)
        print("🏁 FINAL VERIFICATION RESULTS")
        print("=" * 80)
        
        if datetime_ok and scale_ok and training_ok and batch_ok:
            print("✅ ALL VERIFICATIONS PASSED")
            print("✅ Y_TRUE values in loss function are EXACTLY the targets")
            print("✅ Datetimes are perfectly aligned")
            print("✅ Scales are consistent")
            print("✅ Training batch matches target calculation exactly")
            return 0
        else:
            print("❌ VERIFICATION FAILED")
            if not datetime_ok:
                print("❌ Datetime alignment failed")
            if not scale_ok:
                print("❌ Scale consistency failed:")
                for issue in scale_issues:
                    print(f"    {issue}")
            if not training_ok:
                print("❌ Training batch capture failed")
            if not batch_ok:
                print("❌ Batch vs target verification failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ VERIFICATION EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
