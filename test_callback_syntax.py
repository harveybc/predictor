#!/usr/bin/env python3
"""
Quick syntax check for the updated callbacks
"""

import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

def test_syntax():
    """Test that our callback modifications have correct syntax"""
    try:
        # Try to import the modified file
        from predictor_plugins.predictor_plugin_cnn import ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter
        
        # Test that we can instantiate the callbacks with the new interface
        test_metrics = ["val_output_horizon_1_mae_magnitude", "val_output_horizon_2_mae_magnitude"]
        
        early_stopping = EarlyStoppingWithPatienceCounter(
            horizon_metrics=test_metrics,
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateauWithCounter(
            horizon_metrics=test_metrics,
            factor=0.5,
            patience=5,
            cooldown=5,
            min_delta=1e-4,
            verbose=1,
            mode='min'
        )
        
        print("✓ Callback syntax test PASSED!")
        print(f"✓ EarlyStoppingWithPatienceCounter created successfully")
        print(f"✓ ReduceLROnPlateauWithCounter created successfully")
        print(f"✓ Both callbacks configured to monitor: {test_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Syntax test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_syntax()
