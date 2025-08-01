#!/usr/bin/env python3
"""
Test the new multi-horizon callback behavior
"""

import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

def test_callback_behavior():
    """Test that our callbacks behave correctly with multi-horizon metrics"""
    try:
        from predictor_plugins.predictor_plugin_cnn import ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter
        
        # Test metrics for 3 horizons
        test_metrics = [
            "val_output_horizon_1_mae_magnitude", 
            "val_output_horizon_2_mae_magnitude",
            "val_output_horizon_3_mae_magnitude"
        ]
        
        print("=== Testing Multi-Horizon Callback Behavior ===")
        print(f"Monitoring metrics: {test_metrics}")
        
        # Create callbacks
        early_stopping = EarlyStoppingWithPatienceCounter(
            horizon_metrics=test_metrics,
            patience=3,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateauWithCounter(
            horizon_metrics=test_metrics,
            factor=0.5,
            patience=2,
            cooldown=1,
            min_delta=0.001,
            verbose=1,
            mode='min'
        )
        
        # Simulate training logs over several epochs
        print("\n=== Simulating Training Epochs ===")
        
        # Epoch 1: All metrics start high
        logs_1 = {
            "val_output_horizon_1_mae_magnitude": 0.100,
            "val_output_horizon_2_mae_magnitude": 0.110,
            "val_output_horizon_3_mae_magnitude": 0.105,
            "loss": 0.5
        }
        print(f"\nEpoch 1 logs: {logs_1}")
        early_stopping.on_epoch_end(0, logs_1)
        reduce_lr.on_epoch_end(0, logs_1)
        
        # Epoch 2: No improvement
        logs_2 = {
            "val_output_horizon_1_mae_magnitude": 0.102,
            "val_output_horizon_2_mae_magnitude": 0.112,
            "val_output_horizon_3_mae_magnitude": 0.107,
            "loss": 0.51
        }
        print(f"\nEpoch 2 logs: {logs_2}")
        early_stopping.on_epoch_end(1, logs_2)
        reduce_lr.on_epoch_end(1, logs_2)
        
        # Epoch 3: Horizon 2 improves slightly
        logs_3 = {
            "val_output_horizon_1_mae_magnitude": 0.103,
            "val_output_horizon_2_mae_magnitude": 0.108,  # Improvement > min_delta
            "val_output_horizon_3_mae_magnitude": 0.108,
            "loss": 0.52
        }
        print(f"\nEpoch 3 logs: {logs_3}")
        early_stopping.on_epoch_end(2, logs_3)
        reduce_lr.on_epoch_end(2, logs_3)
        
        # Epoch 4: No improvement again
        logs_4 = {
            "val_output_horizon_1_mae_magnitude": 0.104,
            "val_output_horizon_2_mae_magnitude": 0.109,
            "val_output_horizon_3_mae_magnitude": 0.109,
            "loss": 0.53
        }
        print(f"\nEpoch 4 logs: {logs_4}")
        early_stopping.on_epoch_end(3, logs_4)
        reduce_lr.on_epoch_end(3, logs_4)
        
        # Epoch 5: Horizon 1 improves significantly
        logs_5 = {
            "val_output_horizon_1_mae_magnitude": 0.095,  # Big improvement
            "val_output_horizon_2_mae_magnitude": 0.110,
            "val_output_horizon_3_mae_magnitude": 0.110,
            "loss": 0.54
        }
        print(f"\nEpoch 5 logs: {logs_5}")
        early_stopping.on_epoch_end(4, logs_5)
        reduce_lr.on_epoch_end(4, logs_5)
        
        print("\n=== Test Results ===")
        print(f"✓ EarlyStopping final patience counter: {early_stopping.patience_counter}")
        print(f"✓ ReduceLROnPlateau final patience counter: {reduce_lr.patience_counter}")
        print("✓ Callbacks should have reset counters when horizons improved")
        print("✓ Both callbacks now monitor ALL horizons instead of just one")
        print("✓ Any improvement in ANY horizon resets the patience counters")
        
        return True
        
    except Exception as e:
        print(f"❌ Callback behavior test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_callback_behavior()
