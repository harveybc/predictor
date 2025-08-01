#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')
import numpy as np
import pandas as pd

print("Testing target calculation directly...")

try:
    from preprocessor_plugins.target_calculation import TargetCalculationProcessor
    print("✓ Import successful")
    
    # Simple test setup
    processor = TargetCalculationProcessor()
    
    # Simple config
    config = {
        "target_column": "CLOSE",
        "predicted_horizons": [1, 2],
        "window_size": 5,
        "use_returns": True
    }
    
    # Simple test data
    n_points = 20
    close_prices = 100.0 + 0.1 * np.arange(n_points)
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='h')
    
    # Create test dataframes
    df_data = pd.DataFrame({'CLOSE': close_prices}, index=dates)
    
    baseline_data = {
        'x_train_df': df_data,
        'x_val_df': df_data,
        'x_test_df': df_data,
        'y_train_df': df_data,
        'y_val_df': df_data,
        'y_test_df': df_data,
        'dates_train': dates,
        'dates_val': dates,
        'dates_test': dates,
        'norm_json': {}
    }
    
    # Calculate expected samples: n - window_size - max_horizon + 1 = 20 - 5 - 2 + 1 = 14
    windowed_data = {
        'num_samples_train': 14,
        'num_samples_val': 14,
        'num_samples_test': 14
    }
    
    print("✓ Test data created")
    print(f"Data length: {len(close_prices)}")
    print(f"Expected samples: {windowed_data['num_samples_test']}")
    
    # Run target calculation
    result = processor.calculate_targets(baseline_data, windowed_data, config)
    print("✓ Target calculation completed")
    
    # Check results
    print(f"Baseline test shape: {result['baseline_test'].shape}")
    print(f"Target keys: {list(result['y_test'].keys())}")
    
    # Check horizon 1
    target_h1 = result['y_test']['output_horizon_1']
    print(f"Horizon 1 target shape: {target_h1.shape}")
    print(f"Target means: {result['target_returns_mean']}")
    print(f"Target stds: {result['target_returns_std']}")
    
    print("✅ Basic test passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
