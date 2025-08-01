#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')
import numpy as np
import pandas as pd

print("Testing bounds checking fix...")

try:
    from preprocessor_plugins.target_calculation import TargetCalculationProcessor
    print("✓ Import successful")
    
    # Test with realistic parameters that might cause the indexing issue
    processor = TargetCalculationProcessor()
    
    config = {
        "target_column": "CLOSE",
        "predicted_horizons": [1, 3, 6],
        "window_size": 10,
        "use_returns": True
    }
    
    # Create test data that might cause boundary issues
    n_points = 100
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
    
    # Use expected samples that work with the sliding window algorithm
    max_horizon = max(config['predicted_horizons'])
    window_size = config['window_size']
    expected_samples = n_points - window_size - max_horizon + 1
    
    windowed_data = {
        'num_samples_train': expected_samples,
        'num_samples_val': expected_samples,
        'num_samples_test': expected_samples
    }
    
    print(f"Data length: {len(close_prices)}")
    print(f"Dates length: {len(dates)}")
    print(f"Expected samples: {expected_samples}")
    print(f"Max baseline index: {window_size-1+expected_samples-1}")
    
    # Run target calculation
    result = processor.calculate_targets(baseline_data, windowed_data, config)
    print("✓ Target calculation completed successfully")
    
    # Check the results
    print(f"Baseline test shape: {result['baseline_test'].shape}")
    if result['baseline_test_dates'] is not None:
        print(f"Baseline dates shape: {len(result['baseline_test_dates'])}")
    else:
        print("Baseline dates: None")
    
    print("✅ Bounds checking test passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
