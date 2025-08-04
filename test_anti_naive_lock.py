#!/usr/bin/env python3
"""
Test script to verify anti-naive-lock preprocessing is working correctly.
"""

import numpy as np
from preprocessor_plugins.anti_naive_lock import AntiNaiveLockProcessor

# Create test data
num_samples = 100
time_steps = 10
num_features = 5

x_train = np.random.randn(num_samples, time_steps, num_features)
x_val = np.random.randn(50, time_steps, num_features)
x_test = np.random.randn(25, time_steps, num_features)

feature_names = ['OPEN', 'day_of_week', 'stl_trend', 'RSI', 'S&P500_Close']

# Test config with anti-naive-lock enabled
config = {
    'anti_naive_lock_enabled': True,
    'feature_preprocessing_strategy': 'selective',
    'use_log_returns': True,
    'use_cyclic_encoding': True,
    'use_first_differences': True,
    'preserve_stationary_indicators': True,
    'handle_constant_daily_features': True,
    'normalize_after_preprocessing': True,
    
    # Feature categories (simplified from your config)
    'price_features': ['OPEN'],
    'temporal_features': ['day_of_week'],
    'trend_features': ['stl_trend'],
    'stationary_indicators': ['RSI'],
    'constant_daily_features': ['S&P500_Close'],
}

print("="*60)
print("TESTING ANTI-NAIVE-LOCK PREPROCESSING")
print("="*60)

# Create processor and test
processor = AntiNaiveLockProcessor()

print("\nOriginal shapes:")
print(f"  x_train: {x_train.shape}")
print(f"  x_val: {x_val.shape}")
print(f"  x_test: {x_test.shape}")
print(f"  feature_names: {feature_names}")

print(f"\nConfig:")
print(f"  anti_naive_lock_enabled: {config['anti_naive_lock_enabled']}")
print(f"  strategy: {config['feature_preprocessing_strategy']}")
print(f"  use_log_returns: {config['use_log_returns']}")
print(f"  use_cyclic_encoding: {config['use_cyclic_encoding']}")

print("\n" + "-"*60)
print("CALLING ANTI-NAIVE-LOCK PROCESSOR...")
print("-"*60)

# Process the data
x_train_proc, x_val_proc, x_test_proc, stats = processor.process_sliding_windows(
    x_train, x_val, x_test, feature_names, config
)

print("\n" + "-"*60)
print("PROCESSING COMPLETE")
print("-"*60)

print(f"\nProcessed shapes:")
print(f"  x_train: {x_train_proc.shape}")
print(f"  x_val: {x_val_proc.shape}")
print(f"  x_test: {x_test_proc.shape}")

print(f"\nTransforms applied:")
for feature, transform in stats.get('applied_transforms', {}).items():
    print(f"  {feature}: {transform}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
