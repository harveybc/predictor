#!/usr/bin/env python3
"""
Detailed analysis of windowing and target calculation to identify data leakage.
This will trace through exactly what data is being used for inputs vs targets.
"""

import sys
import json
import numpy as np
import pandas as pd

# Add project path
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

def analyze_windowing_and_targets():
    """Analyze the exact windowing and target calculation logic"""
    print("="*80)
    print("DETAILED WINDOWING AND TARGET ANALYSIS")
    print("="*80)
    
    # Simulate the exact logic used in the preprocessor
    print("\n--- Simulating Phase2_6 Windowing Logic ---")
    
    # Create mock data to trace through the logic
    n_data = 1000
    window_size = 144  # Hourly config
    horizons = [1, 2, 3, 4, 5, 6]  # Hourly horizons
    
    # Create mock close prices (simple increasing sequence for easy tracking)
    mock_close = np.arange(1000, 1000 + n_data, dtype=np.float32)
    print(f"Mock data: {n_data} points, close prices from {mock_close[0]} to {mock_close[-1]}")
    print(f"Window size: {window_size}")
    print(f"Horizons: {horizons}")
    
    # Step 1: Simulate the windowing function
    print(f"\n--- Step 1: Windowing Function Analysis ---")
    max_horizon = max(horizons)
    print(f"Max horizon for windowing: {max_horizon}")
    
    # This is the exact logic from create_sliding_windows
    num_possible_windows = n_data - window_size - max_horizon + 1
    print(f"Number of possible windows: {n_data} - {window_size} - {max_horizon} + 1 = {num_possible_windows}")
    
    # Simulate a few windows to see what data is included
    print(f"\nFirst 3 windows:")
    for i in range(min(3, num_possible_windows)):
        window_start = i
        window_end = i + window_size
        window_data_indices = list(range(window_start, window_end))
        print(f"  Window {i}: indices {window_start} to {window_end-1} (data: {mock_close[window_start]:.0f} to {mock_close[window_end-1]:.0f})")
        print(f"    Window covers time t={window_start} to t={window_end-1}")
    
    # Step 2: Simulate the baseline calculation
    print(f"\n--- Step 2: Baseline Calculation ---")
    # From the code: original_offset = effective_stl_window + window_size - 2
    effective_stl_window = 0  # No STL processing in phase2_6
    original_offset = effective_stl_window + window_size - 2
    print(f"Original offset: {effective_stl_window} + {window_size} - 2 = {original_offset}")
    
    # Baseline slice
    baseline_slice_end = original_offset + num_possible_windows
    print(f"Baseline slice: [{original_offset}:{baseline_slice_end}]")
    baseline = mock_close[original_offset:baseline_slice_end]
    print(f"Baseline length: {len(baseline)}")
    print(f"Baseline covers time t={original_offset} to t={baseline_slice_end-1}")
    print(f"Sample baseline values: {baseline[:3]}")
    
    # Step 3: Simulate target calculation for each horizon
    print(f"\n--- Step 3: Target Calculation Analysis ---")
    
    # From the code: target_train = target_train_raw[original_offset:]
    target_base = mock_close[original_offset:]
    print(f"Target base slice: [{original_offset}:] (length: {len(target_base)})")
    
    for h in horizons:
        print(f"\n  Horizon {h}:")
        # From the code: target_train_shifted = target_train[h:]
        target_shifted = target_base[h:]
        print(f"    Target shifted slice: [original_offset+{h}:] = [{original_offset+h}:]")
        print(f"    Target shifted length: {len(target_shifted)}")
        
        # Take first num_samples
        target_h = target_shifted[:num_possible_windows]
        print(f"    Final target length: {len(target_h)}")
        print(f"    Target covers time t={original_offset+h} to t={original_offset+h+len(target_h)-1}")
        print(f"    Sample target values: {target_h[:3]}")
        
        # CRITICAL ANALYSIS: What time does each window correspond to vs its target?
        print(f"    \n    LEAKAGE ANALYSIS for Horizon {h}:")
        for i in range(min(3, num_possible_windows)):
            # Window i uses data from time i to i+window_size-1
            window_time_start = i
            window_time_end = i + window_size - 1
            
            # Baseline i corresponds to time original_offset + i
            baseline_time = original_offset + i
            
            # Target i corresponds to time original_offset + h + i  
            target_time = original_offset + h + i
            
            print(f"      Sample {i}:")
            print(f"        Window uses data: t={window_time_start} to t={window_time_end}")
            print(f"        Baseline time: t={baseline_time}")
            print(f"        Target time: t={target_time}")
            print(f"        Time gap between window end and target: {target_time - window_time_end}")
            
            # Check for leakage
            if target_time <= window_time_end:
                print(f"        🚨 DATA LEAKAGE! Target time {target_time} is <= window end time {window_time_end}")
            else:
                print(f"        ✅ No leakage. Target is {target_time - window_time_end} steps in the future")
    
    # Step 4: Check the STL preprocessor for comparison
    print(f"\n--- Step 4: Compare with STL Preprocessor ---")
    try:
        from preprocessor_plugins.stl_preprocessor import PreprocessorPlugin as STLPreprocessor
        print("STL preprocessor found. Checking target calculation...")
        
        # We'll just read the code to see if there are differences
        
    except Exception as e:
        print(f"Could not load STL preprocessor: {e}")

if __name__ == "__main__":
    analyze_windowing_and_targets()
