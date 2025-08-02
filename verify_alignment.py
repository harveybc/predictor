#!/usr/bin/env python3
"""
CRITICAL INDEX ALIGNMENT VERIFICATION
Verifies exact temporal alignment between windowed features and targets
"""
import json
import numpy as np
import pandas as pd
import sys
import os

# Add app to path for imports
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

def simulate_sliding_window_creation(data, window_size, max_horizon):
    """
    Simulate the exact sliding window creation logic from the code.
    """
    print(f"Simulating sliding windows: data_len={len(data)}, window_size={window_size}, max_horizon={max_horizon}")
    
    windows = []
    n = len(data)
    
    # Exact logic from sliding_windows.py
    max_start_index = n - max_horizon - 1  # Last index where we can place a window baseline
    min_start_index = window_size - 1      # First index where we can have a complete window
    
    print(f"  min_start_index = {min_start_index}")
    print(f"  max_start_index = {max_start_index}")
    print(f"  num_windows = {max_start_index - min_start_index + 1}")
    
    # Create windows: each window ends at current tick t, contains [t-window_size+1:t+1]
    for t in range(min_start_index, max_start_index + 1):
        window_start = t - window_size + 1
        window_end = t + 1
        window = data[window_start:window_end]
        windows.append({
            'window_idx': len(windows),
            'baseline_tick': t,  # This is the "current" tick for this window
            'window_start': window_start,
            'window_end': window_end,
            'window_data': window,
            'baseline_value': data[t]  # The "current" value
        })
    
    return windows

def simulate_target_calculation(data, window_size, horizons, num_windows):
    """
    Simulate the exact target calculation logic from the code.
    """
    print(f"Simulating target calculation: data_len={len(data)}, window_size={window_size}")
    
    # Exact logic from target_calculation.py
    trimmed_start = window_size - 1
    target_trimmed = data[trimmed_start:]
    
    print(f"  trimmed_start = {trimmed_start}")
    print(f"  target_trimmed length = {len(target_trimmed)}")
    
    targets = {}
    for h in horizons:
        print(f"\n  Calculating targets for horizon {h}:")
        
        # Exact logic from target_calculation.py
        baseline_indices = np.arange(0, num_windows)
        future_indices = baseline_indices + h
        
        print(f"    baseline_indices: {baseline_indices[:5]}... (first 5)")
        print(f"    future_indices: {future_indices[:5]}... (first 5)")
        
        if future_indices[-1] >= len(target_trimmed):
            print(f"    ERROR: Not enough trimmed target data for horizon {h}")
            print(f"           need index {future_indices[-1]}, have {len(target_trimmed)}")
            continue
            
        baseline_values = target_trimmed[baseline_indices]
        future_values = target_trimmed[future_indices]
        
        targets[h] = {
            'baseline_values': baseline_values,
            'future_values': future_values,
            'returns': future_values - baseline_values
        }
        
        print(f"    baseline_values[0:3]: {baseline_values[:3]}")
        print(f"    future_values[0:3]: {future_values[:3]}")
        print(f"    returns[0:3]: {(future_values - baseline_values)[:3]}")
    
    return target_trimmed, targets

def verify_alignment():
    """
    Verify exact alignment between windows and targets.
    """
    print("🔍 CRITICAL ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    # Simulate with simple test data
    print("\n1️⃣ CREATING TEST DATA:")
    test_data = np.arange(1000, dtype=float)  # Simple sequential data: [0, 1, 2, 3, ...]
    print(f"Test data: [{test_data[0]}, {test_data[1]}, {test_data[2]}, ..., {test_data[-1]}] (length={len(test_data)})")
    
    window_size = 144  # From config
    horizons = [1, 2, 3, 4, 5, 6]  # From config
    max_horizon = max(horizons)
    
    print(f"Window size: {window_size}")
    print(f"Max horizon: {max_horizon}")
    
    print("\n2️⃣ SIMULATING SLIDING WINDOWS:")
    windows = simulate_sliding_window_creation(test_data, window_size, max_horizon)
    num_windows = len(windows)
    print(f"Created {num_windows} windows")
    
    if num_windows > 0:
        print("\nFirst 3 windows details:")
        for i in range(min(3, num_windows)):
            w = windows[i]
            print(f"  Window {i}: baseline_tick={w['baseline_tick']}, baseline_value={w['baseline_value']}")
            print(f"           window_range=[{w['window_start']}:{w['window_end']}], window_last={w['window_data'][-1]}")
    
    print("\n3️⃣ SIMULATING TARGET CALCULATION:")
    target_trimmed, targets = simulate_target_calculation(test_data, window_size, horizons, num_windows)
    
    print("\n4️⃣ CRITICAL ALIGNMENT CHECK:")
    print("Verifying that window baselines match target baselines...")
    
    if 1 in targets and num_windows > 0:
        target_baselines = targets[1]['baseline_values']
        
        print("\nFirst 5 alignment checks:")
        for i in range(min(5, num_windows)):
            window_baseline = windows[i]['baseline_value']
            target_baseline = target_baselines[i]
            
            # CRITICAL: These should be EXACTLY the same
            if abs(window_baseline - target_baseline) < 1e-10:
                status = "✅ MATCH"
            else:
                status = "❌ MISMATCH"
            
            print(f"  Window {i}: window_baseline={window_baseline}, target_baseline={target_baseline} {status}")
            
            if abs(window_baseline - target_baseline) >= 1e-10:
                print(f"    🚨 ALIGNMENT ERROR: Window {i} baseline doesn't match target baseline!")
                print(f"       Window tick: {windows[i]['baseline_tick']}")
                print(f"       Target index in trimmed: {i}")
                print(f"       Target index in original: {window_size - 1 + i}")
                return False
    
    print("\n5️⃣ HORIZON PREDICTION CHECK:")
    print("Verifying that targets point to correct future values...")
    
    for h in horizons:
        if h in targets and num_windows > 0:
            print(f"\nHorizon {h} check (first 3 windows):")
            target_futures = targets[h]['future_values']
            
            for i in range(min(3, num_windows)):
                window_baseline_tick = windows[i]['baseline_tick']
                expected_future_tick = window_baseline_tick + h
                expected_future_value = test_data[expected_future_tick] if expected_future_tick < len(test_data) else "N/A"
                actual_future_value = target_futures[i]
                
                if expected_future_value != "N/A" and abs(expected_future_value - actual_future_value) < 1e-10:
                    status = "✅ CORRECT"
                else:
                    status = "❌ WRONG"
                
                print(f"  Window {i}: baseline_tick={window_baseline_tick} -> future_tick={expected_future_tick}")
                print(f"           expected_future={expected_future_value}, actual_future={actual_future_value} {status}")
    
    print("\n" + "=" * 80)
    print("✅ ALIGNMENT VERIFICATION COMPLETE")
    return True

if __name__ == "__main__":
    verify_alignment()
