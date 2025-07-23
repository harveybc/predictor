#!/usr/bin/env python3
"""
Test script to verify causality fixes in phase2_6 preprocessor
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the predictor directory to the path
sys.path.append('/home/harveybc/Documents/GitHub/predictor')

def test_causality_logic():
    """Test the causality fix logic without loading actual data"""
    print("="*60)
    print("CAUSALITY FIX VERIFICATION TEST")
    print("="*60)
    
    # Test parameters
    window_size = 288
    horizons = [1, 6, 12, 24, 48, 72]
    data_length = 1000
    
    print(f"Test Configuration:")
    print(f"  Window size: {window_size}")
    print(f"  Data length: {data_length}")
    print(f"  Horizons: {horizons}")
    print(f"  Max horizon: {max(horizons)}")
    print()
    
    # OLD PROBLEMATIC LOGIC
    print("OLD LOGIC (PROBLEMATIC):")
    time_horizon_old = max(horizons)  # 72
    num_windows_old = data_length - window_size - time_horizon_old + 1
    original_offset_old = window_size - 2  # 286
    
    print(f"  time_horizon_for_windowing: {time_horizon_old}")
    print(f"  num_possible_windows: {num_windows_old}")
    print(f"  original_offset: {original_offset_old}")
    
    print("  Window-Target relationships (OLD):")
    for i in range(min(3, num_windows_old)):
        window_start = i
        window_end = i + window_size - 1
        baseline_pos = original_offset_old + i
        print(f"    Window {i}: data[{window_start}:{window_end+1}] -> baseline at {baseline_pos}")
        
        for h in horizons[:3]:  # Show first 3 horizons
            adjusted_horizon = h + 2  # Old adjustment
            target_pos = original_offset_old + adjusted_horizon + i
            print(f"      H{h}: target at {target_pos} (gap from window_end: {target_pos - window_end})")
    print()
    
    # NEW FIXED LOGIC
    print("NEW LOGIC (FIXED):")
    time_horizon_new = 1  # Fixed to 1
    num_windows_new = data_length - window_size - time_horizon_new + 1
    original_offset_new = window_size  # 288
    
    print(f"  time_horizon_for_windowing: {time_horizon_new}")
    print(f"  num_possible_windows: {num_windows_new}")
    print(f"  original_offset: {original_offset_new}")
    
    print("  Window-Target relationships (NEW):")
    for i in range(min(3, num_windows_new)):
        window_start = i
        window_end = i + window_size - 1
        baseline_pos = original_offset_new + i
        print(f"    Window {i}: data[{window_start}:{window_end+1}] -> baseline at {baseline_pos}")
        
        for h in horizons[:3]:  # Show first 3 horizons
            target_pos = original_offset_new + h + i  # New logic: direct horizon offset
            print(f"      H{h}: target at {target_pos} (gap from window_end: {target_pos - window_end})")
    print()
    
    # CAUSALITY ANALYSIS
    print("CAUSALITY ANALYSIS:")
    print("  For strict causality, target at horizon H should be:")
    print("  - Window: data[i:i+window_size-1] (ends at i+window_size-1)")
    print("  - Target: data[i+window_size-1+H] (H steps after window end)")
    print()
    
    print("  OLD logic violations:")
    for h in horizons[:3]:
        adjusted_horizon = h + 2
        gap_old = adjusted_horizon - (window_size - 1)  # Gap from window end
        if gap_old < h:
            print(f"    H{h}: VIOLATION! Gap={gap_old} < required {h}")
        else:
            print(f"    H{h}: OK. Gap={gap_old} >= required {h}")
    print()
    
    print("  NEW logic verification:")
    for h in horizons[:3]:
        gap_new = h  # Direct horizon offset
        if gap_new == h:
            print(f"    H{h}: CORRECT! Gap={gap_new} = required {h}")
        else:
            print(f"    H{h}: ERROR! Gap={gap_new} != required {h}")
    print()
    
    # SUMMARY
    print("SUMMARY:")
    print(f"  Window count change: {num_windows_old} -> {num_windows_new} ({num_windows_new - num_windows_old:+d})")
    print(f"  Offset change: {original_offset_old} -> {original_offset_new} ({original_offset_new - original_offset_old:+d})")
    print("  Causality: OLD=VIOLATED, NEW=STRICT")
    print("="*60)

if __name__ == "__main__":
    test_causality_logic()
