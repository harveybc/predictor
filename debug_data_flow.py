#!/usr/bin/env python3
"""
CRITICAL DATA FLOW ANALYSIS
Expose inconsistencies in train/validation/test data processing
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.append('/home/harveybc/Documents/GitHub/predictor')
sys.path.append('/home/harveybc/Documents/GitHub/preprocessor')

def analyze_raw_data_files():
    """Analyze the raw data files to understand temporal patterns"""
    print("="*80)
    print("LEVEL 1: RAW DATA FILE ANALYSIS")
    print("="*80)
    
    # Load the actual data files
    data_files = {
        'train': 'examples/data/phase_6/normalized_d4.csv',
        'val': 'examples/data/phase_6/normalized_d5.csv', 
        'test': 'examples/data/phase_6/normalized_d6.csv'
    }
    
    # Load normalization config
    with open('examples/data/phase_6/normalization_config_b.json', 'r') as f:
        norm_config = json.load(f)
    
    datasets = {}
    
    for split, file_path in data_files.items():
        print(f"\n--- {split.upper()} DATA ({file_path}) ---")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"CLOSE stats (normalized): min={df['CLOSE'].min():.6f}, max={df['CLOSE'].max():.6f}, mean={df['CLOSE'].mean():.6f}")
        
        # Denormalize CLOSE to see real values
        close_norm = df['CLOSE'].values
        if 'CLOSE' in norm_config:
            close_mean = norm_config['CLOSE']['mean']
            close_std = norm_config['CLOSE']['std']
            close_denorm = close_norm * close_std + close_mean
            print(f"CLOSE stats (denormalized): min={close_denorm.min():.6f}, max={close_denorm.max():.6f}, mean={close_denorm.mean():.6f}")
        
        # Check for temporal continuity issues
        if len(df) > 1:
            time_diffs = df.index[1:] - df.index[:-1]
            unique_diffs = time_diffs.unique()
            print(f"Time intervals: {unique_diffs[:5]} (showing first 5)")
            
        datasets[split] = df
    
    return datasets, norm_config

def analyze_temporal_alignment(datasets):
    """Analyze temporal alignment between splits"""
    print("\n" + "="*80)
    print("LEVEL 2: TEMPORAL ALIGNMENT ANALYSIS")
    print("="*80)
    
    # Check if dates overlap between splits
    train_dates = set(datasets['train'].index)
    val_dates = set(datasets['val'].index)
    test_dates = set(datasets['test'].index)
    
    print(f"\nDate overlap analysis:")
    print(f"Train-Val overlap: {len(train_dates & val_dates)} dates")
    print(f"Train-Test overlap: {len(train_dates & test_dates)} dates")
    print(f"Val-Test overlap: {len(val_dates & test_dates)} dates")
    
    if len(train_dates & val_dates) > 0:
        print("🚨 CRITICAL: Train-Validation date overlap detected!")
    if len(train_dates & test_dates) > 0:
        print("🚨 CRITICAL: Train-Test date overlap detected!")
    if len(val_dates & test_dates) > 0:
        print("🚨 CRITICAL: Validation-Test date overlap detected!")
    
    # Check temporal ordering
    train_last = datasets['train'].index[-1]
    val_first = datasets['val'].index[0]
    val_last = datasets['val'].index[-1] 
    test_first = datasets['test'].index[0]
    
    print(f"\nTemporal ordering:")
    print(f"Train ends: {train_last}")
    print(f"Val starts: {val_first}")
    print(f"Val ends: {val_last}")
    print(f"Test starts: {test_first}")
    
    gap_train_val = val_first - train_last
    gap_val_test = test_first - val_last
    
    print(f"Gap Train->Val: {gap_train_val}")
    print(f"Gap Val->Test: {gap_val_test}")
    
    return {
        'train_val_gap': gap_train_val,
        'val_test_gap': gap_val_test,
        'overlaps': {
            'train_val': len(train_dates & val_dates),
            'train_test': len(train_dates & test_dates),
            'val_test': len(val_dates & test_dates)
        }
    }

def simulate_sliding_window_creation(datasets, config):
    """Simulate sliding window creation for each split"""
    print("\n" + "="*80)
    print("LEVEL 3: SLIDING WINDOW SIMULATION")
    print("="*80)
    
    window_size = 288  # From config
    horizons = [24, 48, 72, 96, 120, 144]  # From config
    max_horizon = max(horizons)
    
    results = {}
    
    for split, df in datasets.items():
        print(f"\n--- {split.upper()} SLIDING WINDOWS ---")
        
        n = len(df)
        print(f"Total data points: {n}")
        
        # Calculate available windows (same logic as sliding_windows.py)
        max_start_index = n - max_horizon - 1
        min_start_index = window_size - 1
        
        if max_start_index < min_start_index:
            print(f"🚨 INSUFFICIENT DATA: Need {window_size + max_horizon}, have {n}")
            num_windows = 0
        else:
            num_windows = max_start_index - min_start_index + 1
            
        print(f"Available windows: {num_windows}")
        print(f"Window range: [{min_start_index}:{max_start_index+1}]")
        
        # Check first few windows
        if num_windows > 0:
            print("\nFirst 3 windows:")
            for i in range(min(3, num_windows)):
                t = min_start_index + i
                window_start = t - window_size + 1
                window_end = t + 1
                baseline_time = t
                
                print(f"  Window {i}: data[{window_start}:{window_end}], baseline@{baseline_time}")
                print(f"    Date: {df.index[baseline_time]}")
                print(f"    CLOSE: {df['CLOSE'].iloc[baseline_time]:.6f}")
                
                # Check targets for this window
                for h in horizons[:3]:  # Just first 3 horizons
                    if baseline_time + h < n:
                        target_time = baseline_time + h
                        print(f"    H{h} target@{target_time}: {df['CLOSE'].iloc[target_time]:.6f}")
        
        results[split] = {
            'num_windows': num_windows,
            'data_length': n,
            'first_window_baseline': min_start_index if num_windows > 0 else None
        }
    
    return results

def main():
    """Main analysis function"""
    print("SYSTEMATIC DATA FLOW ANALYSIS")
    print("="*80)
    
    # Level 1: Raw data analysis
    datasets, norm_config = analyze_raw_data_files()
    
    # Level 2: Temporal alignment 
    temporal_analysis = analyze_temporal_alignment(datasets)
    
    # Level 3: Sliding window simulation
    window_analysis = simulate_sliding_window_creation(datasets, {})
    
    print("\n" + "="*80)
    print("CRITICAL FINDINGS SUMMARY")
    print("="*80)
    
    # Check for temporal issues
    if temporal_analysis['overlaps']['train_val'] > 0:
        print("🚨 FATAL: Train-Validation data overlap = DATA LEAKAGE")
    
    if temporal_analysis['train_val_gap'].total_seconds() < 0:
        print("🚨 FATAL: Validation data precedes training data = TEMPORAL LEAK")
        
    if temporal_analysis['val_test_gap'].total_seconds() < 0:
        print("🚨 FATAL: Test data precedes validation data = TEMPORAL LEAK")
    
    # Check for window availability
    min_windows = min(w['num_windows'] for w in window_analysis.values())
    if min_windows == 0:
        print("🚨 FATAL: Some splits have zero available windows")
    
    # Check for inconsistent window counts
    window_counts = [w['num_windows'] for w in window_analysis.values()]
    if len(set(window_counts)) > 1:
        print(f"⚠️  WARNING: Inconsistent window counts across splits: {window_counts}")
    
    print(f"\nWindow counts: {dict(zip(datasets.keys(), window_counts))}")
    
if __name__ == "__main__":
    main()
