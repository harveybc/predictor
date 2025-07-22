#!/usr/bin/env python3
"""
Verify the phase2_6 daily config windowing and target calculation for long-term predictions.
This will analyze the exact logic for horizons [24,48,72,96,120,144] with window_size=288.
"""

def analyze_daily_config_windowing():
    """Analyze windowing and target calculation for the daily config"""
    print("="*80)
    print("PHASE2_6 DAILY CONFIG - WINDOWING & TARGET ANALYSIS")
    print("="*80)
    
    # Extract parameters from the daily config
    window_size = 288  # From config
    horizons = [24, 48, 72, 96, 120, 144]  # From config
    max_steps_train = 25200  # From config
    
    print(f"Config Parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Horizons: {horizons}")
    print(f"  Max steps train: {max_steps_train}")
    print(f"  Use returns: true")
    
    # Calculate windowing parameters
    max_horizon = max(horizons)
    print(f"\nWindowing Calculations:")
    print(f"  Max horizon for windowing: {max_horizon}")
    
    # Number of possible windows (from create_sliding_windows)
    num_possible_windows = max_steps_train - window_size - max_horizon + 1
    print(f"  Number of possible windows: {max_steps_train} - {window_size} - {max_horizon} + 1 = {num_possible_windows}")
    
    # Original offset calculation (from preprocessor)
    effective_stl_window = 0  # No STL processing in phase2_6
    original_offset = effective_stl_window + window_size - 2
    print(f"  Original offset: {effective_stl_window} + {window_size} - 2 = {original_offset}")
    
    # Analyze each horizon for data leakage
    print(f"\n" + "="*60)
    print("DATA LEAKAGE ANALYSIS FOR EACH HORIZON")
    print("="*60)
    
    for h in horizons:
        print(f"\nHORIZON {h} (Days/Hours):")
        
        # Calculate adjusted horizon (the fix applied)
        adjusted_horizon = h + 2  # The fix: h + 2
        print(f"  Adjusted horizon: {h} + 2 = {adjusted_horizon}")
        
        # For sample 0 (first window):
        sample_i = 0
        
        # Window covers time t=i to t=i+window_size-1
        window_start = sample_i
        window_end = sample_i + window_size - 1
        print(f"  Window {sample_i}: covers time t={window_start} to t={window_end}")
        
        # Baseline time (for returns calculation)
        baseline_time = original_offset + sample_i
        print(f"  Baseline time: t={baseline_time}")
        
        # Target time (after fix)
        target_time = original_offset + adjusted_horizon + sample_i
        print(f"  Target time: t={original_offset} + {adjusted_horizon} + {sample_i} = t={target_time}")
        
        # Calculate gap between window end and target
        gap = target_time - window_end
        print(f"  Time gap: {target_time} - {window_end} = {gap}")
        
        # Check for leakage
        if gap <= 0:
            print(f"  🚨 DATA LEAKAGE! Target overlaps with or is inside window!")
        else:
            print(f"  ✅ NO LEAKAGE: Target is {gap} time steps after window end")
            
        # Verify the target is actually h steps in the future from baseline
        actual_horizon = target_time - baseline_time
        print(f"  Actual horizon: {target_time} - {baseline_time} = {actual_horizon}")
        
        if actual_horizon == h + 2:
            print(f"  ⚠️  Note: Target is {h+2} steps from baseline (not {h}) due to leakage fix")
        
        # Check if we have enough data
        required_data_length = original_offset + adjusted_horizon + num_possible_windows
        print(f"  Required data length: {original_offset} + {adjusted_horizon} + {num_possible_windows} = {required_data_length}")
        
        if required_data_length > max_steps_train:
            print(f"  🚨 WARNING: Not enough data! Need {required_data_length}, have {max_steps_train}")
        else:
            print(f"  ✅ Sufficient data available")
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_gaps = []
    for h in horizons:
        adjusted_horizon = h + 2
        gap = (original_offset + adjusted_horizon) - (window_size - 1)
        all_gaps.append(gap)
    
    min_gap = min(all_gaps)
    print(f"Minimum gap between window end and target: {min_gap}")
    
    if min_gap > 0:
        print("✅ ALL HORIZONS: No data leakage detected")
        print("✅ Targets are properly positioned in the future")
    else:
        print("🚨 SOME HORIZONS: Data leakage detected!")
    
    # Calculate actual prediction horizons after the fix
    print(f"\nActual Prediction Horizons (after fix):")
    for h in horizons:
        actual_h = h + 2
        print(f"  Configured H{h} → Actually predicts H{actual_h}")
    
    return all_gaps

if __name__ == "__main__":
    gaps = analyze_daily_config_windowing()
    print(f"\nTime gaps: {gaps}")
    print("Analysis complete.")
