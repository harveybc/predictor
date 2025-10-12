import numpy as np
import pandas as pd
from .helpers import denormalize, load_normalization_json

def apply_centered_moving_average(data, window_size=2):
    """
    Apply a centered moving average to 1D data and return a numpy array.
    Accepts numpy arrays, pandas Series, or single-column DataFrames.
    """
    if window_size is None or window_size <= 1:
        return np.asarray(data)

    # Convert input to a pandas Series for rolling; if DataFrame, use the first column
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 0:
            return np.array([])
        series = data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        series = data
    else:
        series = pd.Series(np.asarray(data))

    smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.to_numpy()

def calculate_targets_from_baselines(baseline_data, config):
    """
    Target Calculation Module - Step-by-Step Process:
    
    1. Extract baselines from each dataset split (train, val, test)
    2. For each horizon, calculate targets using ONLY baselines:
       - target[t] = log(baseline[t+horizon] / baseline[t])
    3. Normalize targets using training statistics
    4. Return structured target data by horizon and split
    
    Args:
        baseline_data: Dict containing baselines extracted from sliding windows
                      Format: {'baseline_train': array, 'baseline_val': array, 'baseline_test': array}
        config: Configuration dictionary
        
    Returns:
        Dict containing target data and metadata
    """
    print("Calculating targets from provided baselines...")
    
    predicted_horizons = config['predicted_horizons']
    target_column = config.get('target_column', 'CLOSE')
    use_returns = config.get('use_returns', True)
    target_softening = config.get('target_softening', 2)

    target_data = {'train': {}, 'val': {}, 'test': {}}
    baseline_info = {}
    
    # Reset normalization stats for this calculation
    target_returns_means = []
    target_returns_stds = []
    
    if not use_returns:  # If configured to use direct target values instead of returns
        print("Using direct target values (not returns)")  # Log the selected target calculation mode
        # Note: We continue with the same pipeline below; the only difference will be
        # how each per-horizon value is computed (using baseline_future directly).
    
    # Optional smoothing of baselines before target computation
    # IMPORTANT:
    # - For use_returns=True we keep the previous centered moving average (symmetrical) to reduce high-frequency noise.
    # - For use_returns=False (direct-price targets), a centered MA would leak future information into the current baseline.
    #   To avoid leakage that harms training/validation, we switch to a causal (trailing) moving average.
    if target_softening > 1:
        for split in ['train', 'val', 'test']:
            baseline_key = f'baseline_{split}'
            if baseline_key in baseline_data:
                baselines = baseline_data[baseline_key]
                if use_returns:
                    # Symmetric smoothing acceptable in returns-space target derivation
                    baseline_data[baseline_key] = apply_centered_moving_average(baselines, window_size=target_softening)
                else:
                    # Causal (trailing) smoothing to prevent future leakage when predicting absolute prices
                    ser = pd.Series(np.asarray(baselines))
                    baseline_data[baseline_key] = ser.rolling(window=target_softening, center=False, min_periods=1).mean().to_numpy()
       

    print("Calculating targets (returns or direct price depending on use_returns)...")
    
    # Calculate targets for each split using ONLY baselines
    for split in ['train', 'val', 'test']:
        baseline_key = f'baseline_{split}'
        
        if baseline_key not in baseline_data:
            print(f"No baseline data for {split}")
            continue
            
        baselines = baseline_data[baseline_key]
        if len(baselines) == 0:
            # ROOT CAUSE OF LATER PIPELINE KeyError:
            # Previous code just 'continue'd leaving target_data[split] empty (no 'output_horizon_X' keys),
            # so pipeline _extract() raised KeyError when trying to access first horizon.
            # Fix: explicitly create empty arrays for each requested horizon so structure is present.
            print(f"Empty baselines for {split} -> creating empty target arrays for horizons {predicted_horizons}")
            for horizon in predicted_horizons:
                target_data[split][f'output_horizon_{horizon}'] = np.array([])
            continue
        
        # Calculate max horizon to ensure all horizons have same length
        max_horizon = max(predicted_horizons)
        max_samples = len(baselines) - max_horizon  # Ensure we can look ahead max_horizon steps
        
        if max_samples <= 0:
            print(f"  {split}: Insufficient baselines ({len(baselines)}) for max horizon {max_horizon}")
            for horizon in predicted_horizons:
                target_data[split][f'output_horizon_{horizon}'] = np.array([])
            continue
        
        print(f"  {split}: Using {max_samples} samples for ALL horizons (limited by H{max_horizon})")
        target_factor = config.get('target_factor', 1000.0)


        # Calculate targets for each horizon using ONLY baselines
        for i, horizon in enumerate(predicted_horizons):
            horizon_targets = []
            
            # Calculate targets: log(baseline[t+horizon] / baseline[t])
            for t in range(max_samples):
                baseline_current = baselines[t]
                baseline_future = baselines[t + horizon]

                if use_returns:  # Branch: compute returns-based target
                    # Calculate scaled log-return using only baselines
                    # Guard against non-positive values to avoid invalid log operations
                    if baseline_current > 0 and baseline_future > 0:  # Ensure both values are positive
                        # Correct formula: log(future/current); previous version used log(1 + future/current) which biased magnitudes
                        return_value = target_factor * np.log(baseline_future / baseline_current)
                    else:  # If any value is non-positive
                        return_value = 0.0  # Fallback to 0.0 for stability
                else:  # Branch: compute direct-value target (no returns)
                    # Direct target: use the baseline_future value as the target without ratio/log transform
                    if np.isfinite(baseline_future):  # Guard against NaN/Inf values
                        return_value = float(baseline_future)  # Use the raw future baseline value as target
                    else:  # If invalid number
                        return_value = 0.0  # Fallback to 0.0 for stability

                horizon_targets.append(return_value)  # Accumulate per-timestep target for this horizon

            # Store targets for this horizon
            if horizon_targets:
                horizon_targets = np.array(horizon_targets, dtype=np.float32)
                # Remove NaN values
                valid_targets = horizon_targets[~np.isnan(horizon_targets)]
                
                if len(valid_targets) > 0:
                    normalized_targets = valid_targets
                    target_data[split][f'output_horizon_{horizon}'] = normalized_targets
                    print(f"  {split} H{horizon}: {len(normalized_targets)} targets")
                else:
                    print(f"  {split} H{horizon}: No valid targets")
                    target_data[split][f'output_horizon_{horizon}'] = np.array([])
            else:
                print(f"  {split} H{horizon}: No targets calculated")
                target_data[split][f'output_horizon_{horizon}'] = np.array([])
    
    # Store baseline info for output
    for split in ['train', 'val', 'test']:
        baseline_key = f'baseline_{split}'
        if baseline_key in baseline_data:
            baseline_info[baseline_key] = baseline_data[baseline_key]
    
    # Prepare final result
    result = {
        'y_train': target_data['train'],
        'y_val': target_data['val'],
        'y_test': target_data['test'],
        **baseline_info,
        'target_returns_means': target_returns_means,
        'target_returns_stds': target_returns_stds,
        'predicted_horizons': predicted_horizons,
    }
    
    print(f"Target calculation complete. Horizons: {predicted_horizons}")
    print(f"Normalization means: {target_returns_means}")
    print(f"Normalization stds: {target_returns_stds}")
    
    return result
