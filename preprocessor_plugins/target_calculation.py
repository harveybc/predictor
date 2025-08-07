import numpy as np
import pandas as pd
from .helpers import denormalize, load_normalization_json


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
    
    target_data = {'train': {}, 'val': {}, 'test': {}}
    baseline_info = {}
    
    # Reset normalization stats for this calculation
    target_returns_means = []
    target_returns_stds = []
    
    if not use_returns:
        print("Using direct target values (not returns)")
        raise NotImplementedError("Direct target calculation not implemented")
    
    print("Calculating log return targets...")
    
    # Calculate targets for each split using ONLY baselines
    for split in ['train', 'val', 'test']:
        baseline_key = f'baseline_{split}'
        
        if baseline_key not in baseline_data:
            print(f"No baseline data for {split}")
            continue
            
        baselines = baseline_data[baseline_key]
        if len(baselines) == 0:
            print(f"Empty baselines for {split}")
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
        
        # Calculate targets for each horizon using ONLY baselines
        for i, horizon in enumerate(predicted_horizons):
            horizon_targets = []
            
            # Calculate targets: log(baseline[t+horizon] / baseline[t])
            for t in range(max_samples):
                baseline_current = baselines[t]
                baseline_future = baselines[t + horizon]
                
                # Calculate log return using only baselines
                if baseline_current > 0 and baseline_future > 0:
                    log_return = np.log(baseline_future / baseline_current)
                    horizon_targets.append(log_return)
                else:
                    horizon_targets.append(np.nan)
            
            # Store targets for this horizon
            if horizon_targets:
                horizon_targets = np.array(horizon_targets, dtype=np.float32)
                # Remove NaN values
                valid_targets = horizon_targets[~np.isnan(horizon_targets)]
                
                if len(valid_targets) > 0:
                    # Calculate normalization stats for this horizon
                    if split == 'train':  # Only use training data for normalization
                        target_mean = np.mean(valid_targets)
                        target_std = np.std(valid_targets)
                        target_returns_means.append(target_mean)
                        target_returns_stds.append(target_std)
                    else:
                        # Use training stats for other splits
                        if i < len(target_returns_means):
                            target_mean = target_returns_means[i]
                            target_std = target_returns_stds[i]
                        else:
                            target_mean = 0.0
                            target_std = 1.0
                    
                    # Normalize targets
                    if target_std > 0:
                        normalized_targets = (valid_targets - target_mean) / target_std
                    else:
                        normalized_targets = valid_targets - target_mean
                    
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
