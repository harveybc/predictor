import numpy as np
import pandas as pd
from .helpers import denormalize, load_normalization_json


class TargetCalculationProcessor:
    """
    Clean target calculation using baselines from sliding windows.
    """
    
    def __init__(self):
        self.target_returns_means = []
        self.target_returns_stds = []
    
    def calculate_targets(self, baseline_data, windowed_data, config):
        """
        Calculate targets using provided baselines from sliding windows.
        
        Args:
            baseline_data: Dict containing baselines extracted from sliding windows
            windowed_data: Windowed data (not used in this simplified version)
            config: Configuration dictionary
            
        Returns:
            Dict containing target data and metadata
        """
        print("Calculating targets from provided baselines...")
        
        predicted_horizons = config['predicted_horizons']
        target_column = config.get('target_column', 'CLOSE')
        use_returns = config.get('use_returns', True)
        window_size = config.get('window_size', 48)  # CRITICAL FIX: Get window_size from config
        
        # Load normalization parameters
        norm_json = load_normalization_json(config)
        
        target_data = {'train': {}, 'val': {}, 'test': {}}
        baseline_info = {}
        
        # Reset normalization stats for this calculation
        self.target_returns_means = []
        self.target_returns_stds = []
        
        if not use_returns:
            print("Using direct target values (not returns)")
            # Implementation for direct values if needed
            raise NotImplementedError("Direct target calculation not implemented")
        
        print("Calculating log return targets...")
        
        # Calculate targets for each split using provided baselines
        for split in ['train', 'val', 'test']:
            baseline_key = f'baseline_{split}'
            
            if baseline_key not in baseline_data:
                print(f"No baseline data for {split}")
                continue
                
            baselines = baseline_data[baseline_key]
            if len(baselines) == 0:
                print(f"Empty baselines for {split}")
                continue
            
            # CRITICAL FIX: Use the correct data source for target calculation
            # Get the original denormalized data that aligns with sliding window creation
            x_df = baseline_data[f'x_{split}_df']  # Use x_df since it contains the target column
            price_series = x_df[target_column].values.astype(np.float32)
            
            # CRITICAL FIX: Calculate max horizon to ensure all horizons have same length
            max_horizon = max(predicted_horizons)
            max_samples = len(baselines)
            
            # Find maximum number of targets we can create for ALL horizons
            for j in range(len(baselines)):
                baseline_time_idx = j + window_size - 1
                future_time_idx = baseline_time_idx + max_horizon
                if future_time_idx >= len(price_series):
                    max_samples = j
                    break
            
            print(f"  {split}: Using {max_samples} samples for ALL horizons (limited by H{max_horizon})")
            
            # Calculate targets for each horizon with CORRECT temporal alignment
            for i, horizon in enumerate(predicted_horizons):
                horizon_targets = []
                
                # CRITICAL FIX: Use max_samples to ensure all horizons have same length
                for j in range(max_samples):
                    # CORRECT TEMPORAL MAPPING:
                    # Sliding window j spans: [j, j+1, ..., j+window_size-1]
                    # baseline[j] = price_series[j + window_size - 1]
                    baseline_time_idx = j + window_size - 1
                    
                    # The future time index for the target
                    future_time_idx = baseline_time_idx + horizon
                    
                    # We already verified this is safe in max_samples calculation
                    baseline_price = baselines[j]
                    future_price = price_series[future_time_idx]
                    
                    # Calculate log return: log(price[t+h] / baseline[t])
                    if baseline_price > 0 and future_price > 0:
                        log_return = np.log(future_price / baseline_price)
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
                            self.target_returns_means.append(target_mean)
                            self.target_returns_stds.append(target_std)
                        else:
                            # Use training stats for other splits
                            if i < len(self.target_returns_means):
                                target_mean = self.target_returns_means[i]
                                target_std = self.target_returns_stds[i]
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
                baseline_info[f'{baseline_key}_dates'] = baseline_data.get(f'{baseline_key}_dates')
        
        # Prepare final result
        result = {
            'y_train': target_data['train'],
            'y_val': target_data['val'],
            'y_test': target_data['test'],
            **baseline_info,
            'target_returns_means': self.target_returns_means,
            'target_returns_stds': self.target_returns_stds,
            'predicted_horizons': predicted_horizons,
        }
        
        print(f"Target calculation complete. Horizons: {predicted_horizons}")
        print(f"Normalization means: {self.target_returns_means}")
        print(f"Normalization stds: {self.target_returns_stds}")
        
        return result
