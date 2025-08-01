import numpy as np
import pandas as pd
from .helpers import denormalize


class TargetCalculationProcessor:
    """
    Handles target calculation for multiple horizons from baseline data.
    """
    
    def __init__(self):
        self.target_returns_means = []
        self.target_returns_stds = []
    
    def calculate_targets(self, baseline_data, windowed_data, config):
        """
        Calculates normalized target returns for all horizons and splits.
        
        Args:
            baseline_data: Dict containing baseline data for all splits
            windowed_data: Dict containing windowed features data
            config: Configuration dictionary
            
        Returns:
            Dict containing target data and baseline information
        """
        print("\n--- Calculating Targets ---")
        
        target_column = config.get("target_column", "CLOSE")
        predicted_horizons = config['predicted_horizons']
        window_size = config['window_size']
        use_returns = config.get("use_returns", True)
        norm_json = baseline_data['norm_json']
        
        splits = ['train', 'val', 'test']
        target_data = {}
        
        # Reset normalization stats
        self.target_returns_means = []
        self.target_returns_stds = []
        
        # Extract and denormalize target column for each split
        denorm_targets = {}
        baseline_targets = {}
        baseline_dates = {}
        
        for split in splits:
            print(f"\nProcessing {split} targets...")
            
            # Get the target column from baseline and denormalize it
            if split == 'train':
                y_df = baseline_data['y_train_df']
            elif split == 'val':
                y_df = baseline_data['y_val_df']
            else:  # test
                y_df = baseline_data['y_test_df']
            
            # Extract and denormalize target column
            target_raw = y_df[target_column].astype(np.float32).values
            target_denorm = denormalize(target_raw, norm_json, target_column)
            
            # Store the full denormalized target (aligned with windowed features)
            denorm_targets[split] = target_denorm
            baseline_targets[split] = target_denorm
            
            # Use corresponding dates
            dates = baseline_data[f'dates_{split}']
            baseline_dates[split] = dates
            
            print(f"Denormalized {split} target: {len(target_denorm)} samples")
        
        # Now calculate targets for each horizon
        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")
        
        for i, h in enumerate(predicted_horizons):
            print(f"\nCalculating targets for horizon {h}...")
            
            # Calculate returns for training split first to get normalization stats
            split = 'train'
            num_samples = windowed_data[f'num_samples_{split}']
            target_trimmed = denorm_targets[split]
            
            # Calculate baseline and future values with correct alignment
            # Window i uses data [i:i+window_size], so baseline is at index i+window_size-1
            # Target for horizon h is at index i+window_size-1+h
            baseline_indices = np.arange(window_size-1, window_size-1+num_samples)  # window_size-1, window_size, window_size+1, ...
            future_indices = baseline_indices + h  # window_size-1+h, window_size+h, window_size+1+h, ...
            
            # Ensure we have enough data for this horizon
            if future_indices[-1] >= len(target_trimmed):
                raise ValueError(f"Not enough target data for {split} split, horizon {h}: "
                               f"need index {future_indices[-1]}, have {len(target_trimmed)}")
            
            baseline_values = target_trimmed[baseline_indices]
            future_values = target_trimmed[future_indices]
            
            if use_returns:
                # Calculate returns: future[t+h] - baseline[t]
                returns_train = future_values - baseline_values
                
                # Calculate normalization stats from training data
                mean_h = returns_train.mean()
                std_h = returns_train.std() if returns_train.std() >= 1e-8 else 1.0
                
                self.target_returns_means.append(mean_h)
                self.target_returns_stds.append(std_h)
                
                print(f"  Normalizing H={h} with Mean={mean_h:.6f}, Std={std_h:.6f}")
            else:
                mean_h = 0.0
                std_h = 1.0
                self.target_returns_means.append(mean_h)
                self.target_returns_stds.append(std_h)
            
            # Now process all splits with the same normalization
            for split in splits:
                num_samples = windowed_data[f'num_samples_{split}']
                target_trimmed = denorm_targets[split]
                
                # Calculate baseline and future values with correct alignment
                baseline_indices = np.arange(window_size-1, window_size-1+num_samples)
                future_indices = baseline_indices + h
                
                # Ensure we have enough data
                if future_indices[-1] >= len(target_trimmed):
                    raise ValueError(f"Not enough target data for {split} split, horizon {h}: "
                                   f"need index {future_indices[-1]}, have {len(target_trimmed)}")
                
                baseline_values = target_trimmed[baseline_indices]
                future_values = target_trimmed[future_indices]
                
                if use_returns:
                    # Calculate returns: future[t+h] - baseline[t]
                    returns = future_values - baseline_values
                    # Normalize using training stats
                    target_normalized = (returns - mean_h) / std_h
                else:
                    target_normalized = future_values
                
                # Store target for this split and horizon
                horizon_key = f"output_horizon_{h}"
                if split not in target_data:
                    target_data[split] = {}
                target_data[split][horizon_key] = target_normalized.astype(np.float32)
                
                print(f"  {split.capitalize()}: {len(target_normalized)} samples")
        
        # Prepare baseline data (with correct alignment and bounds checking)
        baseline_info = {}
        for split in splits:
            num_samples = windowed_data[f'num_samples_{split}']
            
            # Baseline is the denormalized target column values at the end of each window
            # For window i, baseline is at index window_size-1+i
            baseline_indices = np.arange(window_size-1, window_size-1+num_samples)
            
            # Ensure baseline indices don't exceed the denormalized target array
            max_target_index = len(denorm_targets[split]) - 1
            valid_baseline_indices = baseline_indices[baseline_indices <= max_target_index]
            
            if len(valid_baseline_indices) < len(baseline_indices):
                print(f"WARN: Trimming baseline indices for {split}: {len(baseline_indices)} -> {len(valid_baseline_indices)}")
                # Update num_samples to reflect the actual available data
                actual_num_samples = len(valid_baseline_indices)
            else:
                actual_num_samples = num_samples
                
            baseline_values = denorm_targets[split][valid_baseline_indices]
            baseline_info[f'baseline_{split}'] = baseline_values
            
            # Corresponding dates (aligned with valid baseline indices)
            dates = baseline_dates[split]
            if dates is not None:
                # Ensure date indices don't exceed available dates
                max_date_index = len(dates) - 1
                valid_date_indices = valid_baseline_indices[valid_baseline_indices <= max_date_index]
                if len(valid_date_indices) < len(valid_baseline_indices):
                    print(f"WARN: Further trimming date indices for {split}: {len(valid_baseline_indices)} -> {len(valid_date_indices)}")
                    # Trim baseline values to match available dates
                    baseline_info[f'baseline_{split}'] = baseline_values[:len(valid_date_indices)]
                baseline_info[f'baseline_{split}_dates'] = dates[valid_date_indices]
            else:
                baseline_info[f'baseline_{split}_dates'] = None
        
        # Print normalization summary
        if use_returns:
            print("\nPer-horizon target normalization stats:")
            for i, (mean, std) in enumerate(zip(self.target_returns_means, self.target_returns_stds)):
                horizon = predicted_horizons[i]
                print(f"  Horizon {horizon}: Mean={mean:.6f}, Std={std:.6f}")
        else:
            print("Per-horizon normalization skipped (use_returns=False). Using Mean=0.0, Std=1.0 for all horizons.")
        
        # Combine target data with baseline info
        result = {
            'y_train': target_data.get('train', {}),
            'y_val': target_data.get('val', {}),
            'y_test': target_data.get('test', {}),
            **baseline_info,
            'target_returns_mean': self.target_returns_means,
            'target_returns_std': self.target_returns_stds,
        }
        
        # Add raw test data for evaluation (denormalized, full length)
        result['y_test_raw'] = denormalize(
            baseline_data['y_test_df'][target_column].astype(np.float32).values,
            norm_json, target_column
        )
        
        # Add test close prices for evaluation
        if 'x_test_df' in baseline_data:
            result['test_close_prices'] = denormalize(
                baseline_data['x_test_df']['CLOSE'].astype(np.float32).values[window_size:],
                norm_json, 'CLOSE'
            )[:windowed_data['num_samples_test']]
        else:
            # Fallback to using y_test_df if x_test_df is not available
            result['test_close_prices'] = denormalize(
                baseline_data['y_test_df'][target_column].astype(np.float32).values[window_size:],
                norm_json, target_column
            )[:windowed_data['num_samples_test']]
        
        print("Target calculation complete.")
        return result