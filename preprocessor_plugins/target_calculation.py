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
        
        if use_returns:
            # Calculate GLOBAL normalization stats from ALL returns across ALL horizons
            print("\nCalculating global normalization stats from all horizons...")
            all_returns = []
            
            for h in predicted_horizons:
                # Calculate returns for training split to collect all return values
                split = 'train'
                num_samples = windowed_data[f'num_samples_{split}']
                target_trimmed = denorm_targets[split]
                
                # CRITICAL FIX: Use same baseline calculation as in main loop
                baseline_indices = np.arange(window_size-1, window_size-1+num_samples)
                future_indices = baseline_indices + h
                
                # Ensure we have enough data
                max_required_index = max(baseline_indices[-1], future_indices[-1])
                if max_required_index >= len(target_trimmed):
                    raise ValueError(f"Not enough target data for {split} split, horizon {h}: "
                                   f"need index {max_required_index}, have {len(target_trimmed)}")
                
                baseline_values = target_trimmed[baseline_indices]
                future_values = target_trimmed[future_indices]
                returns = future_values - baseline_values
                all_returns.extend(returns)
            
            # Calculate single global normalization stats
            all_returns = np.array(all_returns)
            global_mean = all_returns.mean()
            global_std = all_returns.std() if all_returns.std() >= 1e-8 else 1.0
            
            print(f"Global normalization stats: Mean={global_mean:.6f}, Std={global_std:.6f}")
            print(f"Global return range: [{all_returns.min():.6f}, {all_returns.max():.6f}]")
            
            # Use the same normalization stats for ALL horizons
            self.target_returns_means = [global_mean] * len(predicted_horizons)
            self.target_returns_stds = [global_std] * len(predicted_horizons)
        else:
            self.target_returns_means = [0.0] * len(predicted_horizons)
            self.target_returns_stds = [1.0] * len(predicted_horizons)
        
        # Now process all horizons with the SAME normalization
        for i, h in enumerate(predicted_horizons):
            print(f"\nCalculating targets for horizon {h}...")
            
            mean_h = self.target_returns_means[i]
            std_h = self.target_returns_stds[i]
            
            # Now process all splits with the same normalization
            for split in splits:
                num_samples = windowed_data[f'num_samples_{split}']
                target_trimmed = denorm_targets[split]
                
                # CRITICAL FIX: Calculate baseline and future indices with correct alignment
                # Sliding windows now start at t=window_size-1, so baseline indices are:
                # Window 0: baseline at index window_size-1
                # Window 1: baseline at index window_size  
                # Window i: baseline at index window_size-1+i
                baseline_indices = np.arange(window_size-1, window_size-1+num_samples)
                future_indices = baseline_indices + h
                
                # Ensure we have enough data for both baseline and future values
                max_required_index = max(baseline_indices[-1], future_indices[-1])
                if max_required_index >= len(target_trimmed):
                    raise ValueError(f"Not enough target data for {split} split, horizon {h}: "
                                   f"need index {max_required_index}, have {len(target_trimmed)}")
                
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
            
            # CRITICAL FIX: Baseline indices must match the sliding windows exactly
            # Each window ends at baseline_index, so window i has baseline at window_size-1+i
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
                # For baseline dates, we need the dates corresponding to the baseline indices
                # These are the dates at the END of each window (current tick)
                valid_date_indices = valid_baseline_indices
                max_date_index = len(dates) - 1
                final_date_indices = valid_date_indices[valid_date_indices <= max_date_index]
                if len(final_date_indices) < len(valid_baseline_indices):
                    print(f"WARN: Further trimming date indices for {split}: {len(valid_baseline_indices)} -> {len(final_date_indices)}")
                    # Trim baseline values to match available dates
                    baseline_info[f'baseline_{split}'] = baseline_values[:len(final_date_indices)]
                baseline_info[f'baseline_{split}_dates'] = dates[final_date_indices]
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