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
        
        # First, prepare denormalized target columns for each split
        denorm_targets = {}
        baseline_targets = {}
        baseline_dates = {}
        
        for split in splits:
            print(f"\nProcessing {split} targets...")
            
            # Get the target column from baseline
            if split == 'train':
                y_df = baseline_data['y_train_df']
            elif split == 'val':
                y_df = baseline_data['y_val_df']
            else:  # test
                y_df = baseline_data['y_test_df']
            
            # Extract and denormalize target column
            target_raw = y_df[target_column].astype(np.float32).values
            target_denorm = denormalize(target_raw, norm_json, target_column)
            
            # Trim first window_size-1 elements to align with sliding windows
            # This ensures we start from the same point as the windowed features
            target_trimmed = target_denorm[window_size-1:]
            
            # Get corresponding dates (also trimmed)
            dates = baseline_data[f'dates_{split}']
            dates_trimmed = dates[window_size-1:] if dates is not None else None
            
            # Store for target calculation
            denorm_targets[split] = target_trimmed
            
            # For baseline return (this will be further trimmed to match actual samples)
            baseline_targets[split] = target_trimmed
            baseline_dates[split] = dates_trimmed
            
            print(f"Denormalized and trimmed {split} target: {len(target_trimmed)} samples")
        
        # Now calculate targets for each horizon
        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")
        
        for i, h in enumerate(predicted_horizons):
            print(f"\nCalculating targets for horizon {h}...")
            
            # Calculate returns for training split first to get normalization stats
            split = 'train'
            num_samples = windowed_data[f'num_samples_{split}']
            target_trimmed = denorm_targets[split]
            
            # Ensure we have enough data for this horizon
            if len(target_trimmed) < num_samples + h:
                raise ValueError(f"Not enough target data for {split} split, horizon {h}: "
                               f"need {num_samples + h}, have {len(target_trimmed)}")
            
            # Calculate baseline and future values
            baseline_values = target_trimmed[:num_samples]  # t=0 to t=num_samples-1
            future_values = target_trimmed[h:num_samples + h]  # t=h to t=num_samples+h-1
            
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
                
                # Ensure we have enough data
                if len(target_trimmed) < num_samples + h:
                    raise ValueError(f"Not enough target data for {split} split, horizon {h}: "
                                   f"need {num_samples + h}, have {len(target_trimmed)}")
                
                # Calculate baseline and future values
                baseline_values = target_trimmed[:num_samples]
                future_values = target_trimmed[h:num_samples + h]
                
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
        
        # Prepare baseline data (trimmed to match actual samples)
        baseline_info = {}
        for split in splits:
            num_samples = windowed_data[f'num_samples_{split}']
            
            # Baseline is the denormalized target column values at t=0 for each sample
            # This corresponds to the "current" price when making predictions
            baseline_values = denorm_targets[split][:num_samples]
            baseline_info[f'baseline_{split}'] = baseline_values
            
            # Corresponding dates
            dates_trimmed = baseline_dates[split]
            if dates_trimmed is not None:
                baseline_info[f'baseline_{split}_dates'] = dates_trimmed[:num_samples]
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
        result['test_close_prices'] = denormalize(
            baseline_data['x_test_df']['CLOSE'].astype(np.float32).values[window_size:],
            norm_json, 'CLOSE'
        )[:windowed_data['num_samples_test']]
        
        print("Target calculation complete.")
        return result