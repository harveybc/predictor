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
            
            # CRITICAL FIX: Simple trimming - remove first window_size-1 rows
            # This aligns with windowed X data which also removes first window_size-1 rows
            window_size = config.get('window_size', 48)
            trimmed_start = window_size - 1
            target_trimmed = target_denorm[trimmed_start:]
            
            # Store the trimmed denormalized target (aligned with windowed features)
            denorm_targets[split] = target_trimmed
            baseline_targets[split] = target_trimmed
            
            # Use corresponding trimmed dates
            dates = baseline_data[f'dates_{split}']
            if dates is not None:
                dates_trimmed = dates[trimmed_start:]
                baseline_dates[split] = dates_trimmed
            else:
                baseline_dates[split] = None
            
            print(f"Trimmed {split} target: {len(target_trimmed)} samples (removed first {trimmed_start} rows)")
        
        # Now calculate targets for each horizon
        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")
        
        if use_returns:
            # UNNORMALIZED RETURNS: Use raw denormalized returns as targets
            print("\nCalculating UNNORMALIZED returns as targets...")
            
            # No normalization applied - targets will be raw denormalized returns
            target_mean = 0.0  # No centering
            target_std = 1.0   # No scaling
            
            print(f"Targets will be RAW denormalized returns (Mean={target_mean:.6f}, Std={target_std:.6f})")
            
            # Calculate sample statistics for reference only
            if 'train' in denorm_targets:
                train_target = denorm_targets['train']
                sample_baseline = train_target[:100] if len(train_target) > 100 else train_target
                sample_future = train_target[1:101] if len(train_target) > 101 else train_target[1:]
                if len(sample_future) == len(sample_baseline):
                    sample_returns = sample_future - sample_baseline
                    actual_mean = np.mean(sample_returns)
                    actual_std = np.std(sample_returns)
                    print(f"Sample denormalized returns statistics (for reference):")
                    print(f"  Mean: {actual_mean:.6f}")
                    print(f"  Std: {actual_std:.6f}")
        else:
            target_mean = 0.0
            target_std = 1.0
        
        # Use NO normalization for all horizons (raw returns)
        self.target_returns_means = [target_mean] * len(predicted_horizons)
        self.target_returns_stds = [target_std] * len(predicted_horizons)
        
        # Now process all horizons with the SAME normalization
        for i, h in enumerate(predicted_horizons):
            print(f"\nCalculating targets for horizon {h}...")
            
            mean_h = self.target_returns_means[i]
            std_h = self.target_returns_stds[i]
            
            # Now process all splits with the same normalization
            for split in splits:
                num_samples = windowed_data[f'num_samples_{split}']
                target_trimmed = denorm_targets[split]
                
                # CORRECT ALIGNMENT: After trimming, targets align with windows
                # Window i ends at tick (window_size-1+i) -> baseline at target_trimmed[i]
                # Window i should predict tick (window_size-1+i+h) -> future at target_trimmed[i+h]
                baseline_indices = np.arange(0, num_samples)
                future_indices = baseline_indices + h
                
                # Ensure we have enough trimmed data for future values
                if future_indices[-1] >= len(target_trimmed):
                    raise ValueError(f"Not enough trimmed target data for {split} split, horizon {h}: "
                                   f"need index {future_indices[-1]}, have {len(target_trimmed)}")
                
                baseline_values = target_trimmed[baseline_indices]
                future_values = target_trimmed[future_indices]
                
                # ALIGNMENT VERIFICATION: Print sample for first horizon and first split
                if i == 0 and split == 'train' and len(baseline_values) > 5:
                    print(f"\nALIGNMENT VERIFICATION for H{h} {split}:")
                    print(f"  Window 0: ends at tick {window_size-1}, baseline={baseline_values[0]:.6f} -> predicts tick {window_size-1+h}, future={future_values[0]:.6f}")
                    print(f"  Window 1: ends at tick {window_size}, baseline={baseline_values[1]:.6f} -> predicts tick {window_size+h}, future={future_values[1]:.6f}")
                    print(f"  Window 2: ends at tick {window_size+1}, baseline={baseline_values[2]:.6f} -> predicts tick {window_size+1+h}, future={future_values[2]:.6f}")
                    print(f"  ✅ CORRECT: Window ending at tick t predicts value at tick t+{h}")
                
                if use_returns:
                    # Calculate RAW returns: future[t+h] - baseline[t] (NO NORMALIZATION)
                    returns = future_values - baseline_values
                    # NO NORMALIZATION: Use raw denormalized returns as targets
                    target_normalized = returns.astype(np.float32)
                    
                    # SAMPLE VERIFICATION: Print sample raw targets
                    if i == 0 and split == 'train' and len(target_normalized) > 5:
                        print(f"  Sample RAW return targets: [{target_normalized[0]:.6f}, {target_normalized[1]:.6f}, {target_normalized[2]:.6f}]")
                        print(f"  Target processing: NO NORMALIZATION (raw denormalized returns)")
                        actual_mean = np.mean(target_normalized)
                        actual_std = np.std(target_normalized)
                        print(f"  Actual target stats: mean={actual_mean:.6f}, std={actual_std:.6f}")
                        print(f"  ✅ TARGETS ARE RAW DENORMALIZED RETURNS")
                else:
                    target_normalized = future_values
                
                # Store target for this split and horizon
                horizon_key = f"output_horizon_{h}"
                if split not in target_data:
                    target_data[split] = {}
                target_data[split][horizon_key] = target_normalized.astype(np.float32)
                
                print(f"  {split.capitalize()}: {len(target_normalized)} samples")
        
        # Prepare baseline data (UNNORMALIZED for direct prediction addition)
        baseline_info = {}
        for split in splits:
            num_samples = windowed_data[f'num_samples_{split}']
            target_trimmed = denorm_targets[split]
            
            # SIMPLE ALIGNMENT: After trimming, baselines align directly with windows
            # Window 0 -> baseline at trimmed_target[0]
            # Window 1 -> baseline at trimmed_target[1]
            # Window i -> baseline at trimmed_target[i]
            baseline_indices = np.arange(0, num_samples)
            
            # Ensure we don't exceed the trimmed target array
            if baseline_indices[-1] >= len(target_trimmed):
                print(f"WARN: Adjusting baseline indices for {split}: need {baseline_indices[-1]}, have {len(target_trimmed)}")
                actual_num_samples = len(target_trimmed)
                baseline_indices = np.arange(0, actual_num_samples)
            else:
                actual_num_samples = num_samples
                
            # BASELINE IS UNNORMALIZED (denormalized target values)
            baseline_values = target_trimmed[baseline_indices]
            baseline_info[f'baseline_{split}'] = baseline_values
            
            if split == 'train':
                print(f"  Baseline {split}: UNNORMALIZED values, mean={np.mean(baseline_values):.6f}, std={np.std(baseline_values):.6f}")
            
            # Corresponding dates (already trimmed)
            dates = baseline_dates[split]
            if dates is not None:
                # Use the same indices for dates (already trimmed to match)
                if len(dates) >= actual_num_samples:
                    baseline_info[f'baseline_{split}_dates'] = dates[baseline_indices]
                else:
                    print(f"WARN: Date array too short for {split}: need {actual_num_samples}, have {len(dates)}")
                    baseline_info[f'baseline_{split}_dates'] = dates[:actual_num_samples] 
            else:
                baseline_info[f'baseline_{split}_dates'] = None
        
        # Print processing summary
        if use_returns:
            print("\nTarget processing summary:")
            print("  ✅ TARGETS: Raw denormalized returns (NO normalization)")
            print("  ✅ BASELINE: Unnormalized denormalized values")
            print("  ✅ PREDICTION: Add predicted returns to baseline for final prediction")
            print("  Target means (all 0.0):", self.target_returns_means)
            print("  Target stds (all 1.0):", self.target_returns_stds)
        else:
            print("Target normalization skipped (use_returns=False). Using raw values.")
        
        # Combine target data with baseline info and processing parameters
        result = {
            'y_train': target_data.get('train', {}),
            'y_val': target_data.get('val', {}),
            'y_test': target_data.get('test', {}),
            **baseline_info,
            'target_returns_means': self.target_returns_means,  # All 0.0 (no normalization)
            'target_returns_stds': self.target_returns_stds,    # All 1.0 (no normalization)
            'predicted_horizons': predicted_horizons,           # For reference
            'use_unnormalized_targets': True,                   # Flag for prediction processing
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