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
        # USE SLIDING WINDOW BASELINES (already denormalized from sliding windows)
        denorm_targets = {}
        baseline_targets = {}
        baseline_dates = {}
        
        print("\n✅ USING SLIDING WINDOW BASELINES (pre-calculated and denormalized)")
        
        for split in splits:
            print(f"\nProcessing {split} targets...")
            
            # Use pre-calculated sliding window baselines (already denormalized)
            sliding_baseline_key = f'sliding_baseline_{split}'
            sliding_dates_key = f'sliding_baseline_{split}_dates'
            
            if sliding_baseline_key not in baseline_data:
                raise ValueError(f"Sliding window baseline not found for {split}. Check sliding window calculation.")
            
            # Get denormalized baselines from sliding windows
            sliding_baselines = baseline_data[sliding_baseline_key]
            sliding_dates = baseline_data[sliding_dates_key]
            
            if len(sliding_baselines) == 0:
                print(f"WARN: No sliding window baselines available for {split}")
                denorm_targets[split] = np.array([])
                baseline_targets[split] = np.array([])
                baseline_dates[split] = np.array([])
                continue
            
            # Store the sliding window baselines (already properly aligned and denormalized)
            denorm_targets[split] = sliding_baselines  # These are the baseline values for each window
            baseline_targets[split] = sliding_baselines
            baseline_dates[split] = sliding_dates
            
            print(f"✅ {split}: {len(sliding_baselines)} sliding window baselines loaded")
            print(f"    Baseline sample: {sliding_baselines[:3]}")
            print(f"    Will calculate targets from sliding window data only")
        
        # NOW CALCULATE FUTURE VALUES FROM SLIDING WINDOW BASELINES ONLY
        # We calculate targets entirely from the sliding window dataset
        
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
                sliding_baselines = denorm_targets[split]  # These are the window baseline values
                
                if len(sliding_baselines) == 0:
                    print(f"WARN: No baselines available for {split}")
                    continue
                
                num_samples = len(sliding_baselines)  # Use actual sliding window count
                
                # CRITICAL: Calculate future values from SLIDING WINDOW BASELINES ONLY
                # For horizon h, future value for window i is the baseline value of window i+h
                # Window i baseline = sliding_baselines[i]
                # Window i should predict = sliding_baselines[i+h] (if it exists)
                
                # Check if we have enough sliding window data for this horizon
                max_valid_samples = num_samples - h
                if max_valid_samples <= 0:
                    print(f"WARN: Insufficient sliding window data for H={h} {split}: need {h} extra windows")
                    continue
                
                # Truncate to valid samples for this horizon
                baseline_values = sliding_baselines[:max_valid_samples]
                future_values = sliding_baselines[h:h+max_valid_samples]  # Future baselines from sliding windows
                
                # Verify alignment
                if len(baseline_values) != len(future_values):
                    print(f"ERROR: Alignment mismatch for H={h} {split}: baseline={len(baseline_values)}, future={len(future_values)}")
                    continue
                
                # ALIGNMENT VERIFICATION: Print sample for first horizon and first split
                if i == 0 and split == 'train' and len(baseline_values) > 5:
                    print(f"\nALIGNMENT VERIFICATION for H{h} {split} (SLIDING WINDOW ONLY):")
                    print(f"  Window 0: baseline=sliding_baselines[0]={baseline_values[0]:.6f} -> predicts sliding_baselines[{h}]={future_values[0]:.6f}")
                    print(f"  Window 1: baseline=sliding_baselines[1]={baseline_values[1]:.6f} -> predicts sliding_baselines[{1+h}]={future_values[1]:.6f}")
                    print(f"  Window 2: baseline=sliding_baselines[2]={baseline_values[2]:.6f} -> predicts sliding_baselines[{2+h}]={future_values[2]:.6f}")
                    print(f"  ✅ CORRECT: Using ONLY sliding window baselines for targets")
                
                if use_returns:
                    # Calculate RAW returns from sliding window baselines: future_baseline[i+h] - baseline[i]
                    returns = future_values - baseline_values
                    # NO NORMALIZATION: Use raw denormalized returns as targets
                    target_normalized = returns.astype(np.float32)
                    
                    # SAMPLE VERIFICATION: Print sample raw targets
                    if i == 0 and split == 'train' and len(target_normalized) > 5:
                        print(f"  Sample RAW return targets from sliding windows: [{target_normalized[0]:.6f}, {target_normalized[1]:.6f}, {target_normalized[2]:.6f}]")
                        print(f"  Target processing: NO NORMALIZATION (raw denormalized returns from sliding windows)")
                        actual_mean = np.mean(target_normalized)
                        actual_std = np.std(target_normalized)
                        print(f"  Actual target stats: mean={actual_mean:.6f}, std={actual_std:.6f}")
                        print(f"  ✅ TARGETS CALCULATED FROM SLIDING WINDOW BASELINES ONLY")
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
            sliding_baselines = denorm_targets[split]  # These are already the correct baselines
            
            if len(sliding_baselines) == 0:
                print(f"WARN: No sliding window baselines for {split}")
                baseline_info[f'baseline_{split}'] = np.array([])
                baseline_info[f'baseline_{split}_dates'] = np.array([])
                continue
            
            # Use sliding window baselines directly (already denormalized and aligned)
            baseline_info[f'baseline_{split}'] = sliding_baselines
            
            if split == 'train':
                print(f"  Baseline {split}: UNNORMALIZED sliding window values, mean={np.mean(sliding_baselines):.6f}, std={np.std(sliding_baselines):.6f}")
            
            # Use corresponding sliding window dates
            sliding_dates_key = f'sliding_baseline_{split}_dates'
            if sliding_dates_key in baseline_data:
                baseline_info[f'baseline_{split}_dates'] = baseline_data[sliding_dates_key]
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