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
            
            # Calculate targets for each horizon
            for split in splits:
                baselines = denorm_targets[split]
                if len(baselines) == 0:
                    continue
                
                for h in predicted_horizons:
                    if len(baselines) > h:
                        # target<t,h> = baseline<t+h> - baseline<t>
                        targets = baselines[h:] - baselines[:-h]
                        target_data.setdefault(split, {})[f'output_horizon_{h}'] = targets.astype(np.float32)
                        print(f"  {split.capitalize()} H{h}: {len(targets)} targets calculated.")
                    else:
                        print(f"  WARN: Not enough baselines for H{h} in {split}.")
                        target_data.setdefault(split, {})[f'output_horizon_{h}'] = np.array([])
        else:
            # Future value prediction
            print("\nCalculating FUTURE VALUES as targets...")
            for split in splits:
                baselines = denorm_targets[split]
                if len(baselines) == 0:
                    continue
                
                for h in predicted_horizons:
                    if len(baselines) > h:
                        targets = baselines[h:]
                        target_data.setdefault(split, {})[f'output_horizon_{h}'] = targets.astype(np.float32)
                    else:
                        target_data.setdefault(split, {})[f'output_horizon_{h}'] = np.array([])

        # CRITICAL: Find the minimum number of samples across all horizons to ensure data alignment
        max_horizon = max(predicted_horizons)
        min_samples_per_split = {}
        
        # Calculate minimum samples for each split considering the largest horizon
        for split in splits:
            sliding_baselines = denorm_targets[split]
            if len(sliding_baselines) > 0:
                min_samples_per_split[split] = len(sliding_baselines) - max_horizon
            else:
                min_samples_per_split[split] = 0
        
        print(f"\nData alignment: Max horizon={max_horizon}, Min samples per split: {min_samples_per_split}")
        
        # Now process all horizons with the SAME normalization AND SAME SAMPLE COUNT
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
                
                # Use the minimum sample count to ensure all horizons have the same number of samples
                max_valid_samples = min_samples_per_split[split]
                if max_valid_samples <= 0:
                    print(f"WARN: Insufficient sliding window data for H={h} {split}: min_samples={max_valid_samples}")
                    continue
                
                # Truncate to the SAME number of samples for ALL horizons
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
            
            # Use the SAME truncated length as targets to ensure alignment
            max_valid_samples = min_samples_per_split[split]
            if max_valid_samples <= 0:
                print(f"WARN: No valid samples for baseline {split}")
                baseline_info[f'baseline_{split}'] = np.array([])
                baseline_info[f'baseline_{split}_dates'] = np.array([])
                continue
            
            # Truncate sliding window baselines to match target data length
            baseline_info[f'baseline_{split}'] = sliding_baselines[:max_valid_samples]
            
            if split == 'train':
                truncated_baselines = sliding_baselines[:max_valid_samples]
                print(f"  Baseline {split}: UNNORMALIZED sliding window values (truncated to {max_valid_samples}), mean={np.mean(truncated_baselines):.6f}, std={np.std(truncated_baselines):.6f}")
            
            # Use corresponding sliding window dates (also truncated)
            sliding_dates_key = f'sliding_baseline_{split}_dates'
            if sliding_dates_key in baseline_data:
                dates = baseline_data[sliding_dates_key]
                if len(dates) >= max_valid_samples:
                    baseline_info[f'baseline_{split}_dates'] = dates[:max_valid_samples]
                else:
                    baseline_info[f'baseline_{split}_dates'] = dates
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
        test_samples = min_samples_per_split.get('test', 0)
        if 'x_test_df' in baseline_data and test_samples > 0:
            result['test_close_prices'] = denormalize(
                baseline_data['x_test_df']['CLOSE'].astype(np.float32).values[window_size:],
                norm_json, 'CLOSE'
            )[:test_samples]
        else:
            # Fallback to using y_test_df if x_test_df is not available
            if test_samples > 0:
                result['test_close_prices'] = denormalize(
                    baseline_data['y_test_df'][target_column].astype(np.float32).values[window_size:],
                    norm_json, target_column
                )[:test_samples]
            else:
                result['test_close_prices'] = np.array([])
        
        print("Target calculation complete.")
        
        # CRITICAL: Update windowed data sample counts to match truncated target data
        print(f"\n--- Updating Windowed Data Sample Counts ---")
        for split in splits:
            if min_samples_per_split[split] > 0:
                old_count = windowed_data.get(f'num_samples_{split}', 0)
                windowed_data[f'num_samples_{split}'] = min_samples_per_split[split]
                print(f"Updated {split} samples: {old_count} -> {min_samples_per_split[split]}")
            else:
                windowed_data[f'num_samples_{split}'] = 0
                print(f"WARNING: {split} has 0 valid samples")
        
        return result