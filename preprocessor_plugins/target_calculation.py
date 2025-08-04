import numpy as np
import pandas as pd
from .helpers import denormalize, load_normalization_json


class TargetCalculationProcessor:
    """
    Handles baseline extraction from sliding windows matrix, target calculation, and date alignment.
    """
    
    def __init__(self):
        self.target_returns_means = []
        self.target_returns_stds = []
    
    def calculate_sliding_window_baselines(self, windowed_data, data_splits, config):
        """
        Calculate sliding window baselines ONLY from the sliding windows matrix.
        
        CRITICAL FLOW:
        1. Extract baselines ONLY from sliding windows matrix - last value of each window for target_column
        2. Denormalize baselines using JSON parameters (ISOLATED from input data)
        3. This denormalized baseline is used for target calculation and final predictions
        4. Extract dates from original CSV and trim to match sliding windows
        
        Args:
            windowed_data: Dict containing sliding windows matrix (X_train, X_val, X_test) and feature_names
            data_splits: Dict containing original CSV data for date extraction ONLY
            config: Configuration dictionary
            
        Returns:
            Dict containing DENORMALIZED baselines and dates for all splits
        """
        print("\n--- Extracting Baselines from Sliding Windows Matrix ONLY ---")
        
        window_size = config.get("window_size", 48)
        target_column = config.get("target_column", "CLOSE") 
        
        # Load normalization JSON for denormalization
        norm_json = load_normalization_json(config)
        
        # Get feature names from windowed data
        feature_names = windowed_data.get('feature_names', [])
        
        # Find the index of the target column in the feature matrix
        if target_column not in feature_names:
            raise ValueError(f"Target column '{target_column}' not found in windowed features: {feature_names}")
        
        target_feature_index = feature_names.index(target_column)
        print(f"✅ Target column '{target_column}' found at feature index {target_feature_index}")
        
        result_data = {}
        
        splits = ['train', 'val', 'test']
        for split in splits:
            print(f"\n📊 Processing {split} split...")
            
            # Get sliding windows matrix for this split
            X_key = f'X_{split}'
            if X_key not in windowed_data:
                print(f"❌ WARNING: No sliding windows data found for {split}")
                result_data[f'sliding_baseline_{split}'] = np.array([])
                result_data[f'sliding_baseline_{split}_dates'] = np.array([])
                continue
            
            # Get sliding windows matrix: shape (num_windows, window_size, num_features)
            X_matrix = windowed_data[X_key]
            print(f"  📈 Sliding windows matrix shape: {X_matrix.shape}")
            
            if X_matrix.shape[0] == 0:
                print(f"  ❌ WARNING: Empty sliding windows matrix for {split}")
                result_data[f'sliding_baseline_{split}'] = np.array([])
                result_data[f'sliding_baseline_{split}_dates'] = np.array([])
                continue
            
            # 🔑 STEP 1: Extract baselines DIRECTLY from sliding windows matrix
            # CRITICAL: Use ONLY the sliding windows matrix, isolated from input data
            print(f"  🎯 Extracting baselines from sliding windows matrix (ISOLATED)...")
            target_windows = X_matrix[:, :, target_feature_index]  # Shape: (num_windows, window_size)
            baselines_normalized = target_windows[:, -1]  # Last value of each window: (num_windows,)
            
            print(f"    ✅ Extracted {len(baselines_normalized)} normalized baselines from windows")
            print(f"    📊 Baseline normalized stats: mean={np.mean(baselines_normalized):.6f}, std={np.std(baselines_normalized):.6f}")
            
            # 🔑 STEP 2: Denormalize baselines using JSON parameters (CRITICAL)
            print(f"  🔄 Denormalizing baselines using {target_column} parameters...")
            
            # STRICT NaN CHECK: Normalized baselines must NEVER have NaN
            nan_count_normalized = np.sum(np.isnan(baselines_normalized))
            if nan_count_normalized > 0:
                print(f"❌ FATAL ERROR: Found {nan_count_normalized} NaN values in NORMALIZED baselines from sliding windows!")
                print(f"    Sliding windows matrix shape: {X_matrix.shape}")
                print(f"    Target feature index: {target_feature_index}")
                print(f"    Sample normalized baselines: {baselines_normalized[:10]}")
                raise ValueError(f"CRITICAL: NaN values detected in normalized baselines from sliding windows matrix - this should NEVER happen!")
            
            baselines_denormalized = denormalize(baselines_normalized, norm_json, target_column)
            
            # STRICT NaN CHECK: Denormalized baselines must NEVER have NaN
            nan_count_denormalized = np.sum(np.isnan(baselines_denormalized))
            if nan_count_denormalized > 0:
                print(f"❌ FATAL ERROR: Found {nan_count_denormalized} NaN values in DENORMALIZED baselines!")
                print(f"    Normalized baselines sample: {baselines_normalized[:10]}")
                print(f"    Denormalized baselines sample: {baselines_denormalized[:10]}")
                print(f"    Normalization JSON for {target_column}: {norm_json.get(target_column, 'NOT FOUND')}")
                raise ValueError(f"CRITICAL: NaN values created during denormalization - check normalization parameters!")
            
            # STRICT ZERO CHECK: Denormalized baselines must NEVER be zero (for log returns)
            zero_count = np.sum(baselines_denormalized == 0)
            if zero_count > 0:
                print(f"❌ FATAL ERROR: Found {zero_count} ZERO values in denormalized baselines!")
                print(f"    Sample denormalized baselines: {baselines_denormalized[:20]}")
                zero_indices = np.where(baselines_denormalized == 0)[0]
                print(f"    Zero indices: {zero_indices[:10]}")
                for idx in zero_indices[:5]:
                    print(f"      Index {idx}: normalized={baselines_normalized[idx]}, denormalized={baselines_denormalized[idx]}")
                raise ValueError(f"CRITICAL: Zero values in denormalized baselines will cause log(0) = -inf in log returns!")
                
            print(f"    ✅ Baseline denormalized stats: mean={np.mean(baselines_denormalized):.6f}, std={np.std(baselines_denormalized):.6f}")
            
            # 🔑 STEP 3: Extract dates from original CSV ONLY for alignment (not for data)
            print(f"  📅 Extracting and aligning dates...")
            df_key = f'x_{split}_df'
            if df_key in data_splits:
                dates_df = data_splits[df_key]
                if isinstance(dates_df.index, pd.DatetimeIndex) and len(dates_df) > window_size:
                    # CRITICAL: Trim first (window_size-1) dates to align with sliding windows
                    # First window baseline is at position window_size-1 in original data
                    dates_raw = dates_df.index[window_size-1:]
                    # Further trim to match baselines length
                    aligned_dates = dates_raw[:len(baselines_denormalized)] if len(dates_raw) >= len(baselines_denormalized) else dates_raw
                    print(f"    ✅ Aligned {len(aligned_dates)} dates with baselines (trimmed first {window_size-1} rows)")
                else:
                    print(f"    ❌ WARNING: No valid dates found for {split}")
                    aligned_dates = None
            else:
                print(f"    ❌ WARNING: No data split found for {split}")
                aligned_dates = None
            
            # 🔑 STEP 4: Store DENORMALIZED results (ready for target calculation)
            result_data[f'sliding_baseline_{split}'] = baselines_denormalized.astype(np.float32)
            result_data[f'sliding_baseline_{split}_dates'] = aligned_dates
            
            print(f"  ✅ {split}: {len(baselines_denormalized)} DENORMALIZED baselines ready for target calculation")
        
        print("✅ Sliding windows baseline extraction complete - ISOLATED from input data")
        return result_data
    
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
        print(f"Processing targets for horizons: {predicted_horizons} (Use Log Returns={use_returns})...")
        
        if use_returns:
            # UNNORMALIZED LOG RETURNS: Use raw denormalized log returns as targets
            print("\nCalculating UNNORMALIZED log returns as targets...")
            
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
        
        # Initialize normalization stats for all horizons (unnormalized: mean=0, std=1)
        if use_returns:
            # For unnormalized returns, all horizons have mean=0.0, std=1.0
            self.target_returns_means = [0.0] * len(predicted_horizons)
            self.target_returns_stds = [1.0] * len(predicted_horizons)
        
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
                
                # STRICT NaN CHECK: baseline_values must NEVER have NaN
                nan_count_baseline = np.sum(np.isnan(baseline_values))
                if nan_count_baseline > 0:
                    print(f"❌ FATAL ERROR: Found {nan_count_baseline} NaN values in baseline_values for H{h} {split}!")
                    print(f"    sliding_baselines sample: {sliding_baselines[:10]}")
                    print(f"    baseline_values sample: {baseline_values[:10]}")
                    print(f"    max_valid_samples: {max_valid_samples}")
                    raise ValueError(f"CRITICAL: NaN values in baseline_values - data corruption detected!")
                
                # STRICT NaN CHECK: future_values must NEVER have NaN
                nan_count_future = np.sum(np.isnan(future_values))
                if nan_count_future > 0:
                    print(f"❌ FATAL ERROR: Found {nan_count_future} NaN values in future_values for H{h} {split}!")
                    print(f"    sliding_baselines sample: {sliding_baselines[:10]}")
                    print(f"    future_values sample: {future_values[:10]}")
                    print(f"    horizon offset: {h}, max_valid_samples: {max_valid_samples}")
                    raise ValueError(f"CRITICAL: NaN values in future_values - data corruption detected!")
                
                # STRICT ZERO CHECK: baseline_values must NEVER be zero (for log returns)
                zero_count_baseline = np.sum(baseline_values == 0)
                if zero_count_baseline > 0:
                    print(f"❌ FATAL ERROR: Found {zero_count_baseline} ZERO values in baseline_values for H{h} {split}!")
                    print(f"    baseline_values sample: {baseline_values[:20]}")
                    zero_indices = np.where(baseline_values == 0)[0]
                    print(f"    Zero indices: {zero_indices[:10]}")
                    raise ValueError(f"CRITICAL: Zero values in baseline_values will cause division by zero!")
                
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
                    # Calculate log returns: ln(future_baseline[i+h]/baseline[i])
                    ratio = future_values / baseline_values
                    
                    # STRICT NaN CHECK: ratio must NEVER have NaN
                    nan_count_ratio = np.sum(np.isnan(ratio))
                    if nan_count_ratio > 0:
                        print(f"❌ FATAL ERROR: Found {nan_count_ratio} NaN values in ratio calculation for H{h} {split}!")
                        print(f"    baseline_values shape: {baseline_values.shape}, dtype: {baseline_values.dtype}")
                        print(f"    future_values shape: {future_values.shape}, dtype: {future_values.dtype}")
                        print(f"    Sample baseline_values: {baseline_values[:10]}")
                        print(f"    Sample future_values: {future_values[:10]}")
                        print(f"    Sample ratios: {ratio[:10]}")
                        nan_indices = np.where(np.isnan(ratio))[0]
                        print(f"    NaN ratio indices: {nan_indices[:10]}")
                        for idx in nan_indices[:5]:
                            print(f"      Index {idx}: baseline={baseline_values[idx]}, future={future_values[idx]}, ratio={ratio[idx]}")
                        raise ValueError(f"CRITICAL: NaN values in ratio calculation - this indicates data corruption!")
                    
                    # STRICT INF CHECK: ratio must NEVER have infinite values
                    inf_count_ratio = np.sum(np.isinf(ratio))
                    if inf_count_ratio > 0:
                        print(f"❌ FATAL ERROR: Found {inf_count_ratio} INFINITE values in ratio calculation for H{h} {split}!")
                        print(f"    Sample ratios: {ratio[:10]}")
                        inf_indices = np.where(np.isinf(ratio))[0]
                        print(f"    Infinite ratio indices: {inf_indices[:10]}")
                        for idx in inf_indices[:5]:
                            print(f"      Index {idx}: baseline={baseline_values[idx]}, future={future_values[idx]}, ratio={ratio[idx]}")
                        raise ValueError(f"CRITICAL: Infinite values in ratio calculation - this indicates division by zero!")
                    
                    # STRICT NEGATIVE/ZERO CHECK: ratio must NEVER be <= 0 (for log returns)
                    negative_zero_count = np.sum(ratio <= 0)
                    if negative_zero_count > 0:
                        print(f"❌ FATAL ERROR: Found {negative_zero_count} NEGATIVE/ZERO values in ratio calculation for H{h} {split}!")
                        print(f"    Sample ratios: {ratio[:10]}")
                        bad_indices = np.where(ratio <= 0)[0]
                        print(f"    Negative/zero ratio indices: {bad_indices[:10]}")
                        for idx in bad_indices[:5]:
                            print(f"      Index {idx}: baseline={baseline_values[idx]}, future={future_values[idx]}, ratio={ratio[idx]}")
                        raise ValueError(f"CRITICAL: Negative or zero ratios will cause log(<=0) = NaN or -inf!")
                    
                    log_returns = np.log(ratio)
                    
                    # STRICT NaN CHECK: log_returns must NEVER have NaN after calculation
                    nan_count_log = np.sum(np.isnan(log_returns))
                    if nan_count_log > 0:
                        print(f"❌ FATAL ERROR: Found {nan_count_log} NaN values in log_returns for H{h} {split}!")
                        print(f"    Sample log_returns: {log_returns[:10]}")
                        nan_indices = np.where(np.isnan(log_returns))[0]
                        print(f"    NaN log_returns indices: {nan_indices[:10]}")
                        for idx in nan_indices[:5]:
                            print(f"      Index {idx}: baseline={baseline_values[idx]}, future={future_values[idx]}, ratio={ratio[idx]}, log={log_returns[idx]}")
                        raise ValueError(f"CRITICAL: NaN values in log_returns - mathematical error detected!")
                    
                    target_normalized = log_returns.astype(np.float32)
                    
                    # FINAL STRICT NaN CHECK: target_normalized must NEVER have NaN
                    nan_count_final = np.sum(np.isnan(target_normalized))
                    if nan_count_final > 0:
                        print(f"❌ FATAL ERROR: Found {nan_count_final} NaN values in final target_normalized for H{h} {split}!")
                        print(f"    Sample target_normalized: {target_normalized[:10]}")
                        raise ValueError(f"CRITICAL: NaN values in final targets - this should be impossible at this point!")
                    
                    # SAMPLE VERIFICATION: Print sample raw targets
                    if i == 0 and split == 'train' and len(target_normalized) > 5:
                        actual_mean = np.mean(target_normalized)
                        actual_std = np.std(target_normalized)
                        print(f"  Sample RAW return targets from sliding windows: [{target_normalized[0]:.6f}, {target_normalized[1]:.6f}, {target_normalized[2]:.6f}]")
                        print(f"  Target processing: NO NORMALIZATION (raw denormalized returns from sliding windows)")
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