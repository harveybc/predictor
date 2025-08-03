import numpy as np
import pandas as pd
from .helpers import normalize_series


class SlidingWindowsProcessor:
    """
    Handles the creation of sliding windows from normalized baseline data.
    """
    
    def __init__(self, scalers=None):
        self.scalers = scalers or {}
    
    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        """
        Creates sliding windows from data with proper temporal alignment.
        
        CRITICAL: First window starts at t=window_size to ensure complete windows only.
        Each window contains data[t-window_size+1:t+1] where t is the current tick.
        The baseline for each window is the LAST value in the window (data[t]).
        
        Args:
            data: 1D array of values
            window_size: Size of each window  
            time_horizon: Maximum time horizon for predictions (affects available windows)
            date_times: Optional datetime index for the data
            
        Returns:
            Tuple of (windows_array, date_windows_array)
        """
        print(f"Creating sliding windows (Size={window_size}, Horizon={time_horizon})...", end="")
        windows = []
        date_windows = []
        n = len(data)
        
        # CRITICAL FIX: Calculate available windows correctly
        # We need at least window_size + time_horizon data points
        # First window starts at index window_size-1 (0-based), last usable data at n-time_horizon-1
        max_start_index = n - time_horizon - 1  # Last index where we can place a window baseline
        min_start_index = window_size - 1      # First index where we can have a complete window
        
        if max_start_index < min_start_index:
            print(f" WARN: Insufficient data ({n}) for Win={window_size}+Horizon={time_horizon}. Need at least {window_size + time_horizon}.")
            return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object)
        
        num_possible_windows = max_start_index - min_start_index + 1
        
        # Create windows: each window ends at current tick t, contains [t-window_size+1:t+1]
        for t in range(min_start_index, max_start_index + 1):
            window_start = t - window_size + 1
            window_end = t + 1
            window = data[window_start:window_end]
            windows.append(window)
            
            # Date corresponds to the current tick (end of window) - this is the baseline time
            if date_times is not None:
                if t < len(date_times): 
                    date_windows.append(date_times[t])
                else: 
                    date_windows.append(None)
        
        if date_times is not None:
            date_windows_arr = np.array(date_windows, dtype=object)
            # CRITICAL FIX: Don't convert to numpy.datetime64 - keep as pandas Timestamp objects for matplotlib compatibility
            # The numpy.datetime64 format with gaps causes matplotlib plotting issues
            if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None):
                print(f"  Keeping {len(date_windows)} dates as pandas Timestamp objects for plotting compatibility")
                # Keep as object array with pandas Timestamps - matplotlib handles these better
                pass  # date_windows_arr already set as object array above
            else:
                print(f"  Warning: Some dates are not pandas Timestamps: {[type(d) for d in date_windows[:3]]}")
        else: 
            date_windows_arr = np.array(date_windows, dtype=object)
        
        print(f" Done ({len(windows)} windows).")
        return np.array(windows, dtype=np.float32), date_windows_arr
    
    def generate_windowed_features(self, baseline_data, config):
        """
        Generates windowed features from baseline data for all splits.
        
        Args:
            baseline_data: Dict containing baseline data for all splits
            config: Configuration dictionary
            
        Returns:
            Dict containing windowed features and metadata
        """
        print("\n--- Generating Windowed Features ---")
        
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        max_horizon = max(predicted_horizons)
        normalize_features = config.get("normalize_features", True)
        
        # Extract data for each split
        splits = ['train', 'val', 'test']
        windowed_data = {}
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            # Get original normalized dataframe - this contains ALL the features we need
            x_df = baseline_data[f'x_{split}_df']
            dates = baseline_data[f'dates_{split}']
            
            # CRITICAL FIX: Use original normalized columns directly from CSV
            # The CSV already contains normalized data - no additional transformations needed
            print("Using original normalized features directly from CSV (no transformations)...")
            
            # Get ALL columns from the original dataframe (all are already normalized)
            feature_columns = [col for col in x_df.columns]  # Use ALL columns as-is
            features = {}
            
            print(f"Available normalized features: {feature_columns}")
            
            # Use all normalized features directly without any transformations
            for col in feature_columns:
                features[col] = x_df[col].values.astype(np.float32)
                print(f"  Added feature: {col} (length: {len(features[col])})")
            
            # All features should have the same length (from same dataframe)
            base_len = len(x_df)
            aligned_features = features  # No alignment needed - all from same dataframe
            
            # Align dates to match features
            dates_aligned = dates[-base_len:] if dates is not None and base_len > 0 else None
            
            # Create sliding windows for each feature
            X_channels = []
            feature_names = []
            x_dates = None
            first_feature_processed = False
            
            # Process features in consistent order (maintain column order from CSV)
            windowing_order = feature_columns  # Use original CSV column order
            print(f"Feature order for windowing: {windowing_order}")
            
            for name in windowing_order:
                series = aligned_features[name]
                print(f"Windowing feature: {name}...", end="")
                
                try:
                    windows, date_windows = self.create_sliding_windows(
                        series, window_size, max_horizon, dates_aligned
                    )
                    
                    if windows.shape[0] > 0:
                        if not first_feature_processed:
                            expected_samples = windows.shape[0]
                            print(f" Initializing sample count: {expected_samples}", end="")
                            first_feature_processed = True
                        
                        if windows.shape[0] == expected_samples:
                            X_channels.append(windows)
                            feature_names.append(name)
                            if x_dates is None:
                                x_dates = date_windows
                            print(" Appended.")
                        else:
                            print(f" Skipping '{name}' due to inconsistent sample count.")
                    else:
                        print(f" Skipping '{name}' (windowing produced 0 samples).")
                        
                except Exception as e:
                    print(f" FAILED windowing '{name}'. Error: {e}. Skipping.")
            
            if not X_channels:
                raise RuntimeError(f"No feature channels available after windowing for {split}!")
            
            # Stack all feature channels
            X_combined = np.stack(X_channels, axis=-1).astype(np.float32)
            
            # Store results for this split
            windowed_data[f'X_{split}'] = X_combined
            windowed_data[f'x_dates_{split}'] = x_dates
            windowed_data[f'num_samples_{split}'] = X_combined.shape[0]
            
            print(f"Final X shape for {split}: {X_combined.shape}")
        
        # Store feature names (same for all splits)
        windowed_data['feature_names'] = feature_names
        
        print(f"Windowed features generated for all splits.")
        print(f"Included features: {feature_names}")
        
        return windowed_data
    
    def calculate_sliding_window_baselines(self, windowed_data, data_splits, config):
        """
        Calculate sliding window baselines and targets from the EXISTING sliding windows matrix.
        
        EXACT REQUIREMENTS:
        1. Extract baseline from sliding windows matrix - last value of each window for target_column
        2. Denormalize baselines using JSON parameters  
        3. Calculate targets as returns: target<t,h> = baseline<t+h> - baseline<t>
        4. Extract dates from original CSV and trim first window_size-1 rows
        
        Args:
            windowed_data: Dict containing sliding windows matrix (X_train, X_val, X_test) and feature_names
            data_splits: Dict containing original CSV data for date extraction
            config: Configuration dictionary
            
        Returns:
            Dict containing baselines, targets, and dates for all splits
        """
        from .helpers import denormalize, load_normalization_json
        
        print("\n--- Calculating Sliding Windows Baselines and Targets ---")
        
        window_size = config.get("window_size", 48)
        target_column = config.get("target_column", "CLOSE") 
        predicted_horizons = config.get('predicted_horizons', [1])
        max_horizon = max(predicted_horizons)
        
        # Load normalization JSON for denormalization
        norm_json = load_normalization_json(config)
        
        # Get feature names from windowed data
        feature_names = windowed_data.get('feature_names', [])
        
        # Find the index of the target column in the feature matrix
        if target_column not in feature_names:
            raise ValueError(f"Target column '{target_column}' not found in windowed features: {feature_names}")
        
        target_feature_index = feature_names.index(target_column)
        print(f"Target column '{target_column}' found at feature index {target_feature_index}")
        
        result_data = {}
        
        splits = ['train', 'val', 'test']
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            # Get sliding windows matrix for this split
            X_key = f'X_{split}'
            if X_key not in windowed_data:
                print(f"WARN: No windowed data found for {split}")
                # Store empty results
                result_data[f'sliding_baseline_{split}'] = np.array([])
                result_data[f'sliding_baseline_{split}_dates'] = np.array([])
                for h in predicted_horizons:
                    result_data[f'targets_{split}_h{h}'] = np.array([])
                continue
            
            # Get sliding windows matrix: shape (num_windows, window_size, num_features)
            X_matrix = windowed_data[X_key]
            print(f"  Sliding windows matrix shape: {X_matrix.shape}")
            
            if X_matrix.shape[0] == 0:
                print(f"    WARN: Empty sliding windows matrix for {split}")
                result_data[f'sliding_baseline_{split}'] = np.array([])
                result_data[f'sliding_baseline_{split}_dates'] = np.array([])
                for h in predicted_horizons:
                    result_data[f'targets_{split}_h{h}'] = np.array([])
                continue
            
            # STEP 1: Extract baselines from sliding windows matrix
            # Extract the target column (CLOSE) from all windows: X_matrix[:, :, target_feature_index]
            # Then take the last value (rightmost) of each window: X_matrix[:, -1, target_feature_index]
            print(f"  Extracting baselines from sliding windows matrix...")
            target_windows = X_matrix[:, :, target_feature_index]  # Shape: (num_windows, window_size)
            baselines_normalized = target_windows[:, -1]  # Last value of each window: (num_windows,)
            
            print(f"    Extracted {len(baselines_normalized)} baselines from windows")
            print(f"    Baseline normalized stats: mean={np.mean(baselines_normalized):.6f}, std={np.std(baselines_normalized):.6f}")
            
            # STEP 2: Denormalize baselines using JSON parameters
            print(f"  Denormalizing baselines...")
            baselines_denormalized = denormalize(baselines_normalized, norm_json, target_column)
            print(f"    Baseline denormalized stats: mean={np.mean(baselines_denormalized):.6f}, std={np.std(baselines_denormalized):.6f}")
            
            # STEP 3: Extract dates from original CSV and align with baselines
            print(f"  Extracting and aligning dates...")
            df_key = f'x_{split}_df'
            if df_key in data_splits:
                df = data_splits[df_key]
                if isinstance(df.index, pd.DatetimeIndex):
                    original_dates = df.index
                    # Trim first window_size-1 rows as required
                    trimmed_dates = original_dates[window_size-1:]
                    
                    # Align dates with baselines (both should have same length)
                    baseline_length = len(baselines_denormalized)
                    if len(trimmed_dates) >= baseline_length:
                        aligned_dates = trimmed_dates[:baseline_length]
                        print(f"    Aligned {len(aligned_dates)} dates with baselines")
                    else:
                        print(f"    WARN: Not enough trimmed dates ({len(trimmed_dates)}) for baselines ({baseline_length})")
                        aligned_dates = trimmed_dates
                else:
                    print(f"    WARN: No datetime index found for {split}")
                    aligned_dates = None
            else:
                print(f"    WARN: No original CSV data found for {split}")
                aligned_dates = None
            
            # STEP 4: Calculate targets as returns for each horizon
            print(f"  Calculating targets for horizons {predicted_horizons}...")
            targets = {}
            
            for h in predicted_horizons:
                # target<t,h> = baseline<t+h> - baseline<t>
                if len(baselines_denormalized) > h:
                    baseline_t = baselines_denormalized[:-h]      # baseline<t>
                    baseline_t_plus_h = baselines_denormalized[h:] # baseline<t+h>
                    
                    # Calculate returns: target<t,h> = baseline<t+h> - baseline<t>
                    target_returns = baseline_t_plus_h - baseline_t
                    targets[h] = target_returns.astype(np.float32)
                    
                    print(f"    Horizon {h}: {len(target_returns)} targets calculated")
                    print(f"      Target stats: mean={np.mean(target_returns):.6f}, std={np.std(target_returns):.6f}")
                else:
                    print(f"    WARN: Not enough baselines for horizon {h}")
                    targets[h] = np.array([])
            
            # STEP 5: Store results
            result_data[f'sliding_baseline_{split}'] = baselines_denormalized
            result_data[f'sliding_baseline_{split}_dates'] = aligned_dates
            
            for h in predicted_horizons:
                result_data[f'targets_{split}_h{h}'] = targets[h]
            
            print(f"  ✅ {split}: {len(baselines_denormalized)} baselines, targets for {len(predicted_horizons)} horizons")
        
        return result_data