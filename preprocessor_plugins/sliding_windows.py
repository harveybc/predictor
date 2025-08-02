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
            if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None):
                try: 
                    date_windows_arr = np.array(date_windows, dtype='datetime64[ns]')
                except (ValueError, TypeError): 
                    pass
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
            
            # Get baseline data for this split
            x_df = baseline_data[f'x_{split}_df']
            dates = baseline_data[f'dates_{split}']
            close_data = baseline_data[f'close_{split}']
            
            # Generate log returns feature from DENORMALIZED close prices
            print("Generating log returns feature...")
            log_ret = np.diff(close_data, prepend=close_data[0])
            # CRITICAL FIX: Don't normalize again - close_data is already denormalized
            # We'll normalize log returns using their own statistics
            log_ret_normalized = normalize_series(
                log_ret, 'log_return', self.scalers, 
                fit=(split == 'train'), 
                normalize_features=normalize_features
            )
            
            # Prepare original X columns (excluding CLOSE which is already processed)
            original_x_cols = [col for col in x_df.columns if col != 'CLOSE']
            features = {'log_return': log_ret_normalized}
            
            if original_x_cols:
                print(f"Including original columns: {original_x_cols}")
                for col in original_x_cols:
                    features[col] = x_df[col].values.astype(np.float32)
            else:
                print("WARN: No original columns found besides 'CLOSE'.")
            
            # Align all features to same length
            base_len = len(log_ret_normalized)
            aligned_features = {}
            for name, series in features.items():
                if len(series) == base_len:
                    aligned_features[name] = series
                elif len(series) > base_len:
                    aligned_features[name] = series[-base_len:]
                else:
                    print(f"WARN: Feature '{name}' too short ({len(series)} vs {base_len})")
                    continue
            
            # Align dates to match features
            dates_aligned = dates[-base_len:] if dates is not None and base_len > 0 else None
            
            # Create sliding windows for each feature
            X_channels = []
            feature_names = []
            x_dates = None
            first_feature_processed = False
            
            # Process features in consistent order
            windowing_order = ['log_return'] + sorted([k for k in aligned_features.keys() if k != 'log_return'])
            print(f"Feature order for windowing: {windowing_order}")
            
            for name in windowing_order:
                if name not in aligned_features:
                    continue
                    
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
    
    def calculate_sliding_window_baselines(self, data_splits, config):
        """
        Calculate sliding window baselines from the target column for all splits.
        Extracts the last tick (baseline) from each sliding window and denormalizes it.
        
        Args:
            data_splits: Dict containing aligned data for all splits
            config: Configuration dictionary
            
        Returns:
            Dict containing sliding window baselines and dates for all splits
        """
        from .helpers import denormalize, load_normalization_json
        
        print("\n--- Calculating Sliding Windows Baseline ---")
        
        window_size = config.get("window_size", 48)
        target_column = config.get("target_column", "CLOSE")
        predicted_horizons = config.get('predicted_horizons', [1])
        max_horizon = max(predicted_horizons)
        
        # Load normalization JSON for denormalization
        norm_json = load_normalization_json(config)
        
        sliding_windows_data = {}
        
        splits = ['train', 'val', 'test']
        for split in splits:
            print(f"\nProcessing {split} sliding windows...")
            
            # Get the target column data (still normalized)
            if split == 'train':
                df_key = 'x_train_df'
            elif split == 'val':
                df_key = 'x_val_df'
            else:  # test
                df_key = 'x_test_df'
                
            if df_key not in data_splits:
                print(f"WARN: No data found for {split}")
                sliding_windows_data[f'sliding_baseline_{split}'] = np.array([])
                sliding_windows_data[f'sliding_baseline_{split}_dates'] = np.array([])
                continue
                
            df = data_splits[df_key]
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in {split} data")
            
            # Extract normalized target column
            target_normalized = df[target_column].astype(np.float32).values
            dates = df.index if isinstance(df.index, pd.DatetimeIndex) else None
            
            # Calculate number of possible windows
            n = len(target_normalized)
            expected_windows = n - (window_size - 1) - max_horizon
            
            if expected_windows <= 0:
                print(f"WARN: Insufficient data for {split}: {n} rows, need {window_size + max_horizon}")
                sliding_windows_data[f'sliding_baseline_{split}'] = np.array([])
                sliding_windows_data[f'sliding_baseline_{split}_dates'] = np.array([])
                continue
            
            print(f"  Data length: {n}, Expected windows: {expected_windows}")
            
            # Create sliding windows and extract last tick (baseline) from each window
            baselines = []
            baseline_dates = []
            
            # Window i ends at tick (window_size-1+i), baseline is data[window_size-1+i]
            for i in range(expected_windows):
                window_end_idx = window_size - 1 + i  # Last tick of window i
                baseline_normalized = target_normalized[window_end_idx]
                baselines.append(baseline_normalized)
                
                if dates is not None and window_end_idx < len(dates):
                    baseline_dates.append(dates[window_end_idx])
                else:
                    baseline_dates.append(None)
            
            # Convert to arrays
            baselines_normalized = np.array(baselines, dtype=np.float32)
            
            # CRITICAL STEP 3: Denormalize using JSON parameters
            print(f"  Denormalizing {len(baselines_normalized)} baseline values using JSON parameters...")
            baselines_denormalized = denormalize(baselines_normalized, norm_json, target_column)
            
            # Store results
            sliding_windows_data[f'sliding_baseline_{split}'] = baselines_denormalized
            sliding_windows_data[f'sliding_baseline_{split}_dates'] = np.array(baseline_dates, dtype=object)
            
            print(f"  ✅ {split}: {len(baselines_denormalized)} denormalized baselines extracted")
            print(f"    Sample baselines: {baselines_denormalized[:3]} (denormalized)")
            print(f"    Baseline stats: mean={np.mean(baselines_denormalized):.6f}, std={np.std(baselines_denormalized):.6f}")
        
        return sliding_windows_data