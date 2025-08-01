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
        Creates sliding windows from data.
        
        Args:
            data: 1D array of values
            window_size: Size of each window
            time_horizon: Time horizon for predictions (affects how many windows can be created)
            date_times: Optional datetime index for the data
            
        Returns:
            Tuple of (windows_array, date_windows_array)
        """
        print(f"Creating sliding windows (Size={window_size}, Horizon={time_horizon})...", end="")
        windows = []
        date_windows = []
        n = len(data)
        num_possible_windows = n - window_size - time_horizon + 1
        
        if num_possible_windows <= 0:
            print(f" WARN: Data short ({n}) for Win={window_size}+Horizon={time_horizon}. No windows.")
            return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object)
        
        for i in range(num_possible_windows):
            window = data[i: i + window_size]
            windows.append(window)
            if date_times is not None:
                date_index = i + window_size - 1
                if date_index < len(date_times): 
                    date_windows.append(date_times[date_index])
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
            
            # Generate log returns feature (starting from window_size-1 to align properly)
            print("Generating log returns feature...")
            log_ret = np.diff(close_data, prepend=close_data[0])
            # Normalize log returns
            log_ret_normalized = normalize_series(
                log_ret, 'log_return', self.scalers, 
                fit=(split == 'train'), 
                normalize_features=normalize_features
            )
            
            # Prepare original X columns (excluding CLOSE)
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