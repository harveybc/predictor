import numpy as np
import pandas as pd
from .helpers import normalize_series


class SlidingWindowsProcessor:
    """
    Handles the creation of sliding windows from normalized baseline data.
    """
    
    def __init__(self, scalers=None):
        self.scalers = scalers or {}
    
    def create_sliding_windows(self, data, window_size, date_times=None):
        """
        Creates sliding windows for feature processing (not limited by horizon).
        
        CRITICAL: Each window contains data[t-window_size+1:t+1] where t is the current tick.
        The baseline is data[t] (last element of window).
        
        Args:
            data: 1D array of values
            window_size: Size of each window  
            date_times: Optional datetime index for the data
            
        Returns:
            Tuple of (windows_array, date_windows_array)
        """
        print(f"Creating sliding windows (Size={window_size})...", end="")
        windows = []
        date_windows = []
        n = len(data)
        
        # Calculate usable range: can create windows from index window_size-1 to n-1
        min_baseline_idx = window_size - 1
        max_baseline_idx = n - 1
        
        if max_baseline_idx < min_baseline_idx:
            print(f" WARN: Insufficient data ({n}) for window size {window_size}. Need at least {window_size}.")
            return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object)
        
        # Create windows: each window ends at baseline time t
        for baseline_idx in range(min_baseline_idx, max_baseline_idx + 1):
            window_start = baseline_idx - window_size + 1
            window_end = baseline_idx + 1
            window = data[window_start:window_end]
            windows.append(window)
            
            # Date corresponds to baseline time (last element of window)
            if date_times is not None and baseline_idx < len(date_times):
                date_windows.append(date_times[baseline_idx])
            else:
                date_windows.append(None)
        
        # Keep dates as pandas Timestamps for matplotlib compatibility
        date_windows_arr = np.array(date_windows, dtype=object)
        
        print(f" Done ({len(windows)} windows).")
        return np.array(windows, dtype=np.float32), date_windows_arr
    
    def generate_windowed_features(self, baseline_data, config):
        """
        Generate windowed features from denormalized data.
        
        Args:
            baseline_data: Dict containing denormalized data for all splits
            config: Configuration dictionary
            
        Returns:
            Dict containing windowed features and metadata
        """
        print("Creating windowed features from denormalized data...")
        
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        # FIXED: Use individual horizon processing instead of max_horizon
        
        windowed_data = {}
        
        for split in ['train', 'val', 'test']:
            print(f"Processing {split} split...")
            
            # Get denormalized dataframe
            x_df = baseline_data[f'x_{split}_df']
            dates = baseline_data[f'dates_{split}']
            
            # Use all features from denormalized data
            feature_columns = list(x_df.columns)
            features = {}
            
            for col in feature_columns:
                features[col] = x_df[col].values.astype(np.float32)
                
            # Create sliding windows for each feature
            X_channels = []
            feature_names = []
            x_dates = None
            
            for name in feature_columns:
                series = features[name]
                
                try:
                    windows, date_windows = self.create_sliding_windows(
                        series, window_size, dates
                    )
                    
                    if windows.shape[0] > 0:
                        X_channels.append(windows)
                        feature_names.append(name)
                        if x_dates is None:
                            x_dates = date_windows
                        print(f"  Added feature: {name}")
                    else:
                        print(f"  Skipped {name} (no valid windows)")
                        
                except Exception as e:
                    print(f"  Failed windowing {name}: {e}")
            
            if not X_channels:
                raise RuntimeError(f"No features available after windowing for {split}")
            
            # Stack all feature channels
            X_combined = np.stack(X_channels, axis=-1).astype(np.float32)
            
            # Store results
            windowed_data[f'X_{split}'] = X_combined
            windowed_data[f'x_dates_{split}'] = x_dates
            windowed_data[f'num_samples_{split}'] = X_combined.shape[0]
            
            print(f"  Final shape: {X_combined.shape}")
        
        # Store feature names
        windowed_data['feature_names'] = feature_names
        
        print("Windowed features created successfully")
        return windowed_data
    
