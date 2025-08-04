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
            
            # Get DENORMALIZED dataframe - these should contain real-scale values from step 2.5
            x_df = baseline_data[f'x_{split}_df']
            dates = baseline_data[f'dates_{split}']
            
            # CRITICAL: Use DENORMALIZED columns from aligned_data (processed in step 2.5)
            # The DataFrames in baseline_data should now contain denormalized data
            print("Using DENORMALIZED features from preprocessor step 2.5...")
            
            # Get ALL columns from the DENORMALIZED dataframe (real-scale values)
            feature_columns = [col for col in x_df.columns]  # Use ALL columns as-is
            features = {}
            
            print(f"Available DENORMALIZED features: {feature_columns}")
            
            # Use all DENORMALIZED features directly without any transformations
            for col in feature_columns:
                features[col] = x_df[col].values.astype(np.float32)
                print(f"  Added feature: {col} (length: {len(features[col])})")
                
                # DEBUG: Check if CLOSE prices are actually denormalized (should be > 0 for prices)
                if col == 'CLOSE':
                    sample_values = features[col][:10]
                    print(f"  🔍 DEBUG: {col} sample values: {sample_values}")
                    print(f"  🔍 DEBUG: {col} stats: min={np.min(features[col]):.6f}, max={np.max(features[col]):.6f}, mean={np.mean(features[col]):.6f}")
                    if np.any(features[col] <= 0):
                        print(f"  ❌ CRITICAL: {col} contains non-positive values - denormalization failed!")
                    else:
                        print(f"  ✅ {col} values are positive - denormalization successful")
            
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
    
