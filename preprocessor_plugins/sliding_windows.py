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
            
            # CRITICAL: Use NORMALIZED columns from aligned_data 
            # The DataFrames in baseline_data contain normalized data from z-score normalization
            print("Using NORMALIZED features for neural network training...")
            
            # Get ALL columns from the NORMALIZED dataframe (z-score normalized values)
            feature_columns = [col for col in x_df.columns]  # Use ALL columns as-is
            features = {}
            
            print(f"Available NORMALIZED features: {feature_columns}")
            
            # Use all NORMALIZED features directly - perfect for neural network training
            for col in feature_columns:
                features[col] = x_df[col].values.astype(np.float32)
                print(f"  Added feature: {col} (length: {len(features[col])})")
                
                # DEBUG: Check if features are properly normalized (should be around mean~0, std~1)
                if col == 'CLOSE':
                    sample_values = features[col][:10]
                    print(f"  🔍 DEBUG: {col} sample values: {sample_values}")
                    print(f"  🔍 DEBUG: {col} stats: min={np.min(features[col]):.6f}, max={np.max(features[col]):.6f}, mean={np.mean(features[col]):.6f}")
                    
                    # Check for reasonable normalized ranges (typical z-score normalized data)
                    if abs(np.mean(features[col])) > 10:
                        print(f"  ⚠️  WARNING: {col} mean is very large for normalized data (mean={np.mean(features[col]):.2f})")
                        print(f"      This may indicate the data is not properly normalized")
                    elif np.std(features[col]) > 10:
                        print(f"  ⚠️  WARNING: {col} std is very large for normalized data (std={np.std(features[col]):.2f})")
                        print(f"      This may indicate high variance in the normalized data")
                    else:
                        print(f"  ✅ {col} values appear properly normalized for neural network training")
            
            # All features should have the same length (from same dataframe)
            base_len = len(x_df)
            normalized_features = features  # No alignment needed - all from same dataframe
            
            # Align dates to match features
            dates_aligned = dates[-base_len:] if dates is not None and base_len > 0 else None
            
            # Create sliding windows for each normalized feature
            X_channels = []
            feature_names = []
            x_dates = None
            first_feature_processed = False
            
            # Process features in consistent order (maintain column order from CSV)
            windowing_order = feature_columns  # Use normalized CSV column order
            print(f"Feature order for windowing: {windowing_order}")
            
            for name in windowing_order:
                series = normalized_features[name]
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
            
            # Stack all normalized feature channels
            X_combined_normalized = np.stack(X_channels, axis=-1).astype(np.float32)
            
            # Store results for this split
            windowed_data[f'X_{split}'] = X_combined_normalized
            windowed_data[f'x_dates_{split}'] = x_dates
            windowed_data[f'num_samples_{split}'] = X_combined_normalized.shape[0]
            
            print(f"Final X shape for {split}: {X_combined_normalized.shape}")
        
        # Store feature names (same for all splits)
        windowed_data['feature_names'] = feature_names
        
        # CRITICAL DIAGNOSTIC: Data is already normalized - verify scales
        print(f"\n=== NORMALIZED DATA VERIFICATION ===")
        print("Input data is already z-score normalized - verifying scales for neural network compatibility")
        
        if len(feature_names) > 0 and 'X_train' in windowed_data:
            X_sample = windowed_data['X_train']
            if X_sample.shape[0] > 0:
                print(f"Analyzing pre-normalized feature scales...")
                for i, fname in enumerate(feature_names):
                    feature_values = X_sample[:, :, i].flatten()
                    valid_values = feature_values[np.isfinite(feature_values)]
                    
                    if len(valid_values) > 0:
                        f_mean = np.mean(valid_values)
                        f_std = np.std(valid_values)
                        f_min = np.min(valid_values)
                        f_max = np.max(valid_values)
                        f_range = f_max - f_min
                        
                        print(f"  {fname}: mean={f_mean:.4f}, std={f_std:.4f}, range=[{f_min:.4f}, {f_max:.4f}]")
                        
                        # Check for properly normalized scales (should be roughly -3 to +3 for z-score)
                        if abs(f_mean) > 5 or f_std > 5:
                            print(f"    ⚠️  SCALE WARNING: Values may not be properly z-score normalized")
                        elif f_range > 15:
                            print(f"    ⚠️  SCALE WARNING: Large range for normalized data")
                        else:
                            print(f"    ✅ Scale appropriate for neural network training (z-score normalized)")
                    else:
                        print(f"  {fname}: No valid values found!")
        
        print(f"✅ Normalized windowed features ready for neural network training.")
        print(f"Features: {feature_names}")
        
        return windowed_data
    
