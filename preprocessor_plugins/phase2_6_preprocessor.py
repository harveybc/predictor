#!/usr/bin/env python
"""
Phase 2.6 Preprocessor Plugin - Row-by-Row Windowing for Pre-processed Data

This preprocessor plugin is designed to work with data that has already been:
1. Feature engineered (all features pre-calculated and included)
2. Pre-processed and split into D1-D6 datasets

Key processing per USER REQUIREMENTS:
- Takes precomputed features row-by-row to compose sliding windows
- No additional feature generation or processing (all done beforehand)
- Strict sliding window: data[t-window_size : t] (EXCLUDES current tick t to prevent data leakage)
- Always calculates targets as returns: CLOSE[t+horizon] - CLOSE[t]
- Maintains strict causality with no data leakage
"""

import numpy as np
import pandas as pd
import os

# Assuming load_csv is correctly imported
try:
    from app.data_handler import load_csv
except ImportError:
    print("CRITICAL ERROR: Could not import 'load_csv' from 'app.data_handler'.")
    raise


class PreprocessorPlugin:
    # Default plugin parameters optimized for phase 2.6 preprocessed data
    plugin_params = {
        # --- File Paths ---
        "x_train_file": "examples/data/phase_2_6/normalized_d4.csv",
        "y_train_file": "examples/data/phase_2_6/normalized_d4.csv",
        "x_validation_file": "examples/data/phase_2_6/normalized_d5.csv",
        "y_validation_file": "examples/data/phase_2_6/normalized_d5.csv",
        "x_test_file": "examples/data/phase_2_6/normalized_d6.csv",
        "y_test_file": "examples/data/phase_2_6/normalized_d6.csv",
        # --- Data Loading ---
        "headers": True,
        "max_steps_train": None, "max_steps_val": None, "max_steps_test": None,
        "target_column": "CLOSE", # Target column for prediction
        # --- Windowing & Horizons ---
        "window_size": 288, # Default window size for phase 2.6
        "predicted_horizons": [24, 48, 72, 96, 120, 144], # Multi-horizon support
        # --- Feature Engineering Flags (legacy - features are already preprocessed) ---
        # REMOVED: All feature generation flags since features are precomputed
        # REMOVED: use_returns flag - targets are always calculated as returns (CLOSE[t+horizon] - CLOSE[t])
        "normalize_features": True, # Keep for compatibility (data is already normalized)
        # --- Phase 2.6 specific parameters ---
        "use_preprocessed_data": True, # Flag indicating preprocessed data
        "expected_feature_count": None, # No strict feature count requirement for preprocessed data
        "date_column": "DATE_TIME", # Date column name
    }
    
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "normalize_features",
        "use_preprocessed_data", "expected_feature_count", "target_column"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        for key, value in kwargs.items(): 
            self.params[key] = value
        # No parameter resolution needed for preprocessed data

    def get_debug_info(self): 
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info): 
        debug_info.update(self.get_debug_info())



    def _load_data(self, file_path, max_rows, headers):
        """Loads CSV file for preprocessed data."""
        print(f"Loading preprocessed data from {file_path}...", end="")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            df = pd.read_csv(file_path, nrows=max_rows, header=0 if headers else None)
            
            # Try to parse DATE_TIME column as datetime index
            if self.params.get("date_column", "DATE_TIME") in df.columns:
                try:
                    df[self.params["date_column"]] = pd.to_datetime(df[self.params["date_column"]])
                    df.set_index(self.params["date_column"], inplace=True)
                    print(" OK (with datetime index).")
                except Exception as e:
                    print(f" OK (datetime parsing failed: {e}).")
            else:
                print(" OK (no date column found).")
            
            # Validate that we have the expected target column
            target_col_name = self.params.get("target_column", "CLOSE")
            if target_col_name not in df.columns:
                raise ValueError(f"Target column '{target_col_name}' not found in {file_path}")
                
            print(f" Shape: {df.shape}, Columns: {len(df.columns)}")
            return df
            
        except FileNotFoundError:
            print(f"\nERROR: File not found: {file_path}")
            raise
        except Exception as e:
            print(f"\nERROR loading/processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None, max_horizon=1):
        """
        Creates sliding windows for feature data with STRICT CAUSALITY per user requirements.
        
        USER REQUIREMENTS:
        - For each tick t (starting from window_size), take previous window_size ticks as sliding window
        - Window: data[t-window_size : t] (EXCLUDES current tick t to prevent data leakage)
        - Prediction timestamp: t (current tick)
        - The sliding window fed to model MUST NOT include the current tick to maintain causality
        - Must ensure enough data remains for target calculation at max_horizon
        
        Args:
            data: 2D numpy array (n_samples, n_features) or 1D array for single feature
            window_size: Size of the sliding window
            time_horizon: Not used in this implementation (targets calculated separately)
            date_times: Optional datetime array
            max_horizon: Maximum prediction horizon to ensure enough future data
        """
        print(f"Creating sliding windows (USER REQUIREMENTS - Size={window_size}, EXCLUDING current tick, max_horizon={max_horizon})...", end="")
        
        # Handle both 1D and 2D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        windows = []
        targets = []
        date_windows = []
        
        # USER REQUIREMENTS: Start from tick window_size (so we have window_size previous ticks)
        # and create windows that EXCLUDE the current tick to prevent data leakage
        # For tick t, window is data[t-window_size : t] (so it has exactly window_size elements BEFORE tick t)
        start_tick = window_size  # First tick where we can create a full window (0-indexed)
        
        # CRITICAL: Ensure we have enough future data for target calculation
        # Last tick we can use is n_samples - max_horizon - 1 (so target at t+max_horizon is valid)
        end_tick = n_samples - max_horizon
        num_possible_windows = end_tick - start_tick
        
        if num_possible_windows <= 0:
             print(f" WARN: Data short ({n_samples}) for window_size={window_size} and max_horizon={max_horizon}. No windows.")
             return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(date_windows, dtype=object)
             
        for t in range(start_tick, end_tick):
            # USER REQUIREMENTS: Window from [t-window_size : t] (EXCLUDES current tick t)
            # This gives us exactly window_size elements: data[t-window_size], ..., data[t-2], data[t-1]
            window_start = t - window_size
            window_end = t  # Exclusive, so we get [t-window_size : t]
            window = data[window_start:window_end]  # Shape: (window_size, n_features)
            
            # Verify window has correct size
            if window.shape[0] != window_size:
                print(f" ERROR: Window at tick {t} has size {window.shape[0]}, expected {window_size}")
                continue
            
            # Target calculation is ignored here - will be done separately with proper horizon logic
            target = 0.0  # Placeholder - not used
            
            windows.append(window)
            targets.append(target)
            
            if date_times is not None:
                # PREDICTION TIMESTAMP: Current tick t
                if t < len(date_times): 
                    date_windows.append(date_times[t])
                else: 
                    date_windows.append(None)
                    
        # Convert to arrays
        if windows:
            windows = np.array(windows, dtype=np.float32)  # Shape: (n_windows, window_size, n_features)
        else:
            windows = np.array([], dtype=np.float32).reshape(0, window_size, n_features)
            
        # Convert dates
        if date_times is not None:
             date_windows_arr = np.array(date_windows, dtype=object)
             if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None):
                  try: date_windows_arr = np.array(date_windows, dtype='datetime64[ns]')
                  except (ValueError, TypeError): pass
        else: 
            date_windows_arr = np.array(date_windows, dtype=object)
            
        print(f" Done ({len(windows)} windows, prediction timestamps from tick {start_tick} to {end_tick-1}).")
        return windows, np.array(targets, dtype=np.float32), date_windows_arr


    def process_data(self, config):
        """
        Processes preprocessed z-score normalized data, but ensures the exact same feature stacking, ordering, and logic as the STL preprocessor.
        """
        print("\n" + "="*15 + " Starting Phase 2.6 Preprocessing (STL-Compatible) " + "="*15)
        self.set_params(**config)
        config = self.params

        # Get key parameters
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons:
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        max_horizon = max(predicted_horizons)

        # --- 1. Load Preprocessed Data ---
        print("\n--- 1. Loading Preprocessed Data ---")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"))
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"))
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"))

        # --- 2. Add close_logreturn Column ---
        print("\n--- 2. Adding close_logreturn Column ---")
        target_column = config["target_column"]
        
        # Calculate log returns for CLOSE: log(CLOSE[t] / CLOSE[t-1]) = log(CLOSE[t]) - log(CLOSE[t-1])
        x_train_df['close_logreturn'] = np.log(x_train_df[target_column]).diff()
        x_val_df['close_logreturn'] = np.log(x_val_df[target_column]).diff()
        x_test_df['close_logreturn'] = np.log(x_test_df[target_column]).diff()
        
        # Fill NaN values (first row) with 0
        x_train_df['close_logreturn'].fillna(0, inplace=True)
        x_val_df['close_logreturn'].fillna(0, inplace=True)
        x_test_df['close_logreturn'].fillna(0, inplace=True)
        
        print(f"Added close_logreturn column to all datasets")
        print(f"Train close_logreturn stats: mean={x_train_df['close_logreturn'].mean():.6f}, std={x_train_df['close_logreturn'].std():.6f}")
        print(f"Val close_logreturn stats: mean={x_val_df['close_logreturn'].mean():.6f}, std={x_val_df['close_logreturn'].std():.6f}")
        print(f"Test close_logreturn stats: mean={x_test_df['close_logreturn'].mean():.6f}, std={x_test_df['close_logreturn'].std():.6f}")





        # --- 3. Extract Target Column and Create Row-by-Row Features ---
        print("\n--- 3. Extract Target and Create Simple Row-by-Row Features ---")
        target_column = config["target_column"]
        if target_column not in x_train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Extract dates
        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None
        
        # Extract target values (CLOSE) - these are already preprocessed and aligned
        close_train = x_train_df[target_column].astype(np.float32).values
        close_val = x_val_df[target_column].astype(np.float32).values
        close_test = x_test_df[target_column].astype(np.float32).values
        
        # Remove target column from features
        feature_columns = [col for col in x_train_df.columns if col != target_column]
        
        # Extract features (already preprocessed and normalized)
        features_train = x_train_df[feature_columns].astype(np.float32).values
        features_val = x_val_df[feature_columns].astype(np.float32).values  
        features_test = x_test_df[feature_columns].astype(np.float32).values
        
        print(f"Loaded preprocessed data with {features_train.shape[1]} features")
        print(f"Feature columns: {feature_columns}")
        print(f"Data shapes - Train: {features_train.shape}, Val: {features_val.shape}, Test: {features_test.shape}")
        print(f"CLOSE shapes - Train: {close_train.shape}, Val: {close_val.shape}, Test: {close_test.shape}")
        
        # Verify close_logreturn is in features and CLOSE is not
        has_close_logreturn = 'close_logreturn' in feature_columns
        has_close = target_column in feature_columns
        print(f"USER REQUIREMENTS VERIFIED:")
        print(f"  - CLOSE column removed from features: {not has_close}")
        print(f"  - close_logreturn included in features: {has_close_logreturn}")
        if not has_close_logreturn:
            raise ValueError("close_logreturn column not found in features!")
        if has_close:
            raise ValueError(f"Target column '{target_column}' should not be in features!")

        # --- 4. Create Sliding Windows (USER REQUIREMENTS) ---
        print("\n--- 4. Creating Sliding Windows (USER REQUIREMENTS) ---")
        
        # USER REQUIREMENTS: For each tick t, window EXCLUDES current tick to prevent data leakage
        # Window: data[t-window_size : t] (EXCLUDES current tick t)
        
        # Create windows for features
        X_train_windows, _, train_dates_windows = self.create_sliding_windows(
            features_train, window_size, 1, dates_train, max_horizon)
        X_val_windows, _, val_dates_windows = self.create_sliding_windows(
            features_val, window_size, 1, dates_val, max_horizon)
        X_test_windows, _, test_dates_windows = self.create_sliding_windows(
            features_test, window_size, 1, dates_test, max_horizon)
        
        # The windows are already in the correct shape: (samples, window_size, features)
        X_train_combined = X_train_windows
        X_val_combined = X_val_windows
        X_test_combined = X_test_windows
        
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # --- 5. Baseline and Target Calculation (USER REQUIREMENTS) ---
        print("\n--- 5. Calculating Baselines and Targets (USER REQUIREMENTS) ---")
        
        # Get number of samples from windowing
        num_samples_train = X_train_combined.shape[0]
        num_samples_val = X_val_combined.shape[0]
        num_samples_test = X_test_combined.shape[0]
        
        # USER REQUIREMENTS: Baseline and Target Calculation
        # Windows start at tick (window_size) and go up to tick (n-1)
        # For each window, the prediction timestamp is the current tick t
        # Baseline should be CLOSE[t] (current tick value)
        # Target should be CLOSE[t+horizon] - CLOSE[t] (future value - current value)
        
        baseline_start_idx = window_size  # Updated to match the new windowing logic
        
        print(f"USER REQUIREMENTS: Baseline calculation")
        print(f"  Window size: {window_size}")
        print(f"  Baseline start index: {baseline_start_idx}")
        print(f"  Number of samples: Train={num_samples_train}, Val={num_samples_val}, Test={num_samples_test}")
        
        # USER REQUIREMENTS: Calculate baseline indices
        # For window i (i=0 to num_samples-1), prediction timestamp is at baseline_start_idx + i
        # Baseline should be CLOSE[prediction_timestamp] = CLOSE[baseline_start_idx + i]
        baseline_train_indices = [baseline_start_idx + i for i in range(num_samples_train)]
        baseline_val_indices = [baseline_start_idx + i for i in range(num_samples_val)]  
        baseline_test_indices = [baseline_start_idx + i for i in range(num_samples_test)]
        
        # Verify indices are valid
        if max(baseline_train_indices) >= len(close_train):
            raise ValueError(f"Baseline train indices out of bounds. Max needed: {max(baseline_train_indices)}, Available: {len(close_train)}")
        if max(baseline_val_indices) >= len(close_val):
            raise ValueError(f"Baseline val indices out of bounds. Max needed: {max(baseline_val_indices)}, Available: {len(close_val)}")
        if max(baseline_test_indices) >= len(close_test):
            raise ValueError(f"Baseline test indices out of bounds. Max needed: {max(baseline_test_indices)}, Available: {len(close_test)}")
        
        baseline_train = close_train[baseline_train_indices]
        baseline_val = close_val[baseline_val_indices]
        baseline_test = close_test[baseline_test_indices]
        
        print(f"Baseline shapes (USER REQUIREMENTS): Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")
        
        # --- 6. Target Calculation (USER REQUIREMENTS - Always Returns) ---
        print("\n--- 6. Target Calculation (USER REQUIREMENTS - Always Returns) ---")
        # USER REQUIREMENTS: Always calculate targets as returns: CLOSE[t+horizon] - CLOSE[t]
        y_train_list = []; y_val_list = []; y_test_list = []
        print(f"Processing targets per USER REQUIREMENTS for horizons: {predicted_horizons} (Always using returns)...")
        print(f"Max horizon: {max_horizon}, ensuring enough future data for all targets")
        
        for h_idx, h in enumerate(predicted_horizons):
            print(f"  Horizon {h_idx+1}/{len(predicted_horizons)}: H={h} - Calculating targets per USER REQUIREMENTS...")
            
            # USER REQUIREMENTS: Target indices
            # For window i (i=0 to num_samples-1), prediction timestamp is at baseline_start_idx + i
            # Target at horizon h is at baseline_start_idx + i + h
            target_train_indices = [baseline_start_idx + i + h for i in range(num_samples_train)]
            target_val_indices = [baseline_start_idx + i + h for i in range(num_samples_val)]
            target_test_indices = [baseline_start_idx + i + h for i in range(num_samples_test)]
            
            # Verify target indices are valid
            if max(target_train_indices) >= len(close_train):
                raise ValueError(f"Target train indices out of bounds for H={h}. Max needed: {max(target_train_indices)}, Available: {len(close_train)}")
            if max(target_val_indices) >= len(close_val):
                raise ValueError(f"Target val indices out of bounds for H={h}. Max needed: {max(target_val_indices)}, Available: {len(close_val)}")
            if max(target_test_indices) >= len(close_test):
                raise ValueError(f"Target test indices out of bounds for H={h}. Max needed: {max(target_test_indices)}, Available: {len(close_test)}")
            
            # Extract target values
            target_train_h_raw = close_train[target_train_indices]
            target_val_h_raw = close_val[target_val_indices]
            target_test_h_raw = close_test[target_test_indices]
            
            # USER REQUIREMENTS: Apply target calculation
            # Target = CLOSE[t+horizon] - CLOSE[t] (future value - current value)
            # This respects causality and prevents future data leakage
            target_train_h = target_train_h_raw - baseline_train
            target_val_h = target_val_h_raw - baseline_val
            target_test_h = target_test_h_raw - baseline_test
            print(f"    USER REQUIREMENTS: target = CLOSE[t+{h}] - CLOSE[t]")
            
            # Validation: Check for any obvious issues
            train_finite = np.isfinite(target_train_h).sum()
            val_finite = np.isfinite(target_val_h).sum()
            test_finite = np.isfinite(target_test_h).sum()
            print(f"    Finite values: Train={train_finite}/{len(target_train_h)}, Val={val_finite}/{len(target_val_h)}, Test={test_finite}/{len(target_test_h)}")
            
            # Add target statistics for verification
            print(f"    Target stats - Train: mean={target_train_h.mean():.6f}, std={target_train_h.std():.6f}")
            print(f"    Target stats - Val: mean={target_val_h.mean():.6f}, std={target_val_h.std():.6f}")
            print(f"    Target stats - Test: mean={target_test_h.mean():.6f}, std={target_test_h.std():.6f}")
            
            y_train_list.append(target_train_h.astype(np.float32))
            y_val_list.append(target_val_h.astype(np.float32))
            y_test_list.append(target_test_h.astype(np.float32))
            
            print(f"    Target shapes: Train={target_train_h.shape}, Val={target_val_h.shape}, Test={target_test_h.shape}")

        # Final verification of target lists
        print(f"\nTARGET VERIFICATION:")
        print(f"  Generated {len(y_train_list)} target sets for {len(predicted_horizons)} horizons")
        print(f"  Expected horizons: {predicted_horizons}")
        print(f"  Target list lengths: Train={len(y_train_list)}, Val={len(y_val_list)}, Test={len(y_test_list)}")
        if len(y_train_list) != len(predicted_horizons):
            raise ValueError(f"Mismatch: {len(y_train_list)} target sets != {len(predicted_horizons)} horizons")
        for i, h in enumerate(predicted_horizons):
            print(f"  Horizon {h}: Train={y_train_list[i].shape}, Val={y_val_list[i].shape}, Test={y_test_list[i].shape}")

        # --- 7. Prepare Date Arrays ---
        y_dates_train = train_dates_windows
        y_dates_val = val_dates_windows
        y_dates_test = test_dates_windows

        # --- 8. Prepare Return Dictionary ---
        print("\n--- 8. Preparing Final Output ---")
        ret = {}
        ret["x_train"] = X_train_combined
        ret["x_val"] = X_val_combined
        ret["x_test"] = X_test_combined
        ret["y_train"] = y_train_list
        ret["y_val"] = y_val_list
        ret["y_test"] = y_test_list
        ret["x_train_dates"] = train_dates_windows
        ret["y_train_dates"] = y_dates_train
        ret["x_val_dates"] = val_dates_windows
        ret["y_val_dates"] = y_dates_val
        ret["x_test_dates"] = test_dates_windows
        ret["y_test_dates"] = y_dates_test
        ret["baseline_train"] = baseline_train  # Current tick CLOSE values per USER REQUIREMENTS
        ret["baseline_val"] = baseline_val      # Current tick CLOSE values per USER REQUIREMENTS
        ret["baseline_test"] = baseline_test    # Current tick CLOSE values per USER REQUIREMENTS
        ret["baseline_train_dates"] = y_dates_train
        ret["baseline_val_dates"] = y_dates_val
        ret["baseline_test_dates"] = y_dates_test
        ret["test_close_prices"] = baseline_test  # USER REQUIREMENTS: baseline_test contains current tick CLOSE values
        ret["feature_names"] = feature_columns
        
        print(f"Final shapes:")
        print(f"  X: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"  Y: {len(y_train_list)} horizons, Train[0]={y_train_list[0].shape}")
        print(f"  Baselines: Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")
        print(f"  Features: {len(ret['feature_names'])}")
        print("\n" + "="*15 + " Phase 2.6 Preprocessing Finished (USER REQUIREMENTS) " + "="*15)
        return ret

    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        # Merge instance defaults with passed config
        run_config = self.params.copy()
        run_config.update(config)
        # Call set_params again AFTER merge to resolve defaults
        self.set_params(**run_config)
        # Run with the fully resolved self.params
        return self.process_data(self.params)

# --- NO if __name__ == '__main__': block ---
