#!/usr/bin/env python
"""
Phase 2.6 Preprocessor Plugin - For Pre-processed Z-Score Normalized Data

This preprocessor plugin is designed to work with data that has already been:
1. Feature engineered (STL decomposition, wavelets, MTM features already included)
2. Z-score normalized using separate normalizers A/B
3. Split into D1-D6 datasets

Key differences from STL preprocessor:
- Assumes data is already preprocessed and normalized
- No STL decomposition, wavelet, or MTM feature generation
- No normalization step needed
- Simplified windowing for already-prepared features
- Direct loading of z-score normalized CSV files
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

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
        # --- Feature Engineering Flags (enabled since config may override) ---
        "use_stl": True, # STL features available in data 
        "use_wavelets": True, # Wavelet features available in data
        "use_multi_tapper": True, # MTM features available in data
        "use_returns": True, # Use returns for prediction
        "normalize_features": False, # Data already normalized
        # --- Phase 2.6 specific parameters ---
        "use_preprocessed_data": True, # Flag indicating preprocessed data
        "expected_feature_count": 55, # Expected number of features in preprocessed data
        "date_column": "DATE_TIME", # Date column name
    }
    
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "use_returns", "normalize_features",
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

    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        """
        Creates sliding windows for preprocessed data.
        Simplified version for already processed features.
        """
        print(f"Creating sliding windows (Size={window_size}, Horizon={time_horizon})...", end="")
        
        if isinstance(data, pd.DataFrame):
            # Multi-feature data (features only, no target column)
            windows = []
            date_windows = []
            
            n = len(data)
            num_possible_windows = n - window_size - time_horizon + 1
            
            if num_possible_windows <= 0:
                print(f" WARN: Data short ({n}) for Win={window_size}+Horizon={time_horizon}. No windows.")
                return np.array(windows, dtype=np.float32), np.array([], dtype=np.float32), np.array(date_windows, dtype=object)
            
            for i in range(num_possible_windows):
                # Create window for all features (no target calculation here)
                window = data.iloc[i:i + window_size].values  # All features
                windows.append(window)
                
                # Date handling
                if date_times is not None:
                    date_index = i + window_size - 1
                    if date_index < len(date_times):
                        date_windows.append(date_times[date_index])
                    else:
                        date_windows.append(None)
                        
        else:
            # Single feature data (univariate)
            windows = []
            date_windows = []
            
            n = len(data)
            num_possible_windows = n - window_size - time_horizon + 1
            
            if num_possible_windows <= 0:
                print(f" WARN: Data short ({n}) for Win={window_size}+Horizon={time_horizon}. No windows.")
                return np.array(windows, dtype=np.float32), np.array([], dtype=np.float32), np.array(date_windows, dtype=object)
                
            for i in range(num_possible_windows):
                window = data[i:i + window_size]
                windows.append(window)
                
                if date_times is not None:
                    date_index = i + window_size - 1
                    if date_index < len(date_times):
                        date_windows.append(date_times[date_index])
                    else:
                        date_windows.append(None)

        # Convert dates
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
        return np.array(windows, dtype=np.float32), np.array([], dtype=np.float32), date_windows_arr

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

        # --- 1b. Rename columns to STL-compatible names ---
        rename_map = {
            "CLOSE_stl_trend": "stl_trend",
            "CLOSE_stl_seasonal": "stl_seasonal",
            "CLOSE_stl_resid": "stl_resid",
            "CLOSE_mtm_band_1_0.000_0.010": "mtm_band_0",
            "CLOSE_mtm_band_2_0.010_0.060": "mtm_band_1",
            "CLOSE_mtm_band_3_0.060_0.200": "mtm_band_2",
            "CLOSE_mtm_band_4_0.200_0.500": "mtm_band_3",
        }
        for df in [x_train_df, x_val_df, x_test_df]:
            df.rename(columns=rename_map, inplace=True)
            # Drop extra wavelet columns not present in STL pipeline
            for col in ["CLOSE_wav_approx_L2", "CLOSE_wav_detail_L1", "CLOSE_wav_detail_L2"]:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)

        # Validate feature count
        expected_features = config.get("expected_feature_count", 55)
        if len(x_train_df.columns) != expected_features:
            print(f"WARN: Expected {expected_features} features, got {len(x_train_df.columns)}")
        print(f"Loaded preprocessed data with {len(x_train_df.columns)} features")
        print(f"Features: {list(x_train_df.columns)}")

        # --- 2. Extract Target Column and Dates ---
        print("\n--- 2. Extracting Target and Dates ---")
        target_column = config["target_column"]
        if target_column not in x_train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        # Extract dates
        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None
        # Extract target values for baseline calculation (keep original CLOSE for baseline)
        close_train = x_train_df[target_column].astype(np.float32).values
        close_val = x_val_df[target_column].astype(np.float32).values
        close_test = x_test_df[target_column].astype(np.float32).values

        # --- 3. Feature Generation: log_return ---
        print("\n--- 3. Generating log_return feature (STL-compatible) ---")
        def compute_log_return(series):
            log_series = np.log1p(np.maximum(0, series))
            return np.diff(log_series, prepend=log_series[0])
        log_return_train = compute_log_return(close_train)
        log_return_val = compute_log_return(close_val)
        log_return_test = compute_log_return(close_test)

        # --- 4. Prepare Feature Set: STL-compatible stacking and ordering ---
        print("\n--- 4. Preparing Feature Set (STL-compatible) ---")
        # Exclude raw CLOSE column from features
        original_feature_columns = [col for col in x_train_df.columns if col != target_column]
        # Build feature dicts
        features_train = {col: x_train_df[col].astype(np.float32).values for col in original_feature_columns}
        features_val = {col: x_val_df[col].astype(np.float32).values for col in original_feature_columns}
        features_test = {col: x_test_df[col].astype(np.float32).values for col in original_feature_columns}
        # Add log_return as first feature (STL always does this)
        features_train = {"log_return": log_return_train, **features_train}
        features_val = {"log_return": log_return_val, **features_val}
        features_test = {"log_return": log_return_test, **features_test}

        # --- 5. Feature Order: Match STL pipeline exactly ---
        stl_feature_order = [
            'log_return',
            'stl_trend', 'stl_seasonal', 'stl_resid',
            'mtm_band_0', 'mtm_band_1', 'mtm_band_2', 'mtm_band_3',
        ]
        # Add the rest of the features (excluding those already in stl_feature_order and log_return), sorted alphabetically
        rest_features = [col for col in original_feature_columns if col not in stl_feature_order and col != 'log_return']
        feature_order = stl_feature_order + sorted(rest_features)
        print(f"Final feature order for windowing: {feature_order}")

        # --- 6. Windowing Features ---
        print("\n--- 6. Creating Sliding Windows (STL-compatible) ---")
        X_train_channels, X_val_channels, X_test_channels = [], [], []
        x_dates_train, x_dates_val, x_dates_test = None, None, None
        first_feature_dates_captured = False
        for name in feature_order:
            arr_train = features_train[name]
            arr_val = features_val[name]
            arr_test = features_test[name]
            # Window each feature
            win_train, _, dates_win_train = self.create_sliding_windows(arr_train, window_size, max_horizon, dates_train)
            win_val, _, dates_win_val = self.create_sliding_windows(arr_val, window_size, max_horizon, dates_val)
            win_test, _, dates_win_test = self.create_sliding_windows(arr_test, window_size, max_horizon, dates_test)
            if win_train.shape[0] == 0 or win_val.shape[0] == 0 or win_test.shape[0] == 0:
                print(f"WARN: Feature '{name}' produced 0 windows in at least one split. Skipping.")
                continue
            X_train_channels.append(win_train)
            X_val_channels.append(win_val)
            X_test_channels.append(win_test)
            if not first_feature_dates_captured:
                x_dates_train, x_dates_val, x_dates_test = dates_win_train, dates_win_val, dates_win_test
                first_feature_dates_captured = True

        if not X_train_channels:
            raise RuntimeError("No feature channels available after windowing!")
        X_train_combined = np.stack(X_train_channels, axis=-1).astype(np.float32)
        X_val_combined = np.stack(X_val_channels, axis=-1).astype(np.float32)
        X_test_combined = np.stack(X_test_channels, axis=-1).astype(np.float32)
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Included features: {feature_order}")

        # --- 7. Baseline and Target Calculation (unchanged) ---
        print("\n--- 7. Calculating Baselines and Targets ---")
        use_returns = config.get("use_returns", False)
        import json
        norm_config_path = config.get("use_normalization_json")
        if norm_config_path and os.path.exists(norm_config_path):
            with open(norm_config_path, 'r') as f:
                norm_params = json.load(f)
            close_mean = norm_params.get("CLOSE", {}).get("mean", 0)
            close_std = norm_params.get("CLOSE", {}).get("std", 1)
            print(f"Loaded CLOSE normalization: mean={close_mean}, std={close_std}")
        else:
            print("WARN: No normalization config found. Using identity transform.")
            close_mean, close_std = 0, 1
        def denormalize_close(normalized_values):
            return normalized_values * close_std + close_mean
        num_train_windows = X_train_combined.shape[0]
        num_val_windows = X_val_combined.shape[0]
        num_test_windows = X_test_combined.shape[0]
        baseline_train = np.zeros(num_train_windows, dtype=np.float32)
        baseline_val = np.zeros(num_val_windows, dtype=np.float32)
        baseline_test = np.zeros(num_test_windows, dtype=np.float32)
        for i in range(num_train_windows):
            baseline_train[i] = close_train[i + window_size - 1]
        for i in range(num_val_windows):
            baseline_val[i] = close_val[i + window_size - 1]
        for i in range(num_test_windows):
            baseline_test[i] = close_test[i + window_size - 1]

        y_train_list = []
        y_val_list = []
        y_test_list = []
        for h in predicted_horizons:
            print(f"Processing horizon {h}...")
            target_start_train = window_size + h - 1
            target_start_val = window_size + h - 1
            target_start_test = window_size + h - 1
            y_train_h = close_train[target_start_train:target_start_train + num_train_windows]
            y_val_h = close_val[target_start_val:target_start_val + num_val_windows]
            y_test_h = close_test[target_start_test:target_start_test + num_test_windows]
            if use_returns:
                y_train_h_denorm = denormalize_close(y_train_h)
                y_val_h_denorm = denormalize_close(y_val_h)
                y_test_h_denorm = denormalize_close(y_test_h)
                baseline_train_denorm = denormalize_close(baseline_train)
                baseline_val_denorm = denormalize_close(baseline_val)
                baseline_test_denorm = denormalize_close(baseline_test)
                y_train_h = y_train_h_denorm - baseline_train_denorm
                y_val_h = y_val_h_denorm - baseline_val_denorm
                y_test_h = y_test_h_denorm - baseline_test_denorm
                print(f"  Calculated returns in denormalized space for horizon {h}")
            else:
                y_train_h = denormalize_close(y_train_h)
                y_val_h = denormalize_close(y_val_h)
                y_test_h = denormalize_close(y_test_h)
                print(f"  Denormalized targets for horizon {h}")
            y_train_list.append(y_train_h.astype(np.float32))
            y_val_list.append(y_val_h.astype(np.float32))
            y_test_list.append(y_test_h.astype(np.float32))

        # --- 8. Prepare Date Arrays ---
        y_dates_train = x_dates_train
        y_dates_val = x_dates_val
        y_dates_test = x_dates_test

        # --- 9. Prepare Return Dictionary ---
        print("\n--- 9. Preparing Final Output ---")
        ret = {}
        ret["x_train"] = X_train_combined
        ret["x_val"] = X_val_combined
        ret["x_test"] = X_test_combined
        ret["y_train"] = y_train_list
        ret["y_val"] = y_val_list
        ret["y_test"] = y_test_list
        ret["x_train_dates"] = x_dates_train
        ret["y_train_dates"] = y_dates_train
        ret["x_val_dates"] = x_dates_val
        ret["y_val_dates"] = y_dates_val
        ret["x_test_dates"] = x_dates_test
        ret["y_test_dates"] = y_dates_test
        ret["baseline_train"] = denormalize_close(baseline_train) if use_returns else denormalize_close(baseline_train)
        ret["baseline_val"] = denormalize_close(baseline_val) if use_returns else denormalize_close(baseline_val)
        ret["baseline_test"] = denormalize_close(baseline_test) if use_returns else denormalize_close(baseline_test)
        ret["baseline_train_dates"] = y_dates_train
        ret["baseline_val_dates"] = y_dates_val
        ret["baseline_test_dates"] = y_dates_test
        ret["test_close_prices"] = denormalize_close(baseline_test)
        ret["feature_names"] = feature_order
        print(f"Final shapes:")
        print(f"  X: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"  Y: {len(y_train_list)} horizons, Train[0]={y_train_list[0].shape}")
        print(f"  Baselines: Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")
        print(f"  Features: {len(ret['feature_names'])}")
        print("\n" + "="*15 + " Phase 2.6 Preprocessing Finished (STL-Compatible) " + "="*15)
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
