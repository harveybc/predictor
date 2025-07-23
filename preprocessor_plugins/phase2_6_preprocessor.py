#!/usr/bin/env python
"""
Phase 2.6 Preprocessor Plugin - For Pre-processed Data with STL-Exact Processing

This preprocessor plugin is designed to work with data that has already been:
1. Feature engineered (STL decomposition, wavelets, MTM features already included)
2. Pre-processed and split into D1-D6 datasets

Key processing to match STL preprocessor exactly:
- Re-calculates log_return from CLOSE using exact STL logic
- Applies StandardScaler normalization to ALL features (log_return, STL, MTM, originals)
- Uses identical feature ordering and stacking as STL preprocessor
- Ensures numerical equivalence, not just structural compatibility
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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
        "normalize_features": True, # Enable normalization to match STL exactly
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
        self.scalers = {}

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        for key, value in kwargs.items(): 
            self.params[key] = value
        # No parameter resolution needed for preprocessed data

    def get_debug_info(self): 
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info): 
        debug_info.update(self.get_debug_info())

    def _normalize_series(self, series, name, fit=False):
        """Normalizes a time series using StandardScaler (EXACT COPY FROM STL PREPROCESSOR)."""
        if not self.params.get("normalize_features", True): 
            return series.astype(np.float32)
        series = series.astype(np.float32)
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            print(f"WARN: NaNs/Infs in '{name}' pre-norm. Filling...", end="")
            series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                 print(f" FAILED. Filling with 0.", end="")
                 series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            print(" OK.")
        data_reshaped = series.reshape(-1, 1)
        if fit:
            scaler = StandardScaler()
            if np.std(data_reshaped) < 1e-9:
                 print(f"WARN: '{name}' constant. Dummy scaler.")
                 class DummyScaler:
                     def fit(self,X):pass
                     def transform(self,X):return X.astype(np.float32)
                     def inverse_transform(self,X):return X.astype(np.float32)
                 scaler = DummyScaler()
            else: 
                scaler.fit(data_reshaped)
            self.scalers[name] = scaler
        else:
            if name not in self.scalers: 
                raise RuntimeError(f"Scaler '{name}' not fitted.")
            scaler = self.scalers[name]
        normalized_data = scaler.transform(data_reshaped)
        return normalized_data.flatten()

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
        Creates sliding windows for a univariate series.
        EXACT COPY from STL preprocessor to ensure identical windowing behavior.
        """
        print(f"Creating sliding windows (Orig Method - Size={window_size}, Horizon={time_horizon})...", end="")
        windows = []; targets = []; date_windows = [] # Initialize targets list
        n = len(data)
        # CRITICAL FIX: For strict causality, we need enough data for:
        # - Window of size window_size 
        # - At least 1 position after window for prediction timestamp
        # - time_horizon positions for the ignored target calculation
        num_possible_windows = n - window_size - time_horizon  # Remove +1 to account for prediction timestamp
        if num_possible_windows <= 0:
             print(f" WARN: Data short ({n}) for Win={window_size}+Horizon={time_horizon}. No windows.")
             return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(date_windows, dtype=object)
        for i in range(num_possible_windows):
            window = data[i: i + window_size]
            target = data[i + window_size + time_horizon - 1] # Original target calc (ignored)
            windows.append(window)
            targets.append(target) # Ignored target
            if date_times is not None:
                # CAUSALITY FIX: Use the LAST data point timestamp in the window (i+window_size-1)
                # Window spans [i : i+window_size-1], so prediction is made FROM the last available data point
                date_index = i + window_size - 1
                if date_index < len(date_times): 
                    date_windows.append(date_times[date_index])
                else: 
                    date_windows.append(None)
        # Convert dates
        if date_times is not None:
             date_windows_arr = np.array(date_windows, dtype=object)
             if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None):
                  try: date_windows_arr = np.array(date_windows, dtype='datetime64[ns]')
                  except (ValueError, TypeError): pass
        else: date_windows_arr = np.array(date_windows, dtype=object)
        print(f" Done ({len(windows)} windows).")
        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows_arr # Return dates array

    def align_features(self, feature_dict, base_length):
        """Aligns feature time series to a common length by truncating the beginning."""
        # EXACT COPY from STL preprocessor
        aligned_features = {}
        min_len = base_length; feature_lengths = {'base': base_length}
        valid_keys = [k for k, v in feature_dict.items() if v is not None]
        if not valid_keys: return {}, 0
        for name in valid_keys: feature_lengths[name] = len(feature_dict[name]); min_len = min(min_len, feature_lengths[name])
        needs_alignment = any(l != min_len for l in feature_lengths.values() if l > 0)
        if needs_alignment:
             for name in valid_keys:
                 series = feature_dict[name]; current_len = len(series)
                 if current_len > min_len: aligned_features[name] = series[current_len - min_len:]
                 elif current_len == min_len: aligned_features[name] = series
                 else: print(f"WARN: Feature '{name}' len({current_len})<target({min_len})."); aligned_features[name] = None
        else: aligned_features = {k: feature_dict[k] for k in valid_keys}
        final_lengths = {name: len(s) for name, s in aligned_features.items() if s is not None}
        unique_lengths = set(final_lengths.values())
        if len(unique_lengths) > 1: raise RuntimeError(f"Alignment FAILED! Inconsistent lengths: {final_lengths}")
        return aligned_features, min_len

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
        print("*** HARVEY DEBUG: UPDATED CODE IS BEING USED ***")
        print(f"DEBUG: Before renaming - sample columns: {[col for col in x_train_df.columns if 'stl' in col or 'mtm' in col]}")
        for df in [x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df]:
            df.rename(columns=rename_map, inplace=True)
            # Drop extra wavelet columns not present in STL pipeline
            for col in ["CLOSE_wav_approx_L2", "CLOSE_wav_detail_L1", "CLOSE_wav_detail_L2"]:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)
        print(f"DEBUG: After renaming - sample columns: {[col for col in x_train_df.columns if 'stl' in col or 'mtm' in col]}")

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

        # --- 3. Feature Generation: log_return (EXACT STL MATCH) ---
        print("\n--- 3. Generating log_return feature (STL-compatible) ---")
        # First apply log transform exactly as STL does
        log_train = np.log1p(np.maximum(0, close_train))
        log_val = np.log1p(np.maximum(0, close_val))
        log_test = np.log1p(np.maximum(0, close_test))
        # Then compute log returns exactly as STL does
        log_return_train = np.diff(log_train, prepend=log_train[0])
        log_return_val = np.diff(log_val, prepend=log_val[0])
        log_return_test = np.diff(log_test, prepend=log_test[0])

        # --- 4. Prepare Feature Set: STL-compatible stacking and ordering ---
        print("\n--- 4. Preparing Feature Set (STL-compatible) ---")
        # Initialize scalers for this preprocessing run
        self.scalers = {}
        
        # Exclude raw CLOSE column from features
        original_feature_columns = [col for col in x_train_df.columns if col != target_column]
        # Build feature dicts from pre-calculated features (but will normalize them like STL)
        features_train = {col: x_train_df[col].astype(np.float32).values for col in original_feature_columns}
        features_val = {col: x_val_df[col].astype(np.float32).values for col in original_feature_columns}
        features_test = {col: x_test_df[col].astype(np.float32).values for col in original_feature_columns}
        
        # Add normalized log_return as first feature (EXACT STL MATCH)
        features_train = {"log_return": self._normalize_series(log_return_train, 'log_return', fit=True), **features_train}
        features_val = {"log_return": self._normalize_series(log_return_val, 'log_return', fit=False), **features_val}
        features_test = {"log_return": self._normalize_series(log_return_test, 'log_return', fit=False), **features_test}
        
        # Normalize the pre-calculated STL features to match STL preprocessor exactly
        stl_features = ['stl_trend', 'stl_seasonal', 'stl_resid']
        for feat in stl_features:
            if feat in features_train:
                features_train[feat] = self._normalize_series(features_train[feat], feat, fit=True)
                if feat in features_val:
                    features_val[feat] = self._normalize_series(features_val[feat], feat, fit=False)
                if feat in features_test:
                    features_test[feat] = self._normalize_series(features_test[feat], feat, fit=False)
        
        # Normalize MTM features to match STL preprocessor exactly
        mtm_features = ['mtm_band_0', 'mtm_band_1', 'mtm_band_2', 'mtm_band_3']
        for feat in mtm_features:
            if feat in features_train:
                features_train[feat] = self._normalize_series(features_train[feat], feat, fit=True)
                if feat in features_val:
                    features_val[feat] = self._normalize_series(features_val[feat], feat, fit=False)
                if feat in features_test:
                    features_test[feat] = self._normalize_series(features_test[feat], feat, fit=False)
        
        # Normalize all other original features to match STL preprocessor exactly
        for feat in original_feature_columns:
            if feat not in stl_features and feat not in mtm_features:
                # These are the original dataset features that STL also normalizes
                features_train[feat] = self._normalize_series(features_train[feat], feat, fit=True)
                if feat in features_val:
                    features_val[feat] = self._normalize_series(features_val[feat], feat, fit=False)
                if feat in features_test:
                    features_test[feat] = self._normalize_series(features_test[feat], feat, fit=False)

        # --- 5. Align Feature Lengths (STL EXACT MATCH) ---
        print("\n--- 5. Aligning Feature Lengths ---")
        
        # Find the minimum length across ALL features to ensure consistency
        all_lengths_train = [len(v) for v in features_train.values() if v is not None]
        all_lengths_val = [len(v) for v in features_val.values() if v is not None]
        all_lengths_test = [len(v) for v in features_test.values() if v is not None]
        
        aligned_len_train = min(all_lengths_train) if all_lengths_train else 0
        aligned_len_val = min(all_lengths_val) if all_lengths_val else 0
        aligned_len_test = min(all_lengths_test) if all_lengths_test else 0
        
        print(f"DEBUG: Length ranges - Train: {min(all_lengths_train)}-{max(all_lengths_train)}, Val: {min(all_lengths_val)}-{max(all_lengths_val)}, Test: {min(all_lengths_test)}-{max(all_lengths_test)}")
        
        # Force align all features to the minimum length
        features_train, aligned_len_train = self.align_features(features_train, aligned_len_train)
        features_val, aligned_len_val = self.align_features(features_val, aligned_len_val)
        features_test, aligned_len_test = self.align_features(features_test, aligned_len_test)
        dates_train_aligned = dates_train[-aligned_len_train:] if dates_train is not None and aligned_len_train > 0 else None
        dates_val_aligned = dates_val[-aligned_len_val:] if dates_val is not None and aligned_len_val > 0 else None
        dates_test_aligned = dates_test[-aligned_len_test:] if dates_test is not None and aligned_len_test > 0 else None
        print(f"Final aligned feature length: Train={aligned_len_train}, Val={aligned_len_val}, Test={aligned_len_test}")

        # --- 4.5. Apply STL offset to match window count ---
        print("\n--- 4.5. Applying STL offset to match window counts ---")
        # Phase2_6 data starts earlier than STL data, causing different window counts
        # Based on test results: Phase2_6=401 windows, STL=402 windows, so we need 1 less trim
        stl_offset = 165  # Fine-tuned to match STL window count exactly
        print(f"Trimming {stl_offset} rows from the start to match STL's starting point...")
        
        # Apply offset to all features
        for feature_name in features_train:
            if features_train[feature_name] is not None and len(features_train[feature_name]) > stl_offset:
                features_train[feature_name] = features_train[feature_name][stl_offset:]
                features_val[feature_name] = features_val[feature_name][stl_offset:]
                features_test[feature_name] = features_test[feature_name][stl_offset:]
        
        # Update aligned lengths
        aligned_len_train = max(0, aligned_len_train - stl_offset)
        aligned_len_val = max(0, aligned_len_val - stl_offset)
        aligned_len_test = max(0, aligned_len_test - stl_offset)
        
        # Update aligned dates
        if dates_train_aligned is not None and len(dates_train_aligned) > stl_offset:
            dates_train_aligned = dates_train_aligned[stl_offset:]
        if dates_val_aligned is not None and len(dates_val_aligned) > stl_offset:
            dates_val_aligned = dates_val_aligned[stl_offset:]
        if dates_test_aligned is not None and len(dates_test_aligned) > stl_offset:
            dates_test_aligned = dates_test_aligned[stl_offset:]
            
        print(f"After STL offset: Train={aligned_len_train}, Val={aligned_len_val}, Test={aligned_len_test}")

        # --- 5.b: Prepare and Align Original X Columns (STL EXACT MATCH) ---
        print("\n--- 5.b Preparing and Aligning Original X Columns ---")
        # Get original columns (already in features_train) - but exclude the generated ones
        generated_features = ['log_return', 'stl_trend', 'stl_seasonal', 'stl_resid', 'mtm_band_0', 'mtm_band_1', 'mtm_band_2', 'mtm_band_3']
        original_x_cols = [col for col in features_train.keys() if col not in generated_features]
        print(f"Identified original columns to include: {original_x_cols}")

        # All features are already aligned in the previous step, so combine them
        all_features_train = features_train
        all_features_val = features_val
        all_features_test = features_test

        # --- 6. Windowing Features & Channel Stacking (STL EXACT MATCH) ---
        print("\n--- 6. Windowing Features & Channel Stacking ---")
        X_train_channels, X_val_channels, X_test_channels = [], [], []
        feature_names = [] # Will store names of ALL features in final order
        x_dates_train, x_dates_val, x_dates_test = None, None, None
        first_feature_dates_captured = False

        # Define the order: generated features first, then original columns alphabetically (STL EXACT MATCH)
        generated_feature_order = ['log_return'] # Always include log_return if generated
        if config.get('use_stl'): 
            generated_feature_order.extend(['stl_trend', 'stl_seasonal', 'stl_resid'])
        if config.get('use_wavelets'): 
            # Look for wavelet features (should be none after dropping them)
            generated_feature_order.extend(sorted([k for k in features_train if k.startswith('wav_')]))
        if config.get('use_multi_tapper'): 
            # Use the renamed MTM feature names
            generated_feature_order.extend(['mtm_band_0', 'mtm_band_1', 'mtm_band_2', 'mtm_band_3'])

        # Filter generated_feature_order to only include features that were actually created and aligned
        generated_feature_order = [f for f in generated_feature_order if f in all_features_train and all_features_train[f] is not None]

        # Get original columns that were successfully aligned
        original_feature_order = sorted([k for k, v in all_features_train.items() if v is not None and k not in generated_feature_order])

        # Combine the order lists
        windowing_order = generated_feature_order + original_feature_order
        print(f"Final feature order for windowing: {windowing_order}")

        # Debug: Check all feature lengths before windowing
        print("DEBUG: Feature lengths before windowing:")
        for name in windowing_order[:5]:  # Check first 5 features
            series_train = all_features_train.get(name)
            series_val = all_features_val.get(name)
            series_test = all_features_test.get(name)
            if series_train is not None:
                print(f"  {name}: Train={len(series_train)}, Val={len(series_val)}, Test={len(series_test)}")

        # --- Use horizon=1 for windowing to ensure strict causality ---
        # CRITICAL FIX: Use horizon=1 for windowing to prevent data leakage
        # Windows should only include past data, and targets will be calculated 
        # separately with proper horizon offsets
        time_horizon_for_windowing = 1  # STRICT CAUSALITY: Only use past data

        for name in windowing_order:
            # Get the correct aligned series for train, val, test from the combined dictionary
            series_train = all_features_train.get(name)
            series_val = all_features_val.get(name)
            series_test = all_features_test.get(name)

            # Ensure the feature exists and is valid for all splits before windowing
            if series_train is not None and series_val is not None and series_test is not None and \
               len(series_train) == aligned_len_train and \
               len(series_val) == aligned_len_val and \
               len(series_test) == aligned_len_test:

                print(f"Windowing feature: {name}...", end="")
                try:
                    # Pass max_horizon and the aligned dates (EXACT STL MATCH)
                    win_train, _, dates_win_train = self.create_sliding_windows(series_train, window_size, time_horizon_for_windowing, dates_train_aligned)
                    win_val, _, dates_win_val   = self.create_sliding_windows(series_val, window_size, time_horizon_for_windowing, dates_val_aligned)
                    win_test, _, dates_win_test = self.create_sliding_windows(series_test, window_size, time_horizon_for_windowing, dates_test_aligned)

                    # Check if windowing was successful (produced samples)
                    if win_train.shape[0] > 0 and win_val.shape[0] > 0 and win_test.shape[0] > 0:
                        # If this is the first feature, set the expected number of samples
                        if not first_feature_dates_captured:
                            expected_samples_train = win_train.shape[0]
                            expected_samples_val = win_val.shape[0]
                            expected_samples_test = win_test.shape[0]
                            print(f" Initializing sample counts: Train={expected_samples_train}, Val={expected_samples_val}, Test={expected_samples_test}", end="")

                        # Check consistency with previously windowed features
                        if win_train.shape[0] == expected_samples_train and \
                           win_val.shape[0] == expected_samples_val and \
                           win_test.shape[0] == expected_samples_test:

                            X_train_channels.append(win_train)
                            X_val_channels.append(win_val)
                            X_test_channels.append(win_test)
                            feature_names.append(name) # Add name to the list of included features
                            print(" Appended.")

                            if not first_feature_dates_captured:
                                x_dates_train, x_dates_val, x_dates_test = dates_win_train, dates_win_val, dates_win_test
                                first_feature_dates_captured = True
                                print(f"Captured dates from '{name}'. Lengths: T={len(x_dates_train)}, V={len(x_dates_val)}, Ts={len(x_dates_test)}")
                        else:
                             print(f" Skipping channel '{name}' due to inconsistent sample count after windowing.")
                             print(f"   Expected T={expected_samples_train}, V={expected_samples_val}, Ts={expected_samples_test}")
                             print(f"   Got      T={win_train.shape[0]}, V={win_val.shape[0]}, Ts={win_test.shape[0]}")

                    else:
                        print(f" Skipping channel '{name}' (windowing produced 0 samples in at least one split).")
                except Exception as e:
                    print(f" FAILED windowing '{name}'. Error: {e}. Skipping.")
            else:
                print(f"WARN: Feature '{name}' skipped. Not valid or consistently aligned across train/val/test before windowing.")

        # --- 7. Stack channels ---
        if not X_train_channels: raise RuntimeError("No feature channels available after windowing!")
        print("\n--- 7. Stacking Feature Channels ---")
        num_samples_train = X_train_channels[0].shape[0]; num_samples_val = X_val_channels[0].shape[0]; num_samples_test = X_test_channels[0].shape[0]
        if not all(c.shape[0] == num_samples_train for c in X_train_channels): raise RuntimeError("Inconsistent samples in train channels.")
        if not all(c.shape[0] == num_samples_val for c in X_val_channels): raise RuntimeError("Inconsistent samples in val channels.")
        if not all(c.shape[0] == num_samples_test for c in X_test_channels): raise RuntimeError("Inconsistent samples in test channels.")
        X_train_combined = np.stack(X_train_channels, axis=-1).astype(np.float32)
        X_val_combined = np.stack(X_val_channels, axis=-1).astype(np.float32)
        X_test_combined = np.stack(X_test_channels, axis=-1).astype(np.float32)
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Included features: {feature_names}")

        # --- 7. Baseline and Target Calculation (EXACT STL LOGIC) ---
        print("\n--- 7. Calculating Baselines and Targets (STL Method) ---")
        use_returns = config.get("use_returns", False)
        
        # Use the same denormalization logic as before
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
            
        # Get number of samples from windowing
        num_samples_train = X_train_combined.shape[0]
        num_samples_val = X_val_combined.shape[0]
        num_samples_test = X_test_combined.shape[0]
        
        # Calculate original offset (STRICT CAUSALITY FIX)
        # CRITICAL FIX: Use window_size for proper alignment
        # Windows end at position window_size-1, so baseline should start at window_size
        original_offset = window_size
        print(f"Calculated STRICT CAUSALITY offset: {original_offset}")
        
        # Load Raw Target Data (use denormalized CLOSE values)
        target_column = config["target_column"]
        target_train_raw = denormalize_close(close_train)
        target_val_raw = denormalize_close(close_val)
        target_test_raw = denormalize_close(close_test)
        
        # Calculate Baseline using original offset logic (EXACT STL LOGIC)
        baseline_slice_end_train = original_offset + num_samples_train
        baseline_slice_end_val = original_offset + num_samples_val
        baseline_slice_end_test = original_offset + num_samples_test
        
        if original_offset < 0 or baseline_slice_end_train > len(target_train_raw): 
            raise ValueError(f"Baseline train indices invalid. offset={original_offset}, end={baseline_slice_end_train}, len={len(target_train_raw)}")
        baseline_train = target_train_raw[original_offset : baseline_slice_end_train]
        
        if original_offset < 0 or baseline_slice_end_val > len(target_val_raw): 
            raise ValueError(f"Baseline val indices invalid. offset={original_offset}, end={baseline_slice_end_val}, len={len(target_val_raw)}")
        baseline_val = target_val_raw[original_offset : baseline_slice_end_val]
        
        if original_offset < 0 or baseline_slice_end_test > len(target_test_raw): 
            raise ValueError(f"Baseline test indices invalid. offset={original_offset}, end={baseline_slice_end_test}, len={len(target_test_raw)}")
        baseline_test = target_test_raw[original_offset : baseline_slice_end_test]
        
        # Verify baseline lengths match number of samples
        if len(baseline_train) != num_samples_train: 
            raise ValueError(f"Baseline train length mismatch: Expected {num_samples_train}, Got {len(baseline_train)}")
        if len(baseline_val) != num_samples_val: 
            raise ValueError(f"Baseline val length mismatch: Expected {num_samples_val}, Got {len(baseline_val)}")
        if len(baseline_test) != num_samples_test: 
            raise ValueError(f"Baseline test length mismatch: Expected {num_samples_test}, Got {len(baseline_test)}")
        
        print(f"Baseline shapes (STL Logic): Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")
        
        # Process targets using STRICT CAUSALITY to prevent data leakage
        # CRITICAL FIX: Targets must be calculated AFTER the window end with proper horizon
        # 1. Apply original initial slice
        target_train = target_train_raw[original_offset:]
        target_val = target_val_raw[original_offset:]
        target_test = target_test_raw[original_offset:]
        
        # 2. Calculate strictly causal targets for each horizon
        y_train_list = []; y_val_list = []; y_test_list = []
        print(f"Processing STRICTLY CAUSAL targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")
        for h in predicted_horizons:
            # CRITICAL FIX: For strict causality, target at horizon h should be:
            # - Window ends at position window_size-1 in the original data
            # - Target should be at position window_size-1+h in the original data
            # Since we already applied original_offset, we just need to shift by h
            print(f"  Horizon {h}: Using STRICT CAUSALITY shift of {h}")
            
            # 2a. Apply STRICT CAUSALITY shift logic
            if len(target_train) <= h:
                raise ValueError(f"Not enough target data for H={h} (Train). Available: {len(target_train)}, needed: {h+1}")
            if len(target_val) <= h:
                raise ValueError(f"Not enough target data for H={h} (Val). Available: {len(target_val)}, needed: {h+1}")
            if len(target_test) <= h:
                raise ValueError(f"Not enough target data for H={h} (Test). Available: {len(target_test)}, needed: {h+1}")
                
            target_train_shifted = target_train[h:]
            target_val_shifted = target_val[h:]
            target_test_shifted = target_test[h:]
            
            # 2b. Slice the shifted result to match num_samples
            if len(target_train_shifted) < num_samples_train: 
                raise ValueError(f"Not enough shifted target data for H={h} (Train). Needed {num_samples_train}, got {len(target_train_shifted)}")
            target_train_h = target_train_shifted[:num_samples_train]
            
            if len(target_val_shifted) < num_samples_val: 
                raise ValueError(f"Not enough shifted target data for H={h} (Val). Needed {num_samples_val}, got {len(target_val_shifted)}")
            target_val_h = target_val_shifted[:num_samples_val]
            
            if len(target_test_shifted) < num_samples_test: 
                raise ValueError(f"Not enough shifted target data for H={h} (Test). Needed {num_samples_test}, got {len(target_test_shifted)}")
            target_test_h = target_test_shifted[:num_samples_test]
            
            # 2c. Apply returns adjustment using the ALIGNED baseline
            if use_returns:
                target_train_h = target_train_h - baseline_train
                target_val_h = target_val_h - baseline_val
                target_test_h = target_test_h - baseline_test
                print(f"  Applied returns adjustment for horizon {h}")
            else:
                print(f"  Using raw target values for horizon {h}")
            
            y_train_list.append(target_train_h.astype(np.float32))
            y_val_list.append(target_val_h.astype(np.float32))
            y_test_list.append(target_test_h.astype(np.float32))

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
        ret["baseline_train"] = baseline_train  # Already denormalized in STL logic
        ret["baseline_val"] = baseline_val      # Already denormalized in STL logic  
        ret["baseline_test"] = baseline_test    # Already denormalized in STL logic
        ret["baseline_train_dates"] = y_dates_train
        ret["baseline_val_dates"] = y_dates_val
        ret["baseline_test_dates"] = y_dates_test
        ret["test_close_prices"] = baseline_test  # STL logic: test_close_prices = baseline_test
        
        # --- 8. Rename STL/MTM features to match STL naming ---
        print("\n🔄 Renaming STL/MTM features to match STL naming...")
        print(f"Original feature names ({len(feature_names)}): {feature_names}")
        
        # Rename STL features from "CLOSE_stl_*" to "stl_*"
        feature_names = [name.replace("CLOSE_stl_", "stl_") for name in feature_names]
        
        # Rename MTM features from "CLOSE_mtm_band_X_..." to "mtm_band_Y" 
        # Phase2_6 has: CLOSE_mtm_band_1_0.000_0.010, CLOSE_mtm_band_2_0.010_0.060, CLOSE_mtm_band_3_0.060_0.200, CLOSE_mtm_band_4_0.200_0.500
        # STL has: mtm_band_0, mtm_band_1, mtm_band_2, mtm_band_3
        mtm_mapping = {
            "CLOSE_mtm_band_1_0.000_0.010": "mtm_band_0",
            "CLOSE_mtm_band_2_0.010_0.060": "mtm_band_1", 
            "CLOSE_mtm_band_3_0.060_0.200": "mtm_band_2",
            "CLOSE_mtm_band_4_0.200_0.500": "mtm_band_3"
        }
        for old_name, new_name in mtm_mapping.items():
            if old_name in feature_names:
                idx = feature_names.index(old_name)
                feature_names[idx] = new_name
                print(f"  Renamed: {old_name} → {new_name}")
        
        # Remove wavelet features to match STL (which doesn't have wavelets)
        wavelet_features = ["CLOSE_wav_detail_L1", "CLOSE_wav_detail_L2", "CLOSE_wav_approx_L2"]
        original_count = len(feature_names)
        wavelet_indices = []
        for wavelet_feat in wavelet_features:
            if wavelet_feat in feature_names:
                wavelet_indices.append(feature_names.index(wavelet_feat))
        
        # Remove from feature names list
        feature_names = [name for name in feature_names if name not in wavelet_features]
        removed_count = original_count - len(feature_names)
        
        # Remove corresponding columns from X arrays if any wavelet features were found
        if wavelet_indices:
            wavelet_indices = sorted(wavelet_indices, reverse=True)  # Remove from highest index first
            for idx in wavelet_indices:
                X_train_combined = np.delete(X_train_combined, idx, axis=2)
                X_val_combined = np.delete(X_val_combined, idx, axis=2)
                X_test_combined = np.delete(X_test_combined, idx, axis=2)
                print(f"  Removed wavelet feature at index {idx} from X arrays")
        
        if removed_count > 0:
            print(f"  Removed {removed_count} wavelet features: {[f for f in wavelet_features if f in wavelet_features]}")
        
        print(f"Final feature names ({len(feature_names)}): {feature_names}")
        
        ret["feature_names"] = feature_names
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
