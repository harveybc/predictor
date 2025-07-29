# Ensure these imports are present at the top of the file
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import os

# Assuming load_csv is correctly imported from app.data_handler
try:
    from app.data_handler import load_csv
except ImportError:
    print("CRITICAL ERROR: Could not import 'load_csv' from 'app.data_handler'.")
    raise

# Include the user-provided verify_date_consistency function
def verify_date_consistency(date_lists, dataset_name):
    """
    Verifies that all date arrays in date_lists have the same first and last elements.
    Prints a warning if any array does not match.
    """
    if not date_lists: return
    valid_date_lists = [d for d in date_lists if d is not None and len(d) > 0]
    if not valid_date_lists: return
    first_len = len(valid_date_lists[0])
    if not all(len(d) == first_len for d in valid_date_lists):
         print(f"WARN: Length mismatch in {dataset_name} dates: {[len(d) for d in valid_date_lists]}")
         return
    try: first = valid_date_lists[0][0]; last = valid_date_lists[0][-1]
    except IndexError: print(f"WARN: Could not access first/last element in {dataset_name} dates."); return
    consistent = True
    for i, d in enumerate(valid_date_lists):
        try:
            if len(d) > 0:
                if d[0] != first or d[-1] != last:
                    print(f"Warning: Date array {i} in {dataset_name} range ({d[0]} to {d[-1]}) does not match first array ({first} to {last}).")
                    consistent = False
            else: print(f"Warning: Date array {i} in {dataset_name} is empty."); consistent = False
        except Exception as e: print(f"Warning: Could not compare dates in {dataset_name} for array {i}. Error: {e}."); consistent = False


class PreprocessorPlugin:
    # Default plugin parameters - Restored working version defaults
    plugin_params = {
        # --- File Paths ---
        "x_train_file": "data/x_train.csv",
        "y_train_file": "data/y_train.csv",
        "x_validation_file": "data/x_val.csv",
        "y_validation_file": "data/y_val.csv",
        "x_test_file": "data/x_test.csv",
        "y_test_file": "data/y_test.csv",
        # --- Data Loading ---
        "headers": True,
        "max_steps_train": None, "max_steps_val": None, "max_steps_test": None,
        "target_column": "TARGET",
        # --- Windowing & Horizons ---
        "window_size": 48,
        "predicted_horizons": [1, 6, 12, 24],
        # --- Feature Engineering Flags ---
        "use_returns": True,
        "normalize_features": True,
        # --- STL Parameters (needed for offset calculation only) ---
        "stl_period": 24,
        "stl_window": None,  # Will be resolved to 2 * stl_period + 1
    }
    
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "use_returns", "normalize_features",
        "target_returns_mean", "target_returns_std"  # These will now be lists
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.scalers = {}

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        for key, value in kwargs.items(): 
            self.params[key] = value
        
        # Resolve STL window for offset calculation (even though we don't use STL)
        config = self.params
        if config.get("stl_period") is not None and config.get("stl_period") > 1:
            if config.get("stl_window") is None: 
                config["stl_window"] = 2 * config["stl_period"] + 1

    def get_debug_info(self):
        """Return debug information for the plugin."""
        debug_info = {}
        for var in self.plugin_debug_vars:
            value = self.params.get(var)
            debug_info[var] = value
        return debug_info

    def add_debug_info(self, debug_info):
        """Add plugin debug info to the provided dictionary."""
        debug_info.update(self.get_debug_info())

    # --- Helper Methods ---
    def _load_data(self, file_path, max_rows, headers):
        print(f"Loading data: {file_path} (Max rows: {max_rows})...", end="")
        try:
            df = load_csv(file_path, headers=headers, max_rows=max_rows)
            if df is None or df.empty: 
                raise ValueError(f"load_csv None/empty for {file_path}")
            print(f" Done. Shape: {df.shape}")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Attempting DatetimeIndex conversion for {file_path}...", end="")
                original_index_name = df.index.name
                try: 
                    df.index = pd.to_datetime(df.index)
                    print(" OK (from index).")
                except Exception:
                    try: 
                        df.index = pd.to_datetime(df.iloc[:, 0])
                        print(" OK (from col 0).")
                    except Exception as e_col: 
                        print(f" FAILED ({e_col}). Dates unavailable.")
                        df.index = None
                if original_index_name: 
                    df.index.name = original_index_name
            
            required_cols = ["CLOSE"]
            target_col_name = self.params.get("target_column", "TARGET")
            if 'y_' in os.path.basename(file_path).lower(): 
                required_cols.append(target_col_name)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols: 
                raise ValueError(f"Missing cols in {file_path}: {missing_cols}")
            return df
        except FileNotFoundError: 
            print(f"\nERROR: File not found: {file_path}.")
            raise
        except Exception as e: 
            print(f"\nERROR loading/processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _normalize_series(self, series, name, fit=False):
        """Normalizes a time series using StandardScaler."""
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

    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        """
        Creates sliding windows for a univariate series.
        Original user version - calculates targets internally but they are ignored.
        Requires a single `time_horizon` argument (use max_horizon here).
        """
        print(f"Creating sliding windows (Orig Method - Size={window_size}, Horizon={time_horizon})...", end="")
        windows = []
        targets = []  # Initialize targets list
        date_windows = []
        n = len(data)
        num_possible_windows = n - window_size - time_horizon + 1
        
        if num_possible_windows <= 0:
             print(f" WARN: Data short ({n}) for Win={window_size}+Horizon={time_horizon}. No windows.")
             return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(date_windows, dtype=object)
        
        for i in range(num_possible_windows):
            window = data[i: i + window_size]
            target = data[i + window_size + time_horizon - 1]  # Original target calc (ignored)
            windows.append(window)
            targets.append(target)  # Ignored target
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
        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows_arr

    def align_features(self, feature_dict, base_length):
        """Aligns feature time series to a common length by truncating the beginning."""
        aligned_features = {}
        min_len = base_length
        feature_lengths = {'base': base_length}
        valid_keys = [k for k, v in feature_dict.items() if v is not None]
        
        if not valid_keys: 
            return {}, 0
        
        for name in valid_keys: 
            feature_lengths[name] = len(feature_dict[name])
            min_len = min(min_len, feature_lengths[name])
        
        needs_alignment = any(l != min_len for l in feature_lengths.values() if l > 0)
        if needs_alignment:
             for name in valid_keys:
                 series = feature_dict[name]
                 current_len = len(series)
                 if current_len > min_len: 
                     aligned_features[name] = series[current_len - min_len:]
                 elif current_len == min_len: 
                     aligned_features[name] = series
                 else: 
                     print(f"WARN: Feature '{name}' len({current_len})<target({min_len}).")
                     aligned_features[name] = None
        else: 
            aligned_features = {k: feature_dict[k] for k in valid_keys}
        
        final_lengths = {name: len(s) for name, s in aligned_features.items() if s is not None}
        unique_lengths = set(final_lengths.values())
        if len(unique_lengths) > 1: 
            raise RuntimeError(f"Alignment FAILED! Inconsistent lengths: {final_lengths}")
        
        return aligned_features, min_len

    def process_data(self, config):
        """
        Processes data using ORIGINAL target calculation and windowing logic.
        Simplified to remove decomposition methods but maintain core logic.
        """
        print("\n" + "="*15 + " Starting Preprocessing " + "="*15)
        self.set_params(**config)
        config = self.params
        self.scalers = {}

        # Get key parameters
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons: 
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        max_horizon = max(predicted_horizons)
        
        # Get stl_window for original offset calculation
        stl_window = config.get('stl_window')
        if stl_window is None: 
            raise ValueError("stl_window parameter must be resolved before processing.")

        # --- 1. Load Data ---
        print("\n--- 1. Loading Data ---")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"))
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"))
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"))

        # --- 2. Initial Prep: Log Transform, Dates ---
        print("\n--- 2. Initial Data Prep ---")
        try: 
            close_train = x_train_df["CLOSE"].astype(np.float32).values
            close_val = x_val_df["CLOSE"].astype(np.float32).values
            close_test = x_test_df["CLOSE"].astype(np.float32).values
        except Exception as e: 
            raise ValueError(f"Error converting 'CLOSE': {e}")
        
        log_train = np.log1p(np.maximum(0, close_train))
        log_val = np.log1p(np.maximum(0, close_val))
        log_test = np.log1p(np.maximum(0, close_test))
        print(f"Log transform applied. Train shape: {log_train.shape}")
        
        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None

        # --- 3. Feature Generation (Only Log Returns) ---
        print("\n--- 3. Feature Generation (Log Returns Only) ---")
        features_train, features_val, features_test = {}, {}, {}
        
        # 3.a. Log Returns (normalized)
        log_ret_train = np.diff(log_train, prepend=log_train[0])
        features_train['log_return'] = self._normalize_series(log_ret_train, 'log_return', fit=True)
        
        log_ret_val = np.diff(log_val, prepend=log_val[0])
        features_val['log_return'] = self._normalize_series(log_ret_val, 'log_return', fit=False)
        
        log_ret_test = np.diff(log_test, prepend=log_test[0])
        features_test['log_return'] = self._normalize_series(log_ret_test, 'log_return', fit=False)
        print("Generated: Log Returns (Normalized)")

        # 3.b. Add Original X Columns (excluding CLOSE)
        print("\n--- 3.b Preparing Original X Columns ---")
        original_x_cols = [col for col in x_train_df.columns if col != 'CLOSE']
        
        if not original_x_cols:
            print("WARN: No original columns found besides 'CLOSE'.")
            aligned_original_train_dict, aligned_original_val_dict, aligned_original_test_dict = {}, {}, {}
        else:
            print(f"Including original columns: {original_x_cols}")
            
            # Get base length from log returns
            base_len_train = len(features_train['log_return'])
            base_len_val = len(features_val['log_return'])
            base_len_test = len(features_test['log_return'])
            
            # Align original columns to match log returns length
            aligned_original_train_df = x_train_df[original_x_cols].iloc[-base_len_train:]
            aligned_original_val_df = x_val_df[original_x_cols].iloc[-base_len_val:]
            aligned_original_test_df = x_test_df[original_x_cols].iloc[-base_len_test:]
            
            # Convert to dictionary of numpy arrays
            aligned_original_train_dict = {col: aligned_original_train_df[col].values.astype(np.float32) for col in original_x_cols}
            aligned_original_val_dict = {col: aligned_original_val_df[col].values.astype(np.float32) for col in original_x_cols}
            aligned_original_test_dict = {col: aligned_original_test_df[col].values.astype(np.float32) for col in original_x_cols}

        # --- 4. Combine Features ---
        print("\n--- 4. Combining Features ---")
        all_features_train = {**aligned_original_train_dict, **features_train}
        all_features_val = {**aligned_original_val_dict, **features_val}
        all_features_test = {**aligned_original_test_dict, **features_test}

        # Get aligned length from log returns
        aligned_len_train = len(features_train['log_return'])
        aligned_len_val = len(features_val['log_return'])
        aligned_len_test = len(features_test['log_return'])
        
        dates_train_aligned = dates_train[-aligned_len_train:] if dates_train is not None and aligned_len_train > 0 else None
        dates_val_aligned = dates_val[-aligned_len_val:] if dates_val is not None and aligned_len_val > 0 else None
        dates_test_aligned = dates_test[-aligned_len_test:] if dates_test is not None and aligned_len_test > 0 else None
        
        print(f"Final aligned feature length: Train={aligned_len_train}, Val={aligned_len_val}, Test={aligned_len_test}")

        # --- 5. Windowing Features & Stacking ---
        print("\n--- 5. Windowing Features & Channel Stacking ---")
        X_train_channels, X_val_channels, X_test_channels = [], [], []
        feature_names = []
        x_dates_train, x_dates_val, x_dates_test = None, None, None
        first_feature_dates_captured = False

        # Define feature order: log_return first, then original columns alphabetically
        windowing_order = ['log_return'] + sorted([k for k, v in aligned_original_train_dict.items() if v is not None])
        print(f"Feature order for windowing: {windowing_order}")

        # Use max_horizon for windowing function
        time_horizon_for_windowing = max_horizon

        for name in windowing_order:
            series_train = all_features_train.get(name)
            series_val = all_features_val.get(name)
            series_test = all_features_test.get(name)

            if series_train is not None and series_val is not None and series_test is not None and \
               len(series_train) == aligned_len_train and \
               len(series_val) == aligned_len_val and \
               len(series_test) == aligned_len_test:

                print(f"Windowing feature: {name}...", end="")
                try:
                    win_train, _, dates_win_train = self.create_sliding_windows(series_train, window_size, time_horizon_for_windowing, dates_train_aligned)
                    win_val, _, dates_win_val = self.create_sliding_windows(series_val, window_size, time_horizon_for_windowing, dates_val_aligned)
                    win_test, _, dates_win_test = self.create_sliding_windows(series_test, window_size, time_horizon_for_windowing, dates_test_aligned)

                    if win_train.shape[0] > 0 and win_val.shape[0] > 0 and win_test.shape[0] > 0:
                        if not first_feature_dates_captured:
                            expected_samples_train = win_train.shape[0]
                            expected_samples_val = win_val.shape[0]
                            expected_samples_test = win_test.shape[0]
                            print(f" Initializing sample counts: Train={expected_samples_train}, Val={expected_samples_val}, Test={expected_samples_test}", end="")

                        if win_train.shape[0] == expected_samples_train and \
                           win_val.shape[0] == expected_samples_val and \
                           win_test.shape[0] == expected_samples_test:

                            X_train_channels.append(win_train)
                            X_val_channels.append(win_val)
                            X_test_channels.append(win_test)
                            feature_names.append(name)
                            print(" Appended.")

                            if not first_feature_dates_captured:
                                x_dates_train, x_dates_val, x_dates_test = dates_win_train, dates_win_val, dates_win_test
                                first_feature_dates_captured = True
                        else:
                             print(f" Skipping '{name}' due to inconsistent sample count.")
                    else:
                        print(f" Skipping '{name}' (windowing produced 0 samples).")
                except Exception as e:
                    print(f" FAILED windowing '{name}'. Error: {e}. Skipping.")
            else:
                 print(f"WARN: Feature '{name}' skipped - not valid across all splits.")

        # --- 6. Stack channels ---
        if not X_train_channels: 
            raise RuntimeError("No feature channels available after windowing!")
        
        print("\n--- 6. Stacking Feature Channels ---")
        num_samples_train = X_train_channels[0].shape[0]
        num_samples_val = X_val_channels[0].shape[0]
        num_samples_test = X_test_channels[0].shape[0]
        
        X_train_combined = np.stack(X_train_channels, axis=-1).astype(np.float32)
        X_val_combined = np.stack(X_val_channels, axis=-1).astype(np.float32)
        X_test_combined = np.stack(X_test_channels, axis=-1).astype(np.float32)
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Included features: {feature_names}")

        # --- 7. Target Processing ---
        print("\n--- 7. Target Processing ---")
        target_column = config["target_column"]
        use_returns = config.get("use_returns", False)

        # Usamos slicing directo: el primer valor válido de baseline está en close[window_size]
        baseline_train = close_train[window_size:window_size + num_samples_train]
        baseline_val = close_val[window_size:window_size + num_samples_val]
        baseline_test = close_test[window_size:window_size + num_samples_test]

        print(f"Baseline shapes: Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")

        # Fechas del baseline alineadas con ese slicing
        baseline_train_dates = dates_train[window_size:window_size + num_samples_train] if dates_train is not None else None
        baseline_val_dates   = dates_val[window_size:window_size + num_samples_val]     if dates_val is not None else None
        baseline_test_dates  = dates_test[window_size:window_size + num_samples_test]   if dates_test is not None else None

        # Target series cruda
        target_train_raw = y_train_df[target_column].astype(np.float32).values
        target_val_raw = y_val_df[target_column].astype(np.float32).values
        target_test_raw = y_test_df[target_column].astype(np.float32).values

        # Final targets (multi-horizon)
        y_train_final_list, y_val_final_list, y_test_final_list = [], [], []
        target_returns_means, target_returns_stds = [], []

        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")

        for h in predicted_horizons:
            # Target real para predicción desde t → t+h
            # Recalcular baseline para alinearse con target desplazado h
            baseline_train_h = baseline_train[:len(baseline_train) - h]
            baseline_val_h   = baseline_val[:len(baseline_val) - h]
            baseline_test_h  = baseline_test[:len(baseline_test) - h]

            target_train_h = target_train_raw[window_size + h : window_size + h + len(baseline_train_h)]
            target_val_h   = target_val_raw[window_size + h : window_size + h + len(baseline_val_h)]
            target_test_h  = target_test_raw[window_size + h : window_size + h + len(baseline_test_h)]


            if use_returns:
                target_train_h = target_train_h - baseline_train_h
                target_val_h   = target_val_h - baseline_val_h
                target_test_h  = target_test_h - baseline_test_h


                # Normalizar con media y std de TRAIN
                mean_h = target_train_h.mean()
                std_h = target_train_h.std()
                if std_h < 1e-8:
                    std_h = 1.0  # Evita división por cero


        y_train_final_list.append(target_train_h.astype(np.float32))
        y_val_final_list.append(target_val_h.astype(np.float32))
        y_test_final_list.append(target_test_h.astype(np.float32))

        # Save normalization stats in params
        self.params['target_returns_mean'] = target_returns_means
        self.params['target_returns_std'] = target_returns_stds

        if use_returns:
            print("Per-horizon target normalization stats:")
            for i, (mean, std) in enumerate(zip(target_returns_means, target_returns_stds)):
                horizon = predicted_horizons[i] if i < len(predicted_horizons) else f"index {i}"
                print(f"  Horizon {horizon}: Mean={mean:.6f}, Std={std:.6f}")
        else:
            print("Per-horizon normalization skipped (use_returns=False). Using Mean=0.0, Std=1.0 for all horizons.")



        # Assign dates based on X windows
        y_dates_train, y_dates_val, y_dates_test = x_dates_train, x_dates_val, x_dates_test
        print("Target processing complete.")

        # --- 8. Final Date Consistency Check ---
        print("\n--- 8. Final Date Consistency Checks ---")
        verify_date_consistency([list(x_dates_train) if x_dates_train is not None else None, 
                                list(y_dates_train) if y_dates_train is not None else None], "Train X/Y Dates")
        verify_date_consistency([list(x_dates_val) if x_dates_val is not None else None, 
                                list(y_dates_val) if y_dates_val is not None else None], "Val X/Y Dates")
        verify_date_consistency([list(x_dates_test) if x_dates_test is not None else None, 
                                list(y_dates_test) if y_dates_test is not None else None], "Test X/Y Dates")

        # --- 9. Prepare Return Dictionary ---
        print("\n--- 9. Preparing Final Output ---")
        ret = {}
        
        # X data
        ret["x_train"] = X_train_combined
        ret["x_val"] = X_val_combined
        ret["x_test"] = X_test_combined

        # Y data (multi-horizon list)
        ret["y_train"] = y_train_final_list
        ret["y_val"] = y_val_final_list
        ret["y_test"] = y_test_final_list

        # Dates
        ret["x_train_dates"] = x_dates_train
        ret["y_train_dates"] = y_dates_train
        ret["x_val_dates"] = x_dates_val
        ret["y_val_dates"] = y_dates_val
        ret["x_test_dates"] = x_dates_test
        ret["y_test_dates"] = y_dates_test
        
        # Baseline data (target values at prediction time)
        ret["baseline_train"] = baseline_train
        ret["baseline_val"] = baseline_val
        ret["baseline_test"] = baseline_test
        ret["baseline_train_dates"] = x_dates_train
        ret["baseline_val_dates"] = x_dates_val
        ret["baseline_test_dates"] = x_dates_test
        
        # Test close prices (for compatibility)
        ret["test_close_prices"] = close_test[window_size:len(close_test)-max_horizon]
        
        # Feature names (now properly extracted from data)
        ret["feature_names"] = feature_names
        
        print(f"Final shapes:")
        #print(f"  X: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print(f"  Y: {len(predicted_horizons)} horizons, Train={len(ret['y_train'][0])}, Val={len(ret['y_val'][0])}, Test={len(ret['y_test'][0])}")
        print(f"  Baselines: Train={len(baseline_train)}, Val={len(baseline_val)}, Test={len(baseline_test)}")
        print(f"  Horizons: {predicted_horizons}")
        print(f"  Features ({len(feature_names)}): {feature_names}")
        print(f"  Target normalization per horizon:")
        self.params["target_returns_mean"] = target_returns_means
        self.params["target_returns_std"] = target_returns_stds

        for i, h in enumerate(predicted_horizons):
            mean_h = target_returns_means[i]
            std_h  = target_returns_stds[i]
            print(f"    Horizon {h}: mean={mean_h:.6f}, std={std_h:.6f}")
        
        # Cleanup
        del x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df
        
        print("\n" + "="*15 + " Preprocessing Finished " + "="*15)

        assert len(target_returns_means) == len(predicted_horizons), "Mismatch: means vs horizons"
        assert len(target_returns_stds) == len(predicted_horizons), "Mismatch: stds vs horizons"

        self.params["target_returns_mean"] = target_returns_means
        self.params["target_returns_std"] = target_returns_stds

        ret["target_returns_mean"] = target_returns_means
        ret["target_returns_std"] = target_returns_stds
        return ret

    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        return processed_data, self.params