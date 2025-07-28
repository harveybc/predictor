# Ensure these imports are present at the top of the file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    # Simplified plugin parameters - removed all decomposition parameters
    plugin_params = {
        # --- File Paths ---
        # --- Data Loading ---
        "headers": True,
        "max_steps_train": None, "max_steps_val": None, "max_steps_test": None,
        "target_column": "TARGET",
        # --- Windowing & Horizons ---
        "window_size": 48,
        "predicted_horizons": [1, 6, 12, 24], # Multi-horizon support
        # --- Feature Engineering Flags ---
        "use_returns": True, # Flag for Y calculation
        "normalize_features": True,
    }
    
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "use_returns", "normalize_features",
        "target_returns_mean", "target_returns_std"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.scalers = {}

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        for key, value in kwargs.items(): 
            self.params[key] = value

    def get_debug_info(self):
        """Return debug information for the plugin."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

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
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Converting to DatetimeIndex for {file_path}...", end="")
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
            
            # Verify required columns
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

    def create_causality_safe_windows_and_targets(self, close_prices, target_values, dates, window_size, horizons):
        """
        Creates causality-safe sliding windows and multi-horizon targets.
        
        For each valid tick t (starting from window_size):
        - Window: uses data from [t-window_size+1 : t+1] (inclusive of current tick t)
        - Target: target_values[t+horizon] - target_values[t] for each horizon
        - Date: dates[t] (the current tick date)
        
        Args:
            close_prices: Array of close prices for feature calculation
            target_values: Array of target column values 
            dates: Array of datetime indices
            window_size: Size of sliding window
            horizons: List of prediction horizons
            
        Returns:
            tuple: (X_windows, Y_targets_dict, window_dates)
        """
        close_prices = np.array(close_prices, dtype=np.float32)
        target_values = np.array(target_values, dtype=np.float32)
        
        if len(close_prices) != len(target_values):
            raise ValueError(f"Length mismatch: close_prices={len(close_prices)}, target_values={len(target_values)}")
        
        max_horizon = max(horizons)
        total_length = len(close_prices)
        
        # Calculate log returns from close prices
        log_prices = np.log1p(np.maximum(0, close_prices))
        log_returns = np.diff(log_prices, prepend=log_prices[0])
        
        # Normalize log returns
        log_returns_norm = self._normalize_series(log_returns, 'log_return', fit=True)
        
        # Determine valid range for windowing
        # Start from window_size (so we have enough history)
        # End early enough so we have targets for max_horizon
        start_idx = window_size
        end_idx = total_length - max_horizon
        
        if start_idx >= end_idx:
            print(f"WARN: Not enough data for windowing. Need at least {window_size + max_horizon} points, got {total_length}")
            return np.array([], dtype=np.float32), {h: np.array([], dtype=np.float32) for h in horizons}, np.array([], dtype=object)
        
        num_windows = end_idx - start_idx
        print(f"Creating {num_windows} causality-safe windows (start={start_idx}, end={end_idx-1})")
        
        # Pre-allocate arrays
        X_windows = np.zeros((num_windows, window_size, 1), dtype=np.float32)  # 1 feature channel
        Y_targets = {h: np.zeros(num_windows, dtype=np.float32) for h in horizons}
        window_dates = []
        
        # Create windows and targets
        for i, t in enumerate(range(start_idx, end_idx)):
            # Window: use data from [t-window_size+1 : t+1] 
            # This includes the current tick t, but no future information
            window_start = t - window_size + 1
            window_end = t + 1
            X_windows[i, :, 0] = log_returns_norm[window_start:window_end]
            
            # Targets: calculate returns for each horizon
            # target[t+h] - target[t] where target is the target column value
            current_target = target_values[t]
            for h in horizons:
                future_target = target_values[t + h]
                Y_targets[h][i] = future_target - current_target
            
            # Date: use the current tick date (t)
            if dates is not None and t < len(dates):
                window_dates.append(dates[t])
            else:
                window_dates.append(None)
        
        # Convert dates to appropriate array
        if dates is not None:
            window_dates_arr = np.array(window_dates, dtype=object)
            if all(isinstance(d, pd.Timestamp) for d in window_dates if d is not None):
                try:
                    window_dates_arr = np.array(window_dates, dtype='datetime64[ns]')
                except (ValueError, TypeError):
                    pass
        else:
            window_dates_arr = np.array(window_dates, dtype=object)
        
        print(f"Created {len(X_windows)} windows with targets for horizons {horizons}")
        return X_windows, Y_targets, window_dates_arr

    def process_data(self, config):
        """
        Processes data with causality-safe windowing and proper datetime alignment.
        """
        print("\n" + "="*15 + " Starting Preprocessing " + "="*15)
        self.set_params(**config)
        config = self.params
        self.scalers = {}

        # Get key parameters
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        target_column = config['target_column']
        use_returns = config.get('use_returns', True)
        
        if not isinstance(predicted_horizons, list) or not predicted_horizons:
            raise ValueError("'predicted_horizons' must be a non-empty list.")

        # --- 1. Load Data ---
        print("\n--- 1. Loading Data ---")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"))
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"))
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"))

        # --- 2. Extract Data Arrays ---
        print("\n--- 2. Extracting Data Arrays ---")
        
        # Verify target column exists
        if target_column not in y_train_df.columns:
            raise ValueError(f"Column '{target_column}' not found in Y train.")
        
        # Extract close prices and target values
        close_train = x_train_df["CLOSE"].astype(np.float32).values
        close_val = x_val_df["CLOSE"].astype(np.float32).values
        close_test = x_test_df["CLOSE"].astype(np.float32).values
        
        target_train = y_train_df[target_column].astype(np.float32).values
        target_val = y_val_df[target_column].astype(np.float32).values
        target_test = y_test_df[target_column].astype(np.float32).values
        
        # Extract dates
        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None

        # --- 3. Create Causality-Safe Windows and Targets ---
        print("\n--- 3. Creating Causality-Safe Windows and Targets ---")
        
        print("Processing TRAIN data...")
        X_train, Y_train_dict, x_dates_train = self.create_causality_safe_windows_and_targets(
            close_train, target_train, dates_train, window_size, predicted_horizons
        )
        
        print("Processing VALIDATION data...")
        X_val, Y_val_dict, x_dates_val = self.create_causality_safe_windows_and_targets(
            close_val, target_val, dates_val, window_size, predicted_horizons
        )
        
        print("Processing TEST data...")
        X_test, Y_test_dict, x_dates_test = self.create_causality_safe_windows_and_targets(
            close_test, target_test, dates_test, window_size, predicted_horizons
        )

        # --- 4. Normalize Targets if using returns ---
        print("\n--- 4. Target Normalization ---")
        if use_returns:
            print("Normalizing target returns using training statistics...")
            
            # Calculate normalization stats from training data for first horizon
            first_horizon = predicted_horizons[0]
            target_returns_mean = Y_train_dict[first_horizon].mean()
            target_returns_std = Y_train_dict[first_horizon].std()
            
            if target_returns_std < 1e-9:
                print("WARN: Target returns have near-zero std. Using dummy normalization.")
                target_returns_std = 1.0
            
            # Store normalization stats
            self.params['target_returns_mean'] = float(target_returns_mean)
            self.params['target_returns_std'] = float(target_returns_std)
            
            print(f"Target normalization stats: mean={target_returns_mean:.6f}, std={target_returns_std:.6f}")
            
            # Apply normalization to all horizons and all splits
            for h in predicted_horizons:
                Y_train_dict[h] = (Y_train_dict[h] - target_returns_mean) / target_returns_std
                Y_val_dict[h] = (Y_val_dict[h] - target_returns_mean) / target_returns_std
                Y_test_dict[h] = (Y_test_dict[h] - target_returns_mean) / target_returns_std
        
        # --- 5. Create Baseline Values ---
        print("\n--- 5. Creating Baseline Values ---")
        # Baseline should be the close price at the time of prediction (current tick)
        # These align exactly with the window dates since we use tick t for both
        
        baseline_train = np.zeros(len(X_train), dtype=np.float32)
        baseline_val = np.zeros(len(X_val), dtype=np.float32)
        baseline_test = np.zeros(len(X_test), dtype=np.float32)
        
        # Extract baseline values - these are the close prices at prediction time
        start_idx_train = window_size
        start_idx_val = window_size  
        start_idx_test = window_size
        
        max_horizon = max(predicted_horizons)
        end_idx_train = len(close_train) - max_horizon
        end_idx_val = len(close_val) - max_horizon
        end_idx_test = len(close_test) - max_horizon
        
        for i, t in enumerate(range(start_idx_train, end_idx_train)):
            baseline_train[i] = close_train[t]
            
        for i, t in enumerate(range(start_idx_val, end_idx_val)):
            baseline_val[i] = close_val[t]
            
        for i, t in enumerate(range(start_idx_test, end_idx_test)):
            baseline_test[i] = close_test[t]
        
        print(f"Baseline shapes: Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")

        # --- 6. Final Date Consistency Check ---
        print("\n--- 6. Final Date Consistency Checks ---")
        verify_date_consistency([
            list(x_dates_train) if x_dates_train is not None else None
        ], "Train Dates")
        verify_date_consistency([
            list(x_dates_val) if x_dates_val is not None else None
        ], "Val Dates")
        verify_date_consistency([
            list(x_dates_test) if x_dates_test is not None else None
        ], "Test Dates")

        # --- 7. Prepare Return Dictionary ---
        print("\n--- 7. Preparing Final Output ---")
        ret = {}
        
        # X data (features)
        ret["x_train"] = X_train
        ret["x_val"] = X_val
        ret["x_test"] = X_test
        
        # Y data (targets as lists for multi-horizon)
        ret["y_train"] = [Y_train_dict[h] for h in predicted_horizons]
        ret["y_val"] = [Y_val_dict[h] for h in predicted_horizons]
        ret["y_test"] = [Y_test_dict[h] for h in predicted_horizons]
        
        # Dates (all aligned to current tick t)
        ret["x_train_dates"] = x_dates_train
        ret["y_train_dates"] = x_dates_train  # Same as X dates
        ret["x_val_dates"] = x_dates_val
        ret["y_val_dates"] = x_dates_val
        ret["x_test_dates"] = x_dates_test
        ret["y_test_dates"] = x_dates_test
        
        # Baseline data (close prices at prediction time)
        ret["baseline_train"] = baseline_train
        ret["baseline_val"] = baseline_val
        ret["baseline_test"] = baseline_test
        ret["baseline_train_dates"] = x_dates_train
        ret["baseline_val_dates"] = x_dates_val
        ret["baseline_test_dates"] = x_dates_test
        
        # Test close prices (for compatibility)
        ret["test_close_prices"] = baseline_test
        
        # Feature names
        ret["feature_names"] = ["log_return"]
        
        print(f"Final shapes:")
        print(f"  X: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print(f"  Y: {len(predicted_horizons)} horizons, Train={len(ret['y_train'][0])}, Val={len(ret['y_val'][0])}, Test={len(ret['y_test'][0])}")
        print(f"  Baselines: Train={len(baseline_train)}, Val={len(baseline_val)}, Test={len(baseline_test)}")
        print(f"  Horizons: {predicted_horizons}")
        
        # Cleanup
        del x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df
        
        print("\n" + "="*15 + " Preprocessing Finished " + "="*15)
        return ret

    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        return processed_data, self.params