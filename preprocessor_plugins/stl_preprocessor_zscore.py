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
    # Plugin parameters
    plugin_params = {
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

    def create_causality_safe_windows_and_targets(self, features_df, target_values, dates, window_size, horizons):
        """
        Creates causality-safe sliding windows and multi-horizon targets with PERFECT alignment.
        
        CRITICAL ALIGNMENT RULES:
        - For tick t: Window contains [t-window_size : t] (EXCLUDES tick t, NO future data)
        - For tick t: Target = target_values[t+horizon] - target_values[t] 
        - For tick t: Date = dates[t] (the current prediction time)
        - All arrays must have EXACT same length and alignment
        
        Args:
            features_df: DataFrame with all feature columns (CLOSE, OPEN, HIGH, LOW, VOLUME, etc.)
            target_values: Array of target column values 
            dates: Array of datetime indices
            window_size: Size of sliding window
            horizons: List of prediction horizons
            
        Returns:
            tuple: (X_windows, Y_targets_dict, window_dates, feature_names)
        """
        # Convert features to numpy array and get feature names
        if isinstance(features_df, pd.DataFrame):
            feature_names = list(features_df.columns)
            features_array = features_df.values.astype(np.float32)
        else:
            # Fallback for array input (backward compatibility)
            feature_names = ["CLOSE"]
            features_array = np.array(features_df, dtype=np.float32).reshape(-1, 1)
        
        target_values = np.array(target_values, dtype=np.float32)
        
        if len(features_array) != len(target_values):
            raise ValueError(f"CRITICAL: Length mismatch features={len(features_array)} vs target_values={len(target_values)}")
        
        total_length = len(features_array)
        max_horizon = max(horizons)
        num_features = features_array.shape[1]
        
        print(f"Feature engineering: {num_features} features detected")
        print(f"Feature names: {feature_names}")
        
        # CRITICAL: Determine exact valid range for tick t
        min_t = window_size - 1  # First valid t (has enough history [t-window_size+1 : t+1])
        max_t = total_length - max_horizon - 1  # Last valid t (has enough future for max horizon)
        
        if min_t > max_t:
            print(f"CRITICAL ERROR: Not enough data. Need {window_size + max_horizon} points, got {total_length}")
            return np.array([], dtype=np.float32), {h: np.array([], dtype=np.float32) for h in horizons}, np.array([], dtype=object), feature_names
        
        # Valid tick range [min_t, max_t] inclusive
        valid_ticks = list(range(min_t, max_t + 1))
        num_windows = len(valid_ticks)
        
        print(f"ALIGNMENT CHECK: Creating {num_windows} windows for ticks [{min_t}, {max_t}]")
        print(f"  Window size: {window_size}, Max horizon: {max_horizon}, Total length: {total_length}")
        print(f"  Features per window: {num_features}")
        
        # Pre-allocate arrays - NOW WITH CORRECT FEATURE DIMENSION
        X_windows = np.zeros((num_windows, window_size, num_features), dtype=np.float32)
        Y_targets = {h: np.zeros(num_windows, dtype=np.float32) for h in horizons}
        window_dates = []
        
        # Create windows and targets with PERFECT alignment
        for i, t in enumerate(valid_ticks):
            # WINDOW: [t-window_size : t] - INCLUDES current tick t, NO future data
            # WINDOW: [t-window_size+1 : t+1] - INCLUDES current tick t, NO future data  
            window_start = t - window_size + 1
            window_end = t + 1

            # Sanity checks
            assert window_start >= 0, f"Window start {window_start} < 0 for tick {t}"
            assert window_end <= total_length, f"Window end {window_end} > {total_length} for tick {t}"
            assert window_end - window_start == window_size, f"Window size mismatch: {window_end - window_start} != {window_size}"
            
            # Fill window with ALL FEATURES (not just log returns)
            X_windows[i, :, :] = features_array[window_start:window_end, :]
            
            # TARGETS: target_values[t+horizon] - target_values[t] for each horizon
            current_target = target_values[t]
            for h in horizons:
                future_idx = t + h
                assert future_idx < total_length, f"Future index {future_idx} >= {total_length} for tick {t}, horizon {h}"
                future_target = target_values[future_idx]
                Y_targets[h][i] = future_target - current_target
            
            # DATE: dates[t] (the current prediction time)
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
        
        print(f"PERFECT ALIGNMENT: Created {len(X_windows)} windows with shape {X_windows.shape}")
        print(f"Window shape breakdown: [samples={X_windows.shape[0]}, timesteps={X_windows.shape[1]}, features={X_windows.shape[2]}]")
        
        # Final verification
        for h in horizons:
            assert len(Y_targets[h]) == num_windows, f"Target length mismatch for horizon {h}"
        assert len(window_dates_arr) == num_windows, "Date length mismatch"
        
        return X_windows, Y_targets, window_dates_arr, feature_names




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
        
        # Prepare feature DataFrames (all columns except target-specific ones)
        feature_cols = [col for col in x_train_df.columns if col != target_column]
        x_train_features = x_train_df[feature_cols]
        x_val_features = x_val_df[feature_cols]
        x_test_features = x_test_df[feature_cols]
        
        print(f"Using {len(feature_cols)} feature columns: {feature_cols}")
        
        print("Processing TRAIN data...")
        X_train, Y_train_dict, x_dates_train, feature_names = self.create_causality_safe_windows_and_targets(
            x_train_features, target_train, dates_train, window_size, predicted_horizons
        )
        
        print("Processing VALIDATION data...")
        X_val, Y_val_dict, x_dates_val, _ = self.create_causality_safe_windows_and_targets(
            x_val_features, target_val, dates_val, window_size, predicted_horizons
        )
        
        print("Processing TEST data...")
        X_test, Y_test_dict, x_dates_test, _ = self.create_causality_safe_windows_and_targets(
            x_test_features, target_test, dates_test, window_size, predicted_horizons
        )

        # --- 4. Calculate and Apply Per-Horizon Target Normalization (Z-Score) ---
        print("\n--- 4. Per-Horizon Target Normalization (Z-Score) ---")
        if use_returns:
            print("Calculating Z-score normalization parameters per horizon from training data...")
            
            # Initialize lists to store mean and std for each horizon
            target_returns_mean = []
            target_returns_std = []
            
            # Calculate normalization stats for each horizon separately
            for i, h in enumerate(predicted_horizons):
                train_targets_h = Y_train_dict[h]
                mean_h = float(np.mean(train_targets_h))
                std_h = float(np.std(train_targets_h))
                
                if std_h < 1e-9:
                    print(f"WARN: Target returns for horizon {h} have near-zero std. Using dummy normalization.")
                    std_h = 1.0
                
                target_returns_mean.append(mean_h)
                target_returns_std.append(std_h)
                
                print(f"Horizon {h}: mean={mean_h:.6f}, std={std_h:.6f}")
            
            # Store normalization parameters in self.params as lists
            self.params['target_returns_mean'] = target_returns_mean
            self.params['target_returns_std'] = target_returns_std
            
            # Apply Z-score normalization per horizon to all splits
            for i, h in enumerate(predicted_horizons):
                mean_h = target_returns_mean[i]
                std_h = target_returns_std[i]
                
                Y_train_dict[h] = (Y_train_dict[h] - mean_h) / std_h
                Y_val_dict[h] = (Y_val_dict[h] - mean_h) / std_h
                Y_test_dict[h] = (Y_test_dict[h] - mean_h) / std_h
                
            print("Per-horizon Z-score normalization applied to all target splits.")
        else:
            # No normalization, but set params to lists of zeros for consistency
            self.params['target_returns_mean'] = [0.0] * len(predicted_horizons)
            self.params['target_returns_std'] = [1.0] * len(predicted_horizons)
            print("Target normalization skipped (use_returns=False).")
        
        # --- 5. Create Baseline Values ---
        print("\n--- 5. Creating Baseline Values ---")
        # Baseline values are the target column values at prediction time (tick t)
        # These correspond exactly to the current target values used in target calculation
        
        baseline_train = np.zeros(len(X_train), dtype=np.float32)
        baseline_val = np.zeros(len(X_val), dtype=np.float32)
        baseline_test = np.zeros(len(X_test), dtype=np.float32)
        
        # Extract baseline values from the target column at tick t
        max_horizon = max(predicted_horizons)
        
        # For training data
        min_t_train = window_size
        max_t_train = len(target_train) - max_horizon - 1
        for i, t in enumerate(range(min_t_train, max_t_train + 1)):
            baseline_train[i] = target_train[t]
            
        # For validation data
        min_t_val = window_size
        max_t_val = len(target_val) - max_horizon - 1
        for i, t in enumerate(range(min_t_val, max_t_val + 1)):
            baseline_val[i] = target_val[t]
            
        # For test data
        min_t_test = window_size
        max_t_test = len(target_test) - max_horizon - 1
        for i, t in enumerate(range(min_t_test, max_t_test + 1)):
            baseline_test[i] = target_test[t]
        
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
        print(f"  X: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print(f"  Y: {len(predicted_horizons)} horizons, Train={len(ret['y_train'][0])}, Val={len(ret['y_val'][0])}, Test={len(ret['y_test'][0])}")
        print(f"  Baselines: Train={len(baseline_train)}, Val={len(baseline_val)}, Test={len(baseline_test)}")
        print(f"  Horizons: {predicted_horizons}")
        print(f"  Features ({len(feature_names)}): {feature_names}")
        print(f"  Target normalization per horizon:")
        for i, h in enumerate(predicted_horizons):
            mean_h = self.params.get('target_returns_mean', [0])[i] if isinstance(self.params.get('target_returns_mean'), list) else 0
            std_h = self.params.get('target_returns_std', [1])[i] if isinstance(self.params.get('target_returns_std'), list) else 1
            print(f"    Horizon {h}: mean={mean_h:.6f}, std={std_h:.6f}")
        
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