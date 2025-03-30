# Ensure these imports are present at the top of the file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import json
from app.data_handler import load_csv # Assuming load_csv is here
# from app.data_handler import write_csv # Not used directly in process_data
from scipy.signal import hilbert
from scipy.stats import shapiro
# Include the user-provided verify_date_consistency function
def verify_date_consistency(date_lists, dataset_name):
    """
    Verifies that all date arrays in date_lists have the same first and last elements.
    Prints a warning if any array does not match.
    """
    if not date_lists:
        print(f"No date lists provided for {dataset_name} consistency check.")
        return

    # Filter out None or empty lists/arrays
    valid_date_lists = [d for d in date_lists if d is not None and len(d) > 0]

    if not valid_date_lists:
        print(f"No valid date arrays with content found for {dataset_name}.")
        return

    # Check lengths first (more fundamental check)
    first_len = len(valid_date_lists[0])
    all_lengths_match = all(len(d) == first_len for d in valid_date_lists)

    if not all_lengths_match:
         lengths = [len(d) for d in valid_date_lists]
         print(f"WARN: Length mismatch in {dataset_name} data: {lengths}")
         # Optionally raise an error if strict alignment is needed
         # raise ValueError(f"Inconsistent lengths found in {dataset_name} data used for date check.")
         return # Stop further date comparison if lengths differ

    # Proceed with first/last element check only if lengths match
    print(f"{dataset_name} data lengths are consistent: {first_len}")
    first = valid_date_lists[0][0]
    last = valid_date_lists[0][-1]
    consistent = True
    for i, d in enumerate(valid_date_lists):
        if d[0] != first or d[-1] != last:
            print(f"Warning: Date array {i} in {dataset_name} does not match the others. First: {d[0]}, Last: {d[-1]}; expected First: {first}, Last: {last}.")
            consistent = False

    if consistent:
        print(f"{dataset_name} date ranges appear consistent.")


class PreprocessorPlugin: # Assuming this is within the PreprocessorPlugin class
    # Default plugin parameters, including stl_trend.
    plugin_params = {
        # Default file paths (adjust as needed)
        "x_train_file": "data/x_train.csv",
        "y_train_file": "data/y_train.csv",
        "x_validation_file": "data/x_val.csv",
        "y_validation_file": "data/y_val.csv",
        "x_test_file": "data/x_test.csv",
        "y_test_file": "data/y_test.csv",
        "headers": True,
        "max_steps_train": None,
        "max_steps_val": None,
        "max_steps_test": None,
        "window_size": 48,
        # "time_horizon": 6, # Deprecated in favor of predicted_horizons
        "predicted_horizons": [1, 6, 12, 24], # Example multi-horizons
        "use_returns": True,
        "stl_period": 24,
        "stl_window": 48,
        "stl_trend": 49,
        "stl_plot_file": "stl_plot.png",
        "target_column": "TARGET" # Added target column default
        # "pos_encoding_dim": 16 # Removed if not used here
    }
    plugin_debug_vars = ["window_size", "predicted_horizons", "use_returns", "stl_period", "stl_window", "stl_trend", "stl_plot_file"]

    def __init__(self):
        self.params = self.plugin_params.copy()

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

    # --- Assume these helper methods exist or are implemented ---
    def _load_data(self, file_path, max_rows, headers):
        # Placeholder: Implement actual data loading logic using app.data_handler
        print(f"Placeholder: Loading data from {file_path}...")
        try:
            df = load_csv(file_path, headers=headers, max_rows=max_rows)
            if not isinstance(df.index, pd.DatetimeIndex):
                try: # Attempt to convert index to datetime
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    print(f"WARN: Could not convert index of {file_path} to DatetimeIndex. Dates may be unavailable.")
                    df.index = None # Set index to None if conversion fails
            return df
        except Exception as e:
             print(f"ERROR loading {file_path}: {e}")
             raise # Re-raise error after logging


    def _rolling_stl(self, series, stl_window, period):
        # Placeholder: Uses original rolling STL logic
        print("Performing rolling STL decomposition...")
        n = len(series)
        num_points = n - stl_window + 1
        if num_points <= 0:
            raise ValueError(f"stl_window ({stl_window}) is too large for the series length ({n}).")
        trend = np.zeros(num_points)
        seasonal = np.zeros(num_points)
        resid = np.zeros(num_points)
        stl_trend_smoother = self.params.get("stl_trend") # Use trend smoother length from params

        for i in tqdm(range(stl_window, n + 1), desc="STL Decomposition", unit="window", disable=None): # Enable progress bar
            window = series[i - stl_window: i]
            # Ensure stl_trend_smoother is odd and suitable for STL window size
            current_stl_trend = stl_trend_smoother
            if current_stl_trend is not None:
                if current_stl_trend >= len(window):
                     current_stl_trend = len(window) - 1 if len(window) > 1 else None # Adjust if too large
                if current_stl_trend is not None and current_stl_trend % 2 == 0:
                     current_stl_trend += 1 # Ensure odd

            try:
                 stl = STL(window, period=period, trend=current_stl_trend, robust=True)
                 result = stl.fit()
                 trend[i - stl_window] = result.trend[-1]
                 seasonal[i - stl_window] = result.seasonal[-1]
                 resid[i - stl_window] = result.resid[-1]
            except Exception as e:
                 print(f"WARN: STL fit failed for window ending at index {i-1}. Error: {e}. Filling with NaN.")
                 trend[i - stl_window] = np.nan
                 seasonal[i - stl_window] = np.nan
                 resid[i - stl_window] = np.nan

        # Handle potential NaNs from failed fits if necessary (e.g., forward fill)
        trend = pd.Series(trend).fillna(method='ffill').fillna(method='bfill').values
        seasonal = pd.Series(seasonal).fillna(method='ffill').fillna(method='bfill').values
        resid = pd.Series(resid).fillna(method='ffill').fillna(method='bfill').values

        return trend, seasonal, resid

    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        # Placeholder: Uses original plotting logic
        print(f"Plotting decomposition to {file_path}...")
        plot_points = 480 # Limit plot points for readability
        if len(series) > plot_points:
            series = series[-plot_points:]
            trend = trend[-plot_points:]
            seasonal = seasonal[-plot_points:]
            resid = resid[-plot_points:]
        try:
            plt.figure(figsize=(12, 9))
            plt.subplot(411); plt.plot(series); plt.title("Log-Transformed Series (Recent)"); plt.grid(True, alpha=0.5)
            plt.subplot(412); plt.plot(trend, color="orange"); plt.title("Trend"); plt.grid(True, alpha=0.5)
            plt.subplot(413); plt.plot(seasonal, color="green"); plt.title("Seasonal"); plt.grid(True, alpha=0.5)
            plt.subplot(414); plt.plot(resid, color="red"); plt.title("Residual"); plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(file_path, dpi=300)
            plt.close()
            print(f"STL decomposition plot saved to {file_path}")
        except Exception as e:
             print(f"WARN: Failed to plot decomposition: {e}")


    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        # Placeholder: Uses original window creation logic
        # Requires time_horizon argument (passed as max_horizon) to limit window creation
        print(f"Creating sliding windows (size={window_size}, max_horizon={time_horizon})...")
        windows = []
        targets = [] # This target list will be ignored later
        date_windows = []
        n = len(data)
        # Stop loop early enough to allow for the maximum horizon
        if n < window_size + time_horizon:
             print(f"WARN: Data length ({n}) too short for window_size ({window_size}) + max_horizon ({time_horizon}). No windows created.")
             return np.array(windows, dtype=np.float32), np.array(date_windows), np.array(targets, dtype=np.float32)

        for i in range(n - window_size - time_horizon + 1):
            window_end_idx = i + window_size
            window = data[i : window_end_idx]
            target_idx = window_end_idx + time_horizon - 1 # Original target index logic
            target = data[target_idx] # This target is ignored
            windows.append(window)
            targets.append(target) # Ignored target
            if date_times is not None:
                # Date corresponds to the last point in the input window
                date_windows.append(date_times[window_end_idx - 1])

        return np.array(windows, dtype=np.float32), np.array(date_windows), np.array(targets, dtype=np.float32)


    def process_data(self, config):
        """
        Processes data for multi-horizon forecasting with STL decomposition.
        Generates multi-channel input features and multi-horizon targets.
        """
        self.params.update(config) # Ensure config overrides defaults
        config = self.params # Use merged params

        headers = config.get("headers", True)

        # 1. Load X and Y data.
        print("Loading datasets...")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), headers)
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), headers)
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), headers)
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), headers)
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), headers)
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), headers)

        # 2. Extract 'CLOSE', apply log transform, get dates.
        if "CLOSE" not in x_train_df.columns: raise ValueError("Column 'CLOSE' not found in training X data.")
        close_train = x_train_df["CLOSE"].astype(np.float32).values
        close_val = x_val_df["CLOSE"].astype(np.float32).values
        close_test = x_test_df["CLOSE"].astype(np.float32).values

        log_train = np.log(close_train)
        log_val = np.log(close_val)
        log_test = np.log(close_test)

        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None
        print("Log transform applied.")

        # 3. Compute causal, rolling STL decomposition.
        stl_period = config.get("stl_period", 24)
        stl_window = config.get("stl_window", config.get("window_size")) # Default to window_size if not set
        if stl_window is None: raise ValueError("STL window size could not be determined (set stl_window or window_size).")
        if stl_window % 2 == 0: stl_window += 1 # Ensure odd window

        print(f"Computing STL decomposition (Period: {stl_period}, Window: {stl_window})...")
        trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period)
        trend_val, seasonal_val, resid_val = self._rolling_stl(log_val, stl_window, stl_period)
        trend_test, seasonal_test, resid_test = self._rolling_stl(log_test, stl_window, stl_period)

        # 4. Plot decomposition and print statistics for training data.
        stl_plot_file = config.get("stl_plot_file", "stl_plot.png")
        # Adjust series for plotting if rolling STL shortens the output
        plot_start_index = len(log_train) - len(trend_train)
        self._plot_decomposition(log_train[plot_start_index:], trend_train, seasonal_train, resid_train, stl_plot_file)

        # Calculate and print detailed STL statistics (keep original logic)
        trend_mean = np.mean(trend_train); trend_std = np.std(trend_train); trend_var = np.var(trend_train)
        seasonal_mean = np.mean(seasonal_train); seasonal_std = np.std(seasonal_train); seasonal_var = np.var(seasonal_train)
        resid_mean = np.mean(resid_train); resid_std = np.std(resid_train); resid_var = np.var(resid_train)
        snr = (trend_var + seasonal_var) / resid_var if resid_var != 0 else np.inf
        seasonal_ac = np.corrcoef(seasonal_train[:-stl_period], seasonal_train[stl_period:])[0,1] if len(seasonal_train) > stl_period else np.nan
        seasonal_fft = np.fft.rfft(seasonal_train); power = np.abs(seasonal_fft)**2
        freqs = np.fft.rfftfreq(len(seasonal_train)); dominant_freq = freqs[np.argmax(power)]
        expected_freq = 1.0 / stl_period; trend_diff_var = np.var(np.diff(trend_train))
        resid_ac = np.corrcoef(resid_train[:-1], resid_train[1:])[0,1] if len(resid_train) > 1 else np.nan
        try: stat, resid_pvalue = shapiro(resid_train)
        except Exception: resid_pvalue = np.nan
        analytic_signal = hilbert(seasonal_train); phase = np.angle(analytic_signal)
        circ_mean = np.angle(np.mean(np.exp(1j * phase)))
        circ_std = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phase)))))

        print("=== STL Decomposition Detailed Statistics (Training Data) ===")
        print(f"Trend     - Mean: {trend_mean:.4f}, Std: {trend_std:.4f}, Variance: {trend_var:.4f}")
        print(f"Seasonal  - Mean: {seasonal_mean:.4f}, Std: {seasonal_std:.4f}, Variance: {seasonal_var:.4f}")
        print(f"Residual  - Mean: {resid_mean:.4f}, Std: {resid_std:.4f}, Variance: {resid_var:.4f}")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.4f}")
        print(f"Seasonal Autocorrelation at lag {stl_period}: {seasonal_ac:.4f}")
        print(f"Dominant Frequency (spectral): {dominant_freq:.4f} (expected: ~{expected_freq:.4f})")
        print(f"Trend Smoothness (variance of first differences): {trend_diff_var:.4f}")
        print(f"Residual Autocorrelation (lag 1): {resid_ac:.4f}")
        print(f"Residual Normality Test p-value: {resid_pvalue:.4f}")
        print(f"Hilbert Phase of Seasonal - Circular Mean: {circ_mean:.4f}, Circular Std: {circ_std:.4f}")
        print("=============================================================")

        # 5. Create sliding windows for Input Features (X).
        window_size = config["window_size"]
        # Use the maximum prediction horizon to ensure enough data for all targets
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not all(isinstance(h, int) and h > 0 for h in predicted_horizons):
            raise TypeError("'predicted_horizons' must be a list of positive integers in config.")
        if not predicted_horizons: raise ValueError("'predicted_horizons' cannot be empty.")
        max_horizon = max(predicted_horizons)

        print(f"Creating sliding windows for inputs (Window Size: {window_size}, Max Horizon: {max_horizon})...")
        # Adjust start index based on length difference after rolling STL
        window_data_start_index = len(log_train) - len(trend_train) # Assumes trend/seasonal/resid have same length

        # Use dates corresponding to the potentially shortened series
        dates_train_win = dates_train[window_data_start_index:] if dates_train is not None else None
        dates_val_win = dates_val[window_data_start_index:] if dates_val is not None else None
        dates_test_win = dates_test[window_data_start_index:] if dates_test is not None else None

        # Create windows for raw log-transformed data
        X_train, x_dates_train, _ = self.create_sliding_windows(log_train[window_data_start_index:], window_size, max_horizon, dates_train_win)
        X_val, x_dates_val, _ = self.create_sliding_windows(log_val[window_data_start_index:], window_size, max_horizon, dates_val_win)
        X_test, x_dates_test, _ = self.create_sliding_windows(log_test[window_data_start_index:], window_size, max_horizon, dates_test_win)
        print(f"Input window shapes created: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

        # Create sliding windows for decomposed channels
        X_train_trend, _, _ = self.create_sliding_windows(trend_train, window_size, max_horizon, dates_train_win)
        X_train_seasonal, _, _ = self.create_sliding_windows(seasonal_train, window_size, max_horizon, dates_train_win)
        X_train_noise, _, _ = self.create_sliding_windows(resid_train, window_size, max_horizon, dates_train_win)
        X_val_trend, _, _ = self.create_sliding_windows(trend_val, window_size, max_horizon, dates_val_win)
        X_val_seasonal, _, _ = self.create_sliding_windows(seasonal_val, window_size, max_horizon, dates_val_win)
        X_val_noise, _, _ = self.create_sliding_windows(resid_val, window_size, max_horizon, dates_val_win)
        X_test_trend, _, _ = self.create_sliding_windows(trend_test, window_size, max_horizon, dates_test_win)
        X_test_seasonal, _, _ = self.create_sliding_windows(seasonal_test, window_size, max_horizon, dates_test_win)
        X_test_noise, _, _ = self.create_sliding_windows(resid_test, window_size, max_horizon, dates_test_win)

        # 7. Compute Baseline Datasets.
        # Baseline is the actual close price at the end of the input window (t).
        # x_dates_* gives the date for the end of each input window.
        # We need the index in the original close_* series that corresponds to these dates.
        num_train_windows = len(X_train)
        num_val_windows = len(X_val)
        num_test_windows = len(X_test)

        # Find the indices in the original 'close' series corresponding to x_dates_*
        # This requires careful index alignment, especially if dates are missing/irregular.
        # Simpler approach: Use the knowledge of how windows were created.
        # The first window uses data up to index (window_data_start_index + window_size - 1).
        # This corresponds to the date x_dates_*[0]. The baseline value is close_*(that index).
        baseline_start_index = window_data_start_index + window_size - 1
        baseline_train = close_train[baseline_start_index : baseline_start_index + num_train_windows]
        baseline_val = close_val[baseline_start_index : baseline_start_index + num_val_windows]
        baseline_test = close_test[baseline_start_index : baseline_start_index + num_test_windows]
        if len(baseline_train) != num_train_windows: raise ValueError("Baseline train length mismatch after calculation.")
        if len(baseline_val) != num_val_windows: raise ValueError("Baseline val length mismatch after calculation.")
        if len(baseline_test) != num_test_windows: raise ValueError("Baseline test length mismatch after calculation.")
        print(f"Baseline shapes computed: Train: {baseline_train.shape}, Val: {baseline_val.shape}, Test: {baseline_test.shape}")

        # 6. Process Multi-Horizon Targets.
        print(f"Processing targets for horizons: {predicted_horizons}...")
        target_column = config["target_column"]
        if target_column not in y_train_df.columns: raise ValueError(f"Column '{target_column}' not found in training Y data.")

        target_train_raw = y_train_df[target_column].astype(np.float32).values
        target_val_raw = y_val_df[target_column].astype(np.float32).values
        target_test_raw = y_test_df[target_column].astype(np.float32).values

        y_train_list_final = []
        y_val_list_final = []
        y_test_list_final = []
        use_returns = config.get("use_returns", False)

        for h in predicted_horizons:
            # Target for input window ending at original index t is at index t+h.
            # First window ends at original index baseline_start_index. Target needed is at baseline_start_index + h.
            # Last window ends at original index baseline_start_index + num_windows - 1. Target needed is at baseline_start_index + num_windows - 1 + h.
            target_start_idx = baseline_start_index + h
            target_end_idx_train = baseline_start_index + num_train_windows -1 + h + 1 # +1 for Python slicing upper bound
            target_end_idx_val   = baseline_start_index + num_val_windows -1 + h + 1
            target_end_idx_test  = baseline_start_index + num_test_windows -1 + h + 1

            if target_end_idx_train > len(target_train_raw): raise ValueError(f"Not enough data in target_train_raw for horizon {h}.")
            target_train_h = target_train_raw[target_start_idx : target_end_idx_train]

            if target_end_idx_val > len(target_val_raw): raise ValueError(f"Not enough data in target_val_raw for horizon {h}.")
            target_val_h = target_val_raw[target_start_idx : target_end_idx_val]

            if target_end_idx_test > len(target_test_raw): raise ValueError(f"Not enough data in target_test_raw for horizon {h}.")
            target_test_h = target_test_raw[target_start_idx : target_end_idx_test]

            # Verify lengths match number of windows
            if len(target_train_h) != num_train_windows: raise ValueError(f"Target train slice length mismatch for H={h}")
            if len(target_val_h) != num_val_windows: raise ValueError(f"Target val slice length mismatch for H={h}")
            if len(target_test_h) != num_test_windows: raise ValueError(f"Target test slice length mismatch for H={h}")

            # Adjust targets by baseline if use_returns is True
            if use_returns:
                target_train_h = target_train_h - baseline_train
                target_val_h = target_val_h - baseline_val
                target_test_h = target_test_h - baseline_test

            # Append the processed target array for this horizon
            y_train_list_final.append(target_train_h.astype(np.float32))
            y_val_list_final.append(target_val_h.astype(np.float32))
            y_test_list_final.append(target_test_h.astype(np.float32))

        # --- Assign Correct Dates for Targets ---
        y_dates_train = x_dates_train # Dates correspond to the end of the input window
        y_dates_val = x_dates_val
        y_dates_test = x_dates_test
        print("Target processing complete.")

        # --- Prepare Test Close Prices for output ---
        # Aligned with baseline/target data points (end of input windows)
        test_close_prices_aligned = close_test[baseline_start_index : baseline_start_index + num_test_windows]

        # --- Reshape X datasets (Original logic preserved) ---
        # This assumes the downstream model expects a specific shape, e.g., adding a channel dimension.
        # If create_sliding_windows already returns 3D, this might be redundant or error-prone.
        # Keeping exactly as provided, assuming windowing function returns 2D (samples, window_size).
        if X_train.ndim == 2:
             print("Reshaping input features to add channel dimension...")
             X_train = X_train.reshape(-1, window_size, 1)
             X_val = X_val.reshape(-1, window_size, 1)
             X_test = X_test.reshape(-1, window_size, 1)

             X_train_trend = X_train_trend.reshape(-1, window_size, 1)
             X_train_seasonal = X_train_seasonal.reshape(-1, window_size, 1)
             X_train_noise = X_train_noise.reshape(-1, window_size, 1)

             X_val_trend = X_val_trend.reshape(-1, window_size, 1)
             X_val_seasonal = X_val_seasonal.reshape(-1, window_size, 1)
             X_val_noise = X_val_noise.reshape(-1, window_size, 1)

             X_test_trend = X_test_trend.reshape(-1, window_size, 1)
             X_test_seasonal = X_test_seasonal.reshape(-1, window_size, 1)
             X_test_noise = X_test_noise.reshape(-1, window_size, 1)
        else:
            print("Input features appear to be already 3D (or higher), skipping default reshape.")


        # 8. Verify date consistency (X dates vs Baseline dates - implicitly Y dates).
        print("Verifying date consistency for training data (Input Dates vs Baseline Length):")
        verify_date_consistency([list(x_dates_train) if x_dates_train is not None else None,
                                 list(baseline_train)], "Training")
        print("Verifying date consistency for validation data (Input Dates vs Baseline Length):")
        verify_date_consistency([list(x_dates_val) if x_dates_val is not None else None,
                                 list(baseline_val)], "Validation")
        print("Verifying date consistency for test data (Input Dates vs Baseline Length):")
        verify_date_consistency([list(x_dates_test) if x_dates_test is not None else None,
                                 list(baseline_test)], "Test")

        # --- Prepare Return Dictionary ---
        ret = {
            # Input Features (Windowed, potentially reshaped)
            "x_train": X_train, "x_val": X_val, "x_test": X_test,
            # Decomposed Input Features (Windowed, potentially reshaped)
            "x_train_trend": X_train_trend, "x_val_trend": X_val_trend, "x_test_trend": X_test_trend,
            "x_train_seasonal": X_train_seasonal, "x_val_seasonal": X_val_seasonal, "x_test_seasonal": X_test_seasonal,
            "x_train_noise": X_train_noise, "x_val_noise": X_val_noise, "x_test_noise": X_test_noise,
            # Multi-Horizon Targets (List of numpy arrays)
            "y_train": y_train_list_final,
            "y_val": y_val_list_final,
            "y_test": y_test_list_final,
            # Dates (Aligned with end of input window)
            "x_train_dates": x_dates_train,
            "y_train_dates": y_dates_train, # Same as x_dates_train
            "x_val_dates": x_dates_val,
            "y_val_dates": y_dates_val,     # Same as x_dates_val
            "x_test_dates": x_dates_test,
            "y_test_dates": y_dates_test,     # Same as x_dates_test
            # Baseline Values (Aligned with end of input window)
            "baseline_train": baseline_train,
            "baseline_val": baseline_val,
            "baseline_test": baseline_test,
            # Dates for baselines (Same as x_dates / y_dates)
            "baseline_train_dates": y_dates_train,
            "baseline_val_dates": y_dates_val,
            "baseline_test_dates": y_dates_test,
            # Test Close Prices (Aligned with end of input window)
            "test_close_prices": test_close_prices_aligned
        }

        print("Data processing finished.")
        return ret

    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        # Merge instance defaults with passed config
        run_config = self.params.copy()
        run_config.update(config)
        return self.process_data(run_config)

# Example placeholder for create_sliding_windows if needed outside class
# def create_sliding_windows_standalone(data, window_size, time_horizon, date_times=None): ...