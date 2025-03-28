#!/usr/bin/env python
"""
Enhanced Preprocessor Plugin with Log Transform, Causal Rolling STL Decomposition,
Progress Bar, Detailed Statistics, Configurable Trend Smoother, and Date Verification.

This plugin processes input data for EUR/USD forecasting by:
  1. Loading separate CSV files for X and Y (train, validation, test).
  2. Extracting the 'CLOSE' column from the X files and applying a log transform.
  3. Performing a causal (rolling) STL decomposition on the log-transformed series using a rolling window.
     The trend smoother length is configurable via 'stl_trend'.
  4. Plotting the decomposition of the training series to a file defined by 'stl_plot_file'.
  5. Printing detailed numerical statistics for the decomposition:
       - Mean, std, and variance for trend, seasonal, and residual.
       - Signal-to-Noise Ratio (SNR) = (Var(trend)+Var(seasonal))/Var(residual) (desired: high).
       - Autocorrelation of seasonal component at lag = stl_period (desired: close to 1).
       - Dominant frequency from spectral analysis of seasonal (expected: ~1/stl_period).
       - Variance of first differences of trend (desired: low).
       - Residual autocorrelation (lag 1, desired: near 0) and Shapiro–Wilk p-value (desired: > 0.05).
       - Hilbert phase statistics (circular mean and std) for seasonal (desired: low dispersion).
  6. Creating sliding windows for the raw log series and for each decomposed channel.
  7. Processing target values from the Y files (assumed to be in the "TARGET" column).
      - If use_returns is True, target windows are adjusted by subtracting the baseline.
  8. Computing a baseline dataset from the original CLOSE values that always contains the current tick's CLOSE.
  9. Verifying that the sliding window date arrays for features (X), targets (Y), and baseline are consistent.
  
The plugin adheres to the standard plugin interface with methods such as set_params, get_debug_info, and add_debug_info.
A tqdm progress bar is used during STL decomposition.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import json
from app.data_handler import load_csv, write_csv  # Ensure these functions are implemented
from scipy.signal import hilbert
from scipy.stats import shapiro

def verify_date_consistency(date_lists, dataset_name):
    """
    Verifies that all date arrays in date_lists have the same first and last elements.
    Prints a warning if any array does not match.
    """
    if not date_lists:
        return
    first = date_lists[0][0] if len(date_lists[0]) > 0 else None
    last = date_lists[0][-1] if len(date_lists[0]) > 0 else None
    for i, d in enumerate(date_lists):
        if len(d) == 0:
            print(f"Warning: {dataset_name} date array {i} is empty.")
            continue
        if d[0] != first or d[-1] != last:
            print(f"Warning: Date array {i} in {dataset_name} does not match the others. First: {d[0]}, Last: {d[-1]}; expected First: {first}, Last: {last}.")

class PreprocessorPlugin:
    # Default plugin parameters, including stl_trend.
    plugin_params = {
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
        "time_horizon": 6,
        "use_returns": True,
        "stl_period": 24,    # For daily seasonality.best:24
        "stl_window": 120,    # Adjusted window. best: 46
        "stl_trend": 121,     # Trend smoother length.best: 49
        "stl_plot_file": "stl_plot.png",
        "pos_encoding_dim": 16
    }
    plugin_debug_vars = ["window_size", "time_horizon", "use_returns", "stl_period", "stl_window", "stl_trend", "stl_plot_file"]

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

    def _load_data(self, file_path, max_rows, headers):
        """
        Helper function to load CSV data.
        """
        df = load_csv(file_path, headers=headers, max_rows=max_rows)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                df.index = None
        return df

    def _rolling_stl(self, series, stl_window, period):
        """
        Computes a causal, rolling STL decomposition on the log-transformed series.
        Uses a rolling window of length 'stl_window', processing only past data.

        Args:
            series (np.ndarray): 1D array of log-transformed data.
            stl_window (int): Length of the rolling window.
            period (int): Seasonal period for STL.

        Returns:
            tuple: (trend, seasonal, resid) arrays of length (n - stl_window + 1).
        """
        n = len(series)
        num_points = n - stl_window + 1
        trend = np.zeros(num_points)
        seasonal = np.zeros(num_points)
        resid = np.zeros(num_points)
        stl_trend = self.params.get("stl_trend", 11)
        for i in tqdm(range(stl_window, n + 1), desc="STL Decomposition", unit="window"):
            window = series[i - stl_window: i]
            stl = STL(window, period=period, trend=stl_trend, robust=True)
            result = stl.fit()
            trend[i - stl_window] = result.trend[-1]
            seasonal[i - stl_window] = result.seasonal[-1]
            resid[i - stl_window] = result.resid[-1]
        return trend, seasonal, resid

    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        """
        Plots the STL decomposition and saves the figure.
        """
        if len(series) > 480:
            series = series[-480:]
            trend = trend[-480:]
            seasonal = seasonal[-480:]
            resid = resid[-480:]
        plt.figure(figsize=(12, 9))
        plt.subplot(411)
        plt.plot(series, label="Log-Transformed Series")
        plt.legend(loc="upper left")
        plt.subplot(412)
        plt.plot(trend, label="Trend", color="orange")
        plt.legend(loc="upper left")
        plt.subplot(413)
        plt.plot(seasonal, label="Seasonal", color="green")
        plt.legend(loc="upper left")
        plt.subplot(414)
        plt.plot(resid, label="Residual", color="red")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"STL decomposition plot saved to {file_path}")

    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        """
        Creates sliding windows for a univariate series.

        Args:
            data (np.ndarray): 1D array of values.
            window_size (int): Length of each window.
            time_horizon (int): Forecast horizon (target index = window_size + time_horizon - 1).
            date_times (iterable, optional): Dates corresponding to the data.

        Returns:
            tuple: (windows, targets, date_windows)
        """
        windows = []
        targets = []
        date_windows = []
        n = len(data)
        for i in range(0, n - window_size - time_horizon + 1):
            window = data[i: i + window_size]
            target = data[i + window_size + time_horizon - 1]
            windows.append(window)
            targets.append(target)
            if date_times is not None:
                date_windows.append(date_times[i + window_size - 1])
        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows

    def process_data(self, config):
        """
        Processes data for EUR/USD forecasting with STL decomposition.
        
        Steps:
          1. Load X and Y CSV files for train, validation, and test.
          2. Extract the 'CLOSE' column from X files and apply a log transform.
          3. Compute a causal, rolling STL decomposition on the log-transformed series.
          4. Plot the decomposition for training data and print detailed numerical statistics.
          5. Create sliding windows for:
             - The raw log-transformed series.
             - Each decomposed channel (trend, seasonal, residual).
          6. Process targets from Y files (assumed to be in the "TARGET" column).
             - If use_returns is True, adjust target windows by subtracting the baseline.
          7. Compute a baseline dataset from the original CLOSE values that always holds the current tick's CLOSE.
          8. Verify that the sliding window date arrays for features (X), targets (Y), and baseline are consistent.
        
        Returns:
            dict: Processed datasets, including:
              - "x_train", "x_train_trend", "x_train_seasonal", "x_train_noise"
              - "y_train" (list and array), and similar for validation and test.
              - "baseline_train", "baseline_val", "baseline_test"
              - Date arrays for X, Y, and baseline for each split.
              - "test_close_prices"
        """
        headers = config.get("headers", self.params["headers"])
        
        # Load X data.
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), headers)
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), headers)
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), headers)
        
        # Load Y data.
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), headers)
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), headers)
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), headers)
        
        # Extract 'CLOSE' and apply log transform.
        if "CLOSE" not in x_train_df.columns:
            raise ValueError("Column 'CLOSE' not found in training X data.")
        close_train = x_train_df["CLOSE"].astype(np.float32).values
        close_val = x_val_df["CLOSE"].astype(np.float32).values
        close_test = x_test_df["CLOSE"].astype(np.float32).values
        
        log_train = np.log(close_train)
        log_val = np.log(close_val)
        log_test = np.log(close_test)
        
        # Get date indices from X data.
        dates_train = x_train_df.index if x_train_df.index is not None else None
        dates_val = x_val_df.index if x_val_df.index is not None else None
        dates_test = x_test_df.index if x_test_df.index is not None else None
        
        # Get STL parameters.
        stl_period = config.get("stl_period", self.params["stl_period"])
        stl_window = config.get("stl_window", config.get("window_size", self.params["window_size"]))
        stl_plot_file = config.get("stl_plot_file", self.params["stl_plot_file"])
        stl_trend = config.get("stl_trend", self.params["stl_trend"])
        
        # Compute STL decomposition on the training log series.
        trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period)
        self._plot_decomposition(log_train[stl_window - 1:], trend_train, seasonal_train, resid_train, stl_plot_file)
        
        # Detailed STL statistics.
        trend_mean = np.mean(trend_train)
        trend_std = np.std(trend_train)
        trend_var = np.var(trend_train)
        seasonal_mean = np.mean(seasonal_train)
        seasonal_std = np.std(seasonal_train)
        seasonal_var = np.var(seasonal_train)
        resid_mean = np.mean(resid_train)
        resid_std = np.std(resid_train)
        resid_var = np.var(resid_train)
        snr = (trend_var + seasonal_var) / resid_var if resid_var != 0 else np.inf
        if len(seasonal_train) > stl_period:
            seasonal_ac = np.corrcoef(seasonal_train[:-stl_period], seasonal_train[stl_period:])[0,1]
        else:
            seasonal_ac = np.nan
        seasonal_fft = np.fft.rfft(seasonal_train)
        power = np.abs(seasonal_fft)**2
        freqs = np.fft.rfftfreq(len(seasonal_train))
        dominant_freq = freqs[np.argmax(power)]
        expected_freq = 1.0 / stl_period
        trend_diff_var = np.var(np.diff(trend_train))
        if len(resid_train) > 1:
            resid_ac = np.corrcoef(resid_train[:-1], resid_train[1:])[0,1]
        else:
            resid_ac = np.nan
        try:
            stat, resid_pvalue = shapiro(resid_train)
        except Exception:
            resid_pvalue = np.nan
        analytic_signal = hilbert(seasonal_train)
        phase = np.angle(analytic_signal)
        circ_mean = np.angle(np.mean(np.exp(1j * phase)))
        circ_std = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phase)))))
        
        print("=== STL Decomposition Detailed Statistics (Training Data) ===")
        print(f"Trend     - Mean: {trend_mean:.4f} (desired: stable), Std: {trend_std:.4f}, Variance: {trend_var:.4f}")
        print(f"Seasonal  - Mean: {seasonal_mean:.4f}, Std: {seasonal_std:.4f}, Variance: {seasonal_var:.4f}")
        print(f"Residual  - Mean: {resid_mean:.4f}, Std: {resid_std:.4f}, Variance: {resid_var:.4f} (desired: low)")
        print(f"Signal-to-Noise Ratio (SNR): {snr:.4f} (desired: high)")
        print(f"Seasonal Autocorrelation at lag {stl_period}: {seasonal_ac:.4f} (desired: close to 1)")
        print(f"Dominant Frequency (spectral): {dominant_freq:.4f} (expected: ~{expected_freq:.4f})")
        print(f"Trend Smoothness (variance of first differences): {trend_diff_var:.4f} (desired: low)")
        print(f"Residual Autocorrelation (lag 1): {resid_ac:.4f} (desired: near 0)")
        print(f"Residual Normality Test p-value: {resid_pvalue:.4f} (desired: > 0.05)")
        print(f"Hilbert Phase of Seasonal - Circular Mean: {circ_mean:.4f}, Circular Std: {circ_std:.4f} (desired: low dispersion)")
        print("=============================================================")
        
        # Compute STL decomposition for validation and test series.
        trend_val, seasonal_val, resid_val = self._rolling_stl(log_val, stl_window, stl_period)
        trend_test, seasonal_test, resid_test = self._rolling_stl(log_test, stl_window, stl_period)
        
        # Create sliding windows on raw log series.
        window_size = config["window_size"]
        time_horizon = config["time_horizon"]
        use_returns = config.get("use_returns", False)
        X_train, x_dates_train, _ = self.create_sliding_windows(log_train[stl_window - 1:], window_size, time_horizon, dates_train)
        X_val, x_dates_val, _ = self.create_sliding_windows(log_val[stl_window - 1:], window_size, time_horizon, dates_val)
        X_test, x_dates_test, _ = self.create_sliding_windows(log_test[stl_window - 1:], window_size, time_horizon, dates_test)
        
        # Create sliding windows for decomposed channels.
        X_train_trend, _, _ = self.create_sliding_windows(trend_train, window_size, time_horizon, dates_train)
        X_train_seasonal, _, _ = self.create_sliding_windows(seasonal_train, window_size, time_horizon, dates_train)
        X_train_noise, _, _ = self.create_sliding_windows(resid_train, window_size, time_horizon, dates_train)
        
        X_val_trend, _, _ = self.create_sliding_windows(trend_val, window_size, time_horizon, dates_val)
        X_val_seasonal, _, _ = self.create_sliding_windows(seasonal_val, window_size, time_horizon, dates_val)
        X_val_noise, _, _ = self.create_sliding_windows(resid_val, window_size, time_horizon, dates_val)
        
        X_test_trend, _, _ = self.create_sliding_windows(trend_test, window_size, time_horizon, dates_test)
        X_test_seasonal, _, _ = self.create_sliding_windows(seasonal_test, window_size, time_horizon, dates_test)
        X_test_noise, _, _ = self.create_sliding_windows(resid_test, window_size, time_horizon, dates_test)
        
        # Compute baseline datasets from original CLOSE values.
        # Since the sliding windows for X start at log_train[stl_window - 1:], then further are created with window_size,
        # the effective offset is (stl_window - 1) + (window_size - 1) = stl_window + window_size - 2.
        baseline_train = close_train[stl_window + window_size - 2 : len(close_train) - time_horizon]
        baseline_val = close_val[stl_window + window_size - 2 : len(close_val) - time_horizon]
        baseline_test = close_test[stl_window + window_size - 2 : len(close_test) - time_horizon]
        
        # Process targets from Y data.
        target_column = config["target_column"]
        if target_column not in y_train_df.columns:
            raise ValueError(f"Column '{target_column}' not found in training Y data.")
        target_train = y_train_df[target_column].astype(np.float32).values
        target_val = y_val_df[target_column].astype(np.float32).values
        target_test = y_test_df[target_column].astype(np.float32).values
        #offset = stl_window + window_size - 2
        target_train = target_train[stl_window + window_size - 2: ]
        target_val = target_val[stl_window + window_size - 2: ]
        target_test = target_test[stl_window + window_size - 2: ]
        y_dates_train = dates_train[stl_window + window_size - 2 : len(dates_train) - time_horizon] if dates_train is not None else None
        y_dates_val = dates_val[stl_window + window_size - 2 : len(dates_val) - time_horizon] if dates_val is not None else None
        y_dates_test = dates_test[stl_window + window_size - 2 : len(dates_test) - time_horizon] if dates_test is not None else None
        #shift the target values but not the dates config['time_horizon'] steps forward in the future
        y_train_sw = target_train[time_horizon:]
        y_val_sw = target_val[time_horizon:]
        y_test_sw = target_test[time_horizon:]


        y_train_array = y_train_sw.reshape(-1, 1)
        y_val_array = y_val_sw.reshape(-1, 1)
        y_test_array = y_test_sw.reshape(-1, 1)
        
        # For test close prices, use the last value from the original CLOSE series.
        test_close_prices = close_test[stl_window + window_size - 2 : len(close_test) - time_horizon]
        
        # If use_returns is True, adjust target windows by subtracting baseline.
        if use_returns:
            y_train_sw = y_train_sw - baseline_train
            y_val_sw = y_val_sw - baseline_val
            y_test_sw = y_test_sw - baseline_test
        
        # Reshape X datasets.
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
        
        # Verify date consistency for train, validation, and test.
        print("Verifying date consistency for training data:")
        verify_date_consistency([list(x_dates_train), list(y_dates_train), list(baseline_train)], "Training")
        print("Verifying date consistency for validation data:")
        verify_date_consistency([list(x_dates_val), list(y_dates_val), list(baseline_val)], "Validation")
        print("Verifying date consistency for test data:")
        verify_date_consistency([list(x_dates_test), list(y_dates_test), list(baseline_test)], "Test")
        
        ret = {
            "x_train": X_train,
            "x_train_trend": X_train_trend,
            "x_train_seasonal": X_train_seasonal,
            "x_train_noise": X_train_noise,
            "y_train": [y_train_sw],
            "y_train_array": y_train_array,
            "x_train_dates": x_dates_train,
            "y_train_dates": y_dates_train,
            "baseline_train": baseline_train,
            "baseline_train_dates": dates_train[stl_window + config["window_size"] - 2 : len(dates_train) - time_horizon] if dates_train is not None else None,
            "x_val": X_val,
            "x_val_trend": X_val_trend,
            "x_val_seasonal": X_val_seasonal,
            "x_val_noise": X_val_noise,
            "y_val": [y_val_sw],
            "y_val_array": y_val_array,
            "x_val_dates": x_dates_val,
            "y_val_dates": y_dates_val,
            "baseline_val": baseline_val,
            "baseline_val_dates": dates_val[stl_window + config["window_size"] - 2 : len(dates_val) - time_horizon] if dates_val is not None else None,
            "x_test": X_test,
            "x_test_trend": X_test_trend,
            "x_test_seasonal": X_test_seasonal,
            "x_test_noise": X_test_noise,
            "y_test": [y_test_sw],
            "y_test_array": y_test_array,
            "x_test_dates": x_dates_test,
            "y_test_dates": y_dates_test,
            "baseline_test": baseline_test,
            "baseline_test_dates": dates_test[stl_window + config["window_size"] - 2 : len(dates_test) - time_horizon] if dates_test is not None else None,
            "test_close_prices": test_close_prices
        }
        return ret

    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        return self.process_data(config)

# Debugging usage example (run directly)
if __name__ == "__main__":
    plugin = PreprocessorPlugin()
    test_config = {
        "x_train_file": "data/x_train.csv",
        "y_train_file": "data/y_train.csv",
        "x_validation_file": "data/x_val.csv",
        "y_validation_file": "data/y_val.csv",
        "x_test_file": "data/x_test.csv",
        "y_test_file": "data/y_test.csv",
        "headers": True,
        "max_steps_train": 1000,
        "max_steps_val": 500,
        "max_steps_test": 500,
        "window_size": 24,
        "time_horizon": 6,
        "use_returns": True,
        "stl_period": 24,
        "stl_window": 24,
        "stl_trend": 49,
        "stl_plot_file": "stl_plot.png",
        "target_column": "TARGET"
    }
    datasets = plugin.process_data(test_config)
    debug_info = plugin.get_debug_info()
    print("Debug Info:", debug_info)
    print("Datasets keys:", list(datasets.keys()))
