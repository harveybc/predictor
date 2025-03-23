#!/usr/bin/env python
"""
Enhanced Preprocessor Plugin with Log Transform, Causal Rolling STL Decomposition,
Progress Bar, and Detailed Statistics with Configurable Trend Smoother.

This plugin processes input data for EUR/USD forecasting by:
  1. Loading separate CSV files for X and Y (train, validation, test).
  2. Extracting the 'CLOSE' column from the X files and applying a log transform.
  3. Performing a causal (rolling) STL decomposition on the log-transformed series using a rolling window.
     The trend smoother length is configurable via 'stl_trend'.
  4. Plotting the decomposition of the training series to a file defined by 'stl_plot_file'.
  5. Printing detailed numerical statistics for the decomposition:
       - Mean, std, and variance for trend, seasonal, and residual components.
       - Signal-to-Noise Ratio (SNR) = (Var(trend)+Var(seasonal))/Var(residual) (desired: high).
       - Autocorrelation of seasonal component at lag=stl_period (desired: close to 1).
       - Dominant frequency from spectral analysis of seasonal (expected: ~1/stl_period).
       - Variance of first differences of trend (desired: low).
       - Residual autocorrelation (lag 1, desired: near 0) and Shapiro–Wilk p-value (desired: > 0.05).
       - Hilbert phase statistics (circular mean and std) for seasonal (desired: stable phase, low dispersion).
  6. Creating sliding windows for the raw log series and for each decomposed channel.
  7. Processing target values from the Y files (assumed to be in the "TARGET" column).

The plugin adheres to the standard plugin interface with methods such as set_params, get_debug_info, and add_debug_info.
A tqdm progress bar is added to the STL decomposition to indicate progress.
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

class PreprocessorPlugin:
    # Default plugin parameters, now including 'stl_trend'
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
        "window_size": 24,
        "time_horizon": 6,
        "use_returns": True,
        "stl_period": 24,# best 72
        "stl_window": 72,#best 96
        "stl_trend": 144, #best 121
        "stl_plot_file": "stl_plot.png",
        "pos_encoding_dim": 16
    }
    # Include the new parameter in the debug variables.
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
        # Try converting the index to datetime.
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
        # Get the trend smoother length from configuration.
        stl_trend = self.params.get("stl_trend", 11)
        for i in tqdm(range(stl_window, n + 1), desc="STL Decomposition", unit="window"):
            window = series[i - stl_window: i]
            stl = STL(window, period=period, trend=stl_trend, robust=True)
            result = stl.fit()
            # Use the last value of the window (causal)
            trend[i - stl_window] = result.trend[-1]
            seasonal[i - stl_window] = result.seasonal[-1]
            resid[i - stl_window] = result.resid[-1]
        return trend, seasonal, resid

    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        """
        Plots the STL decomposition and saves the figure.
        """
        # save the last 120 ticks as a figure
        #limit plotted ticks to 120
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
            date_times (pd.DatetimeIndex, optional): Dates corresponding to the data.

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
          1. Loads X and Y CSV files for train, validation, and test.
          2. Extracts the 'CLOSE' column from X files and applies a log transform.
          3. Applies a causal, rolling STL decomposition on the log-transformed series.
          4. Plots the decomposition for training data to the file specified by stl_plot_file.
             Then, prints detailed numerical statistics:
                - Mean, std, and variance for trend, seasonal, and residual.
                - Signal-to-Noise Ratio (SNR) = (Var(trend)+Var(seasonal))/Var(residual) (desired: high).
                - Autocorrelation of seasonal component at lag = stl_period (desired: close to 1).
                - Dominant frequency from spectral analysis of seasonal (expected: ~1/stl_period).
                - Variance of first differences of trend (desired: low).
                - Residual autocorrelation (lag 1, desired: near 0) and Shapiro–Wilk p-value (desired: > 0.05).
                - Hilbert phase statistics (circular mean and std) for seasonal (desired: low dispersion).
          5. Creates sliding windows for:
             - The raw log-transformed series.
             - Each decomposed channel (trend, seasonal, residual).
          6. Processes targets from Y files (assumed to be in the 'TARGET' column) with sliding windows.
          7. Optionally adjusts targets if use_returns is True.
        
        Returns:
            dict: Processed datasets including:
              - "x_train", "x_train_trend", "x_train_seasonal", "x_train_noise"
              - "y_train" (as list and array), and similar for validation and test.
              - "dates_train", "dates_val", "dates_test", "test_close_prices"
        """
        headers = config.get("headers", self.params["headers"])

        # Load X data for train, validation, and test.
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), headers)
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), headers)
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), headers)

        # Load Y data for train, validation, and test.
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), headers)
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), headers)
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), headers)

        # Extract 'CLOSE' from X data and apply log transformation.
        if "CLOSE" not in x_train_df.columns:
            raise ValueError("Column 'CLOSE' not found in training X data.")
        close_train = x_train_df["CLOSE"].astype(np.float32).values
        close_val = x_val_df["CLOSE"].astype(np.float32).values
        close_test = x_test_df["CLOSE"].astype(np.float32).values

        # Apply log transform to stabilize variance.
        log_train = np.log(close_train)
        log_val = np.log(close_val)
        log_test = np.log(close_test)

        # Get date indices.
        train_dates = x_train_df.index if x_train_df.index is not None else None
        val_dates = x_val_df.index if x_val_df.index is not None else None
        test_dates = x_test_df.index if x_test_df.index is not None else None

        # Get STL parameters.
        stl_period = config.get("stl_period", self.params["stl_period"])
        stl_window = config.get("stl_window", config.get("window_size", self.params["window_size"]))
        stl_plot_file = config.get("stl_plot_file", self.params["stl_plot_file"])
        stl_trend = config.get("stl_trend", self.params["stl_trend"])

        # Compute causal, rolling STL decomposition on the training series.
        trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period)
        # Plot the decomposition for training data.
        self._plot_decomposition(log_train[stl_window - 1:], trend_train, seasonal_train, resid_train, stl_plot_file)
        
        # Compute and print detailed STL statistics for training data.
        # Trend statistics.
        trend_mean = np.mean(trend_train)
        trend_std = np.std(trend_train)
        trend_var = np.var(trend_train)
        # Seasonal statistics.
        seasonal_mean = np.mean(seasonal_train)
        seasonal_std = np.std(seasonal_train)
        seasonal_var = np.var(seasonal_train)
        # Residual statistics.
        resid_mean = np.mean(resid_train)
        resid_std = np.std(resid_train)
        resid_var = np.var(resid_train)
        # Signal-to-Noise Ratio (desired: high)
        snr = (trend_var + seasonal_var) / resid_var if resid_var != 0 else np.inf

        # Autocorrelation of seasonal component at lag = stl_period (desired: close to 1)
        if len(seasonal_train) > stl_period:
            seasonal_ac = np.corrcoef(seasonal_train[:-stl_period], seasonal_train[stl_period:])[0,1]
        else:
            seasonal_ac = np.nan

        # Dominant frequency from spectral analysis of seasonal (expected: ~1/stl_period)
        seasonal_fft = np.fft.rfft(seasonal_train)
        power = np.abs(seasonal_fft)**2
        freqs = np.fft.rfftfreq(len(seasonal_train))
        dominant_freq = freqs[np.argmax(power)]
        expected_freq = 1.0 / stl_period

        # Trend smoothness: variance of first differences (desired: low)
        trend_diff = np.diff(trend_train)
        trend_diff_var = np.var(trend_diff)

        # Residual autocorrelation (lag 1, desired: near 0)
        if len(resid_train) > 1:
            resid_ac = np.corrcoef(resid_train[:-1], resid_train[1:])[0,1]
        else:
            resid_ac = np.nan

        # Residual normality test (Shapiro–Wilk; desired: p-value > 0.05)
        try:
            stat, resid_pvalue = shapiro(resid_train)
        except Exception:
            resid_pvalue = np.nan

        # Hilbert phase of seasonal component.
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

        # Compute rolling STL for validation and test series.
        trend_val, seasonal_val, resid_val = self._rolling_stl(log_val, stl_window, stl_period)
        trend_test, seasonal_test, resid_test = self._rolling_stl(log_test, stl_window, stl_period)

        # Create sliding windows on the raw log series.
        window_size = config["window_size"]
        time_horizon = config["time_horizon"]
        use_returns = config.get("use_returns", False)

        X_train, y_train_sw, dates_train_sw = self.create_sliding_windows(log_train[stl_window - 1:], window_size, time_horizon, train_dates)
        X_val, y_val_sw, dates_val_sw = self.create_sliding_windows(log_val[stl_window - 1:], window_size, time_horizon, val_dates)
        X_test, y_test_sw, dates_test_sw = self.create_sliding_windows(log_test[stl_window - 1:], window_size, time_horizon, test_dates)

        # Create sliding windows for each decomposed channel.
        X_train_trend, _, _ = self.create_sliding_windows(trend_train, window_size, time_horizon, train_dates)
        X_train_seasonal, _, _ = self.create_sliding_windows(seasonal_train, window_size, time_horizon, train_dates)
        X_train_noise, _, _ = self.create_sliding_windows(resid_train, window_size, time_horizon, train_dates)

        X_val_trend, _, _ = self.create_sliding_windows(trend_val, window_size, time_horizon, val_dates)
        X_val_seasonal, _, _ = self.create_sliding_windows(seasonal_val, window_size, time_horizon, val_dates)
        X_val_noise, _, _ = self.create_sliding_windows(resid_val, window_size, time_horizon, val_dates)

        X_test_trend, _, _ = self.create_sliding_windows(trend_test, window_size, time_horizon, test_dates)
        X_test_seasonal, _, _ = self.create_sliding_windows(seasonal_test, window_size, time_horizon, test_dates)
        X_test_noise, _, _ = self.create_sliding_windows(resid_test, window_size, time_horizon, test_dates)

        # If use_returns is True, adjust targets based on the last value of the raw input window.
        if use_returns:
            baseline_train = X_train[:, -1]
            baseline_val = X_val[:, -1]
            baseline_test = X_test[:, -1]
            y_train_sw = y_train_sw - baseline_train
            y_val_sw = y_val_sw - baseline_val
            y_test_sw = y_test_sw - baseline_test

        # Reshape inputs to (samples, window_size, 1) for compatibility.
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

        # Process targets from Y data; assume target column specified by config["target_column"].
        target_column = config["target_column"]
        if target_column not in y_train_df.columns:
            raise ValueError(f"Column '{target_column}' not found in training Y data.")
        target_train = y_train_df[target_column].astype(np.float32).values
        target_val = y_val_df[target_column].astype(np.float32).values
        target_test = y_test_df[target_column].astype(np.float32).values

        # Create sliding windows for targets.
        _, y_train_sw, _ = self.create_sliding_windows(target_train, window_size, time_horizon, train_dates)
        _, y_val_sw, _ = self.create_sliding_windows(target_val, window_size, time_horizon, val_dates)
        _, y_test_sw, _ = self.create_sliding_windows(target_test, window_size, time_horizon, test_dates)

        y_train_array = y_train_sw.reshape(-1, 1)
        y_val_array = y_val_sw.reshape(-1, 1)
        y_test_array = y_test_sw.reshape(-1, 1)

        # For test close prices, use the last value from the original CLOSE series (before log transform).
        test_close_prices = close_test[stl_window + window_size - 2 : len(close_test) - time_horizon]

        # Consolidate all processed data into a dictionary.
        ret = {
            "x_train": X_train,  # Raw log-series windows.
            "x_train_trend": X_train_trend,
            "x_train_seasonal": X_train_seasonal,
            "x_train_noise": X_train_noise,
            "y_train": [y_train_sw],
            "y_train_array": y_train_array,
            "dates_train": train_dates,
            "x_val": X_val,
            "x_val_trend": X_val_trend,
            "x_val_seasonal": X_val_seasonal,
            "x_val_noise": X_val_noise,
            "y_val": [y_val_sw],
            "y_val_array": y_val_array,
            "dates_val": val_dates,
            "x_test": X_test,
            "x_test_trend": X_test_trend,
            "x_test_seasonal": X_test_seasonal,
            "x_test_noise": X_test_noise,
            "y_test": [y_test_sw],
            "y_test_array": y_test_array,
            "dates_test": test_dates,
            "test_close_prices": test_close_prices
        }
        if use_returns:
            ret["baseline_train"] = X_train[:, -1]
            ret["baseline_val"] = X_val[:, -1]
            ret["baseline_test"] = X_test[:, -1]
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
        "stl_plot_file": "stl_plot.png",
        "target_column": "TARGET"
    }
    datasets = plugin.process_data(test_config)
    debug_info = plugin.get_debug_info()
    print("Debug Info:", debug_info)
    print("Datasets keys:", list(datasets.keys()))
