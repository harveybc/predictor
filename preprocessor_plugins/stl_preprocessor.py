#!/usr/bin/env python
"""
Enhanced Preprocessor Plugin with Log Transform and Causal Rolling STL Decomposition

This plugin processes input data for EUR/USD forecasting by:
  1. Loading separate CSV files for X and Y (train, validation, test).
  2. Extracting the 'CLOSE' column from the X files and applying a log transform.
  3. Performing a causal (rolling) STL decomposition on the log-transformed series,
     using a rolling window defined by 'stl_window' and a seasonal period 'stl_period'.
  4. Plotting the decomposition (trend, seasonal, residual) for the training series to 'stl_plot_file'.
  5. Creating sliding windows for the original log-series and each decomposed channel.
  6. Processing targets from the Y files (assumed in column config['target_column']) via sliding windows.
  
The processed outputs (raw, trend, seasonal, and noise channels) are returned as a dictionary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import json
from app.data_handler import load_csv, write_csv  # Ensure these functions are implemented

class PreprocessorPlugin:
    # Default plugin parameters
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
        "window_size": 24,         # Window size for sliding windows (in hours)
        "time_horizon": 6,         # Forecast horizon (e.g., 6 hours ahead)
        "use_returns": False,      # Whether to compute returns for target adjustment
        "stl_period": 120,          # Seasonal period for STL (e.g., 24 for hourly data with daily seasonality)
        "stl_window": 24,          # Rolling window length for causal STL (can be set equal to window_size or longer)
        "stl_plot_file": "stl_plot.png",
        "pos_encoding_dim": 16     # For positional encoding if needed
    }
    plugin_debug_vars = ["window_size", "time_horizon", "use_returns", "stl_period", "stl_window", "stl_plot_file"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Update plugin parameters with global configuration.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Returns debug info for the plugin.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add plugin debug info to the provided dictionary.
        """
        debug_info.update(self.get_debug_info())

    def _load_data(self, file_path, max_rows, headers):
        """
        Helper to load CSV data.
        """
        df = load_csv(file_path, headers=headers, max_rows=max_rows)
        # Try converting index to datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                df.index = None
        return df

    def _rolling_stl(self, series, stl_window, period):
        """
        Computes a causal, rolling STL decomposition over the provided series.
        For each rolling window, computes the STL decomposition and returns the last values.

        Args:
            series (np.ndarray): 1D array (log-transformed series).
            stl_window (int): Length of rolling window for decomposition.
            period (int): Seasonal period for STL.

        Returns:
            tuple: (trend, seasonal, resid) arrays of length (n - stl_window + 1).
        """
        n = len(series)
        num_points = n - stl_window + 1
        trend = np.zeros(num_points)
        seasonal = np.zeros(num_points)
        resid = np.zeros(num_points)
        for i in range(stl_window, n + 1):
            window = series[i - stl_window: i]
            stl = STL(window, period=period, robust=True)
            result = stl.fit()
            # Only use the last value of the rolling window (causal)
            trend[i - stl_window] = result.trend[-1]
            seasonal[i - stl_window] = result.seasonal[-1]
            resid[i - stl_window] = result.resid[-1]
        return trend, seasonal, resid

    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        """
        Plots the STL decomposition and saves the figure.
        """
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

        Returns:
            tuple: (windows, targets, date_windows)
        """
        windows = []
        targets = []
        date_windows = []
        n = len(data)
        for i in range(0, n - window_size - time_horizon + 1):
            window = data[i : i + window_size]
            target = data[i + window_size + time_horizon - 1]
            windows.append(window)
            targets.append(target)
            if date_times is not None:
                date_windows.append(date_times[i + window_size - 1])
        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows

    def process_data(self, config):
        """
        Processes data for EUR/USD forecasting by applying:
          - Log transformation to the "CLOSE" prices.
          - Causal, rolling STL decomposition on the log-transformed series.
          - Generation of sliding windows for the raw log-series and for each decomposed channel.
          - Loading and processing of target values from Y files.

        Args:
            config (dict): Must include keys:
              - x_train_file, y_train_file, x_validation_file, y_validation_file, x_test_file, y_test_file
              - headers, max_steps_*, window_size, time_horizon, use_returns, stl_period, stl_window, stl_plot_file

        Returns:
            dict: Processed datasets including:
              - "x_train", "x_train_trend", "x_train_seasonal", "x_train_noise"
              - "y_train" (as list and array) and similar for validation and test.
              - "dates_train", "dates_val", "dates_test", "test_close_prices"
        """
        headers = config.get("headers", self.params["headers"])

        # Load X data (train, validation, test)
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), headers)
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), headers)
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), headers)

        # Load Y data (train, validation, test)
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), headers)
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), headers)
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), headers)

        # Extract "CLOSE" from X data and apply log transform
        if "CLOSE" not in x_train_df.columns:
            raise ValueError("Column 'CLOSE' not found in training X data.")
        close_train = x_train_df["CLOSE"].astype(np.float32).values
        close_val = x_val_df["CLOSE"].astype(np.float32).values
        close_test = x_test_df["CLOSE"].astype(np.float32).values

        # Apply log transformation
        log_train = np.log(close_train)
        log_val = np.log(close_val)
        log_test = np.log(close_test)

        # Retrieve date indices
        train_dates = x_train_df.index if x_train_df.index is not None else None
        val_dates = x_val_df.index if x_val_df.index is not None else None
        test_dates = x_test_df.index if x_test_df.index is not None else None

        # Get STL parameters
        stl_period = config.get("stl_period", self.params["stl_period"])
        stl_window = config.get("stl_window", config.get("window_size", self.params["window_size"]))
        stl_plot_file = config.get("stl_plot_file", self.params["stl_plot_file"])

        # Compute causal, rolling STL decomposition on the log-transformed training series.
        trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period)
        # Plot the decomposition (for training only)
        self._plot_decomposition(log_train[stl_window - 1:], trend_train, seasonal_train, resid_train, stl_plot_file)

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

        # Adjust targets if using returns (target adjustment on original raw sliding windows)
        if use_returns:
            baseline_train = X_train[:, -1]
            baseline_val = X_val[:, -1]
            baseline_test = X_test[:, -1]
            y_train_sw = y_train_sw - baseline_train
            y_val_sw = y_val_sw - baseline_val
            y_test_sw = y_test_sw - baseline_test

        # Reshape inputs for model compatibility: (samples, window_size, 1)
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

        # Process targets from Y files: assume target column from config["target_column"]
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

        # For test close prices, use the last value from the original X series (before log transform) 
        test_close_prices = close_test[stl_window + window_size - 2 : len(close_test) - time_horizon]

        # Consolidate all processed data into a dictionary.
        ret = {
            "x_train": X_train,  # Log-transformed input windows
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
        """
        Convenience method to execute data processing.
        """
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
        "use_returns": False,
        "stl_period": 24,
        "stl_window": 24,
        "stl_plot_file": "stl_plot.png"
    }
    datasets = plugin.process_data(test_config)
    debug_info = plugin.get_debug_info()
    print("Debug Info:", debug_info)
    print("Datasets keys:", list(datasets.keys()))
