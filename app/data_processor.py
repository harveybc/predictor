import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
import sys
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import logging
from sklearn.metrics import r2_score  # Ensure sklearn is imported at the top
import contextlib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import json
from plugin_loader import load_plugin

# Updated import: use tensorflow.keras instead of keras.
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import Huber

#!/usr/bin/env python
"""
data_processor_nbeats.py

This module implements a revised data processor for a single-step forecast.
It loads the CSV files, extracts the 'CLOSE' column as the univariate series,
and creates sliding-window samples such that each sample's target is a single value:
the future value at (window_size + time_horizon – 1).

Optionally, if config["use_returns"] is True, the target is computed as the difference
between the future value and the last value of the input window.

The output dictionary is structured to be compatible with the existing predictor pipeline.
"""

import pandas as pd
import numpy as np
from app.data_handler import load_csv, write_csv
import json

def create_sliding_windows_single(data, window_size, time_horizon, date_times=None):
    """
    Creates sliding windows for a univariate series with a single-step target.
    
    For each index i, an input window is taken from data[i : i + window_size]
    and the target is the value at index i + window_size + time_horizon - 1.
    
    Args:
        data (np.ndarray): 1D array of data values.
        window_size (int): Number of past time steps used as input.
        time_horizon (int): How many steps ahead to forecast (target = data[i+window_size+time_horizon-1]).
        date_times (pd.DatetimeIndex, optional): Date indices corresponding to the data.
        
    Returns:
        tuple: (windows, targets, date_windows)
          - windows: np.ndarray of shape (n_samples, window_size)
          - targets: np.ndarray of shape (n_samples,) containing single forecast values.
          - date_windows: list of dates corresponding to each input window (if provided).
    """
    windows = []
    targets = []
    date_windows = []
    n = len(data)
    # Ensure that we have enough data points for at least one window and target.
    for i in range(0, n - window_size - time_horizon + 1):
        window = data[i : i + window_size]
        target = data[i + window_size + time_horizon - 1]
        windows.append(window)
        targets.append(target)
        if date_times is not None:
            # Option: Use the date at the last step of the input window
            date_windows.append(date_times[i + window_size - 1])
    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows

def process_data(config):
    """
    Processes data for the N-BEATS plugin using a single-step forecast.
    
    Steps:
      1. Loads CSV files from paths defined in config.
      2. Extracts the 'CLOSE' column as the univariate input.
      3. Converts the series to NumPy arrays (float32).
      4. Creates sliding-window samples with a single forecast target.
         - If config["use_returns"] is True, computes target as (future_value - last_value_of_window).
      5. Reshapes the input windows to (samples, window_size, 1).
      6. Returns a dictionary containing training, validation, and test datasets along with dates.
    
    Args:
        config (dict): Configuration dictionary including:
            - "x_train_file", "x_validation_file", "x_test_file"
            - "headers": boolean for CSV headers.
            - "max_steps_train", "max_steps_val", "max_steps_test": maximum rows to load.
            - "window_size": integer for the sliding window length.
            - "time_horizon": forecast horizon (integer).
            - "use_returns": boolean flag for forecasting returns.
    
    Returns:
        dict: Processed datasets and additional arrays, including:
            "x_train", "y_train", "x_val", "y_val", "x_test", "y_test",
            "dates_train", "dates_val", "dates_test",
            "y_train_array", "y_val_array", "y_test_array",
            "test_close_prices"
    """
    # 1. Load CSV files.
    x_train_df = load_csv(config["x_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    x_val_df   = load_csv(config["x_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_val"))
    x_test_df  = load_csv(config["x_test_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))
    
    # 2. If possible, set the index as datetime.
    for df in (x_train_df, x_val_df, x_test_df):
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                df.index = None
    
    # 3. Extract the 'CLOSE' column.
    if "CLOSE" not in x_train_df.columns:
        raise ValueError("Column 'CLOSE' not found in training data.")
    close_train = x_train_df["CLOSE"].astype(np.float32).values  # shape (n_samples,)
    close_val   = x_val_df["CLOSE"].astype(np.float32).values
    close_test  = x_test_df["CLOSE"].astype(np.float32).values
    
    # Retrieve date indices.
    train_dates = x_train_df.index if x_train_df.index is not None else None
    val_dates   = x_val_df.index   if x_val_df.index is not None else None
    test_dates  = x_test_df.index  if x_test_df.index is not None else None
    
    # 4. Create sliding windows for a single-step forecast.
    window_size = config["window_size"]
    time_horizon = config["time_horizon"]
    use_returns = config.get("use_returns", False)
    
    X_train, y_train, dates_train = create_sliding_windows_single(close_train, window_size, time_horizon, train_dates)
    X_val, y_val, dates_val       = create_sliding_windows_single(close_val, window_size, time_horizon, val_dates)
    X_test, y_test, dates_test    = create_sliding_windows_single(close_test, window_size, time_horizon, test_dates)
    
    # 5. If use_returns is True, compute returns (difference between future and last value of window).
    if use_returns:
        # The baseline is the last value in each window.
        baseline_train = X_train[:, -1]
        baseline_val   = X_val[:, -1]
        baseline_test  = X_test[:, -1]
        y_train = y_train - baseline_train
        y_val   = y_val - baseline_val
        y_test  = y_test - baseline_test
    # 6. Reshape X arrays to (samples, window_size, 1) to denote univariate input.
    X_train = X_train.reshape(-1, window_size, 1)
    X_val   = X_val.reshape(-1, window_size, 1)
    X_test  = X_test.reshape(-1, window_size, 1)
    
    # 7. For compatibility, create y arrays as lists of 1D arrays (each forecast is a scalar).
    y_train_list = [y_train]  # single output per sample
    y_val_list   = [y_val]
    y_test_list  = [y_test]
    
    # Also keep stacked target arrays (with shape (samples, 1)).
    y_train_array = y_train.reshape(-1, 1)
    y_val_array   = y_val.reshape(-1, 1)
    y_test_array  = y_test.reshape(-1, 1)
    
    # 8. For test close prices, use the last value of each input window.
    test_close_prices = close_test[window_size - 1 : len(close_test) - time_horizon]
    
    # Debug messages
    print("Processed datasets:")
    print(" X_train shape:", X_train.shape, " y_train shape:", y_train_array.shape)
    print(" X_val shape:  ", X_val.shape,   " y_val shape:  ", y_val_array.shape)
    print(" X_test shape: ", X_test.shape,  " y_test shape: ", y_test_array.shape)
    print(" Test close prices shape:", test_close_prices.shape)
    
    ret = {
        "x_train": X_train,
        "y_train": y_train_list,
        "x_val": X_val,
        "y_val": y_val_list,
        "x_test": X_test,
        "y_test": y_test_list,
        "dates_train": dates_train,
        "dates_val": dates_val,
        "dates_test": dates_test,
        "y_train_array": y_train_array,
        "y_val_array": y_val_array,
        "y_test_array": y_test_array,
        "test_close_prices": test_close_prices
    }
    if use_returns:
        ret["baseline_train"] = X_train[:, -1].squeeze()
        ret["baseline_val"]   = X_val[:, -1].squeeze()
        ret["baseline_test"]  = X_test[:, -1].squeeze()
    return ret



def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline using training, validation, and test datasets.
    Trains the model (with 5-fold cross-validation), saves metrics, predictions,
    uncertainty estimates, and plots. Predictions (and uncertainties) are denormalized;
    when use_returns is True, predicted returns are converted to close prices by adding
    the corresponding denormalized baseline close. In the plot, only the prediction at
    the horizon given by config['plotted_horizon'] (default=6) is shown.
    """
    import time, numpy as np, pandas as pd, json, matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.model_selection import TimeSeriesSplit

    start_time = time.time()
    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    # Lists for metrics
    training_mae_list, training_r2_list, training_unc_list, training_snr_list, training_profit_list, training_risk_list = [], [], [], [], [], []
    validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list, validation_profit_list, validation_risk_list = [], [], [], [], [], []
    test_mae_list, test_r2_list, test_unc_list, test_snr_list, test_profit_list, test_risk_list = [], [], [], [], [], []



    print("Loading and processing datasets...")
    datasets = process_data(config)
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    x_test, y_test = datasets["x_test"], datasets["y_test"]
    # --- NEW CODE: Retrieve stacked multi-output targets ---
    y_train_array = datasets["y_train_array"]  # shape: (n_samples, time_horizon)
    y_val_array   = datasets["y_val_array"]
    y_test_array  = datasets["y_test_array"]
    print("DEBUG: Retrieved stacked y_train shape:", y_train_array.shape)
    print("DEBUG: Retrieved stacked y_val shape:", y_val_array.shape)
    print("DEBUG: Retrieved stacked y_test shape:", y_test_array.shape)
    # --- END NEW CODE ---

    train_dates = datasets.get("dates_train")
    val_dates = datasets.get("dates_val")
    test_dates = datasets.get("dates_test")
    test_close_prices = datasets.get("test_close_prices")
    # When using returns, process_data returns baseline values
    if config.get("use_returns", False):
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

    # If sliding windows output is a tuple, extract the data.
    if isinstance(x_train, tuple): x_train = x_train[0]
    if isinstance(x_val, tuple): x_val = x_val[0]
    if isinstance(x_test, tuple): x_test = x_test[0]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {[a.shape for a in y_train]}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {[a.shape for a in y_val]}")
    print(f"Test data shapes: x_test: {x_test.shape}, y_test: {[a.shape for a in y_test]}")
    # --- NEW CODE: Stack multi-output target lists into 2D arrays ---
    y_train_array = np.stack(y_train, axis=1)  # Shape: (n_samples, time_horizon)
    y_val_array   = np.stack(y_val, axis=1)
    y_test_array  = np.stack(y_test, axis=1)
    print("DEBUG: Stacked y_train shape:", y_train_array.shape)
    print("DEBUG: Stacked y_val shape:", y_val_array.shape)
    print("DEBUG: Stacked y_test shape:", y_test_array.shape)
    # --- END NEW CODE ---

    # --- NEW CODE: Stack multi-output target lists into arrays ---
    y_train_stacked = np.stack(y_train, axis=1)  # shape: (samples, time_horizon)
    y_val_stacked   = np.stack(y_val, axis=1)
    y_test_stacked  = np.stack(y_test, axis=1)
    print("DEBUG: Stacked y_train shape:", y_train_stacked.shape)
    print("DEBUG: Stacked y_val shape:", y_val_stacked.shape)
    print("DEBUG: Stacked y_test shape:", y_test_stacked.shape)
    # --- END NEW CODE ---
    # --- CHUNK: Training iterations ---
    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")
    if config["plugin"] in ["lstm", "cnn", "transformer","ann"] and window_size is None:
        raise ValueError("`window_size` must be defined for CNN, Transformer and LSTM plugins.")
    print(f"Time Horizon: {time_horizon}")
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    threshold_error = config["threshold_error"]

    # Ensure variables are NumPy arrays.
    for var in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        arr = locals()[var]
        if isinstance(arr, pd.DataFrame):
            locals()[var] = arr.to_numpy().astype(np.float32)

    if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
        if x_train.ndim != 3:
            raise ValueError(f"For CNN and LSTM, x_train must be 3D. Found: {x_train.shape}")
        print("Using pre-processed sliding windows for CNN and LSTM.")
    plugin.set_params(time_horizon=time_horizon)
    
    # Training iterations
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iter_start = time.time()
        if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
            plugin.build_model(input_shape=(window_size, x_train.shape[2]), x_train=x_train, config=config)
        elif config["plugin"] in ["transformer", "transformer_mmd"]:
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)
        else:
            if len(x_train.shape) != 2:
                raise ValueError(f"Expected 2D x_train for {config['plugin']}; got {x_train.shape}")
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)

        history,  train_preds, train_unc, val_preds, val_unc = plugin.train(
            x_train, y_train, epochs=epochs, batch_size=batch_size,
            threshold_error=threshold_error, x_val=x_val, y_val=y_val, config=config
        )

        # === STEP 7: Inverse-scale the predictions if using returns ===
        if config.get("use_returns", False):
            inv_scale_factor = 1.0 / config.get("target_scaling_factor", 100.0)
            print(f"DEBUG: Inversely scaling predictions by factor {inv_scale_factor}.")
            # Multiply the prediction arrays by the inverse scaling factor.
            train_preds = train_preds * inv_scale_factor
            val_preds = val_preds * inv_scale_factor



        # If using returns, recalc r2 based on baseline + predictions.
        # Ensure predictions arrays are correctly squeezed to match the shape of stacked ground truth
        train_preds_squeezed = np.squeeze(train_preds, axis=-1)  # from (samples, horizons, 1) to (samples, horizons)
        val_preds_squeezed = np.squeeze(val_preds, axis=-1)

        if config.get("use_returns", False):
            train_r2 = r2_score(
                (baseline_train[:, -1] + y_train_stacked[:, -1]).flatten(),
                (baseline_train[:, -1] + train_preds_squeezed[:, -1]).flatten()
            )
            val_r2 = r2_score(
                (baseline_val[:, -1] + y_val_stacked[:, -1]).flatten(),
                (baseline_val[:, -1] + val_preds_squeezed[:, -1]).flatten()
            )
        else:
            train_r2 = r2_score(
                y_train_stacked[:, -1].flatten(), 
                train_preds_squeezed[:].flatten()
            )
            val_r2 = r2_score(
                y_val_stacked[:, -1].flatten(), 
                val_preds_squeezed[:].flatten()
            )

        # Debugging statements for verification
        print("DEBUG: train_preds_squeezed shape:", train_preds_squeezed.shape)
        print("DEBUG: val_preds_squeezed shape:", val_preds_squeezed.shape)
        print("DEBUG: baseline_train shape:", baseline_train.shape if config.get("use_returns", False) else "Not using returns")
        print("DEBUG: y_train_stacked shape:", y_train_stacked.shape)




        # Calculate MAE  train_mae = np.mean(np.abs(train_predictions - y_train[:n_test]))
        n_train = train_preds.shape[0]
        n_val = val_preds.shape[0]
        train_mae = np.mean(np.abs(train_preds[:, -1] - y_train_stacked[:n_train, -1]))
        val_mae = np.mean(np.abs(val_preds[:, -1] - y_val_stacked[:n_val, -1]))


        # Save loss plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f"Model Loss for {config['plugin'].upper()} - {iteration}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.savefig(config['loss_plot_file'])
        plt.close()
        print(f"Loss plot saved to {config['loss_plot_file']}")

        print("\nEvaluating on test dataset...")
        #test_predictions = plugin.predict(x_test)
        mc_samples = config.get("mc_samples", 100)
        test_predictions, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
        n_test = test_predictions.shape[0]
        # Convert y_test (which is a list of arrays) to a single NumPy array
        y_test_array = np.stack(y_test, axis=1)  # shape: (n_samples, time_horizon)

        # Squeeze predictions to match shapes properly
        test_predictions_squeezed = np.squeeze(test_predictions, axis=-1)  # shape: (samples, horizons)

        if config.get("use_returns", False):
            test_r2 = r2_score(
                (baseline_test[:, -1] + y_test_array[:n_test, -1]).flatten(),
                (baseline_test[:, -1] + test_predictions_squeezed[:n_test, -1]).flatten()
            )
        else:
            test_r2 = r2_score(
                y_test_array[:n_test, -1].flatten(),
                test_predictions_squeezed[:n_test].flatten()
            )

        # Debugging shapes for verification
        print("DEBUG: test_predictions_squeezed shape:", test_predictions_squeezed.shape)
        print("DEBUG: baseline_test shape:", baseline_test.shape if config.get("use_returns", False) else "Not using returns")
        print("DEBUG: y_test_array shape:", y_test_array.shape)


        test_mae = np.mean(np.abs(test_predictions[:, -1] - y_test_array[:n_test, -1]))


        # calculate mmd for train, val and test
        #print("\nCalculating MMD for train, val and test datasets...")
        #train_mmd = plugin.compute_mmd(train_preds.astype(np.float64) , y_train.astype(np.float64), sigma=1.0, sample_size=mc_samples)
        #val_mmd = plugin.compute_mmd(val_preds.astype(np.float64), y_val.astype(np.float64), sigma=1.0, sample_size=mc_samples)
        #test_mmd = plugin.compute_mmd(test_predictions.astype(np.float64), y_test.astype(np.float64), sigma=1.0, sample_size=mc_samples)
        
        # calculate the mean of the last prediction (time_horizon column) uncertainty values
        train_unc_last = np.mean(train_unc[ : , -1])
        val_unc_last = np.mean(val_unc[ : , -1])
        test_unc_last = np.mean(uncertainty_estimates[ : , -1])
        
        # calcula los promedios de la señal para calcular SNR (la desviación es el uncertainty)
        if config.get("use_returns", False):
            train_mean = np.mean(baseline_train[ : , -1] + train_preds[ : , -1])
            val_mean = np.mean(baseline_val[ : , -1] + val_preds[ : , -1])
            test_mean = np.mean(baseline_test[ : , -1] + test_predictions[ : , -1])
        else: 
            train_mean = np.mean(train_preds[ : , -1])
            val_mean = np.mean(val_preds[ : , -1])
            test_mean = np.mean(test_predictions[ : , -1])
        
            

        # calcula the the SNR as the 1/(uncertainty/mae)^2
        train_snr = 1/(train_unc_last/train_mean)
        val_snr = 1/(val_unc_last/val_mean)
        test_snr = 1/(test_unc_last/test_mean)
        
        # calcula el profit y el risk si se está usando una estrategia de trading
        test_profit = 0.0
        test_risk = 0.0
            
        
        # Append the calculated train values
        training_mae_list.append(train_mae)
        training_r2_list.append(train_r2)
        training_unc_list.append(train_unc_last)
        training_snr_list.append(train_snr)
        training_profit_list.append(0)
        training_risk_list.append(0)
        # Append the calculated validation values
        validation_mae_list.append(val_mae)
        validation_r2_list.append(val_r2)
        validation_unc_list.append(val_unc_last)
        validation_snr_list.append(val_snr)
        validation_profit_list.append(0)
        validation_risk_list.append(0)
        # Append the calculated test values
        test_mae_list.append(test_mae)
        test_r2_list.append(test_r2)
        test_unc_list.append(test_unc_last)
        test_snr_list.append(test_snr)
        test_profit_list.append(test_profit)
        test_risk_list.append(test_risk)
        # print iteration results
        print("************************************************************************")
        print(f"Iteration {iteration} completed.")
        print(f"Training MAE: {train_mae}, Training R²: {train_r2}, Training Uncertainty: {train_unc_last}, Trainign SNR: {train_snr}")
        print(f"Validation MAE: {val_mae}, Validation R²: {val_r2}, Validation Uncertainty: {val_unc_last}, Validation SNR: {val_snr}")
        print(f"Test MAE: {test_mae}, Test R²: {test_r2}, Test Uncertainty: {test_unc_last}, Test SNR: {test_snr}, Test Profit: {test_profit}, Test Risk: {test_risk}")
        print("************************************************************************")
        print(f"Iteration {iteration} completed in {time.time()-iter_start:.2f} seconds")
    # Save consolidated results
    if config.get("use_strategy", False): 
        results = {
            "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR", "Train Profit", "Train Risk", 
                        "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR", "Validation Profit", "Validation Risk",
                        "Test MAE", "Test R²", "Test Uncertainty", "Test SNR", "Test Profit", "Test Risk"],
            "Average": [np.mean(training_mae_list), np.mean(training_r2_list), np.mean(training_unc_list), np.mean(training_snr_list), np.mean(training_profit_list), np.mean(training_risk_list),
                        np.mean(validation_mae_list), np.mean(validation_r2_list), np.mean(validation_unc_list), np.mean(validation_snr_list), np.mean(validation_profit_list), np.mean(validation_risk_list),
                        np.mean(test_mae_list), np.mean(test_r2_list), np.mean(test_unc_list), np.mean(test_snr_list), np.mean(test_profit_list), np.mean(test_risk_list)],
            "Std Dev": [np.std(training_mae_list), np.std(training_r2_list), np.std(training_unc_list), np.std(training_snr_list), np.std(training_profit_list), np.std(training_risk_list),
                        np.std(validation_mae_list), np.std(validation_r2_list), np.std(validation_unc_list), np.std(validation_snr_list), np.std(validation_profit_list), np.std(validation_risk_list),
                        np.std(test_mae_list), np.std(test_r2_list), np.std(test_unc_list), np.std(test_snr_list), np.std(test_profit_list), np.std(test_risk_list)],
            "Max": [np.max(training_mae_list), np.max(training_r2_list), np.max(training_unc_list), np.max(training_snr_list), np.max(training_profit_list), np.max(training_risk_list),
                    np.max(validation_mae_list), np.max(validation_r2_list), np.max(validation_unc_list), np.max(validation_snr_list), np.max(validation_profit_list), np.max(validation_risk_list),
                    np.max(test_mae_list), np.max(test_r2_list), np.max(test_unc_list), np.max(test_snr_list), np.max(test_profit_list), np.max(test_risk_list)],
            "Min": [np.min(training_mae_list), np.min(training_r2_list), np.min(training_unc_list), np.min(training_snr_list), np.min(training_profit_list), np.min(training_risk_list),
                    np.min(validation_mae_list), np.min(validation_r2_list), np.min(validation_unc_list), np.min(validation_snr_list), np.min(validation_profit_list), np.min(validation_risk_list),
                    np.min(test_mae_list), np.min(test_r2_list), np.min(test_unc_list), np.min(test_snr_list), np.min(test_profit_list), np.min(test_risk_list)]
            }
    else:
        results = {
            "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR", 
                        "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                        "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"],
            "Average": [np.mean(training_mae_list), np.mean(training_r2_list), np.mean(training_unc_list), np.mean(training_snr_list),
                        np.mean(validation_mae_list), np.mean(validation_r2_list), np.mean(validation_unc_list), np.mean(validation_snr_list),
                        np.mean(test_mae_list), np.mean(test_r2_list), np.mean(test_unc_list), np.mean(test_snr_list)],
            "Std Dev": [np.std(training_mae_list), np.std(training_r2_list), np.std(training_unc_list), np.std(training_snr_list),
                        np.std(validation_mae_list), np.std(validation_r2_list), np.std(validation_unc_list), np.std(validation_snr_list),
                        np.std(test_mae_list), np.std(test_r2_list), np.std(test_unc_list), np.std(test_snr_list)],
            "Max": [np.max(training_mae_list), np.max(training_r2_list), np.max(training_unc_list), np.max(training_snr_list),
                    np.max(validation_mae_list), np.max(validation_r2_list), np.max(validation_unc_list), np.max(validation_snr_list),
                    np.max(test_mae_list), np.max(test_r2_list), np.max(test_unc_list), np.max(test_snr_list)],
            "Min": [np.min(training_mae_list), np.min(training_r2_list), np.min(training_unc_list), np.min(training_snr_list),
                    np.min(validation_mae_list), np.min(validation_r2_list), np.min(validation_unc_list), np.min(validation_snr_list),
                    np.min(test_mae_list), np.min(test_r2_list), np.min(test_unc_list), np.min(test_snr_list)],
            }
    # Save consolidated results to CSV
    results_file = config.get("results_file", "results.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    # --- Denormalize final test predictions (if normalization provided) ---
    denorm_test_close_prices = test_close_prices
    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if config.get("use_returns", False):
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                # Final predicted close = (predicted_return + baseline)*diff + close_min
                # Squeeze the predictions array to remove extra dimensions (6293, 6, 1) → (6293, 6)
                test_predictions_squeezed = np.squeeze(test_predictions, axis=-1)

                # Expand baseline_test dimensions explicitly to match test_predictions
                baseline_test_expanded = np.expand_dims(baseline_test, axis=-1)  # (6293, 1) → (6293, 1, 1)

                # Perform broadcasting explicitly and safely
                test_predictions = (test_predictions_squeezed + baseline_test_expanded.squeeze(-1)) * diff + close_min

                # --- NEW CODE: Correctly stack y_test into a (n_samples, time_horizon) array ---
                y_test_array = np.stack(y_test, axis=1)  # Ensure y_test is now (n_samples, time_horizon)
                denorm_y_test = (y_test_array + baseline_test) * diff + close_min
                # --- END NEW CODE ---
                
            else:
                print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                # Denormalize the predictions only once
                #test_predictions = test_predictions * (close_max - close_min) + close_min
                # For targets, use the already stacked y_test_array
                denorm_y_test = y_test_array * (close_max - close_min) + close_min
            else:
                print("Warning: 'CLOSE' not found; skipping denormalization for non-returns mode.")
    # Denormalize the test close prices once
    denorm_test_close_prices = test_close_prices * (close_max - close_min) + close_min

    # Save final predictions CSV
    final_test_file = config.get("output_file", "test_predictions.csv")
    test_predictions_df = pd.DataFrame(
        test_predictions, columns=[f"Prediction_{i+1}" for i in range(test_predictions.shape[1])]
    )
    # Use test_dates (which now hold the base dates for each window)
    if test_dates is not None:
        test_predictions_df['DATE_TIME'] = pd.Series(test_dates[:len(test_predictions_df)])
    else:
        test_predictions_df['DATE_TIME'] = pd.NaT
    cols = ['DATE_TIME'] + [col for col in test_predictions_df.columns if col != 'DATE_TIME']
    test_predictions_df = test_predictions_df[cols]
    # Add the denorm_y_test to the existing test_predictions_df dictionary as new columns, it names columsn as Target_1, Target_2, etc
    denorm_y_test_df = pd.DataFrame(
        denorm_y_test, columns=[f"Target_{i+1}" for i in range(denorm_y_test.shape[1])]
    )
    test_predictions_df = pd.concat([test_predictions_df, denorm_y_test_df], axis=1)
    # Add the denorm_test_close_prices to the existing test_predictions_df dictionary as a new column
    test_predictions_df['test_CLOSE'] = denorm_test_close_prices
    # Save the final test predictions to a CSV file
    
    write_csv(file_path=final_test_file, data=test_predictions_df, include_date=False, headers=config.get('headers', True))
    print(f"Final validation predictions saved to {final_test_file}")

    # --- Compute and save uncertainty estimates (denormalized) ---
    print("Computing uncertainty estimates using MC sampling...")
    try:
        mc_samples = config.get("mc_samples", 100)
        _, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
        # Denormalize uncertainties using CLOSE range only (do not add offset)
        if config.get("use_normalization_json") is not None:
            norm_json = config.get("use_normalization_json")
            if isinstance(norm_json, str):
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
            if "CLOSE" in norm_json:
                diff = norm_json["CLOSE"]["max"] - norm_json["CLOSE"]["min"]
                denorm_uncertainty = uncertainty_estimates * diff
            else:
                print("Warning: 'CLOSE' not found; uncertainties remain normalized.")
                denorm_uncertainty = uncertainty_estimates
        else:
            denorm_uncertainty = uncertainty_estimates
        uncertainty_df = pd.DataFrame(
            denorm_uncertainty, columns=[f"Uncertainty_{i+1}" for i in range(denorm_uncertainty.shape[1])]
        )
        if test_dates is not None:
            uncertainty_df['DATE_TIME'] = pd.Series(test_dates[:len(uncertainty_df)])
        else:
            uncertainty_df['DATE_TIME'] = pd.NaT
        cols = ['DATE_TIME'] + [col for col in uncertainty_df.columns if col != 'DATE_TIME']
        uncertainty_df = uncertainty_df[cols]
        uncertainties_file = config.get("uncertainties_file", "test_uncertainties.csv")
        uncertainty_df.to_csv(uncertainties_file, index=False)
        print(f"Uncertainty predictions saved to {uncertainties_file}")
    except Exception as e:
        print(f"Failed to compute or save uncertainty predictions: {e}")

    # --- Plot predictions (only the prediction at the selected horizon) ---
    # Define the plotted horizon (zero-indexed)
    plotted_horizon = config.get("plotted_horizon", 6)
    plotted_idx = plotted_horizon - 1  # Zero-based index for the chosen horizon

    # Ensure indices are valid
    if plotted_idx >= test_predictions.shape[1]:
        raise ValueError(f"Plotted horizon index {plotted_idx} is out of bounds for predictions shape {test_predictions.shape}")

    # Extract predictions for the selected horizon
    pred_plot = test_predictions[:, plotted_idx]

    # Define the test dates for plotting
    
    n_plot = config.get("plot_points",1575)  # Number of points to display
    if len(pred_plot) > n_plot:
        pred_plot = pred_plot[-n_plot:]
        test_dates_plot = test_dates[-n_plot:] if test_dates is not None else np.arange(len(pred_plot))
    else:
        test_dates_plot = test_dates if test_dates is not None else np.arange(len(pred_plot))

    # Extract and correctly denormalize the baseline close value (current tick's true value)
    true_plot = denorm_test_close_prices
    # Ensure true_plot is trimmed to match the number of test dates for plotting
    if len(true_plot) > len(test_dates_plot):
        true_plot = true_plot[-len(test_dates_plot):]
       

    # Extract uncertainty for the plotted horizon
    uncertainty_plot = denorm_uncertainty[:, plotted_idx]
    if len(uncertainty_plot) > n_plot:
        uncertainty_plot = uncertainty_plot[-n_plot:]

    # Plot results
    plot_color_predicted = config.get("plot_color_predicted", "blue")
    plot_color_true = config.get("plot_color_true", "red")  # Default: red
    plot_color_uncertainty = config.get("plot_color_uncertainty", "green")  # Default: green    
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates_plot, pred_plot, label="Predicted Price", color=plot_color_predicted, linewidth=2)
    plt.plot(test_dates_plot, true_plot, label="True Price", color=plot_color_true, linewidth=2)
    # Ensure pred_plot and uncertainty_plot are explicitly squeezed to be 1-dimensional.
    pred_plot = np.squeeze(pred_plot)
    uncertainty_plot = np.squeeze(uncertainty_plot)

    # Check final dimensions to ensure correctness:
    assert pred_plot.ndim == 1, f"pred_plot must be 1-dimensional, got shape {pred_plot.shape}"
    assert uncertainty_plot.ndim == 1, f"uncertainty_plot must be 1-dimensional, got shape {uncertainty_plot.shape}"

    # Now safely plot uncertainty
    plt.fill_between(
        test_dates_plot,
        pred_plot - uncertainty_plot,
        pred_plot + uncertainty_plot,
        color=plot_color_uncertainty,
        alpha=0.15,
        label="Uncertainty"
    )

    if config.get("use_daily", False):    
        plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} days)")
    else:
        plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} hours)")
    plt.xlabel("Close Time")
    plt.ylabel("EUR Price [USD]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    try:
        predictions_plot_file = config.get("predictions_plot_file", "predictions_plot.png")
        plt.savefig(predictions_plot_file, dpi=300)
        plt.close()
        print(f"Prediction plot saved to {predictions_plot_file}")
    except Exception as e:
        print(f"Failed to generate prediction plot: {e}")


    # Plot the model
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(
            plugin.model,
            to_file=config['model_plot_file'],
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=True,
            dpi=300,
            show_layer_activations=True
        )
        print(f"Model plot saved to {config['model_plot_file']}")
    except Exception as e:
        print(f"Failed to generate model plot: {e}")
        print("Download Graphviz from https://graphviz.org/download/")

    save_model_file = config.get("save_model", "pretrained_model.keras")
    try:
        plugin.save(save_model_file)
        print(f"Model saved to {save_model_file}")
    except Exception as e:
        print(f"Failed to save model to {save_model_file}: {e}")
    
    if config.get("use_strategy", False):
        print("*************************************************")
        print("Training Statistics:")
        print(f"MAE - Avg: {results['Average'][0]:.4f}, Std: {results['Std Dev'][0]:.4f}, Max: {results['Max'][0]:.4f}, Min: {results['Min'][0]:.4f}")
        print(f"R²  - Avg: {results['Average'][1]:.4f}, Std: {results['Std Dev'][1]:.4f}, Max: {results['Max'][1]:.4f}, Min: {results['Min'][1]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][2]:.4f}, Std: {results['Std Dev'][2]:.4f}, Max: {results['Max'][2]:.4f}, Min: {results['Min'][2]:.4f}")
        print(f"SNR - Avg: {results['Average'][3]:.4f}, Std: {results['Std Dev'][3]:.4f}, Max: {results['Max'][3]:.4f}, Min: {results['Min'][3]:.4f}")
        print(f"Profit - Avg: {results['Average'][4]:.4f}, Std: {results['Std Dev'][4]:.4f}, Max: {results['Max'][4]:.4f}, Min: {results['Min'][4]:.4f}")
        print(f"Risk - Avg: {results['Average'][5]:.4f}, Std: {results['Std Dev'][5]:.4f}, Max: {results['Max'][5]:.4f}, Min: {results['Min'][5]:.4f}")
        print("*************************************************")
        print("Validation Statistics:")
        print(f"MAE - Avg: {results['Average'][6]:.4f}, Std: {results['Std Dev'][6]:.4f}, Max: {results['Max'][6]:.4f}, Min: {results['Min'][6]:.4f}")
        print(f"R²  - Avg: {results['Average'][7]:.4f}, Std: {results['Std Dev'][7]:.4f}, Max: {results['Max'][7]:.4f}, Min: {results['Min'][7]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][8]:.4f}, Std: {results['Std Dev'][8]:.4f}, Max: {results['Max'][8]:.4f}, Min: {results['Min'][8]:.4f}")
        print(f"SNR - Avg: {results['Average'][9]:.4f}, Std: {results['Std Dev'][9]:.4f}, Max: {results['Max'][9]:.4f}, Min: {results['Min'][9]:.4f}")
        print(f"Profit - Avg: {results['Average'][10]:.4f}, Std: {results['Std Dev'][10]:.4f}, Max: {results['Max'][10]:.4f}, Min: {results['Min'][10]:.4f}")
        print(f"Risk - Avg: {results['Average'][11]:.4f}, Std: {results['Std Dev'][11]:.4f}, Max: {results['Max'][11]:.4f}, Min: {results['Min'][11]:.4f}")
        print("*************************************************")
        print("Test Statistics:")
        print(f"MAE - Avg: {results['Average'][12]:.4f}, Std: {results['Std Dev'][12]:.4f}, Max: {results['Max'][12]:.4f}, Min: {results['Min'][12]:.4f}")
        print(f"R²  - Avg: {results['Average'][13]:.4f}, Std: {results['Std Dev'][13]:.4f}, Max: {results['Max'][13]:.4f}, Min: {results['Min'][13]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][14]:.4f}, Std: {results['Std Dev'][14]:.4f}, Max: {results['Max'][14]:.4f}, Min: {results['Min'][14]:.4f}")
        print(f"SNR - Avg: {results['Average'][15]:.4f}, Std: {results['Std Dev'][15]:.4f}, Max: {results['Max'][15]:.4f}, Min: {results['Min'][15]:.4f}")
        print(f"Profit - Avg: {results['Average'][16]:.4f}, Std: {results['Std Dev'][16]:.4f}, Max: {results['Max'][16]:.4f}, Min: {results['Min'][16]:.4f}")
        print(f"Risk - Avg: {results['Average'][17]:.4f}, Std: {results['Std Dev'][17]:.4f}, Max: {results['Max'][17]:.4f}, Min: {results['Min'][17]:.4f}")
        print("*************************************************")
    else:
        print("*************************************************")
        print("Training Statistics:")
        print(f"MAE - Avg: {results['Average'][0]:.4f}, Std: {results['Std Dev'][0]:.4f}, Max: {results['Max'][0]:.4f}, Min: {results['Min'][0]:.4f}")
        print(f"R²  - Avg: {results['Average'][1]:.4f}, Std: {results['Std Dev'][1]:.4f}, Max: {results['Max'][1]:.4f}, Min: {results['Min'][1]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][2]:.4f}, Std: {results['Std Dev'][2]:.4f}, Max: {results['Max'][2]:.4f}, Min: {results['Min'][2]:.4f}")
        print(f"SNR - Avg: {results['Average'][3]:.4f}, Std: {results['Std Dev'][3]:.4f}, Max: {results['Max'][3]:.4f}, Min: {results['Min'][3]:.4f}")
        print("*************************************************")
        print("Validation Statistics:")
        print(f"MAE - Avg: {results['Average'][4]:.4f}, Std: {results['Std Dev'][4]:.4f}, Max: {results['Max'][4]:.4f}, Min: {results['Min'][4]:.4f}")
        print(f"R²  - Avg: {results['Average'][5]:.4f}, Std: {results['Std Dev'][5]:.4f}, Max: {results['Max'][5]:.4f}, Min: {results['Min'][5]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][6]:.4f}, Std: {results['Std Dev'][6]:.4f}, Max: {results['Max'][6]:.4f}, Min: {results['Min'][6]:.4f}")
        print(f"SNR - Avg: {results['Average'][7]:.4f}, Std: {results['Std Dev'][7]:.4f}, Max: {results['Max'][7]:.4f}, Min: {results['Min'][7]:.4f}") 
        print("*************************************************")
        print("Test Statistics:")
        print(f"MAE - Avg: {results['Average'][8]:.4f}, Std: {results['Std Dev'][8]:.4f}, Max: {results['Max'][8]:.4f}, Min: {results['Min'][8]:.4f}")
        print(f"R²  - Avg: {results['Average'][9]:.4f}, Std: {results['Std Dev'][9]:.4f}, Max: {results['Max'][9]:.4f}, Min: {results['Min'][9]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][10]:.4f}, Std: {results['Std Dev'][10]:.4f}, Max: {results['Max'][10]:.4f}, Min: {results['Min'][10]:.4f}")
        print(f"SNR - Avg: {results['Average'][11]:.4f}, Std: {results['Std Dev'][11]:.4f}, Max: {results['Max'][11]:.4f}, Min: {results['Min'][11]:.4f}")
        print("*************************************************")
    
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")


def load_and_evaluate_model(config, plugin):
    """
    Loads a pre-trained model and evaluates it on the validation data.
    Predictions are denormalized; if use_returns is True, predicted returns are converted
    to predicted close values by adding the corresponding baseline (using the "CLOSE" parameters).
    The final predictions CSV includes a DATE_TIME column.
    """
    import sys, numpy as np, pandas as pd, json
    from tensorflow.keras.models import load_model

    print(f"Loading pre-trained model from {config['load_model']}...")
    try:
        custom_objects = {"combined_loss": combined_loss, "mmd": mmd_metric, "huber": huber_metric}
        plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model from {config['load_model']}: {e}")
        sys.exit(1)

    print("Loading and processing validation data for evaluation...")
    try:
        datasets = process_data(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("dates_val")
        if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
            print("Creating sliding windows for CNN...")
            x_val, val_date_windows = create_sliding_windows(
                x_val, config['window_size'], stride=1, date_times=val_dates
            )
            val_dates = val_date_windows
            print(f"Sliding windows created: x_val: {x_val.shape}")
            if x_val.ndim != 3:
                raise ValueError(f"For CNN and LSTM, x_val must be 3D. Found: {x_val.shape}.")

        print(f"Processed validation data: X shape: {x_val.shape}")
    except Exception as e:
        print(f"Failed to process validation data: {e}")
        sys.exit(1)

    print("Making predictions on validation data...")
    try:
        x_val_array = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()

        #predictions = plugin.predict(x_val_array)
        mc_samples = config.get("mc_samples", 100)
        predictions, uncertainty_estimates = plugin.predict_with_uncertainty(x_val_array, mc_samples=mc_samples)
        
        
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        sys.exit(1)

    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        # Denormalize predictions using CLOSE range.
        if config.get("use_returns", False):
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                if "baseline_val" in datasets:
                    baseline = datasets["baseline_val"]
                    predictions = (predictions + baseline) * diff + close_min
                else:
                    print("Warning: Baseline validation values not found; cannot convert returns to predicted close values.")
            else:
                print("Warning: 'CLOSE' not found; skipping proper denormalization for returns.")
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                predictions = predictions * (close_max - close_min) + close_min

    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    else:
        num_steps = predictions.shape[1]
        pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
        predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    if val_dates is not None:
        predictions_df['DATE_TIME'] = pd.Series(val_dates[:len(predictions_df)])
    else:
        predictions_df['DATE_TIME'] = pd.NaT
        print("Warning: DATE_TIME for validation predictions not captured.")

    cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
    predictions_df = predictions_df[cols]
    evaluate_filename = config['output_file']
    try:
        write_csv(file_path=evaluate_filename, data=predictions_df,
                  include_date=False, headers=config.get('headers', True))
        print(f"Validation predictions with DATE_TIME saved to {evaluate_filename}")
    except Exception as e:
        print(f"Failed to save validation predictions to {evaluate_filename}: {e}")
        sys.exit(1)



def create_sliding_windows(x, window_size, stride=1, date_times=None):
    """
    Creates sliding windows for input features and targets with a specified stride.

    Args:
        x (numpy.ndarray): Input features of shape (N, features).
        y (numpy.ndarray): Targets of shape (N,) or (N, 1).
        window_size (int): Number of past steps to include in each window.
        time_horizon (int): Number of future steps to predict.
        stride (int): Step size between windows.
        date_times (pd.DatetimeIndex, optional): Corresponding date times for each sample.

    Returns:
        tuple:
            - x_windowed (numpy.ndarray): Shaped (samples, window_size, features).
            - y_windowed (numpy.ndarray): Shaped (samples, time_horizon).
            - date_time_windows (list): List of date times for each window (if provided).
    """
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    elif y.ndim > 2:
        raise ValueError("y should be a 1D or 2D array with a single column.")

    x_windowed = []
    y_windowed = []
    date_time_windows = []

    for i in range(0, len(x) - window_size, stride):
        x_window = x[i:i + window_size]
        x_windowed.append(x_window)
        if date_times is not None:
            date_time_windows.append(date_times[i + window_size - 1])

    return np.array(x_windowed), date_time_windows


def generate_positional_encoding(num_features, pos_dim=16):
    """
    Generates positional encoding for a given number of features.

    Args:
        num_features (int): Number of features in the dataset.
        pos_dim (int): Dimension of the positional encoding.

    Returns:
        np.ndarray: Positional encoding of shape (1, num_features * pos_dim).
    """
    position = np.arange(num_features)[:, np.newaxis]
    div_term = np.exp(np.arange(0, pos_dim, 2) * -(np.log(10000.0) / pos_dim))
    pos_encoding = np.zeros((num_features, pos_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    pos_encoding_flat = pos_encoding.flatten().reshape(1, -1)  # Shape: (1, num_features * pos_dim)
    return pos_encoding_flat


def gaussian_kernel_matrix(x, y, sigma):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    x_expanded = tf.reshape(x, [x_size, 1, dim])
    y_expanded = tf.reshape(y, [1, y_size, dim])
    squared_diff = tf.reduce_sum(tf.square(x_expanded - y_expanded), axis=2)
    return tf.exp(-squared_diff / (2.0 * sigma**2))

def combined_loss(y_true, y_pred):
    huber_loss = Huber(delta=1.0)(y_true, y_pred)
    sigma = 1.0
    stat_weight = 1.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
    K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
    K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
    m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
    mmd = tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
    return huber_loss + stat_weight * mmd

def mmd_metric(y_true, y_pred):
    sigma = 1.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
    K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
    K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
    m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
    return tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
mmd_metric.__name__ = "mmd"

def huber_metric(y_true, y_pred):
    return Huber(delta=1.0)(y_true, y_pred)

huber_metric.__name__ = "huber"
