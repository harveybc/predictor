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

def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
    """
    Create sliding windows for input data only.

    Args:
        data (np.ndarray or pd.DataFrame): Input data array of shape (n_samples, n_features).
        window_size (int): The number of time steps in each window.
        stride (int): The stride between successive windows.
        date_times (pd.DatetimeIndex, optional): Corresponding date times for each sample.

    Returns:
        If date_times is provided:
            tuple: (windows, date_time_windows) where windows is an array of shape 
                   (n_windows, window_size, n_features) and date_time_windows is a list of 
                   the DATE_TIME value corresponding to the last time step of each window.
        Otherwise:
            np.ndarray: Array of sliding windows.
    """
    windows = []
    dt_windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i: i + window_size])
        if date_times is not None:
            # Use the date corresponding to the last element in the window
            dt_windows.append(date_times[i + window_size - 1])
    if date_times is not None:
        return np.array(windows), dt_windows
    else:
        return np.array(windows)

def create_multi_step(y_df, horizon, use_returns=False):
    """
    Creates multi-step targets for time-series prediction.
    If use_returns is True, targets are computed as the difference between each future value 
    and the current (baseline) value.

    Args:
        y_df (pd.DataFrame): Target data as a DataFrame.
        horizon (int): Number of future steps to predict.
        use_returns (bool): If True, compute returns instead of absolute values.

    Returns:
        pd.DataFrame: Multi-step targets aligned with the input data, where the index 
                      remains the same as the original (i.e. the date is not shifted)
                      but the target values are taken from the future starting at row i+horizon.
        (if use_returns is True) pd.DataFrame: Baseline values corresponding to each target row.
    """
    blocks = []
    baselines = []
    for i in range(len(y_df) - horizon):
        base = y_df.iloc[i].values.flatten()
        if use_returns:
            window = list(y_df.iloc[i + horizon : i + horizon + horizon].values.flatten() - base)
        else:
            window = list(y_df.iloc[i + horizon : i + horizon + horizon].values.flatten())
        blocks.append(window)
        if use_returns:
            baselines.append(base)
    # Use the same dates as the original rows (up to the row before the last horizon rows)
    df_targets = pd.DataFrame(blocks, index=y_df.index[:len(blocks)])
    if use_returns:
        df_baselines = pd.DataFrame(baselines, index=y_df.index[:len(baselines)])
        return df_targets, df_baselines
    else:
        return df_targets


def create_multi_step_daily(y_df, horizon, use_returns=False):
    """
    Creates multi-step targets for time-series prediction using daily data.
    If use_returns is True, targets are computed as the difference between each future value 
    and the current (baseline) value.

    Args:
        y_df (pd.DataFrame): Target data as a DataFrame.
        horizon (int): Number of future days to predict.
        use_returns (bool): If True, compute returns instead of absolute values.

    Returns:
        pd.DataFrame: Multi-step targets aligned with the input data, with the index remaining
                      the same as the original while the target values are taken from rows at
                      24-tick intervals (i.e. from t+24, t+48, ..., t+24*horizon).
        (if use_returns is True) pd.DataFrame: Baseline values corresponding to each target row.
    """
    blocks = []
    baselines = []
    for i in range(len(y_df) - horizon * 24):
        base = y_df.iloc[i].values.flatten()
        window = []
        for d in range(1, horizon + 1):
            val = y_df.iloc[i + 24 * d].values.flatten()
            if use_returns:
                window.extend(list(val - base))
            else:
                window.extend(list(val))
        blocks.append(window)
        if use_returns:
            baselines.append(base)
    df_targets = pd.DataFrame(blocks, index=y_df.index[:len(blocks)])
    if use_returns:
        df_baselines = pd.DataFrame(baselines, index=y_df.index[:len(baselines)])
        return df_targets, df_baselines
    else:
        return df_targets


def process_data(config):
    """
    Processes data for different plugins, including ANN, CNN, LSTM, and Transformer.
    Loads and processes training, validation, and test datasets; extracts DATE_TIME information,
    and trims each pair (x and y) to their common date range so that they share the same number of rows.
    The multi-step targets are now generated with their index shifted back so that each target corresponds
    to the same date as the input data (or later trimmed to align with sliding window dates).

    Returns:
        dict: Processed datasets for training, validation, and test, along with corresponding
              DATE_TIME arrays. Additionally, if config['use_returns'] is True, the corresponding 
              baseline target values (the original CLOSE values) are also returned.
    """
    import pandas as pd
    # 1) LOAD CSVs for train, validation, and test
    x_train = load_csv(config["x_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    y_train = load_csv(config["y_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    x_val = load_csv(config["x_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_val"))
    y_val = load_csv(config["y_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_val"))
    x_test = load_csv(config["x_test_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))
    y_test = load_csv(config["y_test_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))
    
    # 1a) Trim to common date range if possible.
    if isinstance(x_train.index, pd.DatetimeIndex) and isinstance(y_train.index, pd.DatetimeIndex):
        common_train_index = x_train.index.intersection(y_train.index)
        x_train = x_train.loc[common_train_index]
        y_train = y_train.loc[common_train_index]
    if isinstance(x_val.index, pd.DatetimeIndex) and isinstance(y_val.index, pd.DatetimeIndex):
        common_val_index = x_val.index.intersection(y_val.index)
        x_val = x_val.loc[common_val_index]
        y_val = y_val.loc[common_val_index]
    if isinstance(x_test.index, pd.DatetimeIndex) and isinstance(y_test.index, pd.DatetimeIndex):
        common_test_index = x_test.index.intersection(y_test.index)
        x_test = x_test.loc[common_test_index]
        y_test = y_test.loc[common_test_index]
    
    # Save original DATE_TIME indices AFTER trimming.
    train_dates_orig = x_train.index if isinstance(x_train.index, pd.DatetimeIndex) else None
    val_dates_orig = x_val.index if isinstance(x_val.index, pd.DatetimeIndex) else None
    test_dates_orig = x_test.index if isinstance(x_test.index, pd.DatetimeIndex) else None

    # 2) EXTRACT THE TARGET COLUMN
    target_col = config["target_column"]
    def extract_target(df, col):
        if isinstance(col, str):
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found.")
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("`target_column` must be str or int.")
    y_train = extract_target(y_train, target_col)
    y_val = extract_target(y_val, target_col)
    y_test = extract_target(y_test, target_col)

    # 3) CONVERT EACH DF TO NUMERIC.
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_val = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val = y_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = y_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4) MULTI-STEP TARGETS (using our updated target generators)
    time_horizon = config["time_horizon"]
    if config.get("use_daily", False):
        y_train_ma = y_train.rolling(window=3, center=True, min_periods=1).mean()
        y_val_ma = y_val.rolling(window=3, center=True, min_periods=1).mean()
        y_test_ma = y_test.rolling(window=3, center=True, min_periods=1).mean() 
        if config.get("use_returns", False):
            y_train_multi, baseline_train = create_multi_step_daily(y_train_ma, time_horizon, use_returns=True)
            y_val_multi, baseline_val = create_multi_step_daily(y_val_ma, time_horizon, use_returns=True)
            y_test_multi, baseline_test = create_multi_step_daily(y_test_ma, time_horizon, use_returns=True)
        else:
            y_train_multi = create_multi_step_daily(y_train_ma, time_horizon, use_returns=False)
            y_val_multi = create_multi_step_daily(y_val_ma, time_horizon, use_returns=False)
            y_test_multi = create_multi_step_daily(y_test_ma, time_horizon, use_returns=False)
    else:
        if config.get("use_returns", False):
            y_train_multi, baseline_train = create_multi_step(y_train, time_horizon, use_returns=True)
            y_val_multi, baseline_val = create_multi_step(y_val, time_horizon, use_returns=True)
            y_test_multi, baseline_test = create_multi_step(y_test, time_horizon, use_returns=True)
        else:
            y_train_multi = create_multi_step(y_train, time_horizon, use_returns=False)
            y_val_multi = create_multi_step(y_val, time_horizon, use_returns=False)
            y_test_multi = create_multi_step(y_test, time_horizon, use_returns=False)

    # 5) TRIM x TO MATCH THE LENGTH OF y (for each dataset)
    min_len_train = min(len(x_train), len(y_train_multi))
    x_train = x_train.iloc[:min_len_train]
    y_train_multi = y_train_multi.iloc[:min_len_train]
    if config.get("use_returns", False):
        baseline_train = baseline_train.iloc[:min_len_train]
    min_len_val = min(len(x_val), len(y_val_multi))
    x_val = x_val.iloc[:min_len_val]
    y_val_multi = y_val_multi.iloc[:min_len_val]
    if config.get("use_returns", False):
        baseline_val = baseline_val.iloc[:min_len_val]
    min_len_test = min(len(x_test), len(y_test_multi))
    x_test = x_test.iloc[:min_len_test]
    y_test_multi = y_test_multi.iloc[:min_len_test]
    if config.get("use_returns", False):
        baseline_test = baseline_test.iloc[:min_len_test]
    
    train_dates = train_dates_orig[:min_len_train] if train_dates_orig is not None else None
    val_dates = val_dates_orig[:min_len_val] if val_dates_orig is not None else None
    test_dates = test_dates_orig[:min_len_test] if test_dates_orig is not None else None

    # 6) PER-PLUGIN PROCESSING (including optional sliding windows)
    if config["plugin"] in ["lstm", "cnn", "transformer"]:
        print("Processing data with sliding windows...")
        x_train = x_train.to_numpy().astype(np.float32)
        x_val = x_val.to_numpy().astype(np.float32)
        x_test = x_test.to_numpy().astype(np.float32)
        window_size = config["window_size"]
        def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
            windows = []
            dt_windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
                if date_times is not None:
                    dt_windows.append(date_times[i+window_size-1])
            return np.array(windows), dt_windows if date_times is not None else np.array(windows)
        x_train, train_dates = create_sliding_windows_x(x_train, window_size, stride=1, date_times=train_dates)
        x_val, val_dates = create_sliding_windows_x(x_val, window_size, stride=1, date_times=val_dates)
        x_test, test_dates = create_sliding_windows_x(x_test, window_size, stride=1, date_times=test_dates)
        # Align multi-step targets with sliding windows:
        y_train_multi = y_train_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_val_multi = y_val_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_test_multi = y_test_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        if config.get("use_returns", False):
            baseline_train = baseline_train.iloc[window_size - 1:].to_numpy().astype(np.float32)
            baseline_val = baseline_val.iloc[window_size - 1:].to_numpy().astype(np.float32)
            baseline_test = baseline_test.iloc[window_size - 1:].to_numpy().astype(np.float32)
    else:
        print("Not using sliding windows; converting data to NumPy arrays without windowing.")
        x_train = x_train.to_numpy().astype(np.float32)
        x_val = x_val.to_numpy().astype(np.float32)
        x_test = x_test.to_numpy().astype(np.float32)
        y_train_multi = y_train_multi.to_numpy().astype(np.float32)
        y_val_multi = y_val_multi.to_numpy().astype(np.float32)
        y_test_multi = y_test_multi.to_numpy().astype(np.float32)
        if config.get("use_returns", False):
            baseline_train = baseline_train.to_numpy().astype(np.float32)
            baseline_val = baseline_val.to_numpy().astype(np.float32)
            baseline_test = baseline_test.to_numpy().astype(np.float32)

    print("Processed datasets:")
    print(" x_train:", x_train.shape, " y_train:", y_train_multi.shape)
    print(" x_val:  ", x_val.shape, " y_val:  ", y_val_multi.shape)
    print(" x_test: ", x_test.shape, " y_test: ", y_test_multi.shape)
    
    ret = {
        "x_train": x_train,
        "y_train": y_train_multi,
        "x_val": x_val,
        "y_val": y_val_multi,
        "x_test": x_test,
        "y_test": y_test_multi,
        "dates_train": train_dates,
        "dates_val": val_dates,
        "dates_test": test_dates,
    }
    if config.get("use_returns", False):
        ret["baseline_train"] = baseline_train
        ret["baseline_val"] = baseline_val
        ret["baseline_test"] = baseline_test
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
    train_dates = datasets.get("dates_train")
    val_dates = datasets.get("dates_val")
    test_dates = datasets.get("dates_test")
    if config.get("use_returns", False):
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

        # ---- Verification check before feeding data to the model ----
    try:
        # Load the original y_train from file using the same max_rows as in process_data.
        original_y_train = load_csv(config["y_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
        original_y_train = original_y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        
        # Determine the offset: if sliding windows are used, offset = window_size - 1; otherwise, 0.
        offset = config["window_size"] - 1 if config["plugin"] in ["lstm", "cnn", "transformer"] else 0

        # Process y_train similarly as in process_data.
        if config.get("use_daily", False):
            processed_y = original_y_train.rolling(window=3, center=True, min_periods=1).mean()
        else:
            processed_y = original_y_train

        # Use the row at index 'offset' as the base.
        base_value = processed_y.iloc[offset].values[0]
        expected_values = []
        debug_details = []
        if config.get("use_daily", False):
            # For daily mode: use step = 24 rows.
            for d in range(1, config["time_horizon"] + 1):
                idx = offset + 24 * d
                if idx >= len(processed_y):
                    debug_details.append(f"[DEBUG] Day {d}: index {idx} is out of bounds (length {len(processed_y)})")
                    continue
                future_value = processed_y.iloc[idx].values[0]
                diff_value = future_value - base_value if config.get("use_returns", False) else future_value
                expected_values.append(diff_value)
                debug_details.append(f"[DEBUG] Day {d}: row index {idx}: future value = {future_value:.8f}, base = {base_value:.8f}, diff = {diff_value:.8f}")
        else:
            # For non-daily mode: use step = 1 row.
            for d in range(1, config["time_horizon"] + 1):
                idx = offset + d
                if idx >= len(processed_y):
                    debug_details.append(f"[DEBUG] Tick {d}: index {idx} is out of bounds (length {len(processed_y)})")
                    continue
                future_value = processed_y.iloc[idx].values[0]
                diff_value = future_value - base_value if config.get("use_returns", False) else future_value
                expected_values.append(diff_value)
                debug_details.append(f"[DEBUG] Tick {d}: row index {idx}: future value = {future_value:.8f}, base = {base_value:.8f}, diff = {diff_value:.8f}")
        expected_row = np.array(expected_values)
        
        print(f"[DEBUG] Verification offset: {offset}")
        print(f"[DEBUG] Base row at index {offset}: {processed_y.iloc[offset].values}")
        for detail in debug_details:
            print(detail)
        
        # Retrieve the computed multi-step target row from the processed dataset.
        # Use .iloc if available; otherwise, use standard NumPy indexing.
        if hasattr(datasets["y_train"], 'iloc'):
            computed_row = datasets["y_train"].iloc[offset].values
        else:
            computed_row = datasets["y_train"][offset, :]
        
        print(f"[DEBUG] Expected multi-step target row: {expected_row}")
        print(f"[DEBUG] Computed multi-step target row: {computed_row}")
        
        if not np.allclose(expected_row, computed_row, atol=1e-5):
            print("Verification check failed: the multi-step target does not match the expected future values.")
            sys.exit(1)
        else:
            print("Verification check passed: multi-step target values are correctly shifted without altering dates.")
    except Exception as e:
        print(f"Verification check error: {e}")
        sys.exit(1)
    # ---- End of verification check ----



    # If sliding windows output is a tuple, extract the data.
    if isinstance(x_train, tuple): x_train = x_train[0]
    if isinstance(x_val, tuple): x_val = x_val[0]
    if isinstance(x_test, tuple): x_test = x_test[0]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"Test data shapes: x_test: {x_test.shape}, y_test: {y_test.shape}")

    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")
    if config["plugin"] in ["lstm", "cnn", "transformer"] and window_size is None:
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

    if config["plugin"] in ["lstm", "cnn", "transformer"]:
        if x_train.ndim != 3:
            raise ValueError(f"For CNN and LSTM, x_train must be 3D. Found: {x_train.shape}")
        print("Using pre-processed sliding windows for CNN and LSTM.")
    plugin.set_params(time_horizon=time_horizon)
    tscv = TimeSeriesSplit(n_splits=5)

    # Training iterations
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iter_start = time.time()
        if config["plugin"] in ["lstm", "cnn", "transformer"]:
            plugin.build_model(input_shape=(window_size, x_train.shape[2]), x_train=x_train)
        elif config["plugin"] in ["transformer", "transformer_mmd"]:
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train)
        else:
            if len(x_train.shape) != 2:
                raise ValueError(f"Expected 2D x_train for {config['plugin']}; got {x_train.shape}")
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)

        history, train_preds, train_unc, val_preds, val_unc = plugin.train(
            x_train, y_train, epochs=epochs, batch_size=batch_size,
            threshold_error=threshold_error, x_val=x_val, y_val=y_val, config=config
        )
        # If using returns, recalc r2 based on baseline + predictions.
        if config.get("use_returns", False):
            train_r2 = r2_score((baseline_train[:, -1] + y_train[:, -1]).flatten(), (baseline_train[:, -1] + train_preds[:, -1]).flatten())
            val_r2 = r2_score((baseline_val[:, -1] + y_val[:, -1]).flatten(), (baseline_val[:, -1] + val_preds[:, -1]).flatten())
        else:
            train_r2 = r2_score(y_train[:, -1], train_preds[:, -1])
            val_r2 = r2_score(y_val[:, -1], val_preds[:, -1])

        n_train = train_preds.shape[0]
        n_val = val_preds.shape[0]
        train_mae = np.mean(np.abs(train_preds[:, -1] - y_train[:n_train, -1]))
        val_mae = np.mean(np.abs(val_preds[:, -1] - y_val[:n_val, -1]))

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
        mc_samples = config.get("mc_samples", 100)
        test_predictions, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
        n_test = test_predictions.shape[0]
        if config.get("use_returns", False):
            test_r2 = r2_score((baseline_test[:, -1] + y_test[:n_test, -1]).flatten(), (baseline_test[:, -1] + test_predictions[:, -1]).flatten())
        else:
            test_r2 = r2_score(y_test[:n_test, -1], test_predictions[:, -1])
        test_mae = np.mean(np.abs(test_predictions[:, -1] - y_test[:n_test, -1]))
        train_unc_last = np.mean(train_unc[:, -1])
        val_unc_last = np.mean(val_unc[:, -1])
        test_unc_last = np.mean(uncertainty_estimates[:, -1])
        train_mean = np.mean(baseline_train[:, -1] + train_preds[:, -1])
        val_mean = np.mean(baseline_val[:, -1] + val_preds[:, -1])
        test_mean = np.mean(baseline_test[:, -1] + test_predictions[:, -1])
        train_snr = 1 / (train_unc_last / train_mean)
        val_snr = 1 / (val_unc_last / val_mean)
        test_snr = 1 / (test_unc_last / test_mean)
        test_profit = 0.0
        test_risk = 0.0

        training_mae_list.append(train_mae)
        training_r2_list.append(train_r2)
        training_unc_list.append(train_unc_last)
        training_snr_list.append(train_snr)
        training_profit_list.append(0)
        training_risk_list.append(0)
        validation_mae_list.append(val_mae)
        validation_r2_list.append(val_r2)
        validation_unc_list.append(val_unc_last)
        validation_snr_list.append(val_snr)
        validation_profit_list.append(0)
        validation_risk_list.append(0)
        test_mae_list.append(test_mae)
        test_r2_list.append(test_r2)
        test_unc_list.append(test_unc_last)
        test_snr_list.append(test_snr)
        test_profit_list.append(test_profit)
        test_risk_list.append(test_risk)
        print("************************************************************************")
        print(f"Iteration {iteration} completed.")
        print(f"Training MAE: {train_mae}, Training R²: {train_r2}, Training Uncertainty: {train_unc_last}, Trainign SNR: {train_snr}")
        print(f"Validation MAE: {val_mae}, Validation R²: {val_r2}, Validation Uncertainty: {val_unc_last}, Validation SNR: {val_snr}")
        print(f"Test MAE: {test_mae}, Test R²: {test_r2}, Test Uncertainty: {test_unc_last}, Test SNR: {test_snr}, Test Profit: {test_profit}, Test Risk: {test_risk}")
        print("************************************************************************")
        print(f"Iteration {iteration} completed in {time.time() - iter_start:.2f} seconds")
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
    results_file = config.get("results_file", "results.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # --- Denormalize final test predictions (if normalization provided) ---
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
                test_predictions = (test_predictions + baseline_test) * diff + close_min
            else:
                print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                test_predictions = test_predictions * (close_max - close_min) + close_min

    final_test_file = config.get("output_file", "test_predictions.csv")
    test_predictions_df = pd.DataFrame(
        test_predictions, columns=[f"Prediction_{i+1}" for i in range(test_predictions.shape[1])]
    )
    if test_dates is not None:
        test_predictions_df['DATE_TIME'] = pd.Series(test_dates[:len(test_predictions_df)])
    else:
        test_predictions_df['DATE_TIME'] = pd.NaT
    cols = ['DATE_TIME'] + [col for col in test_predictions_df.columns if col != 'DATE_TIME']
    test_predictions_df = test_predictions_df[cols]
    write_csv(file_path=final_test_file, data=test_predictions_df, include_date=False, headers=config.get('headers', True))
    print(f"Final validation predictions saved to {final_test_file}")

    # --- Compute and save uncertainty estimates (denormalized) ---
    print("Computing uncertainty estimates using MC sampling...")
    try:
        mc_samples = config.get("mc_samples", 100)
        _, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
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
    plotted_horizon = config.get("plotted_horizon", 6)
    plotted_idx = plotted_horizon - 1  # Zero-based index for the chosen horizon
    if plotted_idx >= test_predictions.shape[1]:
        raise ValueError(f"Plotted horizon index {plotted_idx} is out of bounds for predictions shape {test_predictions.shape}")
    pred_plot = test_predictions[:, plotted_idx]
    n_plot = config.get("plot_points", 1575)
    if len(pred_plot) > n_plot:
        pred_plot = pred_plot[-n_plot:]
        test_dates_plot = test_dates[-n_plot:] if test_dates is not None else np.arange(len(pred_plot))
    else:
        test_dates_plot = test_dates if test_dates is not None else np.arange(len(pred_plot))
    if "baseline_test" in datasets:
        baseline_plot = datasets["baseline_test"][:, 0]
        if len(baseline_plot) > n_plot:
            baseline_plot = baseline_plot[-n_plot:]
    else:
        raise ValueError("Baseline test values not found; unable to reconstruct actual predictions.")
    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_min = norm_json["CLOSE"]["min"]
            close_max = norm_json["CLOSE"]["max"]
            diff = close_max - close_min
            true_plot = baseline_plot * diff + close_min  # ✅ Correct
            #pred_plot = true_plot + (pred_plot * diff)  # ✅ Fixing double denormalization (commented out)
        else:
            print("Warning: 'CLOSE' not found; skipping denormalization for predictions.")
            true_plot = baseline_plot
            pred_plot = baseline_plot + pred_plot
    else:
        print("Warning: Normalization JSON not provided; assuming raw values.")
        true_plot = baseline_plot
        pred_plot = baseline_plot + pred_plot
    uncertainty_plot = denorm_uncertainty[:, plotted_idx]
    if len(uncertainty_plot) > n_plot:
        uncertainty_plot = uncertainty_plot[-n_plot:]
    plot_color_predicted = config.get("plot_color_predicted", "blue")
    plot_color_true = config.get("plot_color_true", "red")
    plot_color_uncertainty = config.get("plot_color_uncertainty", "green")
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates_plot, pred_plot, label="Predicted Price", color=plot_color_predicted, linewidth=2)
    plt.plot(test_dates_plot, true_plot, label="True Price", color=plot_color_true, linewidth=2)
    plt.fill_between(test_dates_plot, pred_plot - uncertainty_plot, pred_plot + uncertainty_plot,
                     color=plot_color_uncertainty, alpha=0.15, label="Uncertainty")
    if config.get("use_daily", False):
        plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} days)")
    else:
        plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} hours)")
    plt.xlabel("Close Time")
    plt.ylabel("EUR Price [USD]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        predictions_plot_file = config.get("predictions_plot_file", "predictions_plot.png")
        plt.savefig(predictions_plot_file, dpi=300)
        plt.close()
        print(f"Prediction plot saved to {predictions_plot_file}")
    except Exception as e:
        print(f"Failed to generate prediction plot: {e}")

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
        y_val = datasets["y_val"]
        val_dates = datasets.get("dates_val")
        if config["plugin"] in ["lstm", "cnn", "transformer"]:
            print("Creating sliding windows for CNN...")
            x_val, _, val_date_windows = create_sliding_windows_x(x_val, config['window_size'], stride=1, date_times=val_dates)
            val_dates = val_date_windows
            print(f"Sliding windows created: x_val: {x_val.shape}, y_val: {y_val.shape}")
            if x_val.ndim != 3:
                raise ValueError(f"For CNN and LSTM, x_val must be 3D. Found: {x_val.shape}.")
        print(f"Processed validation data: X shape: {x_val.shape}, Y shape: {y_val.shape}")
    except Exception as e:
        print(f"Failed to process validation data: {e}")
        sys.exit(1)

    print("Making predictions on validation data...")
    try:
        x_val_array = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()
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
        if not config.get("use_returns", False):
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                predictions = predictions * (close_max - close_min) + close_min
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                if "baseline_val" in datasets:
                    baseline = datasets["baseline_val"]
                    predictions = (predictions + baseline) * diff + close_min
                else:
                    print("Warning: Baseline validation values not found; cannot convert returns to predicted close values.")
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



def create_sliding_windows(x, y, window_size, time_horizon, stride=1, date_times=None):
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

    for i in range(0, len(x) - window_size - time_horizon + 1, stride):
        x_window = x[i:i + window_size]
        y_window = y[i + window_size:i + window_size + time_horizon]
        x_windowed.append(x_window)
        y_windowed.append(y_window)
        if date_times is not None:
            date_time_windows.append(date_times[i + window_size + time_horizon - 1])

    return np.array(x_windowed), np.array(y_windowed), date_time_windows


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
