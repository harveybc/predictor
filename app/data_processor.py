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
from sklearn.metrics import r2_score
import contextlib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from plugin_loader import load_plugin

# Use tensorflow.keras instead of keras.
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import Huber


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
    """
    Creates sliding windows for input features.
    For each window, the DATE_TIME is taken from the *first* tick (the "base" tick).
    
    Args:
        data (np.ndarray or pd.DataFrame): Array of shape (n_samples, n_features).
        window_size (int): Number of ticks per window.
        stride (int): Step size between windows.
        date_times (pd.DatetimeIndex, optional): Date/time for each sample.
    
    Returns:
        tuple: (windows, base_dates) where windows is an array of shape 
               (n_windows, window_size, n_features) and base_dates is a list of dates 
               corresponding to the first tick of each window.
    """
    windows = []
    base_dates = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i: i + window_size])
        if date_times is not None:
            base_dates.append(date_times[i])
    if date_times is not None:
        return np.array(windows), base_dates
    else:
        return np.array(windows)


def create_multi_step(y_df, horizon, use_returns=False):
    """
    Creates multi-step targets for hourly data.
    For each row i, the target row contains the values from rows i+horizon to i+horizon+horizon-1.
    The returned DataFrame keeps the date from row i.
    
    Args:
        y_df (pd.DataFrame): DataFrame of target values.
        horizon (int): Number of future ticks to predict.
        use_returns (bool): If True, compute the difference between future and base values.
    
    Returns:
        pd.DataFrame: Multi-step targets (with same index as the base tick).
        If use_returns is True, also returns a DataFrame of baseline values.
    """
    blocks = []
    baselines = []
    # Only compute targets for rows that have enough future ticks.
    for i in range(len(y_df) - 2 * horizon + 1):
        base = y_df.iloc[i].values.flatten()
        future = y_df.iloc[i + horizon: i + horizon + horizon].values.flatten()
        if use_returns:
            window = list(future - base)
        else:
            window = list(future)
        blocks.append(window)
        if use_returns:
            baselines.append(base)
    df_targets = pd.DataFrame(blocks, index=y_df.index[:len(blocks)])
    if use_returns:
        df_baselines = pd.DataFrame(baselines, index=y_df.index[:len(baselines)])
        return df_targets, df_baselines
    else:
        return df_targets


def create_multi_step_daily(y_df, horizon, use_returns=False):
    """
    Creates multi-step targets for daily data.
    For each row i, the target row contains the values at rows i+24, i+48, ..., i+24*horizon.
    The index remains that of row i.
    
    Args:
        y_df (pd.DataFrame): DataFrame of target values.
        horizon (int): Number of future days to predict.
        use_returns (bool): If True, compute the differences (returns).
    
    Returns:
        pd.DataFrame: Multi-step targets with original dates.
        If use_returns is True, also returns a DataFrame of baseline values.
    """
    blocks = []
    baselines = []
    for i in range(len(y_df) - horizon * 24):
        base = y_df.iloc[i].values.flatten()
        window = []
        for d in range(1, horizon + 1):
            idx = i + 24 * d
            if idx >= len(y_df):
                break
            value = y_df.iloc[idx].values.flatten()
            if use_returns:
                diff = value - base
                window.append(diff[0])  # Assuming single target column
            else:
                window.append(value[0])
        if len(window) == horizon:
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
    Loads CSV files, trims data to a common date range, extracts and converts target columns,
    computes multi-step targets (hourly or daily), and (if using sliding windows) creates sliding
    windows for X and aligns Y accordingly.
    
    Returns:
        dict: Contains processed x_train, y_train, x_val, y_val, x_test, y_test,
              and their corresponding date arrays.
              If use_returns is True, also includes baseline_* arrays.
    """
    import pandas as pd

    # --- 1) Load CSV files ---
    x_train = load_csv(config["x_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    y_train = load_csv(config["y_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    x_val = load_csv(config["x_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_val"))
    y_val = load_csv(config["y_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_val"))
    x_test = load_csv(config["x_test_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))
    y_test = load_csv(config["y_test_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))

    # --- 1a) Trim to common date range if indices are datetime ---
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

    # --- Save original DATE_TIME indices ---
    train_dates_orig = x_train.index if isinstance(x_train.index, pd.DatetimeIndex) else None
    val_dates_orig = x_val.index if isinstance(x_val.index, pd.DatetimeIndex) else None
    test_dates_orig = x_test.index if isinstance(x_test.index, pd.DatetimeIndex) else None

    # --- 2) Extract target column ---
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

    # --- 3) Convert DataFrames to numeric ---
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_val = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val = y_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = y_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # --- 4) Compute Multi-Step Targets ---
    time_horizon = config["time_horizon"]
    if config.get("use_daily", False):
        # For daily mode, use raw data (or a rolling average if desired)
        y_train_ma = y_train
        y_val_ma = y_val    
        y_test_ma = y_test
        if config.get("use_returns", False):
            y_train_multi, baseline_train = create_multi_step_daily(y_train_ma, time_horizon, use_returns=True)
            y_val_multi, baseline_val = create_multi_step_daily(y_val_ma, time_horizon, use_returns=True)
            y_test_multi, baseline_test = create_multi_step_daily(y_test_ma, time_horizon, use_returns=True)
        else:
            y_train_multi = create_multi_step_daily(y_train_ma, time_horizon, use_returns=False)
            y_val_multi = create_multi_step_daily(y_val_ma, time_horizon, use_returns=False)
            y_test_multi = create_multi_step_daily(y_test_ma, time_horizon, use_returns=False)
        y_proc = y_train_ma  # For verification use raw daily data.
    else:
        if config.get("use_returns", False):
            y_train_multi, baseline_train = create_multi_step(y_train, time_horizon, use_returns=True)
            y_val_multi, baseline_val = create_multi_step(y_val, time_horizon, use_returns=True)
            y_test_multi, baseline_test = create_multi_step(y_test, time_horizon, use_returns=True)
        else:
            y_train_multi = create_multi_step(y_train, time_horizon, use_returns=False)
            y_val_multi = create_multi_step(y_val, time_horizon, use_returns=False)
            y_test_multi = create_multi_step(y_test, time_horizon, use_returns=False)
        y_proc = y_train

    # --- VERIFICATION BLOCK ---
    try:
        verif_index = 0  # choose the first row
        base_val = y_proc.iloc[verif_index].values[0]
        expected_values = []
        debug_details = []
        if config.get("use_daily", False):
            for d in range(1, time_horizon + 1):
                idx = verif_index + 24 * d
                if idx >= len(y_proc):
                    debug_details.append(f"[DEBUG] Day {d}: index {idx} out of bounds (len {len(y_proc)})")
                    continue
                future_val = y_proc.iloc[idx].values[0]
                diff_val = future_val - base_val if config.get("use_returns", False) else future_val
                expected_values.append(diff_val)
                debug_details.append(f"[DEBUG] Day {d}: index {idx}: future = {future_val:.8f}, base = {base_val:.8f}, diff = {diff_val:.8f}")
        else:
            for d in range(1, time_horizon + 1):
                idx = verif_index + d
                if idx >= len(y_proc):
                    debug_details.append(f"[DEBUG] Tick {d}: index {idx} out of bounds (len {len(y_proc)})")
                    continue
                future_val = y_proc.iloc[idx].values[0]
                diff_val = future_val - base_val if config.get("use_returns", False) else future_val
                expected_values.append(diff_val)
                debug_details.append(f"[DEBUG] Tick {d}: index {idx}: future = {future_val:.8f}, base = {base_val:.8f}, diff = {diff_val:.8f}")
        expected_row = np.array(expected_values)
        if isinstance(y_train_multi, pd.DataFrame):
            computed_row = y_train_multi.iloc[verif_index].values
        else:
            computed_row = y_train_multi[verif_index, :]
        print(f"[DEBUG] Verification index: {verif_index}")
        print(f"[DEBUG] Base row at original index {verif_index}: {y_proc.iloc[verif_index].values}")
        for detail in debug_details:
            print(detail)
        print(f"[DEBUG] Expected multi-step target row: {expected_row}")
        print(f"[DEBUG] Computed multi-step target row: {computed_row}")
        if not np.allclose(expected_row, computed_row, atol=1e-5):
            print("Verification check failed: multi-step target row does not match expected future values.")
            sys.exit(1)
        else:
            print("Verification check passed: multi-step target values are correctly shifted without altering dates.")
    except Exception as e:
        print(f"Verification check error: {e}")
        sys.exit(1)
    # --- End Verification Block ---

    # --- 5) Trim X to match length of Y ---
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

    # --- 6) Per-plugin Processing ---
    if config["plugin"] in ["lstm", "cnn", "transformer"]:
        print("Processing data with sliding windows...")
        x_train = x_train.to_numpy().astype(np.float32)
        x_val = x_val.to_numpy().astype(np.float32)
        x_test = x_test.to_numpy().astype(np.float32)
        window_size = config["window_size"]
        # Create sliding windows for X using base dates.
        x_train, train_dates = create_sliding_windows_x(x_train, window_size, stride=1, date_times=train_dates)
        x_val, val_dates = create_sliding_windows_x(x_val, window_size, stride=1, date_times=val_dates)
        x_test, test_dates = create_sliding_windows_x(x_test, window_size, stride=1, date_times=test_dates)
        # Align Y targets with sliding windows (discard the first window_size-1 rows).
        y_train_multi = y_train_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_val_multi = y_val_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_test_multi = y_test_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        if config.get("use_returns", False):
            baseline_train = baseline_train.iloc[window_size - 1:].to_numpy().astype(np.float32)
            baseline_val = baseline_val.iloc[window_size - 1:].to_numpy().astype(np.float32)
            baseline_test = baseline_test.iloc[window_size - 1:].to_numpy().astype(np.float32)
    else:
        print("Not using sliding windows; converting data to NumPy arrays.")
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
    print(" x_val:", x_val.shape, " y_val:", y_val_multi.shape)
    print(" x_test:", x_test.shape, " y_test:", y_test_multi.shape)

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


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

def run_prediction_pipeline(config, plugin):
    """
    Trains the model and saves predictions.
    
    IMPORTANT: For each input window, the DATE_TIME assigned is the date of the *first* tick 
    (the current tick) from which future predictions are made.
    """
    import time, numpy as np, pandas as pd, json, matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.model_selection import TimeSeriesSplit

    start_time = time.time()
    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

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

    # Convert any remaining DataFrames to NumPy arrays.
    for var in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        if isinstance(locals()[var], pd.DataFrame):
            locals()[var] = locals()[var].to_numpy().astype(np.float32)

    if config["plugin"] in ["lstm", "cnn", "transformer"]:
        if x_train.ndim != 3:
            raise ValueError(f"For CNN and LSTM, x_train must be 3D. Found: {x_train.shape}")
        print("Using sliding window data for CNN/LSTM.")
    plugin.set_params(time_horizon=time_horizon)
    tscv = TimeSeriesSplit(n_splits=5)

    # Training loop.
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iter_start = time.time()
        if config["plugin"] in ["lstm", "cnn", "transformer"]:
            plugin.build_model(input_shape=(window_size, x_train.shape[2]), x_train=x_train)
        elif config["plugin"] in ["transformer", "transformer_mmd"]:
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train)
        else:
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)

        history, train_preds, train_unc, val_preds, val_unc = plugin.train(
            x_train, y_train, epochs=epochs, batch_size=batch_size,
            threshold_error=threshold_error, x_val=x_val, y_val=y_val, config=config
        )
        if config.get("use_returns", False):
            train_r2 = r2_score((baseline_train[:, -1] + y_train[:, -1]).flatten(),
                                (baseline_train[:, -1] + train_preds[:, -1]).flatten())
            val_r2 = r2_score((baseline_val[:, -1] + y_val[:, -1]).flatten(),
                              (baseline_val[:, -1] + val_preds[:, -1]).flatten())
        else:
            train_r2 = r2_score(y_train[:, -1], train_preds[:, -1])
            val_r2 = r2_score(y_val[:, -1], val_preds[:, -1])

        n_train = train_preds.shape[0]
        n_val = val_preds.shape[0]
        train_mae = np.mean(np.abs(train_preds[:, -1] - y_train[:n_train, -1]))
        val_mae = np.mean(np.abs(val_preds[:, -1] - y_val[:n_val, -1]))

        # Save loss plot.
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f"Model Loss ({config['plugin'].upper()}) - Iteration {iteration}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Val"])
        plt.savefig(config['loss_plot_file'])
        plt.close()
        print(f"Loss plot saved to {config['loss_plot_file']}")

        # Test dataset evaluation.
        print("\nEvaluating on test dataset...")
        mc_samples = config.get("mc_samples", 100)
        test_predictions, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
        n_test = test_predictions.shape[0]
        if config.get("use_returns", False):
            test_r2 = r2_score((baseline_test[:, -1] + y_test[:n_test, -1]).flatten(),
                               (baseline_test[:, -1] + test_predictions[:, -1]).flatten())
        else:
            test_r2 = r2_score(y_test[:n_test, -1], test_predictions[:, -1])
        test_mae = np.mean(np.abs(test_predictions[:, -1] - y_test[:n_test, -1]))
        # (Additional metrics can be computed here.)
        print(f"Test MAE: {test_mae}, Test R²: {test_r2}")

        print(f"Iteration {iteration} completed in {time.time() - iter_start:.2f} seconds")

    # --- Denormalize final test predictions if needed ---
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
                test_predictions = (test_predictions + baseline_test) * diff + close_min
            else:
                print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                test_predictions = test_predictions * (close_max - close_min) + close_min

    # --- Save predictions ---
    final_test_file = config.get("output_file", "test_predictions.csv")
    predictions_df = pd.DataFrame(
        test_predictions,
        columns=[f"Prediction_{i+1}" for i in range(test_predictions.shape[1])]
    )
    # IMPORTANT: The DATE_TIME for predictions is taken from the sliding window base dates.
    if test_dates is not None:
        predictions_df['DATE_TIME'] = pd.Series(test_dates[:len(predictions_df)])
    else:
        predictions_df['DATE_TIME'] = pd.NaT
    # Reorder so DATE_TIME is first.
    cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
    predictions_df = predictions_df[cols]
    write_csv(file_path=final_test_file, data=predictions_df, include_date=False, headers=config.get('headers', True))
    print(f"Final predictions saved to {final_test_file}")

    # --- Save uncertainty estimates ---
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
            denorm_uncertainty,
            columns=[f"Uncertainty_{i+1}" for i in range(denorm_uncertainty.shape[1])]
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

    # --- Plot predictions ---
    plotted_horizon = config.get("plotted_horizon", 6)
    plotted_idx = plotted_horizon - 1
    if plotted_idx >= test_predictions.shape[1]:
        raise ValueError(f"Plotted horizon index {plotted_idx} is out of bounds for predictions shape {test_predictions.shape}")
    pred_plot = test_predictions[:, plotted_idx]
    n_plot = config.get("plot_points", 1575)
    if len(pred_plot) > n_plot:
        pred_plot = pred_plot[-n_plot:]
        test_dates_plot = test_dates[-n_plot:] if test_dates is not None else np.arange(len(pred_plot))
    else:
        test_dates_plot = test_dates if test_dates is not None else np.arange(len(pred_plot))
    # For plotting, we denormalize the baseline if using returns.
    if config.get("use_returns", False):
        baseline_plot = baseline_test[:, 0]
    else:
        baseline_plot = None  # Not used in non-returns mode.
    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_min = norm_json["CLOSE"]["min"]
            close_max = norm_json["CLOSE"]["max"]
            diff = close_max - close_min
            true_plot = (baseline_plot * diff + close_min) if baseline_plot is not None else None
        else:
            true_plot = None
    else:
        true_plot = None

    plot_color_predicted = config.get("plot_color_predicted", "blue")
    plot_color_true = config.get("plot_color_true", "red")
    plot_color_uncertainty = config.get("plot_color_uncertainty", "green")
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates_plot, pred_plot, label="Predicted Price", color=plot_color_predicted, linewidth=2)
    if true_plot is not None:
        plt.plot(test_dates_plot, true_plot, label="True Price", color=plot_color_true, linewidth=2)
    plt.fill_between(test_dates_plot, pred_plot - denorm_uncertainty[:, plotted_idx],
                     pred_plot + denorm_uncertainty[:, plotted_idx],
                     color=plot_color_uncertainty, alpha=0.15, label="Uncertainty")
    plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} {'days' if config.get('use_daily', False) else 'hours'})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    predictions_plot_file = config.get("predictions_plot_file", "predictions_plot.png")
    plt.savefig(predictions_plot_file, dpi=300)
    plt.close()
    print(f"Prediction plot saved to {predictions_plot_file}")

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")


# =============================================================================
# MODEL LOADING & EVALUATION (Validation)
# =============================================================================

def load_and_evaluate_model(config, plugin):
    """
    Loads a pre-trained model and evaluates it on the validation data.
    The final predictions CSV uses the DATE_TIME from the input window’s base tick.
    """
    import sys, numpy as np, pandas as pd, json
    from tensorflow.keras.models import load_model

    print(f"Loading pre-trained model from {config['load_model']}...")
    try:
        custom_objects = {"combined_loss": combined_loss, "mmd": mmd_metric, "huber": huber_metric}
        plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model: {e}")
        sys.exit(1)

    print("Loading and processing validation data for evaluation...")
    try:
        datasets = process_data(config)
        x_val = datasets["x_val"]
        y_val = datasets["y_val"]
        val_dates = datasets.get("dates_val")
        if config["plugin"] in ["lstm", "cnn", "transformer"]:
            print("Creating sliding windows for validation data...")
            x_val, base_dates = create_sliding_windows_x(x_val, config['window_size'], stride=1, date_times=val_dates)
            # Use the base dates for predictions.
            val_dates = base_dates
            print(f"Sliding windows created: x_val: {x_val.shape}, y_val: {y_val.shape}")
            if x_val.ndim != 3:
                raise ValueError(f"For CNN/LSTM, x_val must be 3D. Found: {x_val.shape}")
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
                    print("Warning: Baseline validation values not found; cannot convert returns.")
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
        print("Warning: DATE_TIME not captured for validation predictions.")
    cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
    predictions_df = predictions_df[cols]
    evaluate_filename = config['output_file']
    try:
        write_csv(file_path=evaluate_filename, data=predictions_df,
                  include_date=False, headers=config.get('headers', True))
        print(f"Validation predictions saved to {evaluate_filename}")
    except Exception as e:
        print(f"Failed to save validation predictions: {e}")
        sys.exit(1)

# End of code.



def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
    windows = []
    dt_windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i: i + window_size])
        if date_times is not None:
            # Use the date corresponding to the first element in the window
            dt_windows.append(date_times[i])
    if date_times is not None:
        return np.array(windows), dt_windows
    else:
        return np.array(windows)



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
