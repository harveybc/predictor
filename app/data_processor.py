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


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def process_data(config):
    """
    From scratch:
      1. Load (x_train, y_train) and (x_val, y_val), each with DatetimeIndex.
      2. Align x,y on their intersection of timestamps.
      3. Shift y by -1 for next-step forecasting.
      4. Create multi-step targets (time_horizon columns) from y.
      5. Trim x to match the new y length (no extra offsets).
      6. Repeat for training and validation.

    Returns:
        {
            "x_train": DataFrame,
            "y_train": DataFrame,
            "x_val":   DataFrame,
            "y_val":   DataFrame
        }
    """
    print("Loading training and validation CSVs...")

    # 1) Load raw
    x_train = load_csv(config["x_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    y_train = load_csv(config["y_train_file"], headers=config["headers"], max_rows=config.get("max_steps_train"))
    x_val   = load_csv(config["x_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))
    y_val   = load_csv(config["y_validation_file"], headers=config["headers"], max_rows=config.get("max_steps_test"))

    # 2) Extract target column if needed
    target_col = config["target_column"]
    def extract_target(df, col):
        if isinstance(col, str):
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("target_column must be str or int")

    y_train = extract_target(y_train, target_col)
    y_val   = extract_target(y_val,   target_col)

    # Convert to numeric
    for df in [x_train, y_train, x_val, y_val]:
        df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 3) Ensure all have DatetimeIndex
    for name, df in zip(["x_train","y_train","x_val","y_val"], [x_train,y_train,x_val,y_val]):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have DatetimeIndex, found {type(df.index)}.")

    # 4) Align x,y on intersection
    def align_xy(xdf, ydf):
        common_idx = xdf.index.intersection(ydf.index)
        xdf = xdf.loc[common_idx].sort_index()
        ydf = ydf.loc[common_idx].sort_index()
        return xdf, ydf

    x_train, y_train = align_xy(x_train, y_train)
    x_val,   y_val   = align_xy(x_val,   y_val)

    # 5) SHIFT y by -1 => next-step offset
    y_train = y_train.shift(-1)
    y_val   = y_val.shift(-1)

    # Drop last row if it became NaN
    y_train.dropna(inplace=True)
    y_val.dropna(inplace=True)

    # 6) Create multi-step columns in y
    time_horizon = config["time_horizon"]

    def create_multi_step(y_df, horizon):
        # For row i => y[i.. i+horizon-1], flatten into 1D
        y_multi = []
        idx = []
        for i in range(len(y_df) - horizon + 1):
            block = y_df.iloc[i : i + horizon].values.flatten()
            y_multi.append(block)
            idx.append(y_df.index[i])  # keep i-th timestamp as the row index
        return pd.DataFrame(y_multi, index=idx)

    y_train_multi = create_multi_step(y_train, time_horizon)
    y_val_multi   = create_multi_step(y_val,   time_horizon)

    # 7) Trim x so it matches new y length
    def trim_x(x_df, y_df):
        # we keep top len(y_df) rows of x
        max_len = len(y_df)
        x_df = x_df.iloc[:max_len]
        # also align on index if needed
        common_idx = x_df.index.intersection(y_df.index)
        x_df = x_df.loc[common_idx].sort_index()
        y_df = y_df.loc[common_idx].sort_index()
        return x_df, y_df

    x_train, y_train_multi = trim_x(x_train, y_train_multi)
    x_val,   y_val_multi   = trim_x(x_val,   y_val_multi)

    # Final shape checks
    if len(x_train) != len(y_train_multi):
        raise ValueError(f"Train mismatch: x={len(x_train)}, y={len(y_train_multi)}")
    if len(x_val) != len(y_val_multi):
        raise ValueError(f"Val mismatch: x={len(x_val)}, y={len(y_val_multi)}")

    print("process_data => Final shapes:")
    print(" x_train:", x_train.shape, " y_train:", y_train_multi.shape)
    print(" x_val:  ", x_val.shape,   " y_val:  ", y_val_multi.shape)

    return {
        "x_train": x_train,
        "y_train": y_train_multi,
        "x_val":   x_val,
        "y_val":   y_val_multi
    }


def run_prediction_pipeline(config, plugin):
    """
    From scratch:
      1. Load final shapes from process_data.
      2. Convert to numpy, feed EXACT arrays to plugin.train(...).
      3. Evaluate both with Keras's built-in model.evaluate(...) and plugin's external calculate_mae(...).
      4. Return matching training/validation metrics to ensure no mismatch.
    """
    start = time.time()

    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    # For storing iteration metrics
    train_mae_list = []
    val_mae_list   = []
    train_r2_list  = []
    val_r2_list    = []

    # 1) Load final data
    from process_data import process_data
    ds = process_data(config)
    x_train, y_train = ds["x_train"], ds["y_train"]
    x_val,   y_val   = ds["x_val"],   ds["y_val"]

    print("post-process_data => x_train:", x_train.shape, "y_train:", y_train.shape)
    print("                   x_val:   ", x_val.shape,   "y_val:",   y_val.shape)

    # Convert to numpy
    x_train_np = x_train.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32)
    x_val_np   = x_val.to_numpy().astype(np.float32)
    y_val_np   = y_val.to_numpy().astype(np.float32)

    time_horizon = config["time_horizon"]
    window_size  = config.get("window_size")
    plugin.set_params(time_horizon=time_horizon)

    for it in range(1, iterations + 1):
        print(f"\n=== Iteration {it}/{iterations} ===")
        iteration_start = time.time()

        try:
            # 2) Build the model
            if config["plugin"] == "cnn":
                # create sliding windows if needed
                # or assume we do that in process_data. 
                # For simplicity, let's just say we have final arrays already.
                # If you do need sliding windows, do them here (but EXACT the same for training & eval).
                plugin.build_model(input_shape=(window_size, x_train_np.shape[2]))
            elif config["plugin"].lower() == "ann":
                plugin.build_model(input_shape=x_train_np.shape[1])
            else:
                # LSTM/others => typically input_shape=(features,)
                plugin.build_model(input_shape=(x_train_np.shape[1],))

            # 3) Train => pass EXACT arrays
            plugin.train(
                x_train_np,
                y_train_np,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                threshold_error=config["threshold_error"],
                x_val=x_val_np,
                y_val=y_val_np,
            )

            # 4) Evaluate with the EXACT same arrays
            print("Running Keras model.evaluate on training data for direct comparison...")
            keras_train_eval = plugin.model.evaluate(x_train_np, y_train_np, verbose=0)
            # Usually returns [loss, mse, mae], but depends on plugin.model.compile(...)
            # We'll assume it's [loss, mse, mae]
            if len(keras_train_eval) == 3:
                train_loss, train_mse, train_mae_keras = keras_train_eval
            else:
                # fallback if user changed metrics
                train_loss = keras_train_eval[0]
                train_mae_keras = keras_train_eval[-1]

            print(f"Keras train: loss={train_loss}, mae_keras={train_mae_keras}")

            print("Running Keras model.evaluate on validation data for direct comparison...")
            keras_val_eval = plugin.model.evaluate(x_val_np, y_val_np, verbose=0)
            if len(keras_val_eval) == 3:
                val_loss, val_mse, val_mae_keras = keras_val_eval
            else:
                val_loss = keras_val_eval[0]
                val_mae_keras = keras_val_eval[-1]

            print(f"Keras val:   loss={val_loss}, mae_keras={val_mae_keras}")

            # 5) Evaluate externally
            #   a) Predictions with plugin.predict(...)
            print("Predicting with plugin on full train & val sets to measure external MAE/R2...")
            train_preds = plugin.predict(x_train_np)
            val_preds   = plugin.predict(x_val_np)

            #   b) External MAE
            ext_train_mae = plugin.calculate_mae(y_train_np, train_preds)
            ext_val_mae   = plugin.calculate_mae(y_val_np,   val_preds)
            #   c) R^2
            ext_train_r2 = r2_score(y_train_np, train_preds)
            ext_val_r2   = r2_score(y_val_np,   val_preds)

            print(f"External train MAE={ext_train_mae:.4f}, R²={ext_train_r2:.4f}")
            print(f"External  val  MAE={ext_val_mae:.4f}, R²={ext_val_r2:.4f}")

            train_mae_list.append(ext_train_mae)
            val_mae_list.append(ext_val_mae)
            train_r2_list.append(ext_train_r2)
            val_r2_list.append(ext_val_r2)

            iteration_end = time.time()
            print(f"Iteration {it} complete in {iteration_end - iteration_start:.2f} sec.")

        except Exception as e:
            print(f"Iteration {it} failed: {e}")
            continue

    # Aggregation
    results = {
        "Metric": ["Train MAE", "Train R²", "Val MAE", "Val R²"],
        "Average": [
            np.mean(train_mae_list) if train_mae_list else None,
            np.mean(train_r2_list)  if train_r2_list  else None,
            np.mean(val_mae_list)   if val_mae_list   else None,
            np.mean(val_r2_list)    if val_r2_list    else None,
        ],
        "StdDev": [
            np.std(train_mae_list) if train_mae_list else None,
            np.std(train_r2_list)  if train_r2_list  else None,
            np.std(val_mae_list)   if val_mae_list   else None,
            np.std(val_r2_list)    if val_r2_list    else None,
        ]
    }

    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(config.get("results_file", "results.csv"), index=False)
    print("Saved final results to", config.get("results_file", "results.csv"))

    end = time.time()
    print(f"Total pipeline time: {end - start:.2f} sec.")

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

    return np.array(x_windowed), np.array(y_windowed).reshape(-1, time_horizon), date_time_windows


def load_and_evaluate_model(config, plugin):
    """
    Loads a pre-trained model and evaluates it on the validation data.

    This function performs the following steps:
    1. Loads the specified pre-trained model.
    2. Loads and processes the validation data.
    3. Makes predictions using the loaded model.
    4. Saves the predictions to a CSV file for evaluation, including the DATE_TIME column.

    Args:
        config (dict): Configuration dictionary containing parameters for model evaluation.
            Expected keys include:
                - 'load_model' (str): Path to the pre-trained model file.
                - 'x_validation_file' (str): Path to the validation features CSV file.
                - 'y_validation_file' (str): Path to the validation targets CSV file.
                - 'target_column' (str or int): Column name or index to be used as the target.
                - 'headers' (bool): Indicates if CSV files contain headers.
                - 'force_date' (bool): Determines if date should be included in the output CSV.
                - 'evaluate_file' (str): Path to save the evaluation predictions CSV file.
                - 'max_steps_val' (int, optional): Maximum number of rows to read for validation data.

        plugin (Plugin): The ANN predictor plugin to be used for evaluation.

    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        Exception: Propagates any exception that occurs during model loading or data processing.
    """
    # Load the pre-trained model
    print(f"Loading pre-trained model from {config['load_model']}...")
    try:
        plugin.load(config['load_model'])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model from {config['load_model']}: {e}")
        sys.exit(1)

    # Load and process validation data with row limit
    print("Loading and processing validation data for evaluation...")
    try:
        # Assuming process_data can be reused for validation by specifying validation files
        x_val, y_val = process_data(config)
        print(f"Processed validation data: X shape: {x_val.shape}, Y shape: {y_val.shape}")
    except Exception as e:
        print(f"Failed to process validation data: {e}")
        sys.exit(1)

    # Predict using the loaded model
    print("Making predictions on validation data...")
    try:
        predictions = plugin.predict(x_val.to_numpy())
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        sys.exit(1)

    # Convert predictions to DataFrame
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    else:
        num_steps = predictions.shape[1]
        pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
        predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    # Add DATE_TIME column from y_val
    if isinstance(y_val.index, pd.DatetimeIndex):
        predictions_df['DATE_TIME'] = y_val.index[:len(predictions_df)]
    else:
        predictions_df['DATE_TIME'] = pd.NaT  # Assign Not-a-Time if index is not datetime
        print("Warning: DATE_TIME for validation predictions not captured.")

    # Rearrange columns to have DATE_TIME first
    cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
    predictions_df = predictions_df[cols]

    # Save predictions to CSV for evaluation
    evaluate_filename = config['evaluate_file']
    try:
        write_csv(
            file_path=evaluate_filename,
            data=predictions_df,
            include_date=False,  # DATE_TIME is already included
            headers=config.get('headers', True)
        )
        print(f"Validation predictions with DATE_TIME saved to {evaluate_filename}")
    except Exception as e:
        print(f"Failed to save validation predictions to {evaluate_filename}: {e}")
        sys.exit(1)



def create_multi_step_targets(df, time_horizon):
    """
    Creates multi-step targets for time-series prediction.

    Args:
        df (pd.DataFrame): Target data as a DataFrame.
        time_horizon (int): Number of future steps to predict.

    Returns:
        pd.DataFrame: Multi-step targets aligned with the input data.
    """
    y_multi_step = []
    for i in range(len(df) - time_horizon + 1):
        y_multi_step.append(df.iloc[i:i + time_horizon].values.flatten())

    # Create DataFrame with aligned indices
    y_multi_step_df = pd.DataFrame(y_multi_step, index=df.index[:len(y_multi_step)])
    return y_multi_step_df




