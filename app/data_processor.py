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

##########################
# process_data.py
##########################

import pandas as pd
import numpy as np
from app.data_handler import load_csv

def process_data(config):
    """
    1) Loads x_train, y_train, x_val, y_val from CSV.
    2) Extracts the target column in y_*.
    3) Ensures all are numeric, have a DatetimeIndex, and are aligned.
    4) Shifts y by 1 step into the future.
    5) Builds multi-step columns in y.
    6) Slices x to the same final length as y.
    """

    # 1) LOAD ALL CSVs
    x_train = load_csv(
        config["x_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train")
    )
    y_train = load_csv(
        config["y_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train")
    )
    x_val = load_csv(
        config["x_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test")
    )
    y_val = load_csv(
        config["y_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test")
    )

    # 2) EXTRACT THE TARGET COLUMN
    target_col = config["target_column"]

    def extract_target(df, col):
        if isinstance(col, str):
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in DataFrame.")
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("`target_column` must be str or int")

    y_train = extract_target(y_train, target_col)
    y_val   = extract_target(y_val,   target_col)

    # 3) CONVERT EACH DF TO NUMERIC, REALLY REASSIGN TO DF
    #    (Previously was `df.apply(...).fillna(0)` without storing the result => bug!)
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_val   = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val   = y_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4) ENSURE DATETIME INDEX
    for name, df in zip(["x_train","y_train","x_val","y_val"], [x_train,y_train,x_val,y_val]):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex. Found: {type(df.index)}")

    # ALIGN x,y ON INTERSECTION
    def align_xy(x_df, y_df):
        common_idx = x_df.index.intersection(y_df.index)
        x_aligned = x_df.loc[common_idx].sort_index()
        y_aligned = y_df.loc[common_idx].sort_index()
        return x_aligned, y_aligned

    x_train, y_train = align_xy(x_train, y_train)
    x_val,   y_val   = align_xy(x_val,   y_val)

    # SHIFT y BY -1 => NEXT-STEP OFFSET
    y_train = y_train.shift(-1).dropna()
    y_val   = y_val.shift(-1).dropna()

    # BUILD MULTI-STEP COLUMNS
    time_horizon = config["time_horizon"]

    def create_multi_step(y_df, horizon):
        blocks = []
        idx_list = []
        for i in range(len(y_df) - horizon + 1):
            window = y_df.iloc[i : i + horizon].values.flatten()
            blocks.append(window)
            idx_list.append(y_df.index[i])
        return pd.DataFrame(blocks, index=idx_list)

    y_train_multi = create_multi_step(y_train, time_horizon)
    y_val_multi   = create_multi_step(y_val,   time_horizon)

    # SLICE X TO MATCH Y LENGTH
    def slice_to_y(x_df, y_df):
        comm_idx = x_df.index.intersection(y_df.index)
        x_final = x_df.loc[comm_idx].sort_index()
        y_final = y_df.loc[comm_idx].sort_index()
        return x_final, y_final

    x_train_final, y_train_final = slice_to_y(x_train, y_train_multi)
    x_val_final,   y_val_final   = slice_to_y(x_val,   y_val_multi)

    # FINAL CHECKS
    if len(x_train_final) != len(y_train_final):
        raise ValueError("Mismatch in train sets after multi-step")
    if len(x_val_final) != len(y_val_final):
        raise ValueError("Mismatch in val sets after multi-step")

    print("Processed datasets:")
    print(" x_train:", x_train_final.shape, " y_train:", y_train_final.shape)
    print(" x_val:  ", x_val_final.shape,   " y_val:  ", y_val_final.shape)

    return {
        "x_train": x_train_final,
        "y_train": y_train_final,
        "x_val":   x_val_final,
        "y_val":   y_val_final
    }

def run_prediction_pipeline(config, plugin):
    """
    1) Calls process_data once => x_train, y_train, x_val, y_val.
    2) Builds & trains the model on those arrays.
    3) Evaluates on the same arrays, printing training & validation MAE / R².
    """

    start_time = time.time()
    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    train_mae_list = []
    val_mae_list   = []
    train_r2_list  = []
    val_r2_list    = []

    # 1) Load data
    ds = process_data(config)
    x_train_df, y_train_df = ds["x_train"], ds["y_train"]
    x_val_df,   y_val_df   = ds["x_val"],   ds["y_val"]

    print("Training data shapes:", x_train_df.shape, y_train_df.shape)
    print("Validation data shapes:", x_val_df.shape, y_val_df.shape)

    x_train = x_train_df.to_numpy(dtype=np.float32)
    y_train = y_train_df.to_numpy(dtype=np.float32)
    x_val   = x_val_df.to_numpy(dtype=np.float32)
    y_val   = y_val_df.to_numpy(dtype=np.float32)

    time_horizon = config["time_horizon"]
    plugin.set_params(time_horizon=time_horizon)

    # Determine input shape (example for ANN)
    plugin_type = config["plugin"].lower()
    def get_input_shape(x):
        # e.g. for ANN => single integer
        return x.shape[1]

    input_shape = get_input_shape(x_train)
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    threshold_error = config["threshold_error"]

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start = time.time()

        try:
            plugin.build_model(input_shape=input_shape)

            print("Fitting model on training data...")
            plugin.train(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                threshold_error=threshold_error,
                x_val=x_val,
                y_val=y_val
            )

            print("Evaluating on training and validation data...")

            # 2) Predictions
            train_predictions = plugin.predict(x_train)
            val_predictions   = plugin.predict(x_val)

            # 2b) Ensure shape is (N, time_horizon)
            #     If the plugin outputs (N,1) or (N,) => mismatch => raise error
            if train_predictions.shape != y_train.shape:
                raise ValueError(
                    f"train_predictions shape {train_predictions.shape} "
                    f"!= y_train shape {y_train.shape} (time_horizon mismatch?)"
                )
            if val_predictions.shape != y_val.shape:
                raise ValueError(
                    f"val_predictions shape {val_predictions.shape} "
                    f"!= y_val shape {y_val.shape} (time_horizon mismatch?)"
                )

            # Print first 10 lines for debugging
            limit_tr = min(10, len(train_predictions))
            print("\nFirst 10 training predictions vs actual:")
            for i in range(limit_tr):
                print(f"Row {i} => pred={train_predictions[i]}, actual={y_train[i]}")

            limit_va = min(10, len(val_predictions))
            print("\nFirst 10 validation predictions vs actual:")
            for i in range(limit_va):
                print(f"Row {i} => pred={val_predictions[i]}, actual={y_val[i]}")

            # 3) External MAE & R²
            train_mae = float(plugin.calculate_mae(y_train, train_predictions))
            train_r2  = float(r2_score(y_train, train_predictions))
            print(f"Training MAE: {train_mae}")
            print(f"Training R²: {train_r2}")

            val_mae = float(plugin.calculate_mae(y_val, val_predictions))
            val_r2  = float(r2_score(y_val, val_predictions))
            print(f"Validation MAE: {val_mae}")
            print(f"Validation R²: {val_r2}")

            train_mae_list.append(train_mae)
            train_r2_list.append(train_r2)
            val_mae_list.append(val_mae)
            val_r2_list.append(val_r2)

            iteration_end = time.time()
            print(f"Iteration {iteration} completed in {iteration_end - iteration_start:.2f} seconds")

        except Exception as e:
            print(f"Iteration {iteration} failed with error: {e}")
            continue

    # 4) Aggregation
    results = {
        "Metric": ["Train MAE","Train R²","Val MAE","Val R²"],
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

    results_file = config.get("results_file","results.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")


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




