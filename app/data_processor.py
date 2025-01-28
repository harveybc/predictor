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
import math


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

##############################
# process_data.py
##############################

import pandas as pd
import numpy as np
from app.data_handler import load_csv

def process_data(config):
    """
    Simplified process_data function assuming x and y datasets are already aligned row by row.
    """

    # 1) LOAD CSVs
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
                raise ValueError(f"Target column '{col}' not found.")
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("`target_column` must be str or int.")

    y_train = extract_target(y_train, target_col)
    y_val = extract_target(y_val, target_col)

    # 3) CONVERT EACH DF TO NUMERIC, REASSIGN THE RESULT TO AVOID BUGS
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_val = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val = y_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4) MULTI-STEP COLUMNS
    time_horizon = config["time_horizon"]
    def create_multi_step(y_df, horizon):
        """
        Create multi-step targets for time-series prediction.

        Args:
            y_df (pd.DataFrame): Target data as a DataFrame.
            horizon (int): Number of future steps to predict.

        Returns:
            pd.DataFrame: Multi-step targets aligned with the input data.
        """
        blocks = []
        for i in range(len(y_df) - horizon):
            # Collect the next `horizon` ticks starting from the *next* row
            window = y_df.iloc[i + 1 : i + 1 + horizon].values.flatten()
            blocks.append(window)
        # Align index to the input data (exclude the last `horizon` rows)
        return pd.DataFrame(blocks, index=y_df.index[:-horizon])


    y_train_multi = create_multi_step(y_train, time_horizon)
    y_val_multi = create_multi_step(y_val, time_horizon)

    # 5) TRIM x TO MATCH THE LENGTH OF y
    min_len_train = min(len(x_train), len(y_train_multi))
    x_train = x_train.iloc[:min_len_train]
    y_train_multi = y_train_multi.iloc[:min_len_train]

    min_len_val = min(len(x_val), len(y_val_multi))
    x_val = x_val.iloc[:min_len_val]
    y_val_multi = y_val_multi.iloc[:min_len_val]

    print("Processed datasets:")
    print(" x_train:", x_train.shape, " y_train:", y_train_multi.shape)
    print(" x_val:  ", x_val.shape, " y_val:  ", y_val_multi.shape)

    print("x_train index:", x_train.index)
    print("y_train index:", y_train_multi.index)
    assert len(x_train) == len(y_train_multi), "x_train and y_train are misaligned!"
    assert len(x_val) == len(y_val_multi), "x_val and y_val are misaligned!"

    return {
        "x_train": x_train,
        "y_train": y_train_multi,
        "x_val": x_val,
        "y_val": y_val_multi
    }




def run_prediction_pipeline(config, plugin):
    start_time = time.time()
    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    train_mae_list = []
    val_mae_list   = []

    # 1) Process data ONCE
    ds = process_data(config)  # now numeric conversion is correct
    x_train_df, y_train_df = ds["x_train"], ds["y_train"]
    x_val_df,   y_val_df   = ds["x_val"],   ds["y_val"]

    print("Training data shapes:", x_train_df.shape, y_train_df.shape)
    print("Validation data shapes:", x_val_df.shape, y_val_df.shape)

    # Convert to numpy
    x_train = x_train_df.to_numpy(dtype=np.float32)
    y_train = y_train_df.to_numpy(dtype=np.float32)
    x_val   = x_val_df.to_numpy(dtype=np.float32)
    y_val   = y_val_df.to_numpy(dtype=np.float32)

    # 2) Build & train the model
    plugin.set_params(time_horizon=config["time_horizon"])
    input_shape = x_train.shape[1]
    
    for iteration in range(1, iterations+1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        try:
            plugin.build_model(input_shape=input_shape)

            plugin.train(
                x_train, y_train,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                threshold_error=config["threshold_error"],
                x_val=x_val,
                y_val=y_val
            )

            
            # 3) Evaluate Training and Validation Metrics
            print("\n=== Evaluation Results ===")

            # Evaluate on training data
            train_eval_results = plugin.model.evaluate(x_train, y_train, batch_size=config["batch_size"], verbose=0)
            train_loss, train_mse, train_mae = train_eval_results  # Assuming loss, mse, mae order in metrics

            # Calculate R² for training
            train_r2 = r2_score(y_train.flatten(), plugin.model.predict(x_train).flatten())

            print(f"[TRAIN] Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")

            # Evaluate on validation data
            val_eval_results = plugin.model.evaluate(x_val, y_val, batch_size=config["batch_size"], verbose=0)
            val_loss, val_mse, val_mae = val_eval_results  # Assuming loss, mse, mae order in metrics

            # Calculate R² for validation
            val_r2 = r2_score(y_val.flatten(), plugin.model.predict(x_val).flatten())

            print(f"[VAL]   Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")

            # Append results to lists for overall reporting
            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)


        except Exception as e:
            print(f"Iteration {iteration} failed => {e}")
            continue

    # Summaries
    print("Average train MAE:", np.mean(train_mae_list))
    print("Average val MAE:", np.mean(val_mae_list))

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


def replicate_final_epoch_mae(model, x_np, y_np, batch_size):
    """
    Replicate exactly how Keras logs 'mae' in the last epoch:
      - same batch boundaries
      - no shuffle
      - training=True forward pass
      - average of batch-level average errors
    """
    num_samples = x_np.shape[0]
    time_horizon = y_np.shape[1]
    steps = math.ceil(num_samples / batch_size)

    batch_maes = []
    idx = 0
    for step_i in range(steps):
        start = step_i * batch_size
        end = min((step_i+1)*batch_size, num_samples)
        x_batch = x_np[start:end]
        y_batch = y_np[start:end]

        # forward pass in training mode => BN uses ephemeral training stats
        preds = model(x_batch, training=True)

        # mean absolute error for this batch
        batch_mae = tf.reduce_mean(tf.abs(preds - y_batch))
        batch_maes.append(float(batch_mae))

    # Keras logs the average of the batch-level MAEs
    return float(np.mean(batch_maes))