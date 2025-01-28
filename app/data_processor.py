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


def process_data(config):
    """
    Loads and processes training and validation datasets, ensuring correct alignment
    and construction of the training signal for multi-step prediction.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - 'x_train_file', 'y_train_file': Paths to training datasets.
            - 'x_validation_file', 'y_validation_file': Paths to validation datasets.
            - 'target_column': Target column name or index.
            - 'time_horizon': Number of future steps to predict.
            - 'input_offset': Offset for input trimming.
            - 'headers': Whether the CSV files have headers.
            - 'max_steps_train', 'max_steps_test': Maximum rows to load for training and validation.
            - 'plugin': Type of plugin ("cnn", "lstm", etc.).

    Returns:
        dict: Processed datasets with keys 'x_train', 'y_train', 'x_val', 'y_val'.
    """
    datasets = {}

    # Load datasets
    print("Loading training and validation datasets...")
    x_train = load_csv(config['x_train_file'], headers=config['headers'], max_rows=config.get('max_steps_train'))
    y_train = load_csv(config['y_train_file'], headers=config['headers'], max_rows=config.get('max_steps_train'))
    x_val = load_csv(config['x_validation_file'], headers=config['headers'], max_rows=config.get('max_steps_test'))
    y_val = load_csv(config['y_validation_file'], headers=config['headers'], max_rows=config.get('max_steps_test'))

    # Extract target column
    target_column = config['target_column']

    def extract_target(df, col):
        if isinstance(col, str):
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found.")
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("`target_column` must be str or int.")

    y_train = extract_target(y_train, target_column)
    y_val = extract_target(y_val, target_column)

    # Convert to numeric and fill NAs
    for name, df in zip(['x_train', 'y_train', 'x_val', 'y_val'], [x_train, y_train, x_val, y_val]):
        datasets[name] = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Ensure indices are datetime
    for key in ['x_train', 'y_train', 'x_val', 'y_val']:
        if not isinstance(datasets[key].index, pd.DatetimeIndex):
            raise ValueError(f"Dataset '{key}' must have a DatetimeIndex.")

    # Align datasets
    for prefix in ['train', 'val']:
        common_index = datasets[f'x_{prefix}'].index.intersection(datasets[f'y_{prefix}'].index)
        datasets[f'x_{prefix}'] = datasets[f'x_{prefix}'].loc[common_index]
        datasets[f'y_{prefix}'] = datasets[f'y_{prefix}'].loc[common_index]

    # Shift y datasets to align with future predictions
    for prefix in ['train', 'val']:
        datasets[f'y_{prefix}'] = datasets[f'y_{prefix}'].shift(-1).ffill()

    # Trim datasets based on time_horizon and input_offset
    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    total_offset = time_horizon + input_offset

    def trim(x, y):
        return x.iloc[:-time_horizon], y.iloc[total_offset:]

    datasets['x_train'], datasets['y_train'] = trim(datasets['x_train'], datasets['y_train'])
    datasets['x_val'], datasets['y_val'] = trim(datasets['x_val'], datasets['y_val'])

    # Ensure target datasets have the correct shape for multi-step prediction
    def create_multi_step_targets(y, horizon):
        y_multi = []
        for i in range(len(y) - horizon + 1):
            y_multi.append(y.iloc[i:i + horizon].values.flatten())
        return pd.DataFrame(y_multi, index=y.index[:len(y_multi)])

    datasets['y_train'] = create_multi_step_targets(datasets['y_train'], time_horizon)
    datasets['y_val'] = create_multi_step_targets(datasets['y_val'], time_horizon)

    # Adjust x datasets to match y lengths
    for prefix in ['train', 'val']:
        datasets[f'x_{prefix}'] = datasets[f'x_{prefix}'].iloc[:len(datasets[f'y_{prefix}'])]

    # Final shape validation
    for prefix in ['train', 'val']:
        if len(datasets[f'x_{prefix}']) != len(datasets[f'y_{prefix}']):
            raise ValueError(f"Length mismatch: x_{prefix} ({len(datasets[f'x_{prefix}'])}) != y_{prefix} ({len(datasets[f'y_{prefix}'])})")

    print("Processed datasets:")
    print(f"  x_train: {datasets['x_train'].shape}, y_train: {datasets['y_train'].shape}")
    print(f"  x_val:   {datasets['x_val'].shape}, y_val:   {datasets['y_val'].shape}")

    return datasets
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



def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline using both training and validation datasets.
    Iteratively trains and evaluates the model, while saving metrics and predictions.

    Args:
        config (dict): Configuration dictionary containing parameters for training and evaluation.
        plugin (object): Model plugin that implements train, predict, and evaluate methods.
    """
    start_time = time.time()

    iterations = config.get("iterations", 1)
    if iterations <= 0:
        raise ValueError("`iterations` must be a positive integer.")

    print(f"Number of iterations: {iterations}")

    # Lists to store metrics for all iterations
    training_mae_list = []
    training_r2_list = []
    validation_mae_list = []
    validation_r2_list = []

    # Load all datasets
    print("Loading and processing datasets...")
    datasets = process_data(config)
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")

    # Basic config checks
    time_horizon = config.get("time_horizon")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    threshold_error = config.get("threshold_error", None)

    # Convert DataFrame to numpy
    x_train = x_train.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    x_val = x_val.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)

    # If plugin is CNN, create sliding windows
    if config["plugin"] == "cnn":
        if "window_size" not in config:
            raise ValueError("`window_size` must be defined in the configuration for CNN plugin.")
        window_size = config["window_size"]

        print("Creating sliding windows for CNN...")
        x_train, y_train, _ = create_sliding_windows(
            x_train, y_train, window_size, time_horizon, stride=1
        )
        x_val, y_val, _ = create_sliding_windows(
            x_val, y_val, window_size, time_horizon, stride=1
        )
        print("Sliding windows created:")
        print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  x_val:   {x_val.shape}, y_val: {y_val.shape}")

        # CNN expects (window_size, features) per sample
        input_shape = (window_size, x_train.shape[2])

    elif config["plugin"].lower() == "ann":
        # The ANN plugin expects a single integer for input_shape
        # If data is 1D, reshape it to (N,1)
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        if len(x_val.shape) == 1:
            x_val = x_val.reshape(-1, 1)
        input_shape = x_train.shape[1]

    else:
        # For LSTM, Transformers, or others: typically pass a tuple (features,)
        # If data is 1D, reshape it to (N,1)
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        if len(x_val.shape) == 1:
            x_val = x_val.reshape(-1, 1)

        input_shape = (x_train.shape[1],)

    # Pass time_horizon to the plugin (if applicable)
    plugin.set_params(time_horizon=time_horizon)

    # Iterative training and evaluation
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            # Build the model
            plugin.build_model(input_shape=input_shape)

            # Train the model
            plugin.train(
                x_train=x_train,
                y_train=y_train,
                epochs=epochs,
                batch_size=batch_size,
                threshold_error=threshold_error,
                x_val=x_val,
                y_val=y_val,
            )

            print("Evaluating trained model on training and validation data...")

            # Get predictions
            train_predictions = plugin.predict(x_train)
            val_predictions = plugin.predict(x_val)

            # Ensure predictions align with targets
            if train_predictions.shape != y_train.shape:
                raise ValueError(
                    f"Training predictions shape mismatch: {train_predictions.shape} vs {y_train.shape}"
                )
            if val_predictions.shape != y_val.shape:
                raise ValueError(
                    f"Validation predictions shape mismatch: {val_predictions.shape} vs {y_val.shape}"
                )

            # Compute metrics
            train_mae = float(plugin.calculate_mae(y_train, train_predictions))
            train_r2 = float(r2_score(y_train, train_predictions))

            val_mae = float(plugin.calculate_mae(y_val, val_predictions))
            val_r2 = float(r2_score(y_val, val_predictions))

            print(f"Training MAE: {train_mae}, R²: {train_r2}")
            print(f"Validation MAE: {val_mae}, R²: {val_r2}")

            # Store metrics
            training_mae_list.append(train_mae)
            training_r2_list.append(train_r2)
            validation_mae_list.append(val_mae)
            validation_r2_list.append(val_r2)

            iteration_end_time = time.time()
            print(f"Iteration {iteration} completed in {iteration_end_time - iteration_start_time:.2f} seconds")

        except Exception as e:
            print(f"Iteration {iteration} failed: {e}")
            # Continue to next iteration instead of stopping
            continue

    # -----------------------
    # Aggregate and save results
    # -----------------------
    if training_mae_list and validation_mae_list:
        results = {
            "Metric": ["Training MAE", "Training R²", "Validation MAE", "Validation R²"],
            "Average": [
                np.mean(training_mae_list),
                np.mean(training_r2_list),
                np.mean(validation_mae_list),
                np.mean(validation_r2_list),
            ],
            "Std Dev": [
                np.std(training_mae_list),
                np.std(training_r2_list),
                np.std(validation_mae_list),
                np.std(validation_r2_list),
            ],
        }
    else:
        # If all iterations failed, fill with None
        results = {
            "Metric": ["Training MAE", "Training R²", "Validation MAE", "Validation R²"],
            "Average": [None, None, None, None],
            "Std Dev": [None, None, None, None],
        }

    results_file = config.get("results_file", "results.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save final validation predictions (from last successful iteration)
    predictions_file = config.get("output_file", "final_predictions.csv")
    if 'val_predictions' in locals() and val_predictions is not None:
        val_predictions_df = pd.DataFrame(
            val_predictions, 
            columns=[f"Step_{i+1}" for i in range(val_predictions.shape[1])]
        )
        val_predictions_df.to_csv(predictions_file, index=False)
        print(f"Final validation predictions saved to {predictions_file}")
    else:
        print("Warning: No final validation predictions were generated (all iterations may have failed).")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


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




