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
import logging


def process_data(config):
    """
    Processes training data by loading, aligning, and preparing it for model training.

    This function performs the following steps:
    1. Loads training features (X) and targets (Y) from CSV files.
    2. Validates and extracts the specified target column.
    3. Ensures that both X and Y have datetime indices for alignment.
    4. Aligns X and Y based on the intersection of their indices.
    5. Applies time horizon and input offset to prepare multi-step targets.
    6. Limits the number of rows based on `max_steps_train` if specified.
    7. Converts the processed data into appropriate formats for model training.

    Args:
        config (dict): Configuration dictionary containing parameters for data processing.
            Expected keys include:
                - 'x_train_file' (str): Path to the training features CSV file.
                - 'y_train_file' (str): Path to the training targets CSV file.
                - 'target_column' (str or int): Column name or index to be used as the target.
                - 'time_horizon' (int): Number of future steps to predict.
                - 'input_offset' (int): Offset applied to the input data.
                - 'headers' (bool): Indicates if CSV files contain headers.
                - 'max_steps_train' (int, optional): Maximum number of rows to read for training data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - x_train_data (pd.DataFrame): Processed training features.
            - y_train_data (pd.DataFrame): Processed training targets with multi-step horizons.

    Raises:
        ValueError: If any of the validation checks fail, such as missing target columns
                    or insufficient data after applying offsets and horizons.
        Exception: Propagates any exception that occurs during CSV loading or data processing.
    """
    print(f"Loading data from CSV file: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    print(f"Data loaded with shape: {x_train_data.shape}")

    y_train_file = config['y_train_file']
    target_column = config['target_column']

    # Load Y data
    if isinstance(y_train_file, str):
        print(f"Loading y_train data from CSV file: {y_train_file}")
        y_train_data = load_csv(y_train_file, headers=config['headers'])
        print(f"y_train data loaded with shape: {y_train_data.shape}")
    else:
        raise ValueError("`y_train_file` must be specified as a string path to the CSV file.")

    # Extract target column if specified
    if target_column is not None:
        if isinstance(target_column, str):
            if target_column not in y_train_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in y_train_data.")
            y_train_data = y_train_data[[target_column]]
        elif isinstance(target_column, int):
            if target_column < 0 or target_column >= y_train_data.shape[1]:
                raise ValueError(f"Target column index {target_column} is out of range in y_train_data.")
            y_train_data = y_train_data.iloc[:, [target_column]]
        else:
            raise ValueError("`target_column` must be either a string (column name) or an integer index.")
    else:
        raise ValueError("No valid `target_column` was provided for y_train_data.")

    # Convert to numeric, fill NaNs
    x_train_data = x_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Confirm that indices are datetime-like
    if not isinstance(x_train_data.index, pd.DatetimeIndex) or not isinstance(y_train_data.index, pd.DatetimeIndex):
        raise ValueError("Either 'DATE_TIME' column wasn't parsed as datetime or no valid DatetimeIndex found. "
                         "Ensure your CSV has a 'DATE_TIME' column or the first column is recognized as datetime.")

    # Align datasets based on common dates
    common_index = x_train_data.index.intersection(y_train_data.index)
    x_train_data = x_train_data.loc[common_index].sort_index()
    y_train_data = y_train_data.loc[common_index].sort_index()

    # Validate alignment
    if x_train_data.empty or y_train_data.empty:
        raise ValueError(
            "No overlapping dates found (or data is empty after alignment). "
            "Please ensure your CSV files share date ranges."
        )

    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    print(f"Applying time horizon: {time_horizon} and input offset: {input_offset}")
    total_offset = time_horizon + input_offset

    # Shift Y by total_offset and adjust X accordingly
    y_train_data = y_train_data.iloc[total_offset:]
    x_train_data = x_train_data.iloc[:-time_horizon]

    print(f"Data shape after applying offset and time horizon: X: {x_train_data.shape}, Y: {y_train_data.shape}")

    # Validate post-offset data
    if x_train_data.empty or y_train_data.empty:
        raise ValueError(
            "After applying time_horizon and offset, no samples remain. "
            "Check that your dataset is large enough and offsets/time_horizon are correct."
        )

    # Ensure X and Y have the same length
    min_length = min(len(x_train_data), len(y_train_data))
    x_train_data = x_train_data.iloc[:min_length]
    y_train_data = y_train_data.iloc[:min_length]

    # Limit the number of rows based on max_steps_train if specified
    max_steps_train = config.get('max_steps_train')
    if isinstance(max_steps_train, int) and max_steps_train > 0:
        print(f"Limiting training data to first {max_steps_train} rows.")
        x_train_data = x_train_data.iloc[:max_steps_train]
        y_train_data = y_train_data.iloc[:max_steps_train]
        print(f"Training data shape after limiting: X: {x_train_data.shape}, Y: {y_train_data.shape}")

    # Create multi-step Y for time horizon
    Y_list = []
    for i in range(len(y_train_data) - time_horizon + 1):
        row_values = [y_train_data.iloc[i + j].values[0] for j in range(time_horizon)]
        Y_list.append(row_values)

    if not Y_list:
        raise ValueError(
            "After creating multi-step slices, no samples remain. "
            "Check that your data is sufficient for the given time_horizon."
        )

    y_train_data = pd.DataFrame(Y_list)

    # Adjust X to match the number of Y samples
    x_train_data = x_train_data.iloc[:len(y_train_data)].reset_index(drop=True)
    y_train_data = y_train_data.reset_index(drop=True)

    print(f"Returning data of type: {type(x_train_data)}, {type(y_train_data)}")
    print(f"x_train_data shape after adjustments: {x_train_data.shape}")
    print(f"y_train_data shape after adjustments (multi-step): {y_train_data.shape}")

    return x_train_data, y_train_data


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

import logging

import os
import logging
import contextlib

def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline with restored iteration logic, logging, and CSV outputs.
    Predicts the next `time_horizon` ticks with a stride of `time_horizon` for all plugins.
    """
    start_time = time.time()

    iterations = config.get('iterations', 1)
    print(f"Number of iterations: {iterations}")

    # Lists to store MAE and R² values for each iteration
    training_mae_list = []
    training_r2_list = []
    validation_mae_list = []
    validation_r2_list = []

    # Process the training data
    print("Running process_data...")
    x_train, y_train = process_data(config)
    print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")

    # Extract time_horizon from the config
    time_horizon = config.get('time_horizon')
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")

    print(f"Time Horizon: {time_horizon}")

    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold_error = config['threshold_error']

    # Convert x_train and y_train to numpy arrays
    x_train = x_train.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)

    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Set time_horizon in plugin parameters
    plugin.set_params(time_horizon=time_horizon)

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            # Build the model
            plugin.build_model(input_shape=x_train.shape[1])

            # Train the model
            plugin.train(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                threshold_error=threshold_error,
            )

            # Predict with stride logic
            predictions = []
            for i in range(0, len(x_train) - time_horizon + 1, time_horizon):
                stride_input = x_train[i:i + time_horizon]
                if stride_input.shape[0] < time_horizon:
                    break  # Skip incomplete strides
                stride_pred = plugin.predict(stride_input)
                predictions.append(stride_pred)

            predictions = np.vstack(predictions)
            print(f"Concatenated predictions shape: {predictions.shape}")

            # Evaluate training metrics
            mse = float(plugin.calculate_mse(y_train[:len(predictions)], predictions))
            mae = float(plugin.calculate_mae(y_train[:len(predictions)], predictions))
            r2 = float(r2_score(y_train[:len(predictions)], predictions))
            print(f"Training Mean Squared Error: {mse}")
            print(f"Training Mean Absolute Error: {mae}")
            print(f"Training R² Score: {r2}")

            # Save training metrics
            training_mae_list.append(mae)
            training_r2_list.append(r2)

            # Save predictions to CSV (if configured)
            output_file = config.get('output_file', 'predictions.csv')
            predictions_df = pd.DataFrame(predictions, columns=[f'Prediction_{i+1}' for i in range(predictions.shape[1])])
            predictions_df['DATE_TIME'] = range(len(predictions))  # Placeholder for DATE_TIME
            predictions_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")

            iteration_end_time = time.time()
            print(f"Iteration {iteration} completed in {iteration_end_time - iteration_start_time:.2f} seconds")

        except Exception as e:
            print(f"Iteration {iteration} failed: {e}")
            continue  # Continue to the next iteration

    # Aggregate statistics
    if training_mae_list and training_r2_list:
        avg_training_mae = np.mean(training_mae_list)
        std_training_mae = np.std(training_mae_list)
        avg_training_r2 = np.mean(training_r2_list)
        std_training_r2 = np.std(training_r2_list)

        print("\n=== Aggregated Statistics Across Iterations ===")
        print(f"Average Training MAE: {avg_training_mae:.4f} ± {std_training_mae:.4f}")
        print(f"Average Training R²: {avg_training_r2:.4f} ± {std_training_r2:.4f}")

        # Save results to CSV (if configured)
        results_file = config.get('results_file', 'results.csv')
        results_data = {
            'Metric': ['Training MAE', 'Training R²'],
            'Average': [avg_training_mae, avg_training_r2],
            'Std Dev': [std_training_mae, std_training_r2],
        }
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")



def create_sliding_windows(x, y, window_size, time_horizon, stride=1, date_times=None):
    """
    Updated to handle consistent windowing for training and prediction with stride logic.
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




def create_multi_step_targets(y, time_horizon):
    """
    Transforms the target data into multi-step targets based on the specified time horizon.

    Args:
        y (numpy.ndarray): Original target data of shape (N,) or (N, 1).
        time_horizon (int): Number of future steps to predict.

    Returns:
        numpy.ndarray: Transformed target data of shape (N - time_horizon + 1, time_horizon).
    """
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    elif y.ndim > 2:
        raise ValueError("y should be a 1D or 2D array with a single column.")

    Y_list = []
    for i in range(len(y) - time_horizon + 1):
        row = y[i:i + time_horizon]
        Y_list.append(row)
    return np.array(Y_list)



def create_multi_step_targets(y, time_horizon):
    """
    Transforms the target data into multi-step targets based on the specified time horizon.
    
    Args:
        y (numpy.ndarray): Original target data of shape (N,).
        time_horizon (int): Number of future steps to predict.
    
    Returns:
        numpy.ndarray: Transformed target data of shape (N - time_horizon + 1, time_horizon).
    """
    Y_list = []
    for i in range(len(y) - time_horizon + 1):
        row = y[i:i + time_horizon].flatten()
        Y_list.append(row)
    return np.array(Y_list)



