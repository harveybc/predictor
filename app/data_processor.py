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
    Processes training or validation data by loading, aligning, and preparing it for model training or evaluation.

    Args:
        config (dict): Configuration dictionary containing parameters for data processing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed features and targets.
    """
    # Determine if we are processing validation data
    is_validation = 'max_steps_test' in config

    # Select the appropriate files and parameters
    x_file = config['x_validation_file'] if is_validation else config['x_train_file']
    y_file = config['y_validation_file'] if is_validation else config['y_train_file']
    max_steps = config.get('max_steps_test') if is_validation else config.get('max_steps_train')

    # Load X data
    print(f"Loading data from CSV file: {x_file}")
    x_data = load_csv(x_file, headers=config['headers'])
    print(f"Data loaded with shape: {x_data.shape}")

    # Load Y data
    print(f"Loading target data from CSV file: {y_file}")
    y_data = load_csv(y_file, headers=config['headers'])
    print(f"Target data loaded with shape: {y_data.shape}")

    # Extract target column
    target_column = config['target_column']
    if isinstance(target_column, str):
        if target_column not in y_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in target data.")
        y_data = y_data[[target_column]]
    elif isinstance(target_column, int):
        if target_column < 0 or target_column >= y_data.shape[1]:
            raise ValueError(f"Target column index {target_column} is out of range.")
        y_data = y_data.iloc[:, [target_column]]
    else:
        raise ValueError("`target_column` must be either a string (column name) or an integer index.")

    # Convert to numeric and fill NaNs
    x_data = x_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_data = y_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Ensure indices are datetime-like
    if not isinstance(x_data.index, pd.DatetimeIndex) or not isinstance(y_data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a valid DatetimeIndex. Check your input CSV files.")

    # Align datasets
    common_index = x_data.index.intersection(y_data.index)
    x_data = x_data.loc[common_index].sort_index()
    y_data = y_data.loc[common_index].sort_index()

    if x_data.empty or y_data.empty:
        raise ValueError("No overlapping dates found or data is empty after alignment.")

    # Apply offsets and horizons
    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    print(f"Applying time horizon: {time_horizon} and input offset: {input_offset}")
    total_offset = time_horizon + input_offset

    y_data = y_data.iloc[total_offset:]
    x_data = x_data.iloc[:-time_horizon]

    # Validate post-offset data
    if x_data.empty or y_data.empty:
        raise ValueError("After applying time_horizon and offset, no samples remain.")

    # Limit rows based on max_steps
    if isinstance(max_steps, int) and max_steps > 0:
        print(f"Limiting data to first {max_steps} rows.")
        x_data = x_data.iloc[:max_steps]
        y_data = y_data.iloc[:max_steps]

    # Create multi-step Y data
    Y_list = []
    for i in range(len(y_data) - time_horizon + 1):
        row = [y_data.iloc[i + j].values[0] for j in range(time_horizon)]
        Y_list.append(row)

    if not Y_list:
        raise ValueError("After creating multi-step slices, no samples remain.")

    y_data = pd.DataFrame(Y_list)

    # Adjust X data to match Y length
    x_data = x_data.iloc[:len(y_data)].reset_index(drop=True)
    y_data = y_data.reset_index(drop=True)

    print(f"Returning data of type: {type(x_data)}, {type(y_data)}")
    print(f"x_data shape after adjustments: {x_data.shape}")
    print(f"y_data shape after adjustments (multi-step): {y_data.shape}")

    return x_data, y_data



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
    Runs the prediction pipeline with proper handling of training and validation datasets,
    iterative logging of metrics, and output of predictions and statistics to CSV files.
    """
    start_time = time.time()

    iterations = config.get('iterations', 1)
    print(f"Number of iterations: {iterations}")

    # Lists to store metrics for all iterations
    training_mae_list = []
    training_r2_list = []
    validation_mae_list = []
    validation_r2_list = []

    # Process training data
    print("Processing training data...")
    x_train, y_train = process_data(config)
    print(f"Processed training data shape: {x_train.shape}")

    # Process validation data if available
    x_val, y_val = None, None
    if config.get('x_validation_file') and config.get('y_validation_file'):
        val_config = config.copy()
        val_config['x_train_file'] = config['x_validation_file']
        val_config['y_train_file'] = config['y_validation_file']
        x_val, y_val = process_data(val_config)
        print(f"Processed validation data shape: {x_val.shape}")

    # Extract time_horizon and window_size from the config
    time_horizon = config.get('time_horizon')
    window_size = config.get('window_size')

    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")

    if window_size is None and config['plugin'] == 'cnn':
        raise ValueError("`window_size` must be defined in the configuration for CNN plugin.")

    print(f"Time Horizon: {time_horizon}")

    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold_error = config['threshold_error']

    # Convert training data to numpy arrays
    x_train = x_train.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)

    if x_val is not None:
        x_val = x_val.to_numpy().astype(np.float32)
        y_val = y_val.to_numpy().astype(np.float32)

    # Handle CNN-specific sliding windows
    if config['plugin'] == 'cnn':
        print("Creating sliding windows for CNN...")
        x_train, y_train, _ = create_sliding_windows(
            x_train, y_train, window_size, time_horizon, stride=1
        )

        if x_val is not None and y_val is not None:
            x_val, y_val, _ = create_sliding_windows(
                x_val, y_val, window_size, time_horizon, stride=1
            )
        print(f"Sliding windows created: x_train shape {x_train.shape}, y_train shape {y_train.shape}")


    # Ensure x_train is at least 2D for other plugins
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)

    # Set time_horizon in plugin parameters
    plugin.set_params(time_horizon=time_horizon)

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            # Build the model
            if config['plugin'] == 'cnn':
                plugin.build_model(input_shape=(window_size, x_train.shape[2]))
            else:
                plugin.build_model(input_shape=x_train.shape[1])

            # Train the model
            plugin.train(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                threshold_error=threshold_error,
                x_val=x_val,
                y_val=y_val
            )

            print("Evaluating trained model on training and validation data. Please wait...")

            # Suppress TensorFlow/Keras logs during prediction
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                logging.getLogger("tensorflow").setLevel(logging.FATAL)

                # Predict training data
                train_predictions = plugin.predict(x_train)

                # Predict validation data (if available)
                val_predictions = plugin.predict(x_val) if x_val is not None else None

            # Restore TensorFlow logging level
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            logging.getLogger("tensorflow").setLevel(logging.INFO)

            # Evaluate training metrics
            train_mae = float(plugin.calculate_mae(y_train[:len(train_predictions)], train_predictions))
            train_r2 = float(r2_score(y_train[:len(train_predictions)], train_predictions))
            print(f"Training MAE: {train_mae}")
            print(f"Training R²: {train_r2}")

            # Append training metrics
            training_mae_list.append(train_mae)
            training_r2_list.append(train_r2)

            # Evaluate validation metrics
            if x_val is not None and y_val is not None:
                val_mae = float(plugin.calculate_mae(y_val[:len(val_predictions)], val_predictions))
                val_r2 = float(r2_score(y_val[:len(val_predictions)], val_predictions))
                print(f"Validation MAE: {val_mae}")
                print(f"Validation R²: {val_r2}")

                # Append validation metrics
                validation_mae_list.append(val_mae)
                validation_r2_list.append(val_r2)

            iteration_end_time = time.time()
            print(f"Iteration {iteration} completed in {iteration_end_time - iteration_start_time:.2f} seconds")

        except Exception as e:
            print(f"Iteration {iteration} failed: {e}")
            continue  # Proceed to the next iteration

    # Aggregate statistics
    results = {
        'Metric': ['Training MAE', 'Training R²', 'Validation MAE', 'Validation R²'],
        'Average': [
            np.mean(training_mae_list),
            np.mean(training_r2_list),
            np.mean(validation_mae_list) if validation_mae_list else None,
            np.mean(validation_r2_list) if validation_r2_list else None,
        ],
        'Std Dev': [
            np.std(training_mae_list),
            np.std(training_r2_list),
            np.std(validation_mae_list) if validation_mae_list else None,
            np.std(validation_r2_list) if validation_r2_list else None,
        ],
        'Max': [
            np.max(training_mae_list),
            np.max(training_r2_list),
            np.max(validation_mae_list) if validation_mae_list else None,
            np.max(validation_r2_list) if validation_r2_list else None,
        ],
        'Min': [
            np.min(training_mae_list),
            np.min(training_r2_list),
            np.min(validation_mae_list) if validation_mae_list else None,
            np.min(validation_r2_list) if validation_r2_list else None,
        ],
    }

    # Save results to CSV
    results_file = config.get('results_file', 'results.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save final validation predictions (if available)
    if val_predictions is not None:
        final_val_file = config.get('output_file', 'validation_predictions.csv')
        val_predictions_df = pd.DataFrame(val_predictions, columns=[f'Prediction_{i+1}' for i in range(val_predictions.shape[1])])
        val_predictions_df.to_csv(final_val_file, index=False)
        print(f"Final validation predictions saved to {final_val_file}")

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



