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
    print(f"Number of iterations: {iterations}")

    # Lists to store metrics for all iterations
    training_mae_list = []
    training_r2_list = []
    validation_mae_list = []
    validation_r2_list = []

    # Load all datasets
    print("Loading and processing datasets...")
    datasets = process_data(config)  # <-- Rely on the process_data function
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")

    # Extra debug: confirm indices if they are still DataFrames
    if isinstance(x_train, pd.DataFrame) and isinstance(y_train, pd.DataFrame):
        if not x_train.index.equals(y_train.index):
            raise ValueError("TRAIN DATA MISMATCH: x_train and y_train indices do not match. Check alignment.")
        else:
            print("Debug: x_train and y_train indices are aligned.")
    if isinstance(x_val, pd.DataFrame) and isinstance(y_val, pd.DataFrame):
        if not x_val.index.equals(y_val.index):
            raise ValueError("VALIDATION DATA MISMATCH: x_val and y_val indices do not match. Check alignment.")
        else:
            print("Debug: x_val and y_val indices are aligned.")

    # Basic config checks
    time_horizon = config.get("time_horizon")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")

    window_size = config.get("window_size")
    if config["plugin"] == "cnn" and window_size is None:
        raise ValueError("`window_size` must be defined in the configuration for CNN plugin.")

    print(f"Time Horizon: {time_horizon}")

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    threshold_error = config["threshold_error"]

    # Convert datasets to numpy arrays
    x_train_np = x_train.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32)
    x_val_np = x_val.to_numpy().astype(np.float32)
    y_val_np = y_val.to_numpy().astype(np.float32)

    # CNN-specific sliding windows
    if config["plugin"] == "cnn":
        print("Creating sliding windows for CNN...")
        x_train_np, y_train_np, _ = create_sliding_windows(
            x_train_np, y_train_np, window_size, time_horizon, stride=1
        )
        x_val_np, y_val_np, _ = create_sliding_windows(
            x_val_np, y_val_np, window_size, time_horizon, stride=1
        )

        print("Sliding windows created:")
        print(f"  x_train: {x_train_np.shape}, y_train: {y_train_np.shape}")
        print(f"  x_val:   {x_val_np.shape}, y_val: {y_val_np.shape}")

        # Confirm shapes after sliding windows
        if x_train_np.shape[0] != y_train_np.shape[0]:
            raise ValueError("After sliding windows, training x/y have mismatched samples.")
        if x_val_np.shape[0] != y_val_np.shape[0]:
            raise ValueError("After sliding windows, validation x/y have mismatched samples.")

    else:
        # Ensure x_* are at least 2D
        if x_train_np.ndim == 1:
            x_train_np = x_train_np.reshape(-1, 1)
        if x_val_np.ndim == 1:
            x_val_np = x_val_np.reshape(-1, 1)

    # Determine input_shape based on plugin
    if config["plugin"] == "cnn":
        # CNN expects (window_size, features) per sample
        input_shape = (window_size, x_train_np.shape[2])
    elif config["plugin"].lower() == "ann":
        # The ANN plugin expects a single integer for input_shape
        input_shape = x_train_np.shape[1]
    else:
        # For LSTM, Transformers, or others: typically pass a tuple (features,)
        input_shape = (x_train_np.shape[1],)

    print(f"Debug: input_shape determined as: {input_shape}")

    # Pass time_horizon to the plugin (if it uses it)
    plugin.set_params(time_horizon=time_horizon)

    # Iterative training and evaluation
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            # Build the model
            plugin.build_model(input_shape=input_shape)

            print(f"Debug: Starting training. x_train shape: {x_train_np.shape}, y_train shape: {y_train_np.shape}")
            print(f"Debug: Validation sets. x_val shape: {x_val_np.shape}, y_val shape: {y_val_np.shape}")

            # Train the model
            plugin.train(
                x_train_np,
                y_train_np,
                epochs=epochs,
                batch_size=batch_size,
                threshold_error=threshold_error,
                x_val=x_val_np,
                y_val=y_val_np,
            )

            print("Evaluating trained model on training and validation data. Please wait...")

            # Suppress TensorFlow/Keras logs during prediction
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                logging.getLogger("tensorflow").setLevel(logging.FATAL)

                # Predict training data
                train_predictions = plugin.predict(x_train_np)

                # Predict validation data
                val_predictions = plugin.predict(x_val_np)

            # Restore TensorFlow logs
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
            logging.getLogger("tensorflow").setLevel(logging.INFO)

            # -----------------------
            # Dimension checks
            # -----------------------
            print(f"Debug: train_predictions shape: {train_predictions.shape}, val_predictions shape: {val_predictions.shape}")

            if train_predictions.shape[0] != y_train_np.shape[0]:
                raise ValueError(
                    "Mismatch in training samples dimension:\n"
                    f"  train_predictions.shape[0]={train_predictions.shape[0]}, "
                    f"y_train.shape[0]={y_train_np.shape[0]}.\n"
                    "Please ensure data alignment in multi-step logic."
                )
            if train_predictions.shape[1] != y_train_np.shape[1]:
                raise ValueError(
                    "Mismatch in training time_horizon dimension:\n"
                    f"  train_predictions.shape[1]={train_predictions.shape[1]}, "
                    f"y_train.shape[1]={y_train_np.shape[1]}.\n"
                    "Time horizon dimension must match."
                )

            if val_predictions.shape[0] != y_val_np.shape[0]:
                raise ValueError(
                    "Mismatch in validation samples dimension:\n"
                    f"  val_predictions.shape[0]={val_predictions.shape[0]}, "
                    f"y_val.shape[0]={y_val_np.shape[0]}.\n"
                    "Please ensure data alignment for multi-step validation."
                )
            if val_predictions.shape[1] != y_val_np.shape[1]:
                raise ValueError(
                    "Mismatch in validation time_horizon dimension:\n"
                    f"  val_predictions.shape[1]={val_predictions.shape[1]}, "
                    f"y_val.shape[1]={y_val_np.shape[1]}.\n"
                    "Time horizon dimension must match for validation."
                )

            # Evaluate training metrics
            train_mae = float(plugin.calculate_mae(y_train_np, train_predictions))
            train_r2 = float(r2_score(y_train_np, train_predictions))
            print(f"Training MAE: {train_mae}")
            print(f"Training R²: {train_r2}")

            # Save training metrics
            training_mae_list.append(train_mae)
            training_r2_list.append(train_r2)

            # Evaluate validation metrics
            val_mae = float(plugin.calculate_mae(y_val_np, val_predictions))
            val_r2 = float(r2_score(y_val_np, val_predictions))
            print(f"Validation MAE: {val_mae}")
            print(f"Validation R²: {val_r2}")

            # Save validation metrics
            validation_mae_list.append(val_mae)
            validation_r2_list.append(val_r2)

            iteration_end_time = time.time()
            print(f"Iteration {iteration} completed in {iteration_end_time - iteration_start_time:.2f} seconds")

        except Exception as e:
            print(f"Iteration {iteration} failed with error:\n  {e}")
            # Continue to the next iteration if an error occurs
            continue

    # -----------------------
    # Aggregate statistics
    # -----------------------
    results = {
        "Metric": [
            "Training MAE",
            "Training R²",
            "Validation MAE",
            "Validation R²",
        ],
        "Average": [
            np.mean(training_mae_list) if training_mae_list else None,
            np.mean(training_r2_list)  if training_r2_list  else None,
            np.mean(validation_mae_list) if validation_mae_list else None,
            np.mean(validation_r2_list)  if validation_r2_list  else None,
        ],
        "Std Dev": [
            np.std(training_mae_list) if training_mae_list else None,
            np.std(training_r2_list)  if training_r2_list  else None,
            np.std(validation_mae_list) if validation_mae_list else None,
            np.std(validation_r2_list)  if validation_r2_list  else None,
        ],
        "Max": [
            np.max(training_mae_list) if training_mae_list else None,
            np.max(training_r2_list)  if training_r2_list  else None,
            np.max(validation_mae_list) if validation_mae_list else None,
            np.max(validation_r2_list)  if validation_r2_list  else None,
        ],
        "Min": [
            np.min(training_mae_list) if training_mae_list else None,
            np.min(training_r2_list)  if training_r2_list  else None,
            np.min(validation_mae_list) if validation_mae_list else None,
            np.min(validation_r2_list)  if validation_r2_list  else None,
        ],
    }

    # Save results to CSV
    results_file = config.get("results_file", "results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save final validation predictions
    final_val_file = config.get("output_file", "validation_predictions.csv")
    if 'val_predictions' in locals() and val_predictions is not None:
        val_predictions_df = pd.DataFrame(
            val_predictions, 
            columns=[f"Prediction_{i+1}" for i in range(val_predictions.shape[1])]
        )
        val_predictions_df.to_csv(final_val_file, index=False)
        print(f"Final validation predictions saved to {final_val_file}")
    else:
        print("Warning: No final validation predictions were generated (all iterations may have failed).")

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




