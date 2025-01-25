import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
from typing import Tuple, Optional

from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log


def process_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes training data by loading, aligning, and preparing it for model training.

    This function performs the following steps:
    1. Loads training features (X) and targets (Y) from CSV files.
    2. Validates and extracts the specified target column.
    3. Ensures that both X and Y have datetime indices for alignment.
    4. Aligns X and Y based on the intersection of their indices.
    5. Applies time horizon and input offset to prepare multi-step targets.
    6. Converts the processed data into appropriate formats for model training.

    Args:
        config (dict): Configuration dictionary containing parameters for data processing.
            Expected keys include:
                - 'x_train_file' (str): Path to the training features CSV file.
                - 'y_train_file' (str): Path to the training targets CSV file.
                - 'target_column' (str or int): Column name or index to be used as the target.
                - 'time_horizon' (int): Number of future steps to predict.
                - 'input_offset' (int): Offset applied to the input data.
                - 'headers' (bool): Indicates if CSV files contain headers.
                - 'max_steps_train' (int): Maximum number of rows to read for training data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - x_train_data (pd.DataFrame): Processed training features.
            - y_train_data (pd.DataFrame): Processed training targets with multi-step horizons.

    Raises:
        ValueError: If any of the validation checks fail, such as missing target columns
                    or insufficient data after applying offsets and horizons.
        Exception: Propagates any exception that occurs during CSV loading or data processing.

    Example:
        >>> config = {
        ...     'x_train_file': 'path/to/x_train.csv',
        ...     'y_train_file': 'path/to/y_train.csv',
        ...     'target_column': 'Close',
        ...     'time_horizon': 6,
        ...     'input_offset': 0,
        ...     'headers': True,
        ...     'max_steps_train': 6300
        ... }
        >>> x_train, y_train = process_data(config)
    """
    print(f"Loading training features from CSV file: {config['x_train_file']} with max rows: {config['max_steps_train']}")
    x_train_data = load_csv(
        file_path=config['x_train_file'],
        headers=config['headers'],
        max_rows=config['max_steps_train']
    )
    print(f"Training features loaded with shape: {x_train_data.shape}")

    y_train_file = config['y_train_file']
    target_column = config['target_column']

    # Load Y data
    if isinstance(y_train_file, str):
        print(f"Loading training targets from CSV file: {y_train_file} with max rows: {config['max_steps_train']}")
        y_train_data = load_csv(
            file_path=y_train_file,
            headers=config['headers'],
            max_rows=config['max_steps_train']
        )
        print(f"Training targets loaded with shape: {y_train_data.shape}")
    else:
        raise ValueError("`y_train_file` must be specified as a string path to the CSV file.")

    # Extract target column if specified
    if target_column is not None:
        if isinstance(target_column, str):
            if target_column not in y_train_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in training targets.")
            y_train_data = y_train_data[[target_column]]
        elif isinstance(target_column, int):
            if target_column < 0 or target_column >= y_train_data.shape[1]:
                raise ValueError(f"Target column index {target_column} is out of range in training targets.")
            y_train_data = y_train_data.iloc[:, [target_column]]
        else:
            raise ValueError("`target_column` must be either a string (column name) or an integer index.")
    else:
        raise ValueError("No valid `target_column` was provided for training targets.")

    # Convert to numeric types and fill NaNs
    x_train_data = x_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Ensure indices are datetime-like
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

    print(f"Returning processed data: X type: {type(x_train_data)}, Y type: {type(y_train_data)}")
    print(f"X shape after adjustments: {x_train_data.shape}")
    print(f"Y shape after adjustments (multi-step): {y_train_data.shape}")

    return x_train_data, y_train_data


def create_sliding_windows(x: np.ndarray, y: np.ndarray, window_size: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows from the dataset for sequence modeling.

    Parameters:
        x (np.ndarray): Input features of shape (N, features).
        y (np.ndarray): Targets of shape (N, time_horizon).
        window_size (int): Number of time steps in each window.
        step (int): Step size between windows.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x_windows (np.ndarray): Sliding windows of input features.
            - y_windows (np.ndarray): Corresponding sliding windows of targets.

    Example:
        >>> x = np.array([[1], [2], [3], [4], [5], [6]])
        >>> y = np.array([[10], [20], [30], [40], [50], [60]])
        >>> create_sliding_windows(x, y, window_size=2, step=1)
        (array([[[1], [2]],
                [[2], [3]],
                [[3], [4]],
                [[4], [5]]]),
         array([[10, 20],
                [20, 30],
                [30, 40],
                [40, 50]]))
    """
    x_windows = []
    y_windows = []
    for i in range(0, len(x) - window_size - y.shape[1] + 1, step):
        x_window = x[i:i + window_size]
        y_window = y[i + window_size:i + window_size + y.shape[1]].flatten()
        x_windows.append(x_window)
        y_windows.append(y_window)
    return np.array(x_windows), np.array(y_windows)


def run_prediction_pipeline(config: dict, plugin: tf.keras.Model, training_data: pd.DataFrame, validation_data: pd.DataFrame) -> None:
    """
    Executes the feature engineering and model training pipeline.

    This function handles the entire prediction pipeline, including:
    1. Loading and processing training data.
    2. Building and training the model based on the specified plugin.
    3. Handling validation data if provided.
    4. Saving the trained model and predictions.
    5. Logging debug information.

    Args:
        config (dict): Configuration dictionary containing parameters for the pipeline.
            Expected keys include:
                - 'plugin' (str): Name of the plugin to use (e.g., 'cnn', 'transformer').
                - 'x_train_file' (str): Path to the training features CSV file.
                - 'y_train_file' (str): Path to the training targets CSV file.
                - 'x_validation_file' (str): Path to the validation features CSV file.
                - 'y_validation_file' (str): Path to the validation targets CSV file.
                - 'headers' (bool): Indicates if CSV files contain headers.
                - 'max_steps_train' (int): Maximum number of rows to read for training data.
                - 'max_steps_test' (int): Maximum number of rows to read for validation data.
                - 'window_size' (int): Size of the sliding window for sequence models.
                - 'target_column' (str or int): Column name or index to be used as the target.
                - Additional parameters required by the plugin.
        plugin (tf.keras.Model): The machine learning model plugin to be used for training and prediction.
        training_data (pd.DataFrame): Preprocessed training features.
        validation_data (pd.DataFrame): Preprocessed validation features.

    Raises:
        ValueError: If any of the validation checks fail or required configuration parameters are missing.

    Example:
        >>> config = {...}  # Configuration dictionary
        >>> plugin = CNNPlugin()
        >>> run_prediction_pipeline(config, plugin, training_data, validation_data)
    """
    start_time = time.time()

    print("Loading and processing training data...")
    x_train, y_train = process_data(config)
    print(f"Processed training data: X shape: {x_train.shape}, Y shape: {y_train.shape}")

    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    batch_size = config.get('batch_size', 32)  # Default batch size if not specified
    epochs = config.get('epochs', 100)        # Default number of epochs if not specified
    threshold_error = config.get('threshold_error', 0.00004)
    window_size = config.get('window_size', None)
    target_column = config.get('target_column', None)

    print(f"Configured window_size: {window_size}")

    # Convert DataFrame to NumPy arrays for training
    x_train_np = x_train.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().astype(np.float32)

    # Ensure x_train is at least 2D
    if x_train_np.ndim == 1:
        x_train_np = x_train_np.reshape(-1, 1)

    print(f"x_train shape before sliding window: {x_train_np.shape}")
    print(f"y_train shape before sliding window: {y_train_np.shape}")

    # Conditional model preparation based on the plugin type
    if config['plugin'] == 'transformer':
        # Reshape for transformer models
        if x_train_np.ndim == 2:
            x_train_np = x_train_np.reshape((x_train_np.shape[0], x_train_np.shape[1], 1))
            print(f"Reshaped x_train for transformer: {x_train_np.shape}")

        # Build the transformer model with the new input shape
        plugin.build_model(input_shape=x_train_np.shape[1:])

    elif config['plugin'] == 'cnn':
        # Validate window_size for CNN
        if window_size is None:
            raise ValueError("`window_size` must be specified in config for CNN plugin.")

        # Create sliding windows for CNN input
        x_train_windowed, y_train_windowed = create_sliding_windows(x_train_np, y_train_np, window_size)
        print(f"Sliding windows created for training: X shape: {x_train_windowed.shape}, Y shape: {y_train_windowed.shape}")

        # Update plugin's window_size parameter if necessary
        plugin.params['window_size'] = window_size

        # Build the CNN model with the new input shape
        plugin.build_model(input_shape=x_train_windowed.shape[1:])

        # Replace original X and Y with windowed data
        x_train_np = x_train_windowed
        y_train_np = y_train_windowed

    else:
        # Build the model for other plugin types (e.g., ANN, LSTM)
        plugin.build_model(input_shape=x_train_np.shape[1:])

    # Handle validation data if provided
    x_val = None
    y_val = None
    if config.get('x_validation_file') and config.get('y_validation_file'):
        print(f"Loading validation features from {config['x_validation_file']} with max rows: {config['max_steps_test']}")
        x_val_df = load_csv(
            file_path=config['x_validation_file'],
            headers=config['headers'],
            max_rows=config['max_steps_test']
        )
        print(f"Validation features loaded with shape: {x_val_df.shape}")

        y_val_file = config['y_validation_file']
        print(f"Loading validation targets from {y_val_file} with max rows: {config['max_steps_test']}")
        y_val_df = load_csv(
            file_path=y_val_file,
            headers=config['headers'],
            max_rows=config['max_steps_test']
        )
        print(f"Validation targets loaded with shape: {y_val_df.shape}")

        # Extract target column if specified
        if target_column is not None:
            if isinstance(target_column, str):
                if target_column not in y_val_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in validation targets.")
                y_val_df = y_val_df[[target_column]]
            elif isinstance(target_column, int):
                if target_column < 0 or target_column >= y_val_df.shape[1]:
                    raise ValueError(f"Target column index {target_column} is out of range in validation targets.")
                y_val_df = y_val_df.iloc[:, [target_column]]
            else:
                raise ValueError("`target_column` must be either a string (column name) or an integer index.")
        else:
            raise ValueError("No valid `target_column` was provided for validation targets.")

        # Convert to numeric types and fill NaNs
        x_val_np = x_val_df.to_numpy().astype(np.float32)
        y_val_np = y_val_df.to_numpy().astype(np.float32)

        # Ensure x_val is at least 2D
        if x_val_np.ndim == 1:
            x_val_np = x_val_np.reshape(-1, 1)

        # Apply sliding window for CNN
        if config['plugin'] == 'cnn':
            if window_size is None:
                raise ValueError("`window_size` must be specified in config for CNN plugin.")
            x_val_windowed, y_val_windowed = create_sliding_windows(x_val_np, y_val_np, window_size)
            print(f"Sliding windows created for validation: X shape: {x_val_windowed.shape}, Y shape: {y_val_windowed.shape}")
            x_val_np = x_val_windowed
            y_val_np = y_val_windowed

        elif config['plugin'] == 'transformer':
            # Reshape for transformer models
            if x_val_np.ndim == 2:
                x_val_np = x_val_np.reshape((x_val_np.shape[0], x_val_np.shape[1], 1))
                print(f"Reshaped x_val for transformer: {x_val_np.shape}")

        # Assign processed validation data
        x_val = x_val_np
        y_val = y_val_np

    # Train the model with or without validation data
    if config['plugin'] == 'cnn' and x_val is not None and y_val is not None:
        plugin.train(
            x_train=x_train_np,
            y_train=y_train_np,
            epochs=epochs,
            batch_size=batch_size,
            threshold_error=threshold_error,
            x_val=x_val,
            y_val=y_val
        )
    else:
        plugin.train(
            x_train=x_train_np,
            y_train=y_train_np,
            epochs=epochs,
            batch_size=batch_size,
            threshold_error=threshold_error
        )

    # Save the trained model if specified
    if config.get('save_model'):
        plugin.save(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # Make predictions on the training data
    predictions = plugin.predict(x_train_np)

    # Evaluate the model
    mse = float(plugin.calculate_mse(y_train_np, predictions))
    mae = float(plugin.calculate_mae(y_train_np, predictions))
    print(f"Mean Squared Error (Training): {mse}")
    print(f"Mean Absolute Error (Training): {mae}")

    # Convert predictions to DataFrame
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    else:
        num_steps = predictions.shape[1]
        pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
        predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    # Save predictions to CSV
    output_filename = config['output_file']
    write_csv(
        file_path=output_filename,
        data=predictions_df,
        include_date=config.get('force_date', False),
        headers=config.get('headers', True)
    )
    print(f"Predictions saved to {output_filename}")

    # Save debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time_seconds': float(execution_time),
        'training_mse': mse,
        'training_mae': mae
    }

    if config.get('save_log'):
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug information saved to {config['save_log']}")

    if config.get('remote_log'):
        remote_log(
            config=config,
            debug_info=debug_info,
            remote_log_endpoint=config['remote_log'],
            username=config.get('username'),
            password=config.get('password')
        )
        print(f"Debug information remotely logged to {config['remote_log']}")

    print(f"Pipeline execution time: {execution_time:.2f} seconds")


def load_and_evaluate_model(config: dict, plugin: tf.keras.Model) -> None:
    """
    Loads a pre-trained model and evaluates it on the training data.

    This function performs the following steps:
    1. Loads the specified pre-trained model.
    2. Loads and processes the training data.
    3. Makes predictions using the loaded model.
    4. Saves the predictions to a CSV file for evaluation.

    Args:
        config (dict): Configuration dictionary containing parameters for model evaluation.
            Expected keys include:
                - 'load_model' (str): Path to the pre-trained model file.
                - 'x_train_file' (str): Path to the training features CSV file.
                - 'y_train_file' (str): Path to the training targets CSV file.
                - 'headers' (bool): Indicates if CSV files contain headers.
                - 'force_date' (bool): Determines if date should be included in the output CSV.
                - 'headers' (bool): Indicates if CSV files contain headers.
                - 'evaluate_file' (str): Path to save the evaluation predictions CSV file.
                - 'max_steps_train' (int): Maximum number of rows to read for training data.

        plugin (tf.keras.Model): The machine learning model plugin to be used for evaluation.

    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        Exception: Propagates any exception that occurs during model loading or data processing.

    Example:
        >>> config = {
        ...     'load_model': 'path/to/model.keras',
        ...     'x_train_file': 'path/to/x_train.csv',
        ...     'y_train_file': 'path/to/y_train.csv',
        ...     'headers': True,
        ...     'force_date': False,
        ...     'evaluate_file': 'path/to/model_eval.csv',
        ...     'max_steps_train': 6300
        ... }
        >>> plugin = CNNPlugin()
        >>> load_and_evaluate_model(config, plugin)
    """
    # Load the pre-trained model
    print(f"Loading pre-trained model from {config['load_model']}...")
    plugin.load(config['load_model'])
    print("Model loaded successfully.")

    # Load and process training data
    print("Loading and processing training data for evaluation...")
    x_train, _ = process_data(config)
    print(f"Processed training data: X shape: {x_train.shape}")

    # Convert DataFrame to NumPy array for prediction
    x_train_np = x_train.to_numpy().astype(np.float32)

    # Ensure x_train is at least 2D
    if x_train_np.ndim == 1:
        x_train_np = x_train_np.reshape(-1, 1)

    # Make predictions using the loaded model
    print("Making predictions on training data...")
    predictions = plugin.predict(x_train_np)
    print(f"Predictions shape: {predictions.shape}")

    # Convert predictions to DataFrame
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    else:
        num_steps = predictions.shape[1]
        pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
        predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    # Save predictions to CSV for evaluation
    evaluate_filename = config['evaluate_file']
    write_csv(
        file_path=evaluate_filename,
        data=predictions_df,
        include_date=config.get('force_date', False),
        headers=config.get('headers', True)
    )
    print(f"Evaluation predictions saved to {evaluate_filename}")
