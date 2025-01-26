import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import sys

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
    # Load training features
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


def create_sliding_windows(x, y, window_size, step=1):
    """
    Creates sliding windows from the dataset.

    Parameters:
        x (numpy.ndarray): Input features of shape (N, features).
        y (numpy.ndarray): Targets of shape (N, time_horizon).
        window_size (int): Number of time steps in each window.
        step (int): Step size between windows.

    Returns:
        Tuple of numpy.ndarrays: (x_windows, y_windows)
    """
    x_windows = []
    y_windows = []
    for i in range(0, len(x) - window_size - y.shape[1] + 1, step):
        x_windows.append(x[i:i + window_size])
        y_windows.append(y[i + window_size:i + window_size + y.shape[1]].flatten())
    return np.array(x_windows), np.array(y_windows)


def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline with conditional data reshaping for different plugins.

    This function handles the entire prediction pipeline, including:
    1. Loading and processing training data.
    2. Building and training the model based on the specified plugin.
    3. Handling validation data if provided.
    4. Saving the trained model and predictions.
    5. Logging debug information.

    Args:
        config (dict): Configuration dictionary containing parameters for the pipeline.
        plugin (tf.keras.Model): The machine learning model plugin to be used for training and prediction.

    Raises:
        ValueError: If any of the validation checks fail or required configuration parameters are missing.
        Exception: Propagates any exception that occurs during model training or prediction.
    """
    start_time = time.time()

    print("Running process_data...")
    x_train, y_train = process_data(config)
    print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")

    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold_error = config['threshold_error']
    window_size = config.get('window_size', None)  # e.g., 24 for daily patterns
    target_column = config.get('target_column', None)  # Specify the target column

    # Debugging: Print window_size
    print(f"Configured window_size: {window_size}")

    # Ensure x_train and y_train are DataFrame or Series
    if isinstance(x_train, (pd.DataFrame, pd.Series)) and isinstance(y_train, (pd.DataFrame, pd.Series)):
        # Conditional Target Column Selection for CNN
        if config['plugin'] == 'cnn' and target_column is not None:
            if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
                if isinstance(target_column, str):
                    if target_column not in y_train.columns:
                        raise ValueError(f"Target column '{target_column}' not found in y_train.")
                    y_train = y_train[[target_column]]  # Keep it as a DataFrame
                elif isinstance(target_column, int):
                    if target_column < 0 or target_column >= y_train.shape[1]:
                        raise ValueError(f"Target column index {target_column} is out of range in y_train.")
                    y_train = y_train.iloc[:, [target_column]]
                else:
                    raise ValueError("`target_column` must be either a string (column name) or an integer index.")
            else:
                raise ValueError("y_train must be a pandas DataFrame or Series to select target columns by name or index.")

        # Convert to numpy for training
        x_train = x_train.to_numpy().astype(np.float32)
        y_train = y_train.to_numpy().astype(np.float32)

        # Ensure x_train is at least 2D
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)

        # Debug messages
        print(f"x_train shape before sliding window: {x_train.shape}")
        print(f"y_train shape before sliding window: {y_train.shape}")

        # ----------------------------
        # CONDITIONAL RESHAPE FOR TRANSFORMER
        # ----------------------------
        if config['plugin'] == 'transformer':
            # Treat each feature as a separate timestep
            # Reshape from (N, features) to (N, features, 1)
            if x_train.ndim == 2:
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                print(f"Reshaped x_train for transformer: {x_train.shape}")

            # Now we pass a 3D tuple: (samples, seq_len, num_features)
            plugin.build_model(input_shape=x_train.shape[1:])

        elif config['plugin'] == 'cnn':
            # Apply sliding window
            if window_size is None:
                raise ValueError("`window_size` must be specified in config for CNN plugin.")

            # Create sliding windows
            x_train_windowed, y_train_windowed = create_sliding_windows(x_train, y_train, window_size)
            print(f"Sliding windows created: x_train_windowed shape: {x_train_windowed.shape}, y_train_windowed shape: {y_train_windowed.shape}")

            # Update plugin's window_size parameter if necessary
            plugin.params['window_size'] = window_size

            # Build model with window_size
            plugin.build_model(input_shape=x_train_windowed.shape[1:])

            # Replace original x_train and y_train with windowed data
            x_train = x_train_windowed
            y_train = y_train_windowed

        else:
            # Keep old logic for ANN/LSTM
            # Pass a single integer for input_shape
            plugin.build_model(input_shape=x_train.shape[1:])

        # ----------------------------
        # TRAIN THE MODEL
        # ----------------------------
        # Handle validation data if available
        x_val = None
        y_val = None
        if config.get('x_validation_file') and config.get('y_validation_file'):
            print("Preparing validation data...")
            x_val_df = load_csv(config['x_validation_file'], headers=config.get('headers', True))
            y_val_df = load_csv(config['y_validation_file'], headers=config.get('headers', True))

            # Conditional Target Column Selection for CNN
            if config['plugin'] == 'cnn' and target_column is not None:
                if isinstance(y_val_df, pd.DataFrame) or isinstance(y_val_df, pd.Series):
                    if isinstance(target_column, str):
                        if target_column not in y_val_df.columns:
                            raise ValueError(f"Target column '{target_column}' not found in y_val_df.")
                        y_val_df = y_val_df[[target_column]]
                    elif isinstance(target_column, int):
                        if target_column < 0 or target_column >= y_val_df.shape[1]:
                            raise ValueError(f"Target column index {target_column} is out of range in y_val_df.")
                        y_val_df = y_val_df.iloc[:, [target_column]]
                    else:
                        raise ValueError("`target_column` must be either a string (column name) or an integer index.")
                else:
                    raise ValueError("y_val_df must be a pandas DataFrame or Series to select target columns by name or index.")

            # Convert to numpy after selecting the target column
            x_val = x_val_df.to_numpy().astype(np.float32)
            y_val = y_val_df.to_numpy().astype(np.float32)

            # Ensure x_val is at least 2D
            if x_val.ndim == 1:
                x_val = x_val.reshape(-1, 1)

            # Limit the number of rows based on max_steps_test if specified
            max_steps_test = config.get('max_steps_test')
            if isinstance(max_steps_test, int) and max_steps_test > 0:
                print(f"Limiting validation data to first {max_steps_test} rows.")
                x_val = x_val[:max_steps_test]
                y_val = y_val[:max_steps_test]
                print(f"Validation data shape after limiting: X: {x_val.shape}, Y: {y_val.shape}")

            # Apply sliding window for CNN
            if config['plugin'] == 'cnn':
                if window_size is None:
                    raise ValueError("`window_size` must be specified in config for CNN plugin.")
                x_val_windowed, y_val_windowed = create_sliding_windows(x_val, y_val, window_size)
                print(f"Sliding windows created for validation: x_val_windowed shape: {x_val_windowed.shape}, y_val_windowed shape: {y_val_windowed.shape}")
                x_val = x_val_windowed
                y_val = y_val_windowed
            elif config['plugin'] == 'transformer':
                # Reshape for transformer
                if x_val.ndim == 2:
                    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
                    print(f"Reshaped x_val for transformer: {x_val.shape}")
            # No additional processing needed for other plugins

        # Train the model with or without validation data
        if config['plugin'] == 'cnn' and x_val is not None and y_val is not None:
            plugin.train(
                x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                threshold_error=threshold_error,
                x_val=x_val, 
                y_val=y_val
            )
        else:
            plugin.train(
                x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                threshold_error=threshold_error
            )

        # ----------------------------
        # SAVE THE TRAINED MODEL
        # ----------------------------
        if config.get('save_model'):
            plugin.save(config['save_model'])
            print(f"Model saved to {config['save_model']}")

        # ----------------------------
        # PREDICT ON TRAINING DATA
        # ----------------------------
        predictions = plugin.predict(x_train)

        # ----------------------------
        # EVALUATE THE MODEL
        # ----------------------------
        mse = float(plugin.calculate_mse(y_train, predictions))
        mae = float(plugin.calculate_mae(y_train, predictions))
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # ----------------------------
        # CONVERT PREDICTIONS TO DATAFRAME
        # ----------------------------
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        else:
            num_steps = predictions.shape[1]
            pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
            predictions_df = pd.DataFrame(predictions, columns=pred_cols)

        # ----------------------------
        # SAVE PREDICTIONS TO CSV
        # ----------------------------
        output_filename = config['output_file']
        write_csv(
            output_filename, 
            predictions_df, 
            include_date=config.get('force_date', False), 
            headers=config.get('headers', True)
        )
        print(f"Output written to {output_filename}")

        # ----------------------------
        # SAVE DEBUG INFO
        # ----------------------------
        end_time = time.time()
        execution_time = end_time - start_time
        debug_info = {
            'execution_time': float(execution_time),
            'mse': mse,
            'mae': mae
        }

        if config.get('save_log'):
            save_debug_info(debug_info, config['save_log'])
            print(f"Debug info saved to {config['save_log']}")

        if config.get('remote_log'):
            remote_log(
                config, 
                debug_info, 
                config['remote_log'], 
                config.get('username'), 
                config.get('password')
            )
            print(f"Debug info saved to {config['remote_log']}")

        print(f"Execution time: {execution_time} seconds")

        # ----------------------------
        # VALIDATE THE MODEL (IF VALIDATION DATA PROVIDED)
        # ----------------------------
        if config.get('x_validation_file') and config.get('y_validation_file'):
            print("Validating model...")

            x_val_df = load_csv(config['x_validation_file'], headers=config.get('headers', True))
            y_val_df = load_csv(config['y_validation_file'], headers=config.get('headers', True))

            # Conditional Target Column Selection for CNN
            if config['plugin'] == 'cnn' and target_column is not None:
                if isinstance(y_val_df, pd.DataFrame) or isinstance(y_val_df, pd.Series):
                    if isinstance(target_column, str):
                        if target_column not in y_val_df.columns:
                            raise ValueError(f"Target column '{target_column}' not found in y_val_df.")
                        y_val_df = y_val_df[[target_column]]
                    elif isinstance(target_column, int):
                        if target_column < 0 or target_column >= y_val_df.shape[1]:
                            raise ValueError(f"Target column index {target_column} is out of range in y_val_df.")
                        y_val_df = y_val_df.iloc[:, [target_column]]
                    else:
                        raise ValueError("`target_column` must be either a string (column name) or an integer index.")
                else:
                    raise ValueError("y_val_df must be a pandas DataFrame or Series to select target columns by name or index.")

            # Convert to numpy after selecting the target column
            x_val = x_val_df.to_numpy().astype(np.float32)
            y_val = y_val_df.to_numpy().astype(np.float32)

            # Ensure x_val is at least 2D
            if x_val.ndim == 1:
                x_val = x_val.reshape(-1, 1)

            # Limit the number of rows based on max_steps_test if specified
            max_steps_test = config.get('max_steps_test')
            if isinstance(max_steps_test, int) and max_steps_test > 0:
                print(f"Limiting validation data to first {max_steps_test} rows.")
                x_val = x_val[:max_steps_test]
                y_val = y_val[:max_steps_test]
                print(f"Validation data shape after limiting: X: {x_val.shape}, Y: {y_val.shape}")

            # Apply sliding window for CNN
            if config['plugin'] == 'cnn':
                if window_size is None:
                    raise ValueError("`window_size` must be specified in config for CNN plugin.")
                x_val_windowed, y_val_windowed = create_sliding_windows(x_val, y_val, window_size)
                print(f"Sliding windows created for validation: x_val_windowed shape: {x_val_windowed.shape}, y_val_windowed shape: {y_val_windowed.shape}")
                x_val = x_val_windowed
                y_val = y_val_windowed
            elif config['plugin'] == 'transformer':
                # Reshape for transformer
                if x_val.ndim == 2:
                    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
                    print(f"Reshaped x_val for transformer: {x_val.shape}")
            # No additional processing needed for other plugins

            print(f"Validation data shape after adjustments: {x_val.shape}, {y_val.shape}")

            # Predict on the validation data
            validation_predictions = plugin.predict(x_val)
            # Adjust predictions length if necessary
            validation_predictions = validation_predictions[:len(y_val)]

            # Calculate validation errors
            validation_mse = float(plugin.calculate_mse(y_val, validation_predictions))
            validation_mae = float(plugin.calculate_mae(y_val, validation_predictions))
            print(f"Validation Mean Squared Error: {validation_mse}")
            print(f"Validation Mean Absolute Error: {validation_mae}")

            # Convert validation predictions to DataFrame
            if validation_predictions.ndim == 1 or validation_predictions.shape[1] == 1:
                validation_predictions_df = pd.DataFrame(validation_predictions, columns=['Prediction'])
            else:
                val_num_steps = validation_predictions.shape[1]
                val_pred_cols = [f'Prediction_{i+1}' for i in range(val_num_steps)]
                validation_predictions_df = pd.DataFrame(validation_predictions, columns=val_pred_cols)

            # (Optional) Save or further process validation_predictions_df as needed
            # For example:
            # write_csv("validation_predictions.csv", validation_predictions_df)


def load_and_evaluate_model(config, plugin):
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
                - 'evaluate_file' (str): Path to save the evaluation predictions CSV file.
                - 'max_steps_train' (int, optional): Maximum number of rows to read for training data.

        plugin (tf.keras.Model): The machine learning model plugin to be used for evaluation.

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

    # Load and process training data with row limit
    print("Loading and processing training data for evaluation...")
    try:
        x_train, _ = process_data(config)
        print(f"Processed training data: X shape: {x_train.shape}")
    except Exception as e:
        print(f"Failed to process training data: {e}")
        sys.exit(1)

    # Predict using the loaded model
    print("Making predictions on training data...")
    try:
        predictions = plugin.predict(x_train.to_numpy())
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

    # Save predictions to CSV for evaluation
    evaluate_filename = config['evaluate_file']
    try:
        write_csv(
            file_path=evaluate_filename,
            data=predictions_df,
            include_date=config.get('force_date', False),
            headers=config.get('headers', True)
        )
        print(f"Predicted data saved to {evaluate_filename}")
    except Exception as e:
        print(f"Failed to save predictions to {evaluate_filename}: {e}")
        sys.exit(1)
