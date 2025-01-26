import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
import sys
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from sklearn.metrics import r2_score  # Ensure sklearn is imported at the top


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


def create_sliding_windows(x, y, window_size, date_times=None, step=1):
    """
    Creates sliding windows from the dataset along with corresponding DATE_TIME indices.

    Parameters:
        x (numpy.ndarray): Input features of shape (N, features).
        y (numpy.ndarray): Targets of shape (N, time_horizon).
        window_size (int): Number of time steps in each window.
        date_times (pd.DatetimeIndex, optional): Datetime indices corresponding to each sample.
        step (int): Step size between windows.

    Returns:
        Tuple of numpy.ndarrays and list: (x_windows, y_windows, date_time_windows)
    """
    x_windows = []
    y_windows = []
    date_time_windows = []
    
    for i in range(0, len(x) - window_size - y.shape[1] + 1, step):
        x_windows.append(x[i:i + window_size])
        y_windows.append(y[i + window_size:i + window_size + y.shape[1]].flatten())
        if date_times is not None:
            # Assign the DATE_TIME corresponding to the last step in the y window
            date_time_windows.append(date_times[i + window_size + y.shape[1] - 1])
    
    return np.array(x_windows), np.array(y_windows), date_time_windows



def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline with conditional data reshaping for different plugins.
    Ensures row-limiting and displays both Training and Validation MAE and R² with separators.
    Implements multiple iterations and aggregates MAE and R² statistics.
    Saves the aggregated statistics to a CSV file specified by config['results_file'].

    Args:
        config (dict): Configuration dictionary containing parameters for the pipeline.
        plugin (Plugin): The ANN predictor plugin to be used for training and prediction.
    """
    start_time = time.time()

    iterations = config.get('iterations', 1)
    print(f"Number of iterations: {iterations}")

    # Lists to store MAE and R² values for each iteration
    training_mae_list = []
    training_r2_list = []
    validation_mae_list = []
    validation_r2_list = []

    # Set 'time_horizon' in plugin params
    time_horizon = config.get('time_horizon', 1)
    plugin.set_params(time_horizon=time_horizon)

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            print("Running process_data...")
            x_train, y_train = process_data(config)
            print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")

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
                x_train_np = x_train.to_numpy().astype(np.float32)
                y_train_np = y_train.to_numpy().astype(np.float32)

                # Ensure x_train is at least 2D
                if x_train_np.ndim == 1:
                    x_train_np = x_train_np.reshape(-1, 1)

                # Debug messages
                print(f"x_train shape before sliding window: {x_train_np.shape}")
                print(f"y_train shape before sliding window: {y_train_np.shape}")

                # ----------------------------
                # CONDITIONAL RESHAPE FOR TRANSFORMER
                # ----------------------------
                if config['plugin'] == 'transformer':
                    # Treat each feature as a separate timestep
                    # Reshape from (N, features) to (N, features, 1)
                    if x_train_np.ndim == 2:
                        x_train_np = x_train_np.reshape((x_train_np.shape[0], x_train_np.shape[1], 1))
                        print(f"Reshaped x_train for transformer: {x_train_np.shape}")

                    # Now we pass a 3D tuple: (samples, seq_len, num_features)
                    plugin.build_model(input_shape=x_train_np.shape[1:])

                elif config['plugin'] == 'cnn':
                    # Apply sliding window
                    if window_size is None:
                        raise ValueError("`window_size` must be specified in config for CNN plugin.")

                    # Capture DATE_TIME from x_train if available
                    date_times_train = x_train.index if isinstance(x_train, pd.DataFrame) else None

                    # Create sliding windows
                    x_train_windowed, y_train_windowed, date_time_train_windows = create_sliding_windows(
                        x_train_np, y_train_np, window_size, date_times=date_times_train
                    )
                    print(f"Sliding windows created: x_train_windowed shape: {x_train_windowed.shape}, y_train_windowed shape: {y_train_windowed.shape}")

                    # Update plugin's window_size parameter if necessary
                    plugin.set_params(window_size=window_size)

                    # Build model with window_size
                    plugin.build_model(input_shape=x_train_windowed.shape[1:])

                    # Replace original x_train and y_train with windowed data
                    x_train_np = x_train_windowed
                    y_train_np = y_train_windowed

                else:
                    # Handle ANN separately to pass integer input_shape
                    if config['plugin'] == 'ann':
                        input_shape = x_train_np.shape[1]  # Pass integer for ANN
                        print(f"ANN input_shape: {input_shape}")
                    else:
                        input_shape = x_train_np.shape[1:]
                        print(f"{config['plugin'].capitalize()} input_shape: {input_shape}")

                    plugin.build_model(input_shape=input_shape)

                # ----------------------------
                # TRAIN THE MODEL
                # ----------------------------
                # Handle validation data if available
                x_val_np = None
                y_val_np = None
                date_time_val_windows = []  # To store DATE_TIME for validation predictions
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
                    x_val_np = x_val_df.to_numpy().astype(np.float32)
                    y_val_np = y_val_df.to_numpy().astype(np.float32)

                    # Ensure x_val is at least 2D
                    if x_val_np.ndim == 1:
                        x_val_np = x_val_np.reshape(-1, 1)

                    # Limit the number of rows based on max_steps_test if specified
                    max_steps_test = config.get('max_steps_test')
                    if isinstance(max_steps_test, int) and max_steps_test > 0:
                        print(f"Limiting validation data to first {max_steps_test} rows.")
                        x_val_np = x_val_np[:max_steps_test]
                        y_val_np = y_val_np[:max_steps_test]
                        print(f"Validation data shape after limiting: X: {x_val_np.shape}, Y: {y_val_np.shape}")

                    # Apply sliding window for CNN
                    if config['plugin'] == 'cnn':
                        if window_size is None:
                            raise ValueError("`window_size` must be specified in config for CNN plugin.")
                        date_times_val = x_val_df.index if isinstance(x_val_df, pd.DataFrame) else None
                        x_val_windowed, y_val_windowed, date_time_val_windows = create_sliding_windows(
                            x_val_np, y_val_np, window_size, date_times=date_times_val
                        )
                        print(f"Sliding windows created for validation: x_val_windowed shape: {x_val_windowed.shape}, y_val_windowed shape: {y_val_windowed.shape}")
                        x_val_np = x_val_windowed
                        y_val_np = y_val_windowed
                    elif config['plugin'] == 'transformer':
                        # Reshape for transformer
                        if x_val_np.ndim == 2:
                            x_val_np = x_val_np.reshape((x_val_np.shape[0], x_val_np.shape[1], 1))
                            print(f"Reshaped x_val for transformer: {x_val_np.shape}")
                    else:
                        # **New Addition:** Multi-step slicing for non-CNN plugins
                        # Ensure y_val has the same multi-step horizons as y_train
                        if time_horizon > 1:
                            print("Applying multi-step slicing to validation targets...")
                            Y_val_list = []
                            date_time_val_list = []
                            for i in range(len(y_val_np) - time_horizon + 1):
                                row_values = [y_val_np[i + j][0] for j in range(time_horizon)]
                                Y_val_list.append(row_values)
                                # Assign DATE_TIME corresponding to the last step in the time horizon
                                if isinstance(x_val_df.index, pd.DatetimeIndex):
                                    date_time = x_val_df.index[i + time_horizon - 1]
                                else:
                                    date_time = pd.NaT  # Assign Not-a-Time if index is not datetime
                                date_time_val_list.append(date_time)
                            if not Y_val_list:
                                raise ValueError(
                                    "After creating multi-step slices, no validation samples remain. "
                                    "Check that your validation data is sufficient for the given time_horizon."
                                )
                            y_val_np = np.array(Y_val_list)
                            x_val_np = x_val_np[:len(y_val_np)]  # Adjust x_val_np accordingly
                            date_time_val_windows = date_time_val_list
                            print(f"Validation data shape after multi-step slicing: X: {x_val_np.shape}, Y: {y_val_np.shape}")

                # Train the model with or without validation data
                if config['plugin'] == 'cnn' and x_val_np is not None and y_val_np is not None:
                    plugin.train(
                        x_train=x_train_np, 
                        y_train=y_train_np, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        threshold_error=threshold_error,
                        x_val=x_val_np, 
                        y_val=y_val_np
                    )
                else:
                    plugin.train(
                        x_train=x_train_np, 
                        y_train=y_train_np, 
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
                predictions = plugin.predict(x_train_np)

                # ----------------------------
                # EVALUATE THE MODEL
                # ----------------------------
                mse = float(plugin.calculate_mse(y_train_np, predictions))
                mae = float(plugin.calculate_mae(y_train_np, predictions))
                try:
                    r2 = float(plugin.calculate_r2(y_train_np, predictions))
                except AttributeError:
                    # If the plugin does not have calculate_r2, use sklearn
                    r2 = float(r2_score(y_train_np, predictions))
                print(f"Training Mean Squared Error: {mse}")
                print(f"Training Mean Absolute Error: {mae}")
                print(f"Training R² Score: {r2}")

                # Append MAE and R² to lists
                training_mae_list.append(mae)
                training_r2_list.append(r2)

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
                # SAVE PREDICTIONS TO CSV (Training Predictions Removed)
                # ----------------------------
                # Removed saving training predictions as per requirement to save validation predictions only

                # ----------------------------
                # SAVE DEBUG INFO
                # ----------------------------
                end_time_iteration = time.time()
                execution_time_iteration = end_time_iteration - iteration_start_time
                debug_info = {
                    'iteration': iteration,
                    'execution_time': float(execution_time_iteration),
                    'training_mse': mse,
                    'training_mae': mae,
                    'training_r2': r2
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

                print(f"Iteration {iteration} execution time: {execution_time_iteration} seconds")

                # ----------------------------
                # VALIDATE THE MODEL (IF VALIDATION DATA PROVIDED)
                # ----------------------------
                if config.get('x_validation_file') and config.get('y_validation_file'):
                    print("Validating model...")

                    # Predict on the validation data
                    validation_predictions = plugin.predict(x_val_np)
                    # Adjust predictions length if necessary
                    validation_predictions = validation_predictions[:len(y_val_np)]

                    # Calculate validation errors
                    validation_mse = float(plugin.calculate_mse(y_val_np, validation_predictions))
                    validation_mae = float(plugin.calculate_mae(y_val_np, validation_predictions))
                    try:
                        validation_r2 = float(plugin.calculate_r2(y_val_np, validation_predictions))
                    except AttributeError:
                        # If the plugin does not have calculate_r2, use sklearn
                        validation_r2 = float(r2_score(y_val_np, validation_predictions))
                    print(f"Validation Mean Squared Error: {validation_mse}")
                    print(f"Validation Mean Absolute Error: {validation_mae}")
                    print(f"Validation R² Score: {validation_r2}")

                    # Append Validation MAE and R² to lists
                    validation_mae_list.append(validation_mae)
                    validation_r2_list.append(validation_r2)

                    # ----------------------------
                    # CONVERT VALIDATION PREDICTIONS TO DATAFRAME WITH DATE_TIME
                    # ----------------------------
                    if validation_predictions.ndim == 1 or validation_predictions.shape[1] == 1:
                        validation_predictions_df = pd.DataFrame(validation_predictions, columns=['Prediction'])
                    else:
                        num_steps_val = validation_predictions.shape[1]
                        pred_cols_val = [f'Prediction_{i+1}' for i in range(num_steps_val)]
                        validation_predictions_df = pd.DataFrame(validation_predictions, columns=pred_cols_val)

                    # Add DATE_TIME column from date_time_val_windows
                    if date_time_val_windows:
                        validation_predictions_df['DATE_TIME'] = date_time_val_windows
                    else:
                        # If DATE_TIME wasn't captured, assign NaT
                        validation_predictions_df['DATE_TIME'] = pd.NaT
                        print("Warning: DATE_TIME for validation predictions not captured.")

                    # Rearrange columns to have DATE_TIME first
                    cols_val = ['DATE_TIME'] + [col for col in validation_predictions_df.columns if col != 'DATE_TIME']
                    validation_predictions_df = validation_predictions_df[cols_val]

                    # ----------------------------
                    # SAVE VALIDATION PREDICTIONS TO CSV
                    # ----------------------------
                    output_filename = config['output_file']
                    try:
                        write_csv(
                            file_path=output_filename, 
                            data=validation_predictions_df, 
                            include_date=False,  # DATE_TIME is already included
                            headers=config.get('headers', True)
                        )
                        print(f"Validation predictions with DATE_TIME saved to {output_filename}")
                    except Exception as e:
                        print(f"Failed to save validation predictions to {output_filename}: {e}")
                        raise e  # Re-raise to handle in the outer try-except

                    # ----------------------------
                    # PRINT TRAINING AND VALIDATION MAE AND R² WITH SEPARATORS
                    # ----------------------------
                    print("***************************")
                    print(f"Training MAE = {mae}")
                    print(f"Training R² = {r2}")
                    print("***************************")
                    print(f"Validation MAE = {validation_mae}")
                    print(f"Validation R² = {validation_r2}")
                    print("***************************")

        except Exception as e:
            print(f"Iteration {iteration} failed: {e}")
            continue  # Proceed to the next iteration

    # After all iterations, compute aggregated MAE and R² statistics
    if iterations > 0 and training_mae_list and validation_mae_list and training_r2_list and validation_r2_list:
        training_mae_array = np.array(training_mae_list)
        training_r2_array = np.array(training_r2_list)
        validation_mae_array = np.array(validation_mae_list)
        validation_r2_array = np.array(validation_r2_list)

        avg_training_mae = np.mean(training_mae_array)
        std_training_mae = np.std(training_mae_array)
        max_training_mae = np.max(training_mae_array)
        min_training_mae = np.min(training_mae_array)

        avg_training_r2 = np.mean(training_r2_array)
        std_training_r2 = np.std(training_r2_array)
        max_training_r2 = np.max(training_r2_array)
        min_training_r2 = np.min(training_r2_array)

        avg_validation_mae = np.mean(validation_mae_array)
        std_validation_mae = np.std(validation_mae_array)
        max_validation_mae = np.max(validation_mae_array)
        min_validation_mae = np.min(validation_mae_array)

        avg_validation_r2 = np.mean(validation_r2_array)
        std_validation_r2 = np.std(validation_r2_array)
        max_validation_r2 = np.max(validation_r2_array)
        min_validation_r2 = np.min(validation_r2_array)

        # Print aggregated statistics with separators
        print("\n***********************************")
        print("Aggregated MAE and R² Statistics After All Iterations:")
        print("***********************************")
        print(f"Average Training MAE: {avg_training_mae}")
        print(f"Training MAE Std Dev: {std_training_mae}")
        print(f"Training MAE Max: {max_training_mae}")
        print(f"Training MAE Min: {min_training_mae}")
        print(f"Average Training R²: {avg_training_r2}")
        print(f"Training R² Std Dev: {std_training_r2}")
        print(f"Training R² Max: {max_training_r2}")
        print(f"Training R² Min: {min_training_r2}")
        print("***********************************")
        print(f"Average Validation MAE: {avg_validation_mae}")
        print(f"Validation MAE Std Dev: {std_validation_mae}")
        print(f"Validation MAE Max: {max_validation_mae}")
        print(f"Validation MAE Min: {min_validation_mae}")
        print(f"Average Validation R²: {avg_validation_r2}")
        print(f"Validation R² Std Dev: {std_validation_r2}")
        print(f"Validation R² Max: {max_validation_r2}")
        print(f"Validation R² Min: {min_validation_r2}")
        print("***********************************")

        # Save aggregated statistics to results_file in CSV format
        results = {
            'Metric': ['Training MAE', 'Training R²', 'Validation MAE', 'Validation R²'],
            'Average': [avg_training_mae, avg_training_r2, avg_validation_mae, avg_validation_r2],
            'Std Dev': [std_training_mae, std_training_r2, std_validation_mae, std_validation_r2],
            'Max': [max_training_mae, max_training_r2, max_validation_mae, max_validation_r2],
            'Min': [min_training_mae, min_training_r2, min_validation_mae, min_validation_r2]
        }
        results_df = pd.DataFrame(results)
        results_file = config.get('results_file', 'results.csv')
        try:
            results_df.to_csv(results_file, index=False)
            print(f"Aggregated statistics saved to {results_file}")
        except Exception as e:
            print(f"Failed to save aggregated statistics to {results_file}: {e}")
    else:
        print("\n***********************************")
        print("No valid MAE and R² statistics to display.")
        print("***********************************")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time for all iterations: {execution_time} seconds")


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



