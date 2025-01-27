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
    Loads and processes both training and validation datasets (features and targets), tailored for different plugins.

    Args:
        config (dict): Configuration dictionary containing parameters for data processing.

    Returns:
        dict: Dictionary containing processed training and validation data as:
              {
                  "x_train": pd.DataFrame,
                  "y_train": pd.DataFrame,
                  "x_val": pd.DataFrame,
                  "y_val": pd.DataFrame,
              }
    """
    datasets = {}

    # Load and process training data
    print(f"Loading training data from {config['x_train_file']} and {config['y_train_file']}...")
    x_train = load_csv(config['x_train_file'], headers=config['headers'], max_rows=config.get('max_steps_train'))
    y_train = load_csv(config['y_train_file'], headers=config['headers'], max_rows=config.get('max_steps_train'))

    # Process target column for training
    target_column = config['target_column']
    if isinstance(target_column, str):
        if target_column not in y_train.columns:
            raise ValueError(f"Target column '{target_column}' not found in training target data.")
        y_train = y_train[[target_column]]
    elif isinstance(target_column, int):
        y_train = y_train.iloc[:, [target_column]]

    # Load and process validation data
    print(f"Loading validation data from {config['x_validation_file']} and {config['y_validation_file']}...")
    x_val = load_csv(config['x_validation_file'], headers=config['headers'], max_rows=config.get('max_steps_test'))
    y_val = load_csv(config['y_validation_file'], headers=config['headers'], max_rows=config.get('max_steps_test'))

    # Process target column for validation
    if isinstance(target_column, str):
        if target_column not in y_val.columns:
            raise ValueError(f"Target column '{target_column}' not found in validation target data.")
        y_val = y_val[[target_column]]
    elif isinstance(target_column, int):
        y_val = y_val.iloc[:, [target_column]]

    # Convert to numeric and fill NaNs for all datasets
    for ds_name, ds in zip(["x_train", "y_train", "x_val", "y_val"], [x_train, y_train, x_val, y_val]):
        ds_converted = ds.apply(pd.to_numeric, errors='coerce').fillna(0)
        datasets[ds_name] = ds_converted

    # Ensure indices are datetime-like for alignment
    for key in ["x_train", "y_train", "x_val", "y_val"]:
        if not isinstance(datasets[key].index, pd.DatetimeIndex):
            raise ValueError(f"Dataset '{key}' must have a valid DatetimeIndex.")

    # Align training and validation datasets
    common_train_index = datasets["x_train"].index.intersection(datasets["y_train"].index)
    common_val_index = datasets["x_val"].index.intersection(datasets["y_val"].index)
    datasets["x_train"] = datasets["x_train"].loc[common_train_index].sort_index()
    datasets["y_train"] = datasets["y_train"].loc[common_train_index].sort_index()
    datasets["x_val"] = datasets["x_val"].loc[common_val_index].sort_index()
    datasets["y_val"] = datasets["y_val"].loc[common_val_index].sort_index()

    print("Shifting y_train and y_val forward by 1 tick and forward-filling the last row...")
    # Shift forward by 1 so that the next tick is the target
    datasets["y_train"] = datasets["y_train"].shift(-1).ffill()
    datasets["y_val"]   = datasets["y_val"].shift(-1).ffill()

    # Apply time horizon and input offset
    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    total_offset = time_horizon + input_offset
    print(f"Applying time_horizon={time_horizon}, input_offset={input_offset}, total_offset={total_offset}...")

    # Drop first total_offset from y, last time_horizon from x
    # with boundary checks
    # Train
    if len(datasets["y_train"]) > total_offset:
        datasets["y_train"] = datasets["y_train"].iloc[total_offset:]
    else:
        datasets["y_train"] = datasets["y_train"].iloc[0:0]

    if len(datasets["x_train"]) > time_horizon:
        datasets["x_train"] = datasets["x_train"].iloc[:-time_horizon]
    else:
        datasets["x_train"] = datasets["x_train"].iloc[0:0]

    # Validation
    if len(datasets["y_val"]) > total_offset:
        datasets["y_val"] = datasets["y_val"].iloc[total_offset:]
    else:
        datasets["y_val"] = datasets["y_val"].iloc[0:0]

    if len(datasets["x_val"]) > time_horizon:
        datasets["x_val"] = datasets["x_val"].iloc[:-time_horizon]
    else:
        datasets["x_val"] = datasets["x_val"].iloc[0:0]

    # Check emptiness
    for key in ["x_train", "y_train", "x_val", "y_val"]:
        if datasets[key].empty:
            raise ValueError(f"Dataset '{key}' is empty after shape alignment, shift, and offset logic.")

    # Plugin-specific adjustments
    plugin_type = config.get("plugin", "").lower()

    # CNN: multi-step is handled by sliding windows in run_prediction_pipeline
    if plugin_type == "cnn":
        print("Plugin = CNN; no multi-step transform here.")
        pass

    # ANN / LSTM: multi-step transform
    elif plugin_type in ["ann", "lstm"]:
        print(f"Plugin = {plugin_type.upper()} => Creating multi-step targets with horizon={time_horizon}...")
        # Create multi-step Y
        datasets["y_train"] = create_multi_step_targets(datasets["y_train"], time_horizon)
        datasets["y_val"]   = create_multi_step_targets(datasets["y_val"],   time_horizon)

        # Adjust X to match multi-step Y length
        new_len_train = len(datasets["y_train"])
        new_len_val   = len(datasets["y_val"])
        print(f"Trimming x_train to {new_len_train}, x_val to {new_len_val} to match multi-step y shapes...")

        datasets["x_train"] = datasets["x_train"].iloc[:new_len_train].reset_index(drop=True)
        datasets["x_val"]   = datasets["x_val"].iloc[:new_len_val].reset_index(drop=True)
        datasets["y_train"] = datasets["y_train"].reset_index(drop=True)
        datasets["y_val"]   = datasets["y_val"].reset_index(drop=True)

    # Transformers: add positional encoding
    elif plugin_type == "transformers":
        print("Plugin = Transformers => Adding positional encoding columns...")

        def positional_encoding(df_pe):
            # Example: a sine-based encoding
            pos_array = np.arange(len(df_pe)).reshape(-1, 1)
            dimension = df_pe.shape[1]
            encoded = np.zeros((len(df_pe), dimension), dtype=np.float32)
            for c in range(dimension):
                encoded[:, c] = np.sin(pos_array / (10000 ** (2 * c / dimension)))
            # Build new columns
            pe_cols = [f"posenc_{i}" for i in range(dimension)]
            pe_df = pd.DataFrame(encoded, columns=pe_cols, index=df_pe.index)
            return pe_df

        x_train_pe = positional_encoding(datasets["x_train"])
        x_val_pe   = positional_encoding(datasets["x_val"])
        # Concat
        datasets["x_train"] = pd.concat([datasets["x_train"], x_train_pe], axis=1)
        datasets["x_val"]   = pd.concat([datasets["x_val"],   x_val_pe],   axis=1)

    # Final shape prints
    print(
        f"Final shapes:\n"
        f" x_train: {datasets['x_train'].shape},  y_train: {datasets['y_train'].shape}\n"
        f" x_val:   {datasets['x_val'].shape},    y_val:   {datasets['y_val'].shape}"
    )

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
    datasets = process_data(config)
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")

    # Extract time_horizon and window_size from the config
    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")

    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")

    if window_size is None and config["plugin"] == "cnn":
        raise ValueError("`window_size` must be defined in the configuration for CNN plugin.")

    print(f"Time Horizon: {time_horizon}")

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    threshold_error = config["threshold_error"]

    # Convert datasets to numpy arrays
    x_train = x_train.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    x_val = x_val.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)

    # Handle CNN-specific sliding windows
    if config["plugin"] == "cnn":
        print("Creating sliding windows for CNN...")
        x_train, y_train, _ = create_sliding_windows(x_train, y_train, window_size, time_horizon, stride=1)
        x_val, y_val, _ = create_sliding_windows(x_val, y_val, window_size, time_horizon, stride=1)

        print(f"Sliding windows created: x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")

    # Ensure x_train and x_val are at least 2D
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_val.ndim == 1:
        x_val = x_val.reshape(-1, 1)

    # Set time_horizon in plugin parameters
    plugin.set_params(time_horizon=time_horizon)

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            # Build the model
            if config["plugin"] == "cnn":
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
                y_val=y_val,
            )

            print("Evaluating trained model on training and validation data. Please wait...")

            # Suppress TensorFlow/Keras logs during prediction
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                logging.getLogger("tensorflow").setLevel(logging.FATAL)

                # Predict training data
                train_predictions = plugin.predict(x_train)

                # Predict validation data (if available)
                val_predictions = plugin.predict(x_val)

            # Restore TensorFlow logging level
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
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
        "Max": [
            np.max(training_mae_list),
            np.max(training_r2_list),
            np.max(validation_mae_list),
            np.max(validation_r2_list),
        ],
        "Min": [
            np.min(training_mae_list),
            np.min(training_r2_list),
            np.min(validation_mae_list),
            np.min(validation_r2_list),
        ],
    }

    # Save results to CSV
    results_file = config.get("results_file", "results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save final validation predictions
    final_val_file = config.get("output_file", "validation_predictions.csv")
    val_predictions_df = pd.DataFrame(val_predictions, columns=[f"Prediction_{i+1}" for i in range(val_predictions.shape[1])])
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




