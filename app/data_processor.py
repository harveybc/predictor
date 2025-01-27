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
    Loads and processes both training and validation datasets (features and targets), 
    tailored for different plugins (ANN/LSTM, CNN, transformers).

    Steps:
      1) Read training & validation features (x) and targets (y).
      2) Shift y by 1 tick so next-tick becomes the current label.
      3) Forward-fill the last row(s) of y to avoid zeros.
      4) Apply time horizon + offset trimming (drop first 'total_offset' from y, 
         drop last 'time_horizon' from x).
      5) If plugin=ANN/LSTM, create multi-step (time_horizon) columns in y.
      6) If plugin=transformers, add positional encoding to x.
      7) Leave CNN x alone (sliding windows are created later in run_prediction_pipeline).
    """
    datasets = {}

    # 1) LOAD TRAINING DATA
    print(f"Loading training data from {config['x_train_file']} and {config['y_train_file']}...")
    x_train = load_csv(
        config['x_train_file'], 
        headers=config['headers'], 
        max_rows=config.get('max_steps_train')
    )
    y_train = load_csv(
        config['y_train_file'], 
        headers=config['headers'], 
        max_rows=config.get('max_steps_train')
    )

    # 2) LOAD VALIDATION DATA
    print(f"Loading validation data from {config['x_validation_file']} and {config['y_validation_file']}...")
    x_val = load_csv(
        config['x_validation_file'], 
        headers=config['headers'], 
        max_rows=config.get('max_steps_test')
    )
    y_val = load_csv(
        config['y_validation_file'], 
        headers=config['headers'], 
        max_rows=config.get('max_steps_test')
    )

    # Extract the target column
    target_column = config['target_column']

    def extract_target(df, col):
        """Extract the specified target column from DataFrame df."""
        if isinstance(col, str):
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in data.")
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("`target_column` must be str or int.")

    y_train = extract_target(y_train, target_column)
    y_val   = extract_target(y_val,   target_column)

    # Convert to numeric and fill NAs with 0
    for ds_name, ds in zip(["x_train","y_train","x_val","y_val"], [x_train,y_train,x_val,y_val]):
        ds_converted = ds.apply(pd.to_numeric, errors='coerce').fillna(0)
        datasets[ds_name] = ds_converted

    # Ensure datetime indices
    for key in ["x_train","y_train","x_val","y_val"]:
        if not isinstance(datasets[key].index, pd.DatetimeIndex):
            raise ValueError(f"Dataset '{key}' must have a valid DatetimeIndex.")

    # Align each (x, y) pair
    common_train_index = datasets["x_train"].index.intersection(datasets["y_train"].index)
    common_val_index   = datasets["x_val"].index.intersection(datasets["y_val"].index)

    datasets["x_train"] = datasets["x_train"].loc[common_train_index].sort_index()
    datasets["y_train"] = datasets["y_train"].loc[common_train_index].sort_index()
    datasets["x_val"]   = datasets["x_val"].loc[common_val_index].sort_index()
    datasets["y_val"]   = datasets["y_val"].loc[common_val_index].sort_index()

    print("Shifting y datasets by 1 tick forward and filling last row(s):")
    # SHIFT y forward by 1 => next tick is the label
    # Then forward-fill to handle trailing NaNs
    for y_name in ["y_train", "y_val"]:
        ds = datasets[y_name]
        ds_shifted = ds.shift(-1)  # shift up by 1
        ds_filled  = ds_shifted.ffill()
        datasets[y_name] = ds_filled
        print(f"  -> {y_name} shape after shift/fill: {datasets[y_name].shape}")

    # time_horizon & offset trimming
    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    total_offset = time_horizon + input_offset
    print(f"Applying time_horizon={time_horizon}, input_offset={input_offset}, total_offset={total_offset}...")

    def trim_datasets(x_df, y_df, label):
        """Drop first 'total_offset' from y_df, drop last 'time_horizon' from x_df."""
        if len(y_df) > total_offset:
            y_df = y_df.iloc[total_offset:]
        else:
            y_df = y_df.iloc[0:0]  # empty

        if len(x_df) > time_horizon:
            x_df = x_df.iloc[:-time_horizon]
        else:
            x_df = x_df.iloc[0:0]  # empty

        print(f"  -> {label}: x->({x_df.shape}), y->({y_df.shape}) after horizon+offset.")
        return x_df, y_df

    datasets["x_train"], datasets["y_train"] = trim_datasets(datasets["x_train"], datasets["y_train"], "train")
    datasets["x_val"],   datasets["y_val"]   = trim_datasets(datasets["x_val"],   datasets["y_val"],   "val")

    # Ensure none are empty now
    for key in ["x_train","y_train","x_val","y_val"]:
        if datasets[key].empty:
            raise ValueError(f"Dataset '{key}' is empty after offset/horizon logic.")

    # Plugin-specific transformations
    plugin_type = (config.get("plugin") or "").lower()

    # CNN => no multi-step transform here; done later by sliding windows
    if plugin_type == "cnn":
        print("CNN plugin => No multi-step transform in process_data().")

    # ANN / LSTM => multi-step transform for y
    elif plugin_type in ["ann", "lstm"]:
        print(f"{plugin_type.upper()} plugin => Creating multi-step targets for y with time_horizon={time_horizon}...")

        # We expect the final shape of y => (N, time_horizon)
        # Because each row will contain the next time_horizon ticks
        datasets["y_train"] = create_multi_step_targets(datasets["y_train"], time_horizon)
        datasets["y_val"]   = create_multi_step_targets(datasets["y_val"],   time_horizon)

        # Adjust x to match new y lengths
        new_len_train = len(datasets["y_train"])
        new_len_val   = len(datasets["y_val"])

        datasets["x_train"] = datasets["x_train"].iloc[:new_len_train].reset_index(drop=True)
        datasets["x_val"]   = datasets["x_val"].iloc[:new_len_val].reset_index(drop=True)
        datasets["y_train"] = datasets["y_train"].reset_index(drop=True)
        datasets["y_val"]   = datasets["y_val"].reset_index(drop=True)

        # Debug: expected shape for y is (N, time_horizon)
        print(f"  [ANN/LSTM] Expected y to have shape (N, {time_horizon}) after multi-step transform.")
        print(f"  [ANN/LSTM] Actual y_train shape: {datasets['y_train'].shape}")
        print(f"  [ANN/LSTM] Actual y_val   shape: {datasets['y_val'].shape}")
        print(f"  => x_train->({datasets['x_train'].shape}), y_train->({datasets['y_train'].shape})")
        print(f"  => x_val->({datasets['x_val'].shape}),   y_val->({datasets['y_val'].shape})")

    # Transformers => add positional encoding
    elif plugin_type == "transformers":
        print("Transformers plugin => Adding positional encoding to x...")

        def positional_encoding(df):
            """Example sine-based positional encoding; #columns match df.shape[1]."""
            n_rows = len(df)
            n_feat = df.shape[1]
            pos_array = np.arange(n_rows).reshape(-1, 1)
            encoded = np.zeros((n_rows, n_feat), dtype=np.float32)
            for c in range(n_feat):
                encoded[:, c] = np.sin(pos_array[:,0] / (10000 ** (2*c/n_feat)))
            col_names = [f"posenc_{i}" for i in range(n_feat)]
            return pd.DataFrame(encoded, columns=col_names, index=df.index)

        x_train_pe = positional_encoding(datasets["x_train"])
        x_val_pe   = positional_encoding(datasets["x_val"])
        datasets["x_train"] = pd.concat([datasets["x_train"], x_train_pe], axis=1)
        datasets["x_val"]   = pd.concat([datasets["x_val"],   x_val_pe],   axis=1)

        print(f"  => x_train with PE: {datasets['x_train'].shape}, x_val with PE: {datasets['x_val'].shape}")
    else:
        print(f"Plugin '{plugin_type}' => No additional transform applied in process_data().")

    # Final debug prints
    print("Final shapes after all transformations:")
    print(f"  x_train: {datasets['x_train'].shape} | y_train: {datasets['y_train'].shape}")
    print(f"  x_val:   {datasets['x_val'].shape}   | y_val:   {datasets['y_val'].shape}")

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
    datasets = process_data(config)  # <-- We do not modify how process_data works, just rely on it
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")

    # Extract time_horizon and window_size from config
    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")

    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")

    # Confirm CNN plugin requires window_size
    if config["plugin"] == "cnn" and window_size is None:
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

    # CNN-specific sliding windows
    if config["plugin"] == "cnn":
        print("Creating sliding windows for CNN...")
        x_train, y_train, _ = create_sliding_windows(
            x_train, y_train, window_size, time_horizon, stride=1
        )
        x_val, y_val, _ = create_sliding_windows(
            x_val, y_val, window_size, time_horizon, stride=1
        )

        print(f"Sliding windows created:")
        print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  x_val:   {x_val.shape},   y_val:   {y_val.shape}")

    # Ensure x_* are at least 2D
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_val.ndim == 1:
        x_val = x_val.reshape(-1, 1)

    # Pass time_horizon to the plugin (if it uses it)
    plugin.set_params(time_horizon=time_horizon)

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        try:
            # Build the model
            if config["plugin"] == "cnn":
                # CNN expects shape (window_size, features)
                if len(x_train.shape) < 3:
                    raise ValueError(
                        f"For CNN, x_train must be 3D. Found: {x_train.shape}."
                    )
                plugin.build_model(input_shape=(window_size, x_train.shape[2]))
            else:
                # For ANN/LSTM/Transformers: typically shape (features,)
                if len(x_train.shape) != 2:
                    raise ValueError(
                        f"Expected x_train to be 2D for {config['plugin']}. Found: {x_train.shape}."
                    )
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

                # Predict validation data
                val_predictions = plugin.predict(x_val)

            # Restore TensorFlow logs
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
            logging.getLogger("tensorflow").setLevel(logging.INFO)

            # -----------------------
            # Dimension checks
            # -----------------------
            # Expect train_predictions.shape == (num_train_samples, time_horizon)
            #  or at least have the same second dimension as y_train, etc.
            if train_predictions.shape[0] != y_train.shape[0]:
                raise ValueError(
                    "Mismatch in training samples dimension:\n"
                    f"  train_predictions.shape[0]={train_predictions.shape[0]}, "
                    f"y_train.shape[0]={y_train.shape[0]}.\n"
                    "Please ensure data alignment in multi-step logic."
                )
            if train_predictions.shape[1] != y_train.shape[1]:
                raise ValueError(
                    "Mismatch in training time_horizon dimension:\n"
                    f"  train_predictions.shape[1]={train_predictions.shape[1]}, "
                    f"y_train.shape[1]={y_train.shape[1]}.\n"
                    "Time horizon dimension must match."
                )

            # Same check for validation
            if val_predictions.shape[0] != y_val.shape[0]:
                raise ValueError(
                    "Mismatch in validation samples dimension:\n"
                    f"  val_predictions.shape[0]={val_predictions.shape[0]}, "
                    f"y_val.shape[0]={y_val.shape[0]}.\n"
                    "Please ensure data alignment for multi-step validation."
                )
            if val_predictions.shape[1] != y_val.shape[1]:
                raise ValueError(
                    "Mismatch in validation time_horizon dimension:\n"
                    f"  val_predictions.shape[1]={val_predictions.shape[1]}, "
                    f"y_val.shape[1]={y_val.shape[1]}.\n"
                    "Time horizon dimension must match for validation."
                )

            # Evaluate training metrics
            train_mae = float(plugin.calculate_mae(y_train, train_predictions))
            train_r2 = float(r2_score(y_train, train_predictions))
            print(f"Training MAE: {train_mae}")
            print(f"Training R²: {train_r2}")

            # Save training metrics
            training_mae_list.append(train_mae)
            training_r2_list.append(train_r2)

            # Evaluate validation metrics
            val_mae = float(plugin.calculate_mae(y_val, val_predictions))
            val_r2 = float(r2_score(y_val, val_predictions))
            print(f"Validation MAE: {val_mae}")
            print(f"Validation R²: {val_r2}")

            # Save validation metrics
            validation_mae_list.append(val_mae)
            validation_r2_list.append(val_r2)

            iteration_end_time = time.time()
            print(f"Iteration {iteration} completed in {iteration_end_time - iteration_start_time:.2f} seconds")

        except Exception as e:
            print(f"Iteration {iteration} failed with error:\n  {e}")
            continue  # Proceed to the next iteration even if one iteration fails

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
    # Because we are inside a loop, `val_predictions` might not exist if iteration always fails,
    # so we guard with a check:
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




