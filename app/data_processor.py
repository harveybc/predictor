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
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import json

from keras.utils.vis_utils import plot_model
from tensorflow.keras.losses import Huber

def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
    """
    Create sliding windows for input data only.

    Args:
        data (np.ndarray or pd.DataFrame): Input data array of shape (n_samples, n_features).
        window_size (int): The number of time steps in each window.
        stride (int): The stride between successive windows.
        date_times (pd.DatetimeIndex, optional): Corresponding date times for each sample.

    Returns:
        If date_times is provided:
            tuple: (windows, date_time_windows) where windows is an array of shape 
                   (n_windows, window_size, n_features) and date_time_windows is a list of 
                   the DATE_TIME value corresponding to the last time step of each window.
        Otherwise:
            np.ndarray: Array of sliding windows.
    """
    windows = []
    dt_windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i: i + window_size])
        if date_times is not None:
            # Use the date corresponding to the last element in the window
            dt_windows.append(date_times[i + window_size - 1])
    if date_times is not None:
        return np.array(windows), dt_windows
    else:
        return np.array(windows)

def process_data(config):
    """
    Processes data for different plugins, including ANN, CNN, LSTM, and Transformer.

    Args:
        config (dict): Configuration dictionary with dataset paths and parameters.

    Returns:
        dict: Processed datasets for training and validation, along with corresponding DATE_TIME arrays.
    """
    # 1) LOAD CSVs
    x_train = load_csv(
        config["x_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train"),
    )
    y_train = load_csv(
        config["y_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train"),
    )
    x_val = load_csv(
        config["x_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test"),
    )
    y_val = load_csv(
        config["y_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test"),
    )

    # Save original DATE_TIME indices (if available)
    train_dates_orig = x_train.index if isinstance(x_train.index, pd.DatetimeIndex) else None
    val_dates_orig = x_val.index if isinstance(x_val.index, pd.DatetimeIndex) else None

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

    def create_multi_step_daily(y_df, horizon):
        """
        Create multi-step targets for daily predictions in time-series prediction.
        For each row in y_df, returns the predicted values for the same hour over the next `horizon` days.

        Args:
            y_df (pd.DataFrame): Target data as a DataFrame.
            horizon (int): Number of future days to predict.

        Returns:
            pd.DataFrame: Multi-step targets aligned with the input data.
        """
        blocks = []
        # For daily mode, each prediction is offset by 24 ticks (hours)
        for i in range(len(y_df) - horizon * 24):
            window = []
            for d in range(1, horizon + 1):
                # Collect the predicted value at the same hour on the d-th day ahead
                window.extend(y_df.iloc[i + d * 24].values.flatten())
            blocks.append(window)
        # Align index to the input data (exclude the last horizon*24 rows)
        return pd.DataFrame(blocks, index=y_df.index[:-horizon * 24])

    # --- UPDATED DAILY MODE ---
    if config.get("use_daily", False):
        y_train_ma = y_train.rolling(window=48, center=True, min_periods=1).mean()
        y_val_ma = y_val.rolling(window=48, center=True, min_periods=1).mean()
        y_train_multi = create_multi_step_daily(y_train_ma, time_horizon)
        y_val_multi = create_multi_step_daily(y_val_ma, time_horizon)
    else:
        y_train_multi = create_multi_step(y_train, time_horizon)
        y_val_multi = create_multi_step(y_val, time_horizon)
    # --- END UPDATED DAILY MODE ---

    # 5) TRIM x TO MATCH THE LENGTH OF y
    min_len_train = min(len(x_train), len(y_train_multi))
    x_train = x_train.iloc[:min_len_train]
    y_train_multi = y_train_multi.iloc[:min_len_train]

    min_len_val = min(len(x_val), len(y_val_multi))
    x_val = x_val.iloc[:min_len_val]
    y_val_multi = y_val_multi.iloc[:min_len_val]

    # Set initial date variables from the original DataFrame indexes
    train_dates = train_dates_orig[:min_len_train] if train_dates_orig is not None else None
    val_dates = val_dates_orig[:min_len_val] if val_dates_orig is not None else None

    # 6) LSTM-SPECIFIC PROCESSING
    if config["plugin"] == "lstm":
        print("Processing data for LSTM plugin...")

        # Ensure datasets are NumPy arrays
        if not isinstance(x_train, np.ndarray):
            x_train = x_train.to_numpy().astype(np.float32)
        if not isinstance(y_train, np.ndarray):
            y_train = y_train.to_numpy().astype(np.float32)
        if not isinstance(x_val, np.ndarray):
            x_val = x_val.to_numpy().astype(np.float32)
        if not isinstance(y_val, np.ndarray):
            y_val = y_val.to_numpy().astype(np.float32)

        window_size = config["window_size"]  # Ensure `window_size` is in the config

        if config.get("use_daily", False):
            # Define an intra-method function to create sliding windows for x only
            def create_sliding_windows_x(data, window_size, stride=1):
                """
                Create sliding windows for input data only.
                """
                windows = []
                for i in range(0, len(data) - window_size + 1, stride):
                    windows.append(data[i : i + window_size])
                return np.array(windows)

            # Create sliding windows for x_train and x_val without altering y (daily targets)
            x_train = create_sliding_windows_x(x_train, window_size, stride=1)
            x_val = create_sliding_windows_x(x_val, window_size, stride=1)

            # For daily predictions, compute new date windows using the original dates.
            if train_dates is not None:
                train_dates = [train_dates[i + window_size - 1] for i in range(0, len(train_dates) - window_size + 1)]
            if val_dates is not None:
                val_dates = [val_dates[i + window_size - 1] for i in range(0, len(val_dates) - window_size + 1)]
            # Adjust y_train_multi and y_val_multi to match the new x dimensions
            y_train_multi = y_train_multi.iloc[window_size - 1 :].to_numpy().astype(np.float32)
            y_val_multi = y_val_multi.iloc[window_size - 1 :].to_numpy().astype(np.float32)
        else:
            # For hourly predictions with window size 1, do NOT use a sliding window function.
            new_length_train = len(x_train) - time_horizon
            new_length_val = len(x_val) - time_horizon
            x_train = x_train[:new_length_train]
            x_val = x_val[:new_length_val]
            y_train_new = np.array([y_train[i+1:i+1+time_horizon] for i in range(new_length_train)])
            y_val_new = np.array([y_val[i+1:i+1+time_horizon] for i in range(new_length_val)])
            if y_train_new.ndim == 3 and y_train_new.shape[-1] == 1:
                y_train_new = np.squeeze(y_train_new, axis=-1)
            if y_val_new.ndim == 3 and y_val_new.shape[-1] == 1:
                y_val_new = np.squeeze(y_val_new, axis=-1)
            y_train_multi = y_train_new
            y_val_multi = y_val_new
            x_train = x_train.reshape(-1, 1, x_train.shape[1])
            x_val = x_val.reshape(-1, 1, x_val.shape[1])
            if train_dates is not None:
                train_dates = train_dates[:new_length_train]
            if val_dates is not None:
                val_dates = val_dates[:new_length_val]
        print(f"LSTM data shapes after sliding windows:")
        print(f"x_train: {x_train.shape}, y_train: {y_train_multi.shape}")
        print(f"x_val:   {x_val.shape}, y_val:   {y_val_multi.shape}")

    # 7) TRANSFORMER-SPECIFIC PROCESSING
    if config["plugin"] in ["transformer", "transformer_mmd"]:
        print("Processing data for Transformer plugin...")
        if not isinstance(x_train, np.ndarray):
            x_train = x_train.to_numpy().astype(np.float32)
        if not isinstance(x_val, np.ndarray):
            x_val = x_val.to_numpy().astype(np.float32)

        pos_dim = config.get("positional_encoding_dim", 16)
        num_features = x_train.shape[1]

        pos_encoding_train = generate_positional_encoding(num_features, pos_dim)
        pos_encoding_val = generate_positional_encoding(x_val.shape[1], pos_dim)

        pos_encoding_train = np.tile(pos_encoding_train, (x_train.shape[0], 1))
        pos_encoding_val = np.tile(pos_encoding_val, (x_val.shape[0], 1))

        x_train = np.concatenate([x_train, pos_encoding_train], axis=1)
        x_val = np.concatenate([x_val, pos_encoding_val], axis=1)

        print(f"Positional encoding concatenated:")
        print(f"  x_train: {x_train.shape}, y_train: {y_train_multi.shape}")
        print(f"  x_val:   {x_val.shape},   y_val: {y_val_multi.shape}")

    print("Processed datasets:")
    print(" x_train:", x_train.shape, " y_train:", y_train_multi.shape)
    print(" x_val:  ", x_val.shape, " y_val:  ", y_val_multi.shape)

    if config["plugin"] not in ["lstm", "cnn", "cnn_mmd", "transformer", "transformer_mmd"]:
        if train_dates is None:
            train_dates = y_train_multi.index
        if val_dates is None:
            val_dates = y_val_multi.index

    return {
        "x_train": x_train,
        "y_train": y_train_multi,
        "x_val": x_val,
        "y_val": y_val_multi,
        "dates_train": train_dates,
        "dates_val": val_dates,
    }



def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline using both training and validation datasets.
    Iteratively trains and evaluates the model, saving metrics and predictions.
    """
    start_time = time.time()

    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    # Lists to store metrics for all iterations
    training_mae_list, training_r2_list = [], []
    validation_mae_list, validation_r2_list = [], []

    # Load datasets
    print("Loading and processing datasets...")
    datasets = process_data(config)
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    train_dates, val_dates = datasets.get("dates_train"), datasets.get("dates_val")

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")

    # Extract key parameters
    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")
    if config["plugin"] in ["cnn", "cnn_mmd"] and window_size is None:
        raise ValueError("`window_size` must be defined for CNN plugins.")

    print(f"Time Horizon: {time_horizon}")
    batch_size, epochs = config["batch_size"], config["epochs"]
    threshold_error = config["threshold_error"]

    # Ensure NumPy arrays
    x_train, y_train = np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
    x_val, y_val = np.array(x_val, dtype=np.float32), np.array(y_val, dtype=np.float32)

    # Plugin-specific data reshaping
    if config["plugin"] in ["cnn", "cnn_mmd"]:
        print("Creating sliding windows for CNN...")
        x_train, _, train_dates = create_sliding_windows(
            x_train, y_train, window_size, time_horizon, stride=1, date_times=train_dates
        )
        x_val, _, val_dates = create_sliding_windows(
            x_val, y_val, window_size, time_horizon, stride=1, date_times=val_dates
        )
        print(f"Sliding windows created:")
        print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  x_val:   {x_val.shape},   y_val:   {y_val.shape}")

    elif config["plugin"] == "lstm":
        print("Using LSTM data from process_data (window size 1, no date shift).")
        if x_train.ndim != 3:
            raise ValueError(f"LSTM requires 3D x_train. Found: {x_train.shape}.")

    elif config["plugin"] in ["transformer", "transformer_mmd"]:
        if x_train.ndim != 2:
            raise ValueError(f"Transformer requires 2D x_train. Found: {x_train.shape}.")

    # Set train_size automatically here (the crucial addition)
    train_size = x_train.shape[0]

    # Set plugin parameters
    plugin.set_params(time_horizon=time_horizon)

    # Model training iterations
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iteration_start_time = time.time()

        # Build model with dynamically calculated train_size
        if config["plugin"] in ["cnn", "cnn_mmd"]:
            plugin.build_model(input_shape=(window_size), x_train=x_train)
        elif config["plugin"] == "lstm":
            plugin.build_model(input_shape=(x_train.shape[1]), x_train=x_train)
        else:
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train)

        # Train model
        history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions = plugin.train(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            threshold_error=threshold_error,
            x_val=x_val, y_val=y_val
        )

        # Continue rest of pipeline normally...



def load_and_evaluate_model(config, plugin):
    """
    Loads a pre-trained model and evaluates it on the validation data.

    Steps:
    1. Loads the specified pre-trained model with custom metrics.
    2. Processes validation data according to plugin requirements.
    3. Generates predictions using the loaded model.
    4. Denormalizes predictions using provided normalization parameters.
    4. Saves predictions along with DATE_TIME to CSV.

    Args:
        config (dict): Configuration parameters.
        plugin (Plugin): Predictor plugin instance.

    Raises:
        Exception: If loading model or data processing fails.
    """
    from keras.models import load_model

    # Load the trained model with custom objects
    custom_objects = {
        "combined_loss": combined_loss,
        "mmd_metric": mmd_metric,
        "huber_metric": huber_metric
    }
    try:
        plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Process validation data
    datasets = process_data(config)
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    val_dates = datasets.get("dates_val")

    # Plugin-specific adjustments
    if config["plugin"] in ["cnn", "cnn_mmd"]:
        x_val, _, val_dates = create_sliding_windows(
            x_val, y_val, config['window_size'],
            config['time_horizon'], stride=1, date_times=val_dates
        )
    if config["plugin"] == "lstm" and x_val.ndim != 3:
        raise ValueError(f"LSTM expects 3D data, found: {x_val.shape}")
    if config["plugin"] in ["transformer", "transformer_mmd"] and x_val.ndim != 2:
        raise ValueError(f"Transformer expects 2D data, found: {x_val.shape}")

    # Generate predictions
    try:
        predictions = plugin.predict(x_val)
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(1)

    # Denormalize predictions if normalization parameters provided
    norm_json = config.get("use_normalization_json")
    if norm_json:
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            min_val, max_val = norm_json["CLOSE"]["min"], norm_json["CLOSE"]["max"]
            predictions = predictions * (max_val - min_val) + min_val

    # Save predictions to DataFrame
    pred_cols = [f'Prediction_{i+1}' for i in range(predictions.shape[1])]
    predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    # Include DATE_TIME
    if val_dates is not None:
        predictions_df['DATE_TIME'] = val_dates[:len(predictions_df)]
    else:
        predictions_df['DATE_TIME'] = pd.NaT

    # Reorder columns
    predictions_df = predictions_df[['DATE_TIME'] + pred_cols]

    # Save predictions CSV
    try:
        predictions_df.to_csv(config['output_file'], index=False)
        print(f"Predictions saved: {config['output_file']}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)

    print("Model evaluation completed successfully.")



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

def generate_positional_encoding(num_features, pos_dim=16):
    """
    Generates positional encoding for a given number of features.

    Args:
        num_features (int): Number of features in the dataset.
        pos_dim (int): Dimension of the positional encoding.

    Returns:
        np.ndarray: Positional encoding of shape (1, num_features * pos_dim).
    """
    position = np.arange(num_features)[:, np.newaxis]
    div_term = np.exp(np.arange(0, pos_dim, 2) * -(np.log(10000.0) / pos_dim))
    pos_encoding = np.zeros((num_features, pos_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    pos_encoding_flat = pos_encoding.flatten().reshape(1, -1)  # Shape: (1, num_features * pos_dim)
    return pos_encoding_flat

def generate_positional_encoding(num_features, pos_dim=16):
    """
    Generates positional encoding for a given number of features.

    Args:
        num_features (int): Number of features in the dataset.
        pos_dim (int): Dimension of the positional encoding.

    Returns:
        np.ndarray: Positional encoding of shape (1, num_features * pos_dim).
    """
    position = np.arange(num_features)[:, np.newaxis]
    div_term = np.exp(np.arange(0, pos_dim, 2) * -(np.log(10000.0) / pos_dim))
    pos_encoding = np.zeros((num_features, pos_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    pos_encoding_flat = pos_encoding.flatten().reshape(1, -1)  # Shape: (1, num_features * pos_dim)
    return pos_encoding_flat


def gaussian_kernel_matrix(self, x, y, sigma):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    x_expanded = tf.reshape(x, [x_size, 1, dim])
    y_expanded = tf.reshape(y, [1, y_size, dim])
    squared_diff = tf.reduce_sum(tf.square(x_expanded - y_expanded), axis=2)
    return tf.exp(-squared_diff / (2.0 * sigma**2))

def combined_loss(y_true, y_pred):
    huber_loss = Huber(delta=1.0)(y_true, y_pred)
    sigma = 1.0
    stat_weight = 1.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
    K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
    K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
    m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
    mmd = tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
    return huber_loss + stat_weight * mmd

def mmd_metric(y_true, y_pred):
    sigma = 1.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
    K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
    K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
    m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
    return tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
mmd_metric.__name__ = "mmd"

def huber_metric(y_true, y_pred):
    return Huber(delta=1.0)(y_true, y_pred)

huber_metric.__name__ = "huber"
