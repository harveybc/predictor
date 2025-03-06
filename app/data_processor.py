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
    Loads and processes training, validation, and test datasets; extracts DATE_TIME information,
    and trims each pair (x and y) to their common date range so that they share the same number of rows.
    
    Returns:
        dict: Processed datasets for training, validation, and test, along with corresponding
              DATE_TIME arrays.
    """
    import pandas as pd
    # 1) LOAD CSVs for train, validation, and test
    x_train = load_csv(
        config["x_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train")
    )
    y_train = load_csv(
        config["y_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train")
    )
    x_val = load_csv(
        config["x_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_val")
    )
    y_val = load_csv(
        config["y_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_val")
    )
    x_test = load_csv(
        config["x_test_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test")
    )
    y_test = load_csv(
        config["y_test_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test")
    )
    
    # 1a) Trim each pair to their common date range (if both have a DatetimeIndex)
    if isinstance(x_train.index, pd.DatetimeIndex) and isinstance(y_train.index, pd.DatetimeIndex):
        common_train_index = x_train.index.intersection(y_train.index)
        x_train = x_train.loc[common_train_index]
        y_train = y_train.loc[common_train_index]
    if isinstance(x_val.index, pd.DatetimeIndex) and isinstance(y_val.index, pd.DatetimeIndex):
        common_val_index = x_val.index.intersection(y_val.index)
        x_val = x_val.loc[common_val_index]
        y_val = y_val.loc[common_val_index]
    if isinstance(x_test.index, pd.DatetimeIndex) and isinstance(y_test.index, pd.DatetimeIndex):
        common_test_index = x_test.index.intersection(y_test.index)
        x_test = x_test.loc[common_test_index]
        y_test = y_test.loc[common_test_index]
    
    # Save original DATE_TIME indices AFTER trimming.
    train_dates_orig = x_train.index if isinstance(x_train.index, pd.DatetimeIndex) else None
    val_dates_orig = x_val.index if isinstance(x_val.index, pd.DatetimeIndex) else None
    test_dates_orig = x_test.index if isinstance(x_test.index, pd.DatetimeIndex) else None

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
    y_test = extract_target(y_test, target_col)

    # 3) CONVERT EACH DF TO NUMERIC.
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_val = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val = y_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = y_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4) MULTI-STEP TARGETS
    time_horizon = config["time_horizon"]
    def create_multi_step(y_df, horizon):
        blocks = []
        for i in range(len(y_df) - horizon):
            window = list(y_df.iloc[i + 1: i + 1 + horizon].values.flatten())
            blocks.append(window)
        return pd.DataFrame(blocks, index=y_df.index[:-horizon])
    def create_multi_step_daily(y_df, horizon):
        blocks = []
        for i in range(len(y_df) - horizon * 24):
            window = []
            for d in range(1, horizon + 1):
                window.extend(list(y_df.iloc[i + d * 24].values.flatten()))
            blocks.append(window)
        return pd.DataFrame(blocks, index=y_df.index[:-horizon * 24])
    if config.get("use_daily", False):
        y_train_ma = y_train.rolling(window=48, center=True, min_periods=1).mean()
        y_val_ma = y_val.rolling(window=48, center=True, min_periods=1).mean()
        y_test_ma = y_test.rolling(window=48, center=True, min_periods=1).mean()
        y_train_multi = create_multi_step_daily(y_train_ma, time_horizon)
        y_val_multi = create_multi_step_daily(y_val_ma, time_horizon)
        y_test_multi = create_multi_step_daily(y_test_ma, time_horizon)
    else:
        y_train_multi = create_multi_step(y_train, time_horizon)
        y_val_multi = create_multi_step(y_val, time_horizon)
        y_test_multi = create_multi_step(y_test, time_horizon)

    # 5) TRIM x TO MATCH THE LENGTH OF y (for each dataset)
    min_len_train = min(len(x_train), len(y_train_multi))
    x_train = x_train.iloc[:min_len_train]
    y_train_multi = y_train_multi.iloc[:min_len_train]
    min_len_val = min(len(x_val), len(y_val_multi))
    x_val = x_val.iloc[:min_len_val]
    y_val_multi = y_val_multi.iloc[:min_len_val]
    min_len_test = min(len(x_test), len(y_test_multi))
    x_test = x_test.iloc[:min_len_test]
    y_test_multi = y_test_multi.iloc[:min_len_test]
    
    train_dates = train_dates_orig[:min_len_train] if train_dates_orig is not None else None
    val_dates = val_dates_orig[:min_len_val] if val_dates_orig is not None else None
    test_dates = test_dates_orig[:min_len_test] if test_dates_orig is not None else None

    # 6) PER-PLUGIN PROCESSING
    if config["plugin"] == "lstm":
        print("Processing data for LSTM plugin...")
        x_train = x_train.to_numpy().astype(np.float32)
        x_val = x_val.to_numpy().astype(np.float32)
        x_test = x_test.to_numpy().astype(np.float32)
        window_size = config["window_size"]
        # Create sliding windows for x only
        def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
            windows = []
            dt_windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
                if date_times is not None:
                    dt_windows.append(date_times[i+window_size-1])
            return np.array(windows), dt_windows if date_times is not None else np.array(windows)
        x_train, train_dates = create_sliding_windows_x(x_train, window_size, stride=1, date_times=train_dates)
        x_val, val_dates = create_sliding_windows_x(x_val, window_size, stride=1, date_times=val_dates)
        x_test, test_dates = create_sliding_windows_x(x_test, window_size, stride=1, date_times=test_dates)
        y_train_multi = y_train_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_val_multi = y_val_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_test_multi = y_test_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
    elif config["plugin"] in ["cnn", "cnn_mmd"]:
        print("Processing data for CNN plugin using sliding windows...")
        def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
            windows = []
            dt_windows = []
            for i in range(0, len(data) - window_size + 1, stride):
                windows.append(data[i:i+window_size])
                if date_times is not None:
                    dt_windows.append(date_times[i+window_size-1])
            return np.array(windows), dt_windows if date_times is not None else np.array(windows)
        x_train, train_dates = create_sliding_windows_x(x_train, config['window_size'], stride=1, date_times=train_dates)
        x_val, val_dates = create_sliding_windows_x(x_val, config['window_size'], stride=1, date_times=val_dates)
        x_test, test_dates = create_sliding_windows_x(x_test, config['window_size'], stride=1, date_times=test_dates)
        # Trim target arrays so that their lengths match the sliding-windowed x arrays.
        trim_amount = config['window_size'] - 1
        y_train_multi = y_train_multi.to_numpy().astype(np.float32)[trim_amount:]
        y_val_multi = y_val_multi.to_numpy().astype(np.float32)[trim_amount:]
        y_test_multi = y_test_multi.to_numpy().astype(np.float32)[trim_amount:]
    elif config["plugin"] in ["transformer", "transformer_mmd"]:
        x_train = x_train.to_numpy().astype(np.float32)
        x_val = x_val.to_numpy().astype(np.float32)
        x_test = x_test.to_numpy().astype(np.float32)
        pos_dim = config.get("positional_encoding_dim", 16)
        num_features = x_train.shape[1]
        pos_encoding_train = generate_positional_encoding(num_features, pos_dim)
        pos_encoding_val = generate_positional_encoding(x_val.shape[1], pos_dim)
        pos_encoding_test = generate_positional_encoding(x_test.shape[1], pos_dim)
        pos_encoding_train = np.tile(pos_encoding_train, (x_train.shape[0], 1))
        pos_encoding_val = np.tile(pos_encoding_val, (x_val.shape[0], 1))
        pos_encoding_test = np.tile(pos_encoding_test, (x_test.shape[0], 1))
        x_train = np.concatenate([x_train, pos_encoding_train], axis=1)
        x_val = np.concatenate([x_val, pos_encoding_val], axis=1)
        x_test = np.concatenate([x_test, pos_encoding_test], axis=1)
    # For other plugins, data remains unchanged.
    
    print("Processed datasets:")
    print(" x_train:", x_train.shape, " y_train:", y_train_multi.shape)
    print(" x_val:  ", x_val.shape, " y_val:  ", y_val_multi.shape)
    print(" x_test: ", x_test.shape, " y_test: ", y_test_multi.shape)
    
    return {
        "x_train": x_train,
        "y_train": y_train_multi,
        "x_val": x_val,
        "y_val": y_val_multi,
        "x_test": x_test,
        "y_test": y_test_multi,
        "dates_train": train_dates,
        "dates_val": val_dates,
        "dates_test": test_dates,
    }


def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline using training, validation, and test datasets.
    Iteratively trains and evaluates the model with 5-fold cross-validation,
    saves metrics and predictions, and prints/exports test evaluation metrics.
    """
    import time
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.model_selection import TimeSeriesSplit
    import matplotlib.pyplot as plt
    start_time = time.time()

    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    # Lists to store metrics for all iterations
    training_mae_list = []
    training_r2_list = []
    validation_mae_list = []
    validation_r2_list = []
    test_mae_list = []
    test_r2_list = []

    print("Loading and processing datasets...")
    datasets = process_data(config)
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    x_test, y_test = datasets["x_test"], datasets["y_test"]
    train_dates = datasets.get("dates_train")
    val_dates = datasets.get("dates_val")
    test_dates = datasets.get("dates_test")

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"Test data shapes: x_test: {x_test.shape}, y_test: {y_test.shape}")

    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")
    if config["plugin"] in ["cnn", "cnn_mmd"] and window_size is None:
        raise ValueError("`window_size` must be defined in the configuration for CNN plugins.")

    print(f"Time Horizon: {time_horizon}")
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    threshold_error = config["threshold_error"]

    # Ensure data are NumPy arrays
    for var_name in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        arr = locals()[var_name]
        if isinstance(arr, pd.DataFrame):
            locals()[var_name] = arr.to_numpy().astype(np.float32)

    if config["plugin"] in ["cnn", "cnn_mmd"]:
        if x_train.ndim != 3:
            raise ValueError(f"For CNN plugins, x_train must be 3D. Found: {x_train.shape}.")
        print("Using pre-processed sliding windows for CNN (no reprocessing).")

    plugin.set_params(time_horizon=time_horizon)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iter_start_time = time.time()
        if config["plugin"] in ["cnn", "cnn_mmd"]:
            plugin.build_model(input_shape=(window_size, x_train.shape[2]), config=config)
        elif config["plugin"] == "lstm":
            plugin.build_model(input_shape=(x_train.shape[1], x_train.shape[2]), config=config)
        elif config["plugin"] in ["transformer", "transformer_mmd"]:
            plugin.build_model(input_shape=x_train.shape[1], config=config)
        else:
            if len(x_train.shape) != 2:
                raise ValueError(f"Expected x_train to be 2D for {config['plugin']}. Found: {x_train.shape}.")
            plugin.build_model(input_shape=x_train.shape[1], config=config)
        
        history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions = plugin.train(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            threshold_error=threshold_error,
            x_val=x_val,
            y_val=y_val,
            config=config
        )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss for ' + f"{config['plugin'].upper()} - {iteration}")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(config['loss_plot_file'])
        plt.close()
        print(f"Loss plot saved to {config['loss_plot_file']}")

        print("*************************************************")
        print(f"Iteration {iteration} completed.")
        print(f"Training MAE: {train_mae}")
        print(f"Training R²: {train_r2}")
        print(f"Validation MAE: {val_mae}")
        print(f"Validation R²: {val_r2}")
        print("*************************************************")
        
        training_mae_list.append(train_mae)
        training_r2_list.append(train_r2)
        validation_mae_list.append(val_mae)
        validation_r2_list.append(val_r2)
        
        iter_end_time = time.time()
        print(f"Iteration {iteration} completed in {iter_end_time - iter_start_time:.2f} seconds")
    
    # Evaluate on test dataset
    print("\nEvaluating on test dataset...")
    test_predictions = plugin.predict(x_test)
    from sklearn.metrics import r2_score
    test_mae = np.mean(np.abs(test_predictions - y_test[:x_test.shape[0]]))
    test_r2 = r2_score(y_test[:x_test.shape[0]], test_predictions)
    print("*************************************************")
    print(f"Test MAE: {test_mae}")
    print(f"Test R²: {test_r2}")
    print("*************************************************")
    test_mae_list.append(test_mae)
    test_r2_list.append(test_r2)
    
    # Save test predictions to CSV, including DATE_TIME if available.
    test_output_file = config.get("output_test_file", "test_predictions.csv")
    test_predictions_df = pd.DataFrame(test_predictions, columns=[f"Prediction_{i+1}" for i in range(test_predictions.shape[1])])
    if test_dates is not None:
        test_predictions_df.insert(0, "DATE_TIME", pd.Series(test_dates[:len(test_predictions_df)]))
    else:
        test_predictions_df['DATE_TIME'] = pd.NaT
    cols = ['DATE_TIME'] + [col for col in test_predictions_df.columns if col != 'DATE_TIME']
    test_predictions_df = test_predictions_df[cols]
    test_predictions_df.to_csv(test_output_file, index=False)
    print(f"Test predictions saved to {test_output_file}")
    
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")



def load_and_evaluate_model(config, plugin):
    """
    Loads a pre-trained model and evaluates it on the validation data.

    This function performs the following steps:
    1. Loads the specified pre-trained model.
    2. Loads and processes the validation data.
    3. Makes predictions using the loaded model.
    4. Denormalizes predictions if a normalization JSON is provided.
    5. Saves the predictions to a CSV file for evaluation, including the DATE_TIME column.

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
                - 'use_normalization_json' (str or dict, optional): Path to a JSON file or dictionary for normalization parameters.
        plugin (Plugin): The ANN predictor plugin to be used for evaluation.

    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        Exception: Propagates any exception that occurs during model loading or data processing.
    """
    # Load the pre-trained model using custom_objects for the custom loss and metrics.
    print(f"Loading pre-trained model from {config['load_model']}...")
    try:
        from keras.models import load_model
        custom_objects = {
            "combined_loss": combined_loss,
            "mmd": mmd_metric,
            "huber": huber_metric
        }
        plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model from {config['load_model']}: {e}")
        sys.exit(1)

    # Load and process validation data with row limit
    print("Loading and processing validation data for evaluation...")
    try:
        datasets = process_data(config)
        x_val = datasets["x_val"]
        y_val = datasets["y_val"]
        val_dates = datasets.get("dates_val")
        # For CNN plugins, create sliding windows and update dates accordingly
        if config["plugin"] in ["cnn", "cnn_mmd"]:
            print("Creating sliding windows for CNN...")
            x_val, _, val_date_windows = create_sliding_windows(
                x_val, y_val, config['window_size'], config['time_horizon'], stride=1, date_times=val_dates
            )
            val_dates = val_date_windows
            print(f"Sliding windows created:")
            print(f"  x_val:   {x_val.shape},   y_val:   {y_val.shape}")
        if config["plugin"] == "lstm":
            print("Using LSTM data from process_data (window size 1, no date shift).")
            if x_val.ndim != 3:
                raise ValueError(f"For LSTM, x_val must be 3D. Found: {x_val.shape}.")
        if config["plugin"] in ["transformer", "transformer_mmd"]:
            if x_val.ndim != 2:
                raise ValueError(f"For Transformer plugins, x_val must be 2D. Found: {x_val.shape}.")
        print(f"Processed validation data: X shape: {x_val.shape}, Y shape: {y_val.shape}")
    except Exception as e:
        print(f"Failed to process validation data: {e}")
        sys.exit(1)

    # Predict using the loaded model
    print("Making predictions on validation data...")
    try:
        # If x_val is already a NumPy array, pass it directly.
        x_val_array = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()
        predictions = plugin.predict(x_val_array)
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        sys.exit(1)

    # Denormalize predictions if a normalization JSON is provided.
    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
            except Exception as e:
                print(f"Error loading normalization JSON from {norm_json}: {e}")
                norm_json = {}
        elif not isinstance(norm_json, dict):
            print("Error: Normalization JSON is not a valid dictionary.")
            norm_json = {}
        # If "CLOSE" exists in the normalization JSON, denormalize predictions.
        if "CLOSE" in norm_json:
            min_val = norm_json["CLOSE"].get("min", 0)
            max_val = norm_json["CLOSE"].get("max", 1)
            predictions = predictions * (max_val - min_val) + min_val

    # Convert predictions to DataFrame
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    else:
        num_steps = predictions.shape[1]
        pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
        predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    # Add DATE_TIME column using the stored dates if available
    if val_dates is not None:
        predictions_df['DATE_TIME'] = pd.Series(val_dates[:len(predictions_df)])
    else:
        predictions_df['DATE_TIME'] = pd.NaT
        print("Warning: DATE_TIME for validation predictions not captured.")

    # Rearrange columns to have DATE_TIME first
    cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
    predictions_df = predictions_df[cols]

    # Save predictions to CSV for evaluation
    evaluate_filename = config['output_file']
    try:
        write_csv(
            file_path=evaluate_filename,
            data=predictions_df,
            include_date=False,
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
