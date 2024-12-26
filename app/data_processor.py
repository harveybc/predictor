import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log

def process_data(config):
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
        raise ValueError("Either y_train_file must be specified as a string path to the CSV file.")

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
            raise ValueError("target_column must be either a string (column name) or an integer index.")
    else:
        raise ValueError("No valid target_column was provided for y_train_data.")

    # Convert to numeric, fill NaNs just in case (the load_csv might have done it, but we ensure again)
    x_train_data = x_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # At this point, load_csv() should have already parsed 'DATE_TIME' and set it as index if found.
    # But to be safe, confirm that they have an index of type datetime-like.
    # If not, we cannot align by date properly. 
    # (If you prefer a different date column name, adjust accordingly.)

    if not isinstance(x_train_data.index, pd.DatetimeIndex) or not isinstance(y_train_data.index, pd.DatetimeIndex):
        raise ValueError("Either 'DATE_TIME' column wasn't parsed as datetime or no valid DatetimeIndex found. "
                         "Ensure your CSV has a 'DATE_TIME' column or the first column is recognized as datetime.")

    # Now align by intersection of dates (this is what you want).
    common_index = x_train_data.index.intersection(y_train_data.index)
    x_train_data = x_train_data.loc[common_index].sort_index()
    y_train_data = y_train_data.loc[common_index].sort_index()

    # If no overlap, raise an error
    if x_train_data.empty or y_train_data.empty:
        raise ValueError(
            "No overlapping dates found (or data is empty after alignment). "
            "Please ensure your CSV files truly share date ranges."
        )

    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    print(f"Applying time horizon: {time_horizon} and input offset: {input_offset}")
    total_offset = time_horizon + input_offset

    # Shift y by total_offset, remove last time_horizon from x
    # e.g., we want to predict future Y
    y_train_data = y_train_data.iloc[total_offset:]
    x_train_data = x_train_data.iloc[:-time_horizon]

    print(f"Data shape after applying offset and time horizon: {x_train_data.shape}, {y_train_data.shape}")

    # If the offset leads to zero rows, error out
    if x_train_data.empty or y_train_data.empty:
        raise ValueError(
            "After applying time_horizon and offset, no samples remain. "
            "Check that your dataset is large enough and offsets/time_horizon are correct."
        )

    # Ensure the same min length
    min_length = min(len(x_train_data), len(y_train_data))
    x_train_data = x_train_data.iloc[:min_length]
    y_train_data = y_train_data.iloc[:min_length]

    # Build multi-step Y
    Y_list = []
    for i in range(len(y_train_data) - time_horizon + 1):
        row_values = []
        for j in range(time_horizon):
            row_values.append(y_train_data.iloc[i + j].values[0])
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





import numpy as np
import pandas as pd

import time
import pandas as pd
import numpy as np

def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline for various plugins (models) without CNN-specific changes.
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
    
    # Convert to numpy for training
    if isinstance(x_train, pd.DataFrame) or isinstance(x_train, pd.Series):
        x_train = x_train.to_numpy().astype(np.float32)
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy().astype(np.float32)

    # Ensure x_train is at least 2D
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    
    # Debug messages
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Build the model with the appropriate input shape
    plugin.build_model(input_shape=x_train.shape[1:])
    
    # Train the model
    plugin.train(
        x_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        threshold_error=threshold_error
    )
    
    # Save the model if required
    if config.get('save_model'):
        plugin.save(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # Predict on training data
    predictions = plugin.predict(x_train)

    # Evaluate the model
    mse = float(plugin.calculate_mse(y_train, predictions))
    mae = float(plugin.calculate_mae(y_train, predictions))
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

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
        output_filename, 
        predictions_df, 
        include_date=config.get('force_date', False), 
        headers=config.get('headers', True)
    )
    print(f"Output written to {output_filename}")

    # Save debug info
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

    # Validation (if available)
    if config.get('x_validation_file') and config.get('y_validation_file'):
        print("Validating model...")
        x_val_df = load_csv(config['x_validation_file'], headers=config.get('headers', True))
        y_val_df = load_csv(config['y_validation_file'], headers=config.get('headers', True))

        # Convert to numpy
        if isinstance(x_val_df, pd.DataFrame) or isinstance(x_val_df, pd.Series):
            x_val = x_val_df.to_numpy().astype(np.float32)
        if isinstance(y_val_df, pd.DataFrame) or isinstance(y_val_df, pd.Series):
            y_val = y_val_df.to_numpy().astype(np.float32)

        # Ensure x_val is at least 2D
        if x_val.ndim == 1:
            x_val = x_val.reshape(-1, 1)

        # Predict on validation data
        validation_predictions = plugin.predict(x_val)

        # Evaluate validation
        validation_mse = float(plugin.calculate_mse(y_val, validation_predictions))
        validation_mae = float(plugin.calculate_mae(y_val, validation_predictions))
        print(f"Validation Mean Squared Error: {validation_mse}")
        print(f"Validation Mean Absolute Error: {validation_mae}")

    else:
        print("No validation data provided.")








def load_and_evaluate_model(config, plugin):
    # Load the model
    plugin.load(config['load_model'])

    # Load the input data
    x_train, _ = process_data(config)

    # Predict using the loaded model
    predictions = plugin.predict(x_train.to_numpy())

    # Save the predictions to CSV
    evaluate_filename = config['evaluate_file']
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    write_csv(evaluate_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Predicted data saved to {evaluate_filename}")