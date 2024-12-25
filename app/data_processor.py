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

    if isinstance(y_train_file, str):
        print(f"Loading y_train data from CSV file: {y_train_file}")
        y_train_data = load_csv(y_train_file, headers=config['headers'])
        print(f"y_train data loaded with shape: {y_train_data.shape}")
    else:
        raise ValueError("Either y_train_file must be specified as a string path to the CSV file.")

    # If the user specified a target_column, extract that column from y_train_data
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

    # Align x_train_data and y_train_data on their date/index
    common_index = x_train_data.index.intersection(y_train_data.index)
    x_train_data = x_train_data.loc[common_index].sort_index()
    y_train_data = y_train_data.loc[common_index].sort_index()

    # If there's no overlap at all, we won't be able to train
    if x_train_data.empty or y_train_data.empty:
        raise ValueError(
            "No overlapping dates found (or data is empty after alignment). "
            "Please ensure your CSV files have matching date ranges and that the target_column is correct."
        )

    # Convert data to numeric, fill NaNs
    x_train_data = x_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train_data = y_train_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    print(f"Applying time horizon: {time_horizon} and input offset: {input_offset}")
    total_offset = time_horizon + input_offset

    # Apply original offset logic
    y_train_data = y_train_data.iloc[total_offset:]
    x_train_data = x_train_data.iloc[:-time_horizon]

    print(f"Data shape after applying offset and time horizon: {x_train_data.shape}, {y_train_data.shape}")

    # If one of them is now empty, we won't be able to train
    if x_train_data.empty or y_train_data.empty:
        raise ValueError(
            "No data remaining after applying time_horizon and offsets. "
            "Check that your data is long enough and your offsets/time_horizon are appropriate."
        )

    # Ensure the same min length
    min_length = min(len(x_train_data), len(y_train_data))
    x_train_data = x_train_data.iloc[:min_length]
    y_train_data = y_train_data.iloc[:min_length]

    # Multi-step output creation
    Y_list = []
    for i in range(len(y_train_data) - time_horizon + 1):
        row_values = []
        for j in range(time_horizon):
            row_values.append(y_train_data.iloc[i + j].values[0])
        Y_list.append(row_values)

    # If time_horizon slicing leads to zero rows, raise an error
    if not Y_list:
        raise ValueError(
            "After creating multi-step slices, no samples remain. "
            "Check that your data is sufficient for the given time_horizon."
        )

    y_train_data = pd.DataFrame(Y_list)

    # Adjust x_train_data to match new y_train_data length
    x_train_data = x_train_data.iloc[:len(y_train_data)].reset_index(drop=True)
    y_train_data = y_train_data.reset_index(drop=True)

    print(f"Returning data of type: {type(x_train_data)}, {type(y_train_data)}")
    print(f"x_train_data shape after adjustments: {x_train_data.shape}")
    print(f"y_train_data shape after adjustments (multi-step): {y_train_data.shape}")

    return x_train_data, y_train_data




def run_prediction_pipeline(config, plugin):
    """
    Loads and processes the training data, trains the model, optionally saves it,
    generates predictions on the training data, writes them to CSV, and if validation data
    is provided, loads and processes validation data, applies the same offset/time_horizon logic,
    and evaluates the model on the validation set.
    """

    # Helper function for sinusoidal positional encoding (only used if plugin == 'transformer')
    def apply_positional_encoding(x):
        """
        Implements the standard sinusoidal positional encoding used in the original Transformer paper.
        x is assumed to be a 2D NumPy array of shape (batch_size, features).
        This returns x + positional_encoding_of_same_shape.
        """
        import numpy as np
        seq_len, d_model = x.shape
        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)

        # Generate positional encodings
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pos_encoding[pos, i] = np.sin(angle)
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(angle)

        # Add to original input
        return x + pos_encoding

    start_time = time.time()
    
    print("Running process_data...")
    x_train, y_train = process_data(config)
    print(f"Processed data received of type: {type(x_train)} and shape: {x_train.shape}")
    
    time_horizon = config['time_horizon']
    input_offset = config['input_offset']
    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold_error = config['threshold_error']

    # Ensure x_train and y_train are DataFrame or Series
    if isinstance(x_train, (pd.DataFrame, pd.Series)) and isinstance(y_train, (pd.DataFrame, pd.Series)):
        # Convert to numpy for training
        x_train = x_train.to_numpy().astype(np.float32)
        y_train = y_train.to_numpy().astype(np.float32)

        # If the plugin is 'transformer', add positional encoding to x_train
        if config['plugin'] == 'transformer':
            print("Applying positional encoding since plugin is 'transformer'.")
            x_train = apply_positional_encoding(x_train)

        # Ensure x_train is 2D
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        
        # Debug messages
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        # Train the model
        plugin.build_model(input_shape=x_train.shape[1])
        plugin.train(x_train, y_train, epochs=epochs, batch_size=batch_size, threshold_error=threshold_error)

        # Save the trained model
        if config['save_model']:
            plugin.save(config['save_model'])
            print(f"Model saved to {config['save_model']}")

        # Predict on the training data
        predictions = plugin.predict(x_train)

        # Evaluate on the training data
        mse = float(plugin.calculate_mse(y_train, predictions))
        mae = float(plugin.calculate_mae(y_train, predictions))
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # Convert predictions to a DataFrame and save to CSV
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        output_filename = config['output_file']
        write_csv(output_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
        print(f"Output written to {output_filename}")

        # Save final debug info
        end_time = time.time()
        execution_time = end_time - start_time
        debug_info = {
            'execution_time': float(execution_time),
            'mse': mse,
            'mae': mae
        }

        # Save debug info
        if config.get('save_log'):
            save_debug_info(debug_info, config['save_log'])
            print(f"Debug info saved to {config['save_log']}")

        # Remote log debug info and config
        if config.get('remote_log'):
            remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
            print(f"Debug info saved to {config['remote_log']}")

        print(f"Execution time: {execution_time} seconds")

        # Validate the model if validation data is provided
        if config['x_validation_file'] and config['y_validation_file']:
            print("Validating model...")

            # Load the validation data as DataFrames (similar to process_data, but inline)
            x_val_df = load_csv(config['x_validation_file'], headers=config['headers'])
            y_val_df = load_csv(config['y_validation_file'], headers=config['headers'])

            # If the user specified a target_column, extract that column from y_val_df
            target_column = config['target_column']
            if target_column is not None:
                if isinstance(target_column, str):
                    if target_column not in y_val_df.columns:
                        raise ValueError(f"Target column '{target_column}' not found in y_val_df.")
                    y_val_df = y_val_df[[target_column]]
                elif isinstance(target_column, int):
                    if target_column < 0 or target_column >= y_val_df.shape[1]:
                        raise ValueError(f"Target column index {target_column} is out of range in y_val_df.")
                    y_val_df = y_val_df.iloc[:, [target_column]]
                else:
                    raise ValueError("target_column must be either a string (column name) or an integer index.")

            # Align x_val_df and y_val_df by date/index
            common_index = x_val_df.index.intersection(y_val_df.index)
            x_val_df = x_val_df.loc[common_index].sort_index()
            y_val_df = y_val_df.loc[common_index].sort_index()

            # Convert data to numeric
            x_val_df = x_val_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            y_val_df = y_val_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Apply the same offset/time_horizon logic
            total_offset = time_horizon + config['input_offset']
            y_val_df = y_val_df.iloc[total_offset:]
            x_val_df = x_val_df.iloc[:-time_horizon]

            # Ensure the same min length
            min_length = min(len(x_val_df), len(y_val_df))
            x_val_df = x_val_df.iloc[:min_length]
            y_val_df = y_val_df.iloc[:min_length]

            # Multi-step processing for validation
            Y_val_list = []
            for i in range(len(y_val_df) - time_horizon + 1):
                row_values = []
                for j in range(time_horizon):
                    row_values.append(y_val_df.iloc[i + j].values[0])
                Y_val_list.append(row_values)

            y_val_df = pd.DataFrame(Y_val_list).reset_index(drop=True)
            x_val_df = x_val_df.iloc[:len(y_val_df)].reset_index(drop=True)

            # Convert validation data to numpy
            x_validation = x_val_df.to_numpy().astype(np.float32)
            y_validation = y_val_df.to_numpy().astype(np.float32)

            # If transformer plugin is used, also apply positional encoding to validation X
            if config['plugin'] == 'transformer':
                print("Applying positional encoding to validation data since plugin is 'transformer'.")
                x_validation = apply_positional_encoding(x_validation)

            # Ensure x_validation is 2D
            if x_validation.ndim == 1:
                x_validation = x_validation.reshape(-1, 1)

            print(f"Validation data shape after adjustments: {x_validation.shape}, {y_validation.shape}")

            # Predict on the validation data
            validation_predictions = plugin.predict(x_validation)

            # Adjust predictions length if necessary
            validation_predictions = validation_predictions[:len(y_validation)]

            # Calculate validation errors
            validation_mse = float(plugin.calculate_mse(y_validation, validation_predictions))
            validation_mae = float(plugin.calculate_mae(y_validation, validation_predictions))
            print(f"Validation Mean Squared Error: {validation_mse}")
            print(f"Validation Mean Absolute Error: {validation_mae}")

    else:
        print(f"Invalid data type returned: {type(x_train)}, {type(y_train)}")
        raise ValueError("Processed data is not in the correct format (DataFrame or Series).")



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