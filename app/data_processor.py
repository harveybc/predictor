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





def run_prediction_pipeline(config, plugin):
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

        # Evaluate the model
        mse = float(plugin.calculate_mse(y_train, predictions))
        mae = float(plugin.calculate_mae(y_train, predictions))
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # Create a DataFrame from predictions. If multi-step (time_horizon>1),
        # we assign multiple columns. Otherwise, just one.
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        else:
            # Multi-step output
            num_steps = predictions.shape[1]
            pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
            predictions_df = pd.DataFrame(predictions, columns=pred_cols)

        # Save to CSV
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

            # Create a DataFrame from validation predictions
            if validation_predictions.ndim == 1 or validation_predictions.shape[1] == 1:
                validation_predictions_df = pd.DataFrame(validation_predictions, columns=['Prediction'])
            else:
                # Multi-step output
                val_num_steps = validation_predictions.shape[1]
                val_pred_cols = [f'Prediction_{i+1}' for i in range(val_num_steps)]
                validation_predictions_df = pd.DataFrame(validation_predictions, columns=val_pred_cols)

            # (Optional) You can save or process validation_predictions_df as needed
            # For instance:
            # write_csv("validation_predictions.csv", validation_predictions_df)
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