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





import time
import pandas as pd
import numpy as np

import time
import pandas as pd
import numpy as np

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
                    raise ValueError("target_column must be either a string (column name) or an integer index.")
            else:
                raise ValueError("y_train must be a pandas DataFrame or Series to select target columns by name or index.")
        
        # Convert to numpy for training
        x_train = x_train.to_numpy().astype(np.float32)
        y_train = y_train.to_numpy().astype(np.float32)

        # Ensure x_train is at least 2D
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        
        # Debug messages
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

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
                raise ValueError("window_size must be specified in config for CNN plugin.")
            
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
                        raise ValueError("target_column must be either a string (column name) or an integer index.")
                else:
                    raise ValueError("y_val_df must be a pandas DataFrame or Series to select target columns by name or index.")
            
            # Convert to numpy after selecting the target column
            x_val = x_val_df.to_numpy().astype(np.float32)
            y_val = y_val_df.to_numpy().astype(np.float32)
            
            # Ensure x_val is at least 2D
            if x_val.ndim == 1:
                x_val = x_val.reshape(-1, 1)
            
            # Apply sliding window for CNN
            if config['plugin'] == 'cnn':
                if window_size is None:
                    raise ValueError("window_size must be specified in config for CNN plugin.")
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
                        raise ValueError("target_column must be either a string (column name) or an integer index.")
                else:
                    raise ValueError("y_val_df must be a pandas DataFrame or Series to select target columns by name or index.")
            
            # Convert to numpy after selecting the target column
            x_val = x_val_df.to_numpy().astype(np.float32)
            y_val = y_val_df.to_numpy().astype(np.float32)
            
            # Ensure x_val is at least 2D
            if x_val.ndim == 1:
                x_val = x_val.reshape(-1, 1)
            
            # Apply sliding window for CNN
            if config['plugin'] == 'cnn':
                if window_size is None:
                    raise ValueError("window_size must be specified in config for CNN plugin.")
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