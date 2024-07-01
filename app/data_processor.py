import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log

def process_data(config):
    print(f"Loading data from CSV file: {config['input_file']}")
    data = load_csv(config['input_file'], headers=config['headers'])
    print(f"Data loaded with shape: {data.shape}")

    input_timeseries = config['input_timeseries']
    if isinstance(input_timeseries, str):
        print(f"Loading input timeseries from CSV file: {input_timeseries}")
        timeseries_data = load_csv(input_timeseries, headers=config['headers'])
        print(f"Input timeseries loaded with shape: {timeseries_data.shape}")
    else:
        raise ValueError("Input timeseries must be specified as a CSV file path in the configuration.")

    # Ensure data is numeric
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    timeseries_data = timeseries_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    return data, timeseries_data

def run_prediction_pipeline(config, plugin):
    start_time = time.time()
    
    print("Running process_data...")
    input_data, timeseries_data = process_data(config)
    print("Processed data received.")
    
    time_horizon = config['time_horizon']
    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold_error = config['threshold_error']

    # Prepare data for training
    x_train = input_data[:-time_horizon].to_numpy()
    y_train = timeseries_data[time_horizon:].to_numpy().flatten()

    # Adjust the length of y_train to match x_train
    y_train = y_train[:len(x_train)]

    # Ensure x_train is a 2D array
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    
    # Train the model
    plugin.build_model(input_shape=x_train.shape[1])
    plugin.train(x_train, y_train, epochs=epochs, batch_size=batch_size, threshold_error=threshold_error)

    # Save the trained model
    if config['save_model']:
        plugin.save(config['save_model'])
        print(f"Model saved to {config['save_model']}")

    # Predict using the trained model
    predictions = plugin.predict(x_train)

    # Evaluate the model
    mse = plugin.calculate_mse(y_train, predictions)
    mae = plugin.calculate_mae(y_train, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    # Convert predictions to a DataFrame and save to CSV
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    output_filename = config['output_file']
    write_csv(output_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Output written to {output_filename}")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': execution_time,
        'mse': mse,
        'mae': mae
    }

    # Save debug info
    if config.get('save_log'):
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")

    # Remote log debug info and config
    if config.get('remote_log'):
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        print(f"Debug info saved to {config['remote_log']}.")

    print(f"Execution time: {execution_time} seconds")

def load_and_evaluate_model(config, plugin):
    # Load the model
    plugin.load(config['load_model'])

    # Load the input data
    input_data, _ = process_data(config)

    # Predict using the loaded model
    predictions = plugin.predict(input_data.to_numpy())

    # Save the predictions to CSV
    evaluate_filename = config['evaluate_file']
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    write_csv(evaluate_filename, predictions_df, include_date=config['force_date'], headers=config['headers'])
    print(f"Predicted data saved to {evaluate_filename}")
