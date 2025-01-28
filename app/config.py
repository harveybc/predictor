# config.py

DEFAULT_VALUES = {
    #'x_train_file': './tests/data/encoder_eval_d2.csv',
    'x_train_file': './tests/data/normalized_d2.csv',
    'y_train_file': './tests/data/normalized_d2.csv',
    #'x_validation_file': './tests/data/encoder_eval_d3.csv',
    'x_validation_file': './tests/data/normalized_d3.csv',
    'y_validation_file': './tests/data/normalized_d3.csv',
    'target_column': 'CLOSE',
    'output_file': './csv_output.csv',
    'results_file': './results.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'evaluate_file': './model_eval.csv',
    'plugin': 'cnn',
    'time_horizon': 1,
    'threshold_error': 0.00001,
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None,
    'load_config': None,
    'save_config': './config_out.json',
    'save_log': './debug_out.json',
    'quiet_mode': False,
    'force_date': False,
    'headers': True,
    'input_offset': 0,
    'window_size': 128,  # Number of time steps in each window (e.g., 24 for daily patterns)
    'l2_reg': 1e-4,          # L2 regularization factor
    'patience': 10,           # Early stopping patience
    'max_steps_train': 6300,
    'max_steps_test': 6300,
    'iterations': 10,
    'epochs': 200
}
