# config.py

DEFAULT_VALUES = {
    'x_train_file': '..\\Documents\\encoder_eval_d2_indicators_128.csv',
    'y_train_file': './tests/data/target_column_d2.csv',
    'x_validation_file': '..\\Documents\\encoder_eval_d3_indicators_128.csv',
    'y_validation_file': './tests/data/target_column_d3.csv',
    'target_column': None,
    'output_file': './csv_output.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'evaluate_file': './model_eval.csv',
    'plugin': 'cnn',
    'time_horizon': 1,
    'threshold_error': 0.00004,
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
    'input_offset': 127  
}
