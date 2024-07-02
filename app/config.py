# config.py

DEFAULT_VALUES = {
    'input_file': './tests/data/encoder_eval.csv',
    'input_timeseries': './tests/data/csv_sel_unb_norm_512.csv',
    'target_column': None,
    'output_file': './csv_output.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'evaluate_file': './model_eval.csv',
    'plugin': 'ann',
    'time_horizon': 12,  # Set an appropriate default value
    'threshold_error': 0.003,
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
}
