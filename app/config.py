# config.py

DEFAULT_VALUES = {
    'input_file': './csv_input.csv',
    'input_timeseries': None,
    'target_column': None,
    'output_file': './csv_output.csv',
    'save_model': './predictor_model.h5',
    'load_model': None,
    'evaluate_file': './model_eval.csv',
    'plugin': 'default',
    'time_horizon': 10,  # Assuming a default value of 10 ticks ahead
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
    'headers': False,
}
