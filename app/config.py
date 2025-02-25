# config.py

DEFAULT_VALUES = {
    "use_normalization_json": "examples\\config\\phase_2_normalizer_debug_out.json",


    "x_train_file": "examples\\data\\phase_1\\phase_1_normalized_d2.csv",
    #"x_train_file": "examples\\data\\phase_2\\normalized_d2.csv",
    #"x_train_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "y_train_file": "examples\\data\\phase_1\\phase_1_normalized_d2.csv",
    
    "x_validation_file": "examples\\data\\phase_1\\phase_1_normalized_d3.csv",
    #"x_validation_file": "examples\\data\\phase_2\\normalized_d3.csv",
    #"x_validation_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "y_validation_file": "examples\\data\\phase_1\\phase_1_normalized_d3.csv",
    
    'target_column': 'CLOSE',
    'output_file': './prediction.csv',
    'results_file': './results.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'loss_plot_file': './loss_plot.png',
    'model_plot_file': './model_plot.png',	
    'plugin': 'ann',
    'time_horizon': 6,
    'use_daily': False, # isntead of predicting the next time_horizon hours, predict the next time_horizon days.
    'threshold_error': 0.0001,
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
    'window_size': 256,  # Number of time steps in each window (e.g., 24 for daily patterns)
    'l2_reg': 1e-4,          # L2 regularization factor
    'patience': 30,           # Early stopping patience
    'max_steps_train': 6300,
    'max_steps_test': 6300,
    'iterations': 3,
    'epochs': 1000
    
}
