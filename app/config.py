# config.py

DEFAULT_VALUES = {
    "use_normalization_json": "examples\\config\\phase_3_normalizer_debug_out.json",


    "x_train_file": "examples\\data\\phase_3\\exp_2\\phase_3_encoder_eval_d1.csv",
    "y_train_file": "examples\\data\\phase_3\\exp_2\\normalized_d1.csv",
    
    #"x_train_file": "examples\\data\\phase_1\\phase_1_normalized_d2.csv",
    #"x_train_file": "examples\\data\\phase_2\\normalized_d2.csv",
    #"x_train_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    
    
    #"x_validation_file": "examples\\data\\phase_1\\phase_1_normalized_d3.csv",
    #"x_validation_file": "examples\\data\\phase_2\\normalized_d3.csv",
    #"x_validation_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "x_validation_file": "examples\\data\\phase_3\\exp_2\\phase_3_encoder_eval_d3.csv",
    "y_validation_file": "examples\\data\\phase_3\\exp_2\\normalized_d3.csv",

    "x_test_file": "examples\\data\\phase_3\\exp_2\\phase_3_encoder_eval_d2.csv",
    "y_test_file": "examples\\data\\phase_3\\exp_2\\normalized_d2.csv",
    
    'target_column': 'CLOSE',
    'output_file': './prediction.csv',
    'results_file': './results.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'loss_plot_file': './loss_plot.png',
    'model_plot_file': './model_plot.png',	
    'plugin': 'cnn',
    'input_offset': 0,
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
    'learning_rate': 1e-4,
    'time_horizon': 6,
    'window_size': 48,  # Number of time steps in each window (e.g., 24 for daily patterns)
    'initial_layer_size': 48,
    'intermediate_layers': 3,
    'layer_size_divisor': 2,
    'l2_reg': 1e-6,          # L2 regularization factor, best 1e-3 = TestMAE:0.0103 R2:0.9921
    'early_patience': 45,           # Early stopping patience
    'max_steps_train': 20000,
    'max_steps_test': 20000,
    'iterations': 3,
    'epochs': 1000,
    'use_mmd': True
    
}
