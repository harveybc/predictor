# config.py

DEFAULT_VALUES = {
    "use_normalization_json": "examples\\config\\phase_2_normalizer_debug_out.json",


    "x_train_file": "examples\\data\\phase_3\\phase_3_encoder_eval_d2.csv",
    #"x_train_file": "examples\\data\\phase_2\\normalized_d2.csv",
    #"x_train_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "y_train_file": "examples\\data\\phase_2\\exp_4\\normalized_d2.csv",
    
    "x_validation_file": "examples\\data\\phase_3\\phase_3_encoder_eval_d3.csv",
    #"x_validation_file": "examples\\data\\phase_2\\normalized_d3.csv",
    #"x_validation_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "y_validation_file": "examples\\data\\phase_2\\exp_4\\normalized_d3.csv",
    
    'target_column': 'CLOSE',
    'output_file': './prediction.csv',
    'results_file': './results.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'loss_plot_file': './loss_plot.png',
    'model_plot_file': './model_plot.png',	
    'plugin': 'ann',
    'use_daily': False, # isntead of predicting the next time_horizon hours, predict the next time_horizon days.
    'threshold_error': 0.000000001,
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
    'early_patience': 30,           # Early stopping patience
    'max_steps_train': 6300,
    'max_steps_val': 6300,
    'max_steps_test': 6300,
    'iterations': 3,
    'epochs': 1000,
    'uncertainty_file': 'prediction_uncertainties.csv',
    'batch_size': 32,
    'use_sliding_window' : False,
    "kl_weight": 1e-6,
    "kl_anneal_epochs": 100,        
    "mmd_lambda": 0.1,
    "overfitting_penalty": 0.1,
    "use_returns": True,
    "mc_samples":100,
    "plotted_horizon": 6,
    "min_delta": 1e-4,
    "start_from_epoch": 10,
    "plot_color_predicted": "orange",
    "plot_color_true": "blue",
    "plot_color_uncertainty": "green",
    "uncertainty_color_alpha": 0.01,
    "min_delta": 1e-5,
    "plot_points": 240,
    "use_strategy": False,
    "strategy_plugin_group": "heuristic_strategy.plugins",
    "strategy_plugin_name": "ls_pred_strategy",
    "strategy_1h_prediction": "examples/results/phase_1/phase_1_cnn_25200_1h_prediction.csv",
    "strategy_1h_uncertainty": "examples/results/phase_1/phase_1_cnn_25200_1h_uncertanties.csv",
    "strategy_base_dataset": "examples/data/phase_1/phase_1_base_d3.csv",
    "strategy_load_parameters": "examples/data/phase_1/strategy_parameters.json",
    "target_scaling_factor":1000,
    "optimizer_output_file": "optimizer_output.json",
    "penalty_close_lambda":0.0001, # penalty in thel loss function for the predicted value being 0 (Naive)
    "penalty_far_lambda":0.0001,    # penalty in thel loss function for the predicted value being far from the target value in the opposite dicection of the 0 (Naive)
    "feature_extractor_file": None,
    "train_fe" : False
}
