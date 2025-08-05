#!/usr/bin/env python
"""
STL Pipeline Plugin - Z-Score Version 

Updated to work with the modular preprocessor that uses z-score normalization.
Properly handles target_returns_means and target_returns_stds for each horizon.
Maintains all original functionality, outputs, and printed messages.
"""

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# Conditional import for plot_model
try:
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None
# Assuming write_csv is correctly imported
from app.data_handler import write_csv


class STLPipelinePlugin:
    # Default parameters (kept from previous version)
    plugin_params = {
        "iterations": 1, "batch_size": 32, "epochs": 50, "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png", "output_file": "test_predictions.csv",
        "uncertainties_file": "test_uncertainties.csv", "model_plot_file": "model_plot.png",
        "predictions_plot_file": "predictions_plot.png", "results_file": "results.csv",
        "plot_points": 480, "plotted_horizon": 6,
        "predicted_horizons": [1, 6, 12, 24], "use_returns": True, "normalize_features": True,
        "window_size": 48, "target_column": "TARGET", "use_normalization_json": None,
        "mc_samples": 100,
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error", "output_file", "uncertainties_file", "results_file", "plotted_horizon", "plot_points"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items(): 
            self.params[key] = value

    def get_debug_info(self): 
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info): 
        debug_info.update(self.get_debug_info())

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin):
        start_time = time.time()
        run_config = self.params.copy()
        run_config.update(config)
        config = run_config
        iterations = config.get("iterations", 1)
        print(f"Iterations: {iterations}")

        # Init metric storage
        predicted_horizons = config.get('predicted_horizons')
        num_outputs = len(predicted_horizons)
        metric_names = ["MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {ds: {mn: {h: [] for h in predicted_horizons} for mn in metric_names} for ds in data_sets}

        # 1. Get datasets from preprocessor
        print("Loading/processing datasets via Preprocessor...")
        datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
        print("Preprocessor finished.")
        
        X_train = datasets["x_train"]
        X_val = datasets["x_val"]
        X_test = datasets["x_test"]
        
        # Extract targets from target dictionaries (structured by horizon)
        y_train_list = []
        y_val_list = []
        y_test_list = []
        
        for idx, h in enumerate(predicted_horizons):
            # Target data is structured as dictionaries with horizon keys
            horizon_key = f'output_horizon_{h}'
            y_train_list.append(datasets["y_train"][horizon_key])
            y_val_list.append(datasets["y_val"][horizon_key])
            y_test_list.append(datasets["y_test"][horizon_key])
        
        train_dates = datasets.get("y_train_dates")
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")
        
        # --- COLUMN EXCLUSION: Remove specified columns from sliding windows AFTER targets are calculated ---
        excluded_columns = config.get("excluded_columns", [])
        if excluded_columns:
            print(f"\n--- Removing excluded columns from sliding windows: {excluded_columns} ---")
            
            # Get column names from datasets (returned by preprocessor as feature_names)
            column_names = datasets.get("feature_names", None)
            if column_names is None:
                print("WARNING: No feature_names available from preprocessor, cannot exclude columns by name")
            else:
                print(f"Available columns: {column_names}")
                
                # Find indices of columns to exclude
                excluded_indices = []
                for col_name in excluded_columns:
                    if col_name in column_names:
                        excluded_indices.append(column_names.index(col_name))
                        print(f"  Excluding column '{col_name}' at index {column_names.index(col_name)}")
                    else:
                        print(f"  WARNING: Column '{col_name}' not found in dataset columns")
                
                if excluded_indices:
                    # Remove excluded columns from all datasets (last dimension = features)
                    remaining_indices = [i for i in range(len(column_names)) if i not in excluded_indices]
                    print(f"  Keeping {len(remaining_indices)} columns out of {len(column_names)} original columns")
                    
                    print(f"  Original shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
                    X_train = X_train[:, :, remaining_indices]
                    X_val = X_val[:, :, remaining_indices]
                    X_test = X_test[:, :, remaining_indices]
                    print(f"  New shapes after exclusion: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
                    
                    # Update column names in datasets and preprocessor_params for reference
                    new_column_names = [column_names[i] for i in remaining_indices]
                    datasets["feature_names"] = new_column_names
                    preprocessor_params["feature_names"] = new_column_names
                    print(f"  Updated column names: {new_column_names}")
                else:
                    print("  No valid columns to exclude")
        else:
            print("No columns specified for exclusion")
        
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")
        
        # Get target normalization stats from preprocessor (lists per horizon)
        if "target_returns_means" not in preprocessor_params or "target_returns_stds" not in preprocessor_params:
            raise ValueError("Preprocessor did not return target normalization stats. Check preprocessor execution.")
        target_returns_means = preprocessor_params["target_returns_means"]
        target_returns_stds = preprocessor_params["target_returns_stds"]
        
        use_returns = config.get("use_returns", False)
        if use_returns and (baseline_train is None or baseline_val is None or baseline_test is None): 
            raise ValueError("Baselines required for log return calculation when use_returns=True.")
        
        # Verify test dates are available for output
        if test_dates is None or len(test_dates) == 0:
            print(f"WARNING: test_dates not available from preprocessor - using indices for output")

        # Config Validation & Setup
        plotted_horizon = config.get('plotted_horizon')
        plotted_index = predicted_horizons.index(plotted_horizon)
        output_names = [f"output_horizon_{h}" for h in predicted_horizons]

        # Prepare Target Dicts for Training
        y_train_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_val_list)}

        print(f"Input shapes: Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")
        print(f"Target shapes(H={predicted_horizons[0]}): Train:{y_train_list[0].shape}, Val:{y_val_list[0].shape}, Test:{y_test_list[0].shape}")
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        print(f"Predicting Horizons: {predicted_horizons}, Plotting: H={plotted_horizon}")

        # --- Iteration Loop ---
        list_test_preds = None
        list_test_unc = None  # For last iteration results
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()

            # Build & Train
            input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],)
            predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)


            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
                X_train, y_train_dict, epochs=epochs, batch_size=batch_size, 
                threshold_error=config.get("threshold_error", 0.001),
                x_val=X_val, y_val=y_val_dict, config=config
            )

            # --- Convert predictions and targets to real-world prices for evaluation ---
            print("\n--- Converting z-score normalized predictions to real-world prices for evaluation ---")
            
            # Store original predictions and targets for later reference
            original_train_preds = [pred.copy() for pred in list_train_preds]
            original_val_preds = [pred.copy() for pred in list_val_preds]
            original_train_targets = [target.copy() for target in y_train_list]
            original_val_targets = [target.copy() for target in y_val_list]
            original_train_unc = [unc.copy() for unc in list_train_unc]
            original_val_unc = [unc.copy() for unc in list_val_unc]
            
            # Convert to real-world prices for metrics calculation
            real_train_price_preds = []
            real_val_price_preds = []
            real_train_price_targets = []
            real_val_price_targets = []
            real_train_price_uncertainties = []
            real_val_price_uncertainties = []
            
            for idx, h in enumerate(predicted_horizons):
                # Get horizon-specific normalization stats
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Convert predictions: z-score normalized -> log returns -> prices
                train_log_normalized_preds = original_train_preds[idx].flatten()
                val_log_normalized_preds = original_val_preds[idx].flatten()
                
                train_log_returns = train_log_normalized_preds * target_std + target_mean
                val_log_returns = val_log_normalized_preds * target_std + target_mean
                
                # Convert targets: z-score normalized -> log returns -> prices  
                train_log_normalized_targets = original_train_targets[idx].flatten()
                val_log_normalized_targets = original_val_targets[idx].flatten()
                
                train_target_log_returns = train_log_normalized_targets * target_std + target_mean
                val_target_log_returns = val_log_normalized_targets * target_std + target_mean
                
                # Convert uncertainties: z-score normalized -> log returns -> price uncertainty
                train_log_normalized_unc = original_train_unc[idx].flatten()
                val_log_normalized_unc = original_val_unc[idx].flatten()
                
                train_unc_log_returns = train_log_normalized_unc * target_std
                val_unc_log_returns = val_log_normalized_unc * target_std
                
                # CRITICAL FIX: Baselines and predictions are perfectly aligned
                # Both come from the same sliding windows matrix, so use them directly
                num_train = len(train_log_returns)
                num_val = len(val_log_returns)
                
                # Use the corresponding baselines (perfectly aligned with predictions)
                train_baselines = baseline_train[:num_train]
                val_baselines = baseline_val[:num_val]
                
                train_prices = train_baselines * np.exp(train_log_returns)
                val_prices = val_baselines * np.exp(val_log_returns)
                train_target_prices = train_baselines * np.exp(train_target_log_returns)
                val_target_prices = val_baselines * np.exp(val_target_log_returns)
                
                # Convert uncertainty to price scale: uncertainty = baseline * (exp(log_return + unc) - exp(log_return))
                # For small uncertainties: ≈ baseline * exp(log_return) * unc = price * unc
                # Convert uncertainty to price scale: uncertainty represents range around prediction
                # Since prediction_price = baseline * exp(log_return), uncertainty in price scale is:
                # uncertainty_price = prediction_price * uncertainty_log_returns (for small uncertainties)
                train_price_uncertainties = train_prices * np.abs(train_unc_log_returns)
                val_price_uncertainties = val_prices * np.abs(val_unc_log_returns)
                
                real_train_price_preds.append(train_prices)
                real_val_price_preds.append(val_prices)
                real_train_price_targets.append(train_target_prices)
                real_val_price_targets.append(val_target_prices)
                real_train_price_uncertainties.append(train_price_uncertainties)
                real_val_price_uncertainties.append(val_price_uncertainties)
                
                print(f"  H{h}: Converted {num_train} train, {num_val} val points to real prices")
            
            print("✅ All data converted to real-world prices for evaluation")

            # Calculate Train/Validation metrics using real-world prices
            can_calc_train_stats = all(len(lst) == num_outputs for lst in [real_train_price_preds, real_train_price_uncertainties])
            if can_calc_train_stats:
                print("Calculating Train/Validation metrics (all horizons)...")
                for idx, h in enumerate(predicted_horizons):
                    try:
                        # Use real-world price data for metrics
                        train_preds_h = real_train_price_preds[idx].flatten()
                        train_target_h = real_train_price_targets[idx].flatten()
                        train_unc_h = real_train_price_uncertainties[idx].flatten()
                        val_preds_h = real_val_price_preds[idx].flatten()
                        val_target_h = real_val_price_targets[idx].flatten()
                        val_unc_h = real_val_price_uncertainties[idx].flatten()
                        
                        # Ensure consistent lengths
                        min_train_len = min(len(train_preds_h), len(train_target_h), len(train_unc_h))
                        min_val_len = min(len(val_preds_h), len(val_target_h), len(val_unc_h))
                        
                        train_preds_h = train_preds_h[:min_train_len]
                        train_target_h = train_target_h[:min_train_len]
                        train_unc_h = train_unc_h[:min_train_len]
                        
                        val_preds_h = val_preds_h[:min_val_len]
                        val_target_h = val_target_h[:min_val_len]
                        val_unc_h = val_unc_h[:min_val_len]
                        
                        # Calculate metrics in real-world price scale
                        train_mae_h = np.mean(np.abs(train_preds_h - train_target_h))
                        val_mae_h = np.mean(np.abs(val_preds_h - val_target_h))
                        
                        train_r2_h = r2_score(train_target_h, train_preds_h)
                        val_r2_h = r2_score(val_target_h, val_preds_h)
                        
                        # Uncertainty metrics (already in price scale)
                        train_unc_mean_h = np.mean(train_unc_h)
                        val_unc_mean_h = np.mean(val_unc_h)
                        
                        # SNR: signal (price level) to noise (uncertainty) ratio
                        train_snr_h = np.mean(train_preds_h) / (train_unc_mean_h + 1e-9)
                        val_snr_h = np.mean(val_preds_h) / (val_unc_mean_h + 1e-9)
                        
                        metrics_results["Train"]["MAE"][h].append(train_mae_h)
                        metrics_results["Train"]["R2"][h].append(train_r2_h)
                        metrics_results["Train"]["Uncertainty"][h].append(train_unc_mean_h)
                        metrics_results["Train"]["SNR"][h].append(train_snr_h)
                        
                        metrics_results["Validation"]["MAE"][h].append(val_mae_h)
                        metrics_results["Validation"]["R2"][h].append(val_r2_h)
                        metrics_results["Validation"]["Uncertainty"][h].append(val_unc_mean_h)
                        metrics_results["Validation"]["SNR"][h].append(val_snr_h)
                        
                    except Exception as e: 
                        print(f"WARN: Error Train/Val metrics H={h}: {e}")
                        [metrics_results[ds][m][h].append(np.nan) for ds in ["Train", "Validation"] for m in metric_names]
            else: 
                print("WARN: Skipping Train/Val stats calculation.")

            # Save Loss Plot
            loss_plot_file = config.get("loss_plot_file")
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Val')
            plt.title(f"Loss-Iter {iteration}")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True, alpha=0.6)
            plt.savefig(loss_plot_file)
            plt.close()
            print(f"Loss plot saved: {loss_plot_file}")

            # Evaluate Test & Calc Metrics (All Horizons)
            print("Evaluating test set & calculating metrics...")
            mc_samples = config.get("mc_samples", 100)
            list_test_preds, list_test_unc = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)  # Assign results
            if not all(len(lst) == num_outputs for lst in [list_test_preds, list_test_unc]): 
                raise ValueError("Predictor predict mismatch outputs.")
            
            # --- Convert test predictions to real-world prices ---
            print("\n--- Converting test predictions to real-world prices ---")
            
            # Convert test predictions to real-world prices
            real_test_price_preds = []
            real_test_price_targets = []
            real_test_price_uncertainties = []
            
            for idx, h in enumerate(predicted_horizons):
                # Get horizon-specific normalization stats
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Convert predictions: z-score normalized -> log returns -> prices
                # Note: predict_with_uncertainty returns predictions in same scale as targets (z-score normalized)
                test_log_normalized_preds = list_test_preds[idx].flatten()
                test_log_returns = test_log_normalized_preds * target_std + target_mean
                
                # Convert targets: z-score normalized -> log returns -> prices
                test_log_normalized_targets = y_test_list[idx].flatten()
                test_target_log_returns = test_log_normalized_targets * target_std + target_mean
                
                # Convert uncertainties: z-score normalized -> log returns -> price uncertainty
                test_log_normalized_unc = list_test_unc[idx].flatten()
                test_unc_log_returns = test_log_normalized_unc * target_std
                
                # CRITICAL FIX: Baselines and predictions are perfectly aligned  
                num_test = len(test_log_returns)
                
                # Use the corresponding baselines (perfectly aligned with predictions)
                test_baselines = baseline_test[:num_test]
                
                # Convert log returns to real prices using aligned baselines
                test_prices = test_baselines * np.exp(test_log_returns)
                test_target_prices = test_baselines * np.exp(test_target_log_returns)
                
                # Convert uncertainty to price scale: uncertainty represents range around prediction
                # uncertainty_price = prediction_price * uncertainty_log_returns (for small uncertainties)
                test_price_unc = test_prices * np.abs(test_unc_log_returns)
                
                real_test_price_preds.append(test_prices)
                real_test_price_targets.append(test_target_prices)
                real_test_price_uncertainties.append(test_price_unc)
                
                print(f"  H{h}: Converted {num_test} test points to real prices")
            
            print("✅ Test predictions and targets converted to real-world prices")

            # Calculate test metrics using real-world price data
            for idx, h in enumerate(predicted_horizons):
                try:
                    # Use real-world price data for test metrics
                    test_preds_h = real_test_price_preds[idx].flatten()
                    test_target_h = real_test_price_targets[idx].flatten()
                    test_unc_h = real_test_price_uncertainties[idx].flatten()
                    
                    # Ensure consistent lengths
                    min_test_len = min(len(test_preds_h), len(test_target_h), len(test_unc_h))
                    test_preds_h = test_preds_h[:min_test_len]
                    test_target_h = test_target_h[:min_test_len]
                    test_unc_h = test_unc_h[:min_test_len]
                    
                    # Calculate metrics in real-world price scale
                    test_mae_h = np.mean(np.abs(test_preds_h - test_target_h))
                    test_r2_h = r2_score(test_target_h, test_preds_h)
                    test_unc_mean_h = np.mean(test_unc_h)
                    test_snr_h = np.mean(test_preds_h) / (test_unc_mean_h + 1e-9)
                    
                    metrics_results["Test"]["MAE"][h].append(test_mae_h)
                    metrics_results["Test"]["R2"][h].append(test_r2_h)
                    metrics_results["Test"]["Uncertainty"][h].append(test_unc_mean_h)
                    metrics_results["Test"]["SNR"][h].append(test_snr_h)
                    
                except Exception as e: 
                    print(f"WARN: Error Test metrics H={h}: {e}")
                    [metrics_results["Test"][m][h].append(np.nan) for m in metric_names]

            # Print Iteration Summary (using PLOTTED horizon)
            try:
                can_calc_train_val_stats = all(len(lst) == num_outputs for lst in [list_val_preds, list_val_unc])
                train_mae_plot = metrics_results["Train"]["MAE"][plotted_horizon][-1] if can_calc_train_val_stats else np.nan
                train_r2_plot = metrics_results["Train"]["R2"][plotted_horizon][-1] if can_calc_train_val_stats else np.nan
                val_mae_plot = metrics_results["Validation"]["MAE"][plotted_horizon][-1] if can_calc_train_val_stats else np.nan
                val_r2_plot = metrics_results["Validation"]["R2"][plotted_horizon][-1] if can_calc_train_val_stats else np.nan
                test_mae_plot = metrics_results["Test"]["MAE"][plotted_horizon][-1]
                test_r2_plot = metrics_results["Test"]["R2"][plotted_horizon][-1]
                test_unc_plot = metrics_results["Test"]["Uncertainty"][plotted_horizon][-1]
                test_snr_plot = metrics_results["Test"]["SNR"][plotted_horizon][-1]
                
                print("*" * 72)
                print(f"Iter {iteration} Done|Time:{time.time() - iter_start:.2f}s|Plot H:{plotted_horizon}")
                print(f"  Train MAE:{train_mae_plot:.6f}|R²:{train_r2_plot:.4f} -- Valid MAE:{val_mae_plot:.6f}|R²:{val_r2_plot:.4f}")
                print(f"  Test  MAE:{test_mae_plot:.6f}|R²:{test_r2_plot:.4f}|Unc:{test_unc_plot:.6f}|SNR:{test_snr_plot:.2f}")
                print("*" * 72)
            except Exception as e: 
                print(f"WARN: Error printing iter summary: {e}")
            # --- End of Iteration Loop ---

        # --- Consolidate results across iterations FOR ALL HORIZONS (Avg/Std/Min/Max) ---
        print("\n--- Aggregating Results Across Iterations (All Horizons) ---")
        results_list = []
        # (Logic confirmed correct and includes Min/Max)
        for ds in data_sets:
            for mn in metric_names:
                for h in predicted_horizons:
                    values = metrics_results[ds][mn][h]
                    valid_values = [v for v in values if not np.isnan(v)]
                    if valid_values: 
                        results_list.append({
                            "Metric": f"{ds} {mn} H{h}", 
                            "Average": np.mean(valid_values), 
                            "Std Dev": np.std(valid_values), 
                            "Min": np.min(valid_values), 
                            "Max": np.max(valid_values)
                        })
                    else: 
                        results_list.append({
                            "Metric": f"{ds} {mn} H{h}", 
                            "Average": np.nan, 
                            "Std Dev": np.nan, 
                            "Min": np.nan, 
                            "Max": np.nan
                        })
        
        results_df = pd.DataFrame(results_list)
        results_file = config.get("results_file", self.params["results_file"])
        try: 
            results_df.to_csv(results_file, index=False, float_format='%.6f')
            print(f"Aggregated results saved: {results_file}")
            print(results_df.to_string())
        except Exception as e: 
            print(f"ERROR saving results: {e}")

        # --- Save Final Test Outputs (Predictions & Uncertainties) ---
        print("\n--- Saving Final Test Outputs ---")
        try:
            # Use last iteration's results
            if list_test_preds is None or list_test_unc is None: 
                raise ValueError("Test predictions/uncertainties unavailable from last iteration.")
            final_predictions = list_test_preds
            final_uncertainties = list_test_unc

            # Determine consistent length for output
            num_test_points = min(len(final_predictions[0]), len(baseline_test))
            if test_dates is not None:
                num_test_points = min(num_test_points, len(test_dates))
            print(f"Output length: {num_test_points}")
            
            # Prepare output dates
            if test_dates is not None and len(test_dates) > 0:
                final_dates = list(test_dates[:num_test_points])
                print(f"Using {len(final_dates)} dates from preprocessor")
            else:
                print("ERROR: No test_dates from preprocessor! Using fallback indices.")
                final_dates = list(range(num_test_points))
                
            # CRITICAL FIX: Baselines and predictions are perfectly aligned
            final_baseline = baseline_test[:num_test_points].flatten()

            # Validate baseline data
            if len(final_baseline) == 0:
                print(f"ERROR: Empty baseline_test - cannot generate outputs!")
                if len(list_test_preds[0]) > 0:
                    fallback_value = np.mean(list_test_preds[0][:10]) if len(list_test_preds[0]) > 10 else 1000.0
                    final_baseline = np.full(num_test_points, fallback_value)
                    print(f"Using fallback baseline value: {fallback_value}")
                else:
                    raise ValueError("No baseline data available and no fallback possible")

            # Prepare output dictionaries
            output_data = {"DATE_TIME": final_dates}
            uncertainty_data = {"DATE_TIME": final_dates}

            # Add baseline price column
            output_data["test_CLOSE"] = final_baseline.flatten()

            # Process each horizon - add real-world price data to output
            for idx, h in enumerate(predicted_horizons):
                # Use real-world price data (already converted correctly)
                pred_prices = real_test_price_preds[idx][:num_test_points].flatten()
                target_prices = real_test_price_targets[idx][:num_test_points].flatten()
                price_uncertainties = real_test_price_uncertainties[idx][:num_test_points].flatten()
                
                # Add to output dictionaries
                output_data[f"Target_H{h}"] = target_prices
                output_data[f"Prediction_H{h}"] = pred_prices
                uncertainty_data[f"Uncertainty_H{h}"] = price_uncertainties

            # --- Save Predictions CSV ---
            output_file = config.get("output_file", self.params["output_file"])
            try:
                # Validate consistent lengths
                lengths = [len(v) for v in output_data.values()]
                if len(set(lengths)) > 1: 
                    raise ValueError(f"Length mismatch in output data: {dict(zip(output_data.keys(), lengths))}")
                
                # Create and save DataFrame
                output_df = pd.DataFrame(output_data)
                cols_order = ['DATE_TIME', 'test_CLOSE']
                for h in predicted_horizons:
                    cols_order.extend([f"Target_H{h}", f"Prediction_H{h}"])
                output_df = output_df.reindex(columns=[c for c in cols_order if c in output_df.columns])
                
                write_csv(file_path=output_file, data=output_df, include_date=False, headers=True)
                print(f"Predictions saved: {output_file} ({len(output_df)} rows)")
            except Exception as e: 
                print(f"ERROR saving predictions CSV: {e}")

            # --- Save Uncertainties CSV ---
            uncertainties_file = config.get("uncertainties_file", self.params.get("uncertainties_file"))
            if uncertainties_file:
                try:
                    # Validate consistent lengths
                    lengths = [len(v) for v in uncertainty_data.values()]
                    if len(set(lengths)) > 1: 
                        raise ValueError(f"Length mismatch in uncertainty data: {dict(zip(uncertainty_data.keys(), lengths))}")
                    
                    # Create and save DataFrame
                    uncertainty_df = pd.DataFrame(uncertainty_data)
                    cols_order = ['DATE_TIME'] + [f"Uncertainty_H{h}" for h in predicted_horizons]
                    uncertainty_df = uncertainty_df.reindex(columns=[c for c in cols_order if c in uncertainty_df.columns])
                    
                    write_csv(file_path=uncertainties_file, data=uncertainty_df, include_date=False, headers=True)
                    print(f"Uncertainties saved: {uncertainties_file} ({len(uncertainty_df)} rows)")
                except Exception as e: 
                    print(f"ERROR saving uncertainties CSV: {e}")
            else: 
                print("No uncertainties_file specified - skipping uncertainty output.")
        except Exception as e: 
            print(f"ERROR during final output saving: {e}")

        # --- Generate Prediction Plot for plotted_horizon ---
        print(f"\nGenerating prediction plot for H={plotted_horizon}...")
        try:
            # Use real-world price data for plotting (already correctly converted)
            pred_prices_plot = real_test_price_preds[plotted_index][:num_test_points].flatten()
            target_prices_plot = real_test_price_targets[plotted_index][:num_test_points].flatten()
            price_uncertainties_plot = real_test_price_uncertainties[plotted_index][:num_test_points].flatten()
            baseline_prices_plot = final_baseline.flatten()

            # Prepare plot data with proper slicing
            n_plot = config.get("plot_points", self.params["plot_points"])
            plot_start = max(0, len(pred_prices_plot) - n_plot)
            plot_slice = slice(plot_start, len(pred_prices_plot))

            # Extract plot data
            dates_plot = final_dates[plot_slice] if final_dates else list(range(len(pred_prices_plot)))[plot_slice]
            pred_plot = pred_prices_plot[plot_slice]
            target_plot = target_prices_plot[plot_slice]
            baseline_plot = baseline_prices_plot[plot_slice]
            uncertainty_plot = price_uncertainties_plot[plot_slice]

            # Create plot
            plt.figure(figsize=(14, 7))
            
            plt.plot(dates_plot, pred_plot, label=f"Pred Price H{plotted_horizon}", color=config.get("plot_color_predicted", "red"), lw=1.5, zorder=3)
            plt.plot(dates_plot, target_plot, label=f"Target Price H{plotted_horizon}", color=config.get("plot_color_target", "orange"), lw=1.5, zorder=2)
            plt.plot(dates_plot, baseline_plot, label="Actual Price", color=config.get("plot_color_true", "blue"), lw=1, ls='--', alpha=0.7, zorder=1)
            plt.fill_between(dates_plot, pred_plot - np.abs(uncertainty_plot), pred_plot + np.abs(uncertainty_plot),
                             color=config.get("plot_color_uncertainty", "green"), alpha=0.2, label=f"Uncertainty H{plotted_horizon}", zorder=0)
            plt.title(f"Predictions vs Target/Actual (H={plotted_horizon})")
            plt.xlabel("Time Steps")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.6)
            plt.tight_layout()
            
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300)
            plt.close()
            print(f"Prediction plot saved: {predictions_plot_file}")
        except Exception as e: 
            print(f"ERROR generating prediction plot: {e}")
            import traceback
            traceback.print_exc()
            plt.close()

                # --- Save Model Plot (if available) ---
        if plot_model is not None and hasattr(predictor_plugin, 'model') and predictor_plugin.model is not None:
            try: 
                model_plot_file = config.get('model_plot_file', 'model_plot.png')
                plot_model(predictor_plugin.model, to_file=model_plot_file, show_shapes=True, show_layer_names=True, dpi=300)
                print(f"Model plot saved: {model_plot_file}")
            except Exception as e: 
                print(f"WARN: Failed model plot: {e}")

        # --- Save Model ---
        if hasattr(predictor_plugin, 'save') and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try: 
                predictor_plugin.save(save_model_file)
                print(f"Model saved: {save_model_file}")
            except Exception as e: 
                print(f"ERROR saving model: {e}")

        print(f"\n=== Pipeline Complete (Total: {time.time() - start_time:.2f}s) ===")
        return {}
    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin):
        from tensorflow.keras.models import load_model
        print(f"Loading pre-trained model from {config['load_model']}...")
        try: 
            custom_objects = {}
            predictor_plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
            print("Model loaded.")
        except Exception as e: 
            print(f"Failed load model: {e}")
            return
        
        print("Loading/processing validation data for evaluation...")
        datasets, preprocessor_params = preprocessor_plugin.run_preprocessing(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("y_val_dates")
        baseline_val_eval = datasets.get("baseline_val")
        
        # Get target normalization stats from preprocessor_params
        if "target_returns_means" not in preprocessor_params or "target_returns_stds" not in preprocessor_params:
            raise ValueError("Preprocessor did not return 'target_returns_means' or 'target_returns_stds' for evaluation. Check preprocessor configuration and execution.")
        target_returns_means = preprocessor_params["target_returns_means"]
        target_returns_stds = preprocessor_params["target_returns_stds"]
        
        print(f"Validation data X shape: {x_val.shape}")
        print("Making predictions on validation data...")
        try: 
            mc_samples = config.get("mc_samples", 100)
            list_predictions, _ = predictor_plugin.predict_with_uncertainty(x_val, mc_samples=mc_samples)
            print(f"Preds list length: {len(list_predictions)}")
            
            # Convert predictions to real-world prices using same logic as main pipeline
            print("Converting validation predictions to real-world prices...")
            for idx, h in enumerate(config['predicted_horizons']):
                # Get horizon-specific normalization stats
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Convert predictions: z-score normalized -> log returns -> prices
                pred_normalized = list_predictions[idx].flatten()
                pred_log_returns = pred_normalized * target_std + target_mean
                
                # CRITICAL FIX: Baselines and predictions are perfectly aligned
                num_val = len(pred_log_returns)
                
                # Use the corresponding baselines (perfectly aligned with predictions)
                val_baselines = baseline_val_eval[:num_val]
                
                # Convert log returns to real prices using aligned baselines
                pred_prices = val_baselines * np.exp(pred_log_returns)
                
                # Update predictions with real prices
                list_predictions[idx] = pred_prices.reshape(-1, 1)
                
                print(f"  H{h}: Converted {num_val} validation points to real prices")
                    
        except Exception as e: 
            print(f"Failed predictions: {e}")
            return
        
        try:
            num_val_points = len(list_predictions[0])
            final_dates = list(val_dates[:num_val_points]) if val_dates is not None else list(range(num_val_points))
            output_data = {"DATE_TIME": final_dates}
            
            for idx, h in enumerate(config['predicted_horizons']):
                # Predictions are already real-world prices
                pred_prices = list_predictions[idx][:num_val_points].flatten()
                output_data[f"Prediction_H{h}"] = pred_prices
            
            evaluate_df = pd.DataFrame(output_data)
            evaluate_filename = config.get('output_file', 'eval_predictions.csv')
            write_csv(file_path=evaluate_filename, data=evaluate_df, include_date=False, headers=True)
            print(f"Validation predictions saved: {evaluate_filename}")
        except ImportError: 
            print(f"WARN: write_csv not found.")
        except Exception as e: 
            print(f"Failed save validation predictions: {e}")

# --- NO if __name__ == '__main__': block ---