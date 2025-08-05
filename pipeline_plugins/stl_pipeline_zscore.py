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
        y_train_list = [datasets["y_train"][f"output_horizon_{h}"] for h in predicted_horizons]
        y_val_list = [datasets["y_val"][f"output_horizon_{h}"] for h in predicted_horizons]
        y_test_list = [datasets["y_test"][f"output_horizon_{h}"] for h in predicted_horizons]
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
        test_close_prices = datasets.get("test_close_prices")  # Future prices for target calculation
        
        # Verify test dates are available for output
        if test_dates is None or len(test_dates) == 0:
            print(f"WARNING: test_dates not available from preprocessor - using indices for output")
        
        # Get target normalization stats from preprocessor_params (now lists per horizon)
        if "target_returns_means" not in preprocessor_params or "target_returns_stds" not in preprocessor_params:
            raise ValueError("Preprocessor did not return 'target_returns_means' or 'target_returns_stds'. Check preprocessor configuration and execution.")
        target_returns_means = preprocessor_params["target_returns_means"]
        target_returns_stds = preprocessor_params["target_returns_stds"]
        
        use_returns = config.get("use_returns", False)
        if use_returns and (baseline_train is None or baseline_val is None or baseline_test is None): 
            raise ValueError("Baselines required when use_returns=True.")

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

            # --- STEP 10: DE-TRANSFORM PREDICTIONS IMMEDIATELY AFTER TRAIN METHOD ---
            print("\n--- STEP 10: De-transforming predictions immediately after training ---")
            
            # CRITICAL FIX: Use per-horizon normalization stats for proper denormalization
            list_train_denorm_preds = []
            list_val_denorm_preds = []
            
            for idx, h in enumerate(predicted_horizons):
                # Use horizon-specific normalization stats
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Denormalize predictions from normalized space back to log returns
                train_normalized_preds = list_train_preds[idx].flatten()
                train_log_returns = train_normalized_preds * target_std + target_mean
                list_train_denorm_preds.append(train_log_returns)
                
                val_normalized_preds = list_val_preds[idx].flatten()
                val_log_returns = val_normalized_preds * target_std + target_mean
                list_val_denorm_preds.append(val_log_returns)
                
                print(f"  H{h}: Denormalized predictions using mean={target_mean:.6f}, std={target_std:.6f}")
            
            # --- STEP 10B: DENORMALIZE TARGETS TO MATCH DENORMALIZED PREDICTIONS ---
            print("\n--- STEP 10B: Denormalizing targets to match predictions ---")
            
            list_train_denorm_targets = []
            list_val_denorm_targets = []
            
            for idx, h in enumerate(predicted_horizons):
                # Use horizon-specific normalization stats for targets too
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Train targets: denormalize from normalized space back to log returns
                train_normalized_targets = y_train_list[idx].flatten()
                train_log_return_targets = train_normalized_targets * target_std + target_mean
                list_train_denorm_targets.append(train_log_return_targets)
                
                # Val targets: denormalize from normalized space back to log returns
                val_normalized_targets = y_val_list[idx].flatten()
                val_log_return_targets = val_normalized_targets * target_std + target_mean
                list_val_denorm_targets.append(val_log_return_targets)
                
                print(f"  H{h}: Denormalized targets using mean={target_mean:.6f}, std={target_std:.6f}")
            
            # --- CRITICAL FIX: Use consistent length and baseline slicing for both predictions and targets ---
            print("\n--- STEP 10C: Ensuring consistent baseline alignment for predictions and targets ---")
            
            list_train_full_preds = []
            list_val_full_preds = []
            list_train_full_targets = []
            list_val_full_targets = []
            
            for idx, h in enumerate(predicted_horizons):
                # Get denormalized log returns
                train_log_returns = list_train_denorm_preds[idx]
                val_log_returns = list_val_denorm_preds[idx]
                train_log_return_targets = list_train_denorm_targets[idx]
                val_log_return_targets = list_val_denorm_targets[idx]
                
                # CRITICAL: Use the minimum length across predictions, targets, and baselines
                num_train_pts = min(len(train_log_returns), len(train_log_return_targets), len(baseline_train))
                num_val_pts = min(len(val_log_returns), len(val_log_return_targets), len(baseline_val))
                
                # Slice ALL arrays to the same consistent length
                train_log_returns = train_log_returns[:num_train_pts]
                train_log_return_targets = train_log_return_targets[:num_train_pts]
                train_baselines = baseline_train[:num_train_pts]
                
                val_log_returns = val_log_returns[:num_val_pts]
                val_log_return_targets = val_log_return_targets[:num_val_pts]
                val_baselines = baseline_val[:num_val_pts]
                
                # Transform to full prices using SAME baseline slice for both predictions and targets
                train_full_prices = train_baselines * np.exp(train_log_returns)
                train_full_price_targets = train_baselines * np.exp(train_log_return_targets)
                
                val_full_prices = val_baselines * np.exp(val_log_returns)
                val_full_price_targets = val_baselines * np.exp(val_log_return_targets)
                
                list_train_full_preds.append(train_full_prices)
                list_train_full_targets.append(train_full_price_targets)
                list_val_full_preds.append(val_full_prices)
                list_val_full_targets.append(val_full_price_targets)
                
                print(f"  H{h}: Train {num_train_pts} aligned points -> Full prices, Val {num_val_pts} aligned points -> Full prices")
            
            # Replace predictions and targets with de-transformed full price versions
            list_train_preds = list_train_full_preds
            list_val_preds = list_val_full_preds
            y_train_list = list_train_full_targets
            y_val_list = list_val_full_targets
            
            print("✅ All predictions and targets are now FULL PRICES with consistent baseline alignment")

            # POST-PROCESSING SECTION OF PREDICTION PIPELINE
            # NOTE: Predictions are now FULL PRICES after de-transformation above
            # END OF POST-PROCESSING SECTION


            # Check outputs & Calc Train/Val Metrics (All Horizons)
            can_calc_train_stats = all(len(lst) == num_outputs for lst in [list_train_preds, list_train_unc])
            if can_calc_train_stats:
                print("Calculating Train/Validation metrics (all horizons)...")
                for idx, h in enumerate(predicted_horizons):
                    try:
                        # --- Ensure inputs are flattened BEFORE potential addition ---
                        train_preds_h = list_train_preds[idx].flatten()
                        train_target_h = y_train_list[idx].flatten()
                        train_unc_h = list_train_unc[idx].flatten()
                        val_preds_h = list_val_preds[idx].flatten()
                        val_target_h = y_val_list[idx].flatten()
                        val_unc_h = list_val_unc[idx].flatten()
                        
                        num_train_pts = min(len(train_preds_h), len(train_target_h), len(baseline_train))
                        num_val_pts = min(len(val_preds_h), len(val_target_h), len(baseline_val))
                        
                        train_preds_h = train_preds_h[:num_train_pts]
                        train_target_h = train_target_h[:num_train_pts]
                        train_unc_h = train_unc_h[:num_train_pts]
                        baseline_train_h = baseline_train[:num_train_pts].flatten()  # Flatten baseline too
                        
                        val_preds_h = val_preds_h[:num_val_pts]
                        val_target_h = val_target_h[:num_val_pts]
                        val_unc_h = val_unc_h[:num_val_pts]
                        baseline_val_h = baseline_val[:num_val_pts].flatten()  # Flatten baseline too
                        
                        # Predictions and targets are already FULL PRICES from Step 10B denormalization
                        # No additional denormalization needed - use directly for all metrics
                        train_target_price = train_target_h  # Already full prices
                        train_pred_price = train_preds_h     # Already full prices
                        val_target_price = val_target_h      # Already full prices
                        val_pred_price = val_preds_h         # Already full prices
                        
                        # CRITICAL FIX: Use horizon-specific uncertainty denormalization
                        # Get the horizon-specific target std for this horizon
                        horizon_target_std = target_returns_stds[idx]
                        
                        # Uncertainties represent log return std, scale by horizon-specific target_std
                        train_unc_denorm = train_unc_h * horizon_target_std  # Convert from normalized to log return scale
                        val_unc_denorm = val_unc_h * horizon_target_std    # Convert from normalized to log return scale
                        
                        # MAE should be calculated on the full prices (not log returns)
                        train_mae_h = np.mean(np.abs(train_pred_price - train_target_price))
                        val_mae_h = np.mean(np.abs(val_pred_price - val_target_price))
                        
                        # Metrics: MAE , R² in price space
                        train_r2_h = r2_score(train_target_price, train_pred_price)
                        train_unc_mean_h = np.mean(np.abs(train_unc_denorm))
                        train_snr_h = np.mean(train_pred_price) / (train_unc_mean_h + 1e-9)
                        
                        val_r2_h = r2_score(val_target_price, val_pred_price)
                        val_unc_mean_h = np.mean(np.abs(val_unc_denorm))
                        val_snr_h = np.mean(val_pred_price) / (val_unc_mean_h + 1e-9)
                        
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
            
            # --- STEP 11: DE-TRANSFORM TEST PREDICTIONS USING SAME LOGIC AS TRAINING/VALIDATION ---
            print("\n--- STEP 11: De-transforming test predictions using consistent logic ---")
            
            # CRITICAL FIX: Use per-horizon normalization stats for test predictions
            list_test_denorm_preds = []
            for idx, h in enumerate(predicted_horizons):
                # Use horizon-specific normalization stats
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Denormalize predictions from normalized space back to log returns
                test_normalized_preds = list_test_preds[idx].flatten()
                test_log_returns = test_normalized_preds * target_std + target_mean
                list_test_denorm_preds.append(test_log_returns)
                print(f"  H{h}: Denormalized test predictions using mean={target_mean:.6f}, std={target_std:.6f}")
            
            # --- STEP 11B: DENORMALIZE TEST TARGETS TO MATCH PREDICTIONS ---
            print("\n--- STEP 11B: Denormalizing test targets to match predictions ---")
            
            list_test_denorm_targets = []
            for idx, h in enumerate(predicted_horizons):
                # Use horizon-specific normalization stats for test targets too
                target_mean = target_returns_means[idx]
                target_std = target_returns_stds[idx]
                
                # Test targets: denormalize from normalized space back to log returns
                test_normalized_targets = y_test_list[idx].flatten()
                test_log_return_targets = test_normalized_targets * target_std + target_mean
                list_test_denorm_targets.append(test_log_return_targets)
                print(f"  H{h}: Denormalized test targets using mean={target_mean:.6f}, std={target_std:.6f}")
            
            # --- STEP 11C: CONSISTENT BASELINE ALIGNMENT FOR TEST DATA ---
            print("\n--- STEP 11C: Ensuring consistent baseline alignment for test predictions and targets ---")
            
            list_test_full_preds = []
            list_test_full_targets = []
            
            for idx, h in enumerate(predicted_horizons):
                # Get denormalized log returns
                test_log_returns = list_test_denorm_preds[idx]
                test_log_return_targets = list_test_denorm_targets[idx]
                
                # CRITICAL: Use the minimum length across predictions, targets, and baselines
                num_test_pts = min(len(test_log_returns), len(test_log_return_targets), len(baseline_test))
                
                # Slice ALL arrays to the same consistent length
                test_log_returns = test_log_returns[:num_test_pts]
                test_log_return_targets = test_log_return_targets[:num_test_pts]
                test_baselines = baseline_test[:num_test_pts]
                
                # Transform to full prices using SAME baseline slice for both predictions and targets
                test_full_prices = test_baselines * np.exp(test_log_returns)
                test_full_price_targets = test_baselines * np.exp(test_log_return_targets)
                
                list_test_full_preds.append(test_full_prices)
                list_test_full_targets.append(test_full_price_targets)
                
                print(f"  H{h}: Test {num_test_pts} aligned points -> Full prices")
            
            # Replace test predictions and targets with de-transformed full price versions
            list_test_preds = list_test_full_preds
            y_test_list = list_test_full_targets
            
            print("✅ All test predictions and targets are now FULL PRICES with consistent baseline alignment")
            
            for idx, h in enumerate(predicted_horizons):
                try:
                    # --- Ensure inputs are flattened BEFORE potential addition ---
                    test_preds_h = list_test_preds[idx].flatten()
                    test_target_h = y_test_list[idx].flatten()
                    test_unc_h = list_test_unc[idx].flatten()
                    
                    num_test_pts = min(len(test_preds_h), len(test_target_h), len(baseline_test))
                    test_preds_h = test_preds_h[:num_test_pts]
                    test_target_h = test_target_h[:num_test_pts]
                    test_unc_h = test_unc_h[:num_test_pts]
                    baseline_test_h = baseline_test[:num_test_pts].flatten()  # Flatten baseline too
                    
                    # Predictions and targets are already FULL PRICES from Step 11C denormalization
                    # No additional denormalization needed - use directly for all metrics
                    test_target_price = test_target_h  # Already full prices
                    test_pred_price = test_preds_h     # Already full prices
                    
                    # CRITICAL FIX: Use horizon-specific uncertainty denormalization for test
                    # Get the horizon-specific target std for this horizon
                    horizon_target_std = target_returns_stds[idx]
                    
                    # Uncertainties represent log return std, scale by horizon-specific target_std
                    # Do NOT multiply by baseline (price level) - that was causing massive uncertainty inflation
                    unc_normalized = test_unc_h
                    test_unc_denorm = unc_normalized * horizon_target_std  # Convert from normalized to log return scale
                    
                    # MAE should be calculated on the full prices (consistent with training/validation)
                    test_mae_h = np.mean(np.abs(test_pred_price - test_target_price))
                    
                    # Metrics: MAE and R² in price space (consistent with training/validation)
                    test_r2_h = r2_score(test_target_price, test_pred_price)
                    test_unc_mean_h = np.mean(np.abs(test_unc_denorm))
                    test_snr_h = np.mean(test_pred_price) / (test_unc_mean_h + 1e-9)
                    
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

        # --- Save Final Test Outputs (Separate Files - CORRECTED & VERIFIED) ---
        print("\n--- Saving Final Test Outputs (Predictions & Uncertainties Separately) ---")
        try:
            # Use last iteration's results stored in loop-scoped variables
            if list_test_preds is None or list_test_unc is None: 
                raise ValueError("Test preds/unc from last iter unavailable.")
            final_predictions = list_test_preds
            final_uncertainties = list_test_unc

            # Determine consistent length
            arrays_to_check_len = [final_predictions[0], baseline_test]
            if test_dates is not None:
                arrays_to_check_len.append(test_dates)
            num_test_points = min(len(arr) for arr in arrays_to_check_len if arr is not None)
            print(f"Determined consistent output length: {num_test_points}")
            
            # Use proper dates from preprocessor (last element timestamps from sliding windows)
            if test_dates is not None and len(test_dates) > 0:
                final_dates = list(test_dates[:num_test_points])
                print(f"Using proper dates from preprocessor: {len(final_dates)} dates")
                print(f"Sample preprocessor dates: {final_dates[:3] if len(final_dates) >= 3 else final_dates}")
            else:
                print("ERROR: No test_dates from preprocessor! Preprocessor should return last element timestamps from sliding windows.")
                print("This indicates a problem in the preprocessor - dates should be extracted from DATE_TIME column")
                print("Creating fallback range as last resort...")
                final_dates = list(range(num_test_points))
            final_baseline = baseline_test[:num_test_points].flatten() if baseline_test is not None else None  # Flatten baseline here

            # Check if final_baseline is valid
            if final_baseline is None or len(final_baseline) == 0:
                print(f"ERROR: baseline_test is empty - plotting may fail!")
                # Try to create a fallback baseline from predictions
                if len(list_test_preds[0]) > 0:
                    fallback_value = np.mean(list_test_preds[0][:10]) if len(list_test_preds[0]) > 10 else 1000.0
                    final_baseline = np.full(num_test_points, fallback_value)
                    print(f"  Using fallback baseline with value: {fallback_value}")
                else:
                    raise ValueError("No baseline data available for plotting and no fallback possible")

            # Prepare dictionaries
            output_data = {"DATE_TIME": final_dates}
            uncertainty_data = {"DATE_TIME": final_dates}

            # Add denormalized test CLOSE price (baseline is already denormalized)
            try: 
                denorm_test_close = final_baseline if final_baseline is not None else np.full(num_test_points, np.nan)
            except Exception as e: 
                print(f"WARN: Error using test_CLOSE: {e}")
                denorm_test_close = np.full(num_test_points, np.nan)
            output_data["test_CLOSE"] = denorm_test_close.flatten()

            # Process each horizon
            for idx, h in enumerate(predicted_horizons):
                # Get raw results (sliced) & FLATTEN for correct addition
                preds_raw = final_predictions[idx][:num_test_points].flatten()
                target_raw = y_test_list[idx][:num_test_points].flatten()
                unc_raw = final_uncertainties[idx][:num_test_points].flatten()

                pred_price_denorm = np.full(num_test_points, np.nan)
                target_price_denorm = np.full(num_test_points, np.nan)
                unc_denorm = np.full(num_test_points, np.nan)
                
                try:
                    # Predictions and targets are already FULL PRICES from Step 11C
                    # No additional transformation needed - use directly
                    pred_price_denorm = preds_raw.copy()  # Already full prices
                    target_price_denorm = target_raw.copy()  # Already full prices
                    
                    # Uncertainties are already converted to price scale in metrics calculation
                    unc_denorm = unc_raw.copy()  # Already price-scale uncertainties
                    
                except Exception as e: 
                    print(f"WARN: Error denorm H={h}: {e}")
                
                # Add flattened results
                output_data[f"Target_H{h}"] = target_price_denorm
                output_data[f"Prediction_H{h}"] = pred_price_denorm
                uncertainty_data[f"Uncertainty_H{h}"] = unc_denorm

            # --- Save Predictions DataFrame (output_file) ---
            output_file = config.get("output_file", self.params["output_file"])
            try:
                if len(set(len(v) for v in output_data.values())) > 1: 
                    raise ValueError("Length mismatch (Predictions).")
                
                output_df = pd.DataFrame(output_data)
                cols_order = ['DATE_TIME', 'test_CLOSE'] if 'test_CLOSE' in output_df else ['DATE_TIME']
                [cols_order.extend([f"Target_H{h}", f"Prediction_H{h}"]) for h in predicted_horizons]
                output_df = output_df.reindex(columns=[c for c in cols_order if c in output_df.columns])
                
                write_csv(file_path=output_file, data=output_df, include_date=False, headers=True)
                print(f"Predictions/Targets saved: {output_file} ({len(output_df)} rows)")
            except ImportError: 
                print(f"WARN: write_csv not found. Skip save: {output_file}.")
            except ValueError as ve: 
                print(f"ERROR creating/saving predictions CSV: {ve}")
            except Exception as e: 
                print(f"ERROR saving predictions CSV: {e}")

            # --- Save Uncertainties DataFrame (uncertainties_file) ---
            uncertainties_file = config.get("uncertainties_file", self.params.get("uncertainties_file"))
            if uncertainties_file:
                try:
                    if len(set(len(v) for v in uncertainty_data.values())) > 1: 
                        raise ValueError("Length mismatch (Uncertainty).")
                    
                    uncertainty_df = pd.DataFrame(uncertainty_data)
                    cols_order = ['DATE_TIME']
                    [cols_order.append(f"Uncertainty_H{h}") for h in predicted_horizons]
                    uncertainty_df = uncertainty_df.reindex(columns=[c for c in cols_order if c in uncertainty_df.columns])
                    
                    write_csv(file_path=uncertainties_file, data=uncertainty_df, include_date=False, headers=True)
                    print(f"Uncertainties saved: {uncertainties_file} ({len(uncertainty_df)} rows)")
                except ImportError: 
                    print(f"WARN: write_csv not found. Skip save: {uncertainties_file}.")
                except ValueError as ve: 
                    print(f"ERROR creating/saving uncertainties CSV: {ve}")
                except Exception as e: 
                    print(f"ERROR saving uncertainties CSV: {e}")
            else: 
                print("INFO: No 'uncertainties_file' specified.")
        except Exception as e: 
            print(f"ERROR during final CSV saving: {e}")

        # --- Plot Predictions for 'plotted_horizon' (CORRECTED - Using EXACT working code from stl_pipeline.py) ---
        print(f"\nGenerating prediction plot for H={plotted_horizon}...")
        try:
            # Use CORRECT variable names from last iteration, sliced
            preds_plot_raw = list_test_preds[plotted_index][:num_test_points]  # Shape (num_test_points,) or (num_test_points, 1)
            target_plot_raw = y_test_list[plotted_index][:num_test_points]  # Shape (num_test_points,) or (num_test_points, 1)
            unc_plot_raw = list_test_unc[plotted_index][:num_test_points]  # Shape (num_test_points,) or (num_test_points, 1)
            baseline_plot = final_baseline  # Already sliced, shape (num_test_points,)

            # Data is already FULL PRICES from Step 11C - no additional transformation needed!
            pred_plot_price_flat = preds_plot_raw.flatten()  # Already full prices
            target_plot_price_flat = target_plot_raw.flatten()  # Already full prices
            
            # Uncertainties and baseline are already in correct scale
            unc_plot_denorm_flat = unc_plot_raw.flatten()
            true_plot_price_flat = baseline_plot.flatten()

            # Determine plot points and slice FLATTENED arrays
            n_plot = config.get("plot_points", self.params["plot_points"])
            num_avail_plot = len(pred_plot_price_flat)  # Length of data available for plot
            plot_slice = slice(max(0, num_avail_plot - n_plot), num_avail_plot)

            # Handle pandas Timestamp objects properly for plotting
            if len(final_dates) > 0 and hasattr(final_dates[0], 'to_pydatetime'):
                try:
                    dates_for_plot = [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in final_dates]
                    # Check for large time gaps that might cause plotting issues
                    if len(dates_for_plot) > 2:
                        gap1 = dates_for_plot[1] - dates_for_plot[0]
                        gap2 = dates_for_plot[2] - dates_for_plot[1]
                        if abs(gap2.total_seconds() - gap1.total_seconds()) > 3600:  # More than 1 hour difference
                            dates_for_plot = list(range(len(final_dates)))
                except Exception:
                    dates_for_plot = list(range(len(final_dates)))
            else:
                dates_for_plot = list(range(len(final_dates)))
            
            dates_plot_final = dates_for_plot[plot_slice]
            pred_plot_final = pred_plot_price_flat[plot_slice]
            target_plot_final = target_plot_price_flat[plot_slice]
            true_plot_final = true_plot_price_flat[plot_slice]
            unc_plot_final = unc_plot_denorm_flat[plot_slice]  # This is now 1D

            # Plotting
            plt.figure(figsize=(14, 7))
            
            plt.plot(dates_plot_final, pred_plot_final, label=f"Pred Price H{plotted_horizon}", color=config.get("plot_color_predicted", "red"), lw=1.5, zorder=3)
            plt.plot(dates_plot_final, target_plot_final, label=f"Target Price H{plotted_horizon}", color=config.get("plot_color_target", "orange"), lw=1.5, zorder=2)
            plt.plot(dates_plot_final, true_plot_final, label="Actual Price", color=config.get("plot_color_true", "blue"), lw=1, ls='--', alpha=0.7, zorder=1)
            plt.fill_between(dates_plot_final, pred_plot_final - abs(unc_plot_final), pred_plot_final + abs(unc_plot_final),
                             color=config.get("plot_color_uncertainty", "green"), alpha=0.2, label=f"Uncertainty H{plotted_horizon}", zorder=0)
            plt.title(f"Predictions vs Target/Actual (H={plotted_horizon})")
            plt.xlabel("Time Steps")  # Changed from "Time" since we're using indices
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

        # --- Plot/Save Model --- (Keep as is)
        if plot_model is not None and hasattr(predictor_plugin, 'model') and predictor_plugin.model is not None:
            try: 
                model_plot_file = config.get('model_plot_file', 'model_plot.png')
                plot_model(predictor_plugin.model, to_file=model_plot_file, show_shapes=True, show_layer_names=True, dpi=300)
                print(f"Model plot saved: {model_plot_file}")
            except Exception as e: 
                print(f"WARN: Failed model plot: {e}")
        else: 
            print("INFO: Skipping model plot.")

        # --- Save Model --- (Keep as is)
        if hasattr(predictor_plugin, 'save') and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try: 
                predictor_plugin.save(save_model_file)
                print(f"Model saved: {save_model_file}")
            except Exception as e: 
                print(f"ERROR saving model: {e}")
        else: 
            print("WARN: Predictor has no save method.")

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")

    # --- load_and_evaluate_model (Updated for z-score) ---
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
            
            # De-transform validation predictions from log returns to prices
            use_returns_eval = config.get("use_returns", False)
            if use_returns_eval and baseline_val_eval is not None:
                print("De-transforming validation predictions...")
                for idx in range(len(list_predictions)):
                    pred_log_returns = list_predictions[idx].flatten()
                    baseline_values = baseline_val_eval[:len(pred_log_returns)]
                    
                    # Convert log returns to prices
                    pred_log_returns_clipped = np.clip(pred_log_returns, -10, 10)
                    pred_prices = baseline_values * np.exp(pred_log_returns_clipped)
                    
                    # Update list with de-transformed values
                    list_predictions[idx] = pred_prices.reshape(list_predictions[idx].shape)
                    
        except Exception as e: 
            print(f"Failed predictions: {e}")
            return
        
        try:
            num_val_points = len(list_predictions[0])
            final_dates = list(val_dates[:num_val_points]) if val_dates is not None else list(range(num_val_points))
            output_data = {"DATE_TIME": final_dates}
            use_returns_eval = config.get("use_returns", False)
            
            if use_returns_eval and baseline_val_eval is None: 
                raise ValueError("Baseline needed.")
            
            for idx, h in enumerate(config['predicted_horizons']):
                preds_raw = list_predictions[idx][:num_val_points].flatten()  # Flatten preds
                
                # Predictions are already de-transformed to prices if use_returns
                if use_returns_eval:
                    pred_price = preds_raw.copy()  # Already converted to prices
                else:
                    pred_price = preds_raw  # Direct values
                
                output_data[f"Prediction_H{h}"] = pred_price.flatten()
            
            evaluate_df = pd.DataFrame(output_data)
            evaluate_filename = config.get('output_file', 'eval_predictions.csv')
            write_csv(file_path=evaluate_filename, data=evaluate_df, include_date=False, headers=True)
            print(f"Validation predictions saved: {evaluate_filename}")
        except ImportError: 
            print(f"WARN: write_csv not found.")
        except Exception as e: 
            print(f"Failed save validation predictions: {e}")

# --- NO if __name__ == '__main__': block ---