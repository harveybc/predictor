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
import os # Needed for basename check

# Conditional import for plot_model
try:
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None

import tensorflow as tf
import tensorflow.keras.backend as K
# Assuming write_csv is correctly imported
from app.data_handler import write_csv


# --- Z-Score Denormalization Functions ---
def denormalize(data, config):
    """Denormalizes price using z-score normalization parameters from JSON."""
    data = np.asarray(data)
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f: 
                    norm_json = json.load(f)
            except Exception as e: 
                print(f"WARN: Failed load norm JSON {norm_json}: {e}")
                return data
        if isinstance(norm_json, dict) and "CLOSE" in norm_json:
            try:
                if "mean" in norm_json["CLOSE"] and "std" in norm_json["CLOSE"]:
                    # Z-score denormalization: value = (normalized * std) + mean
                    close_mean = norm_json["CLOSE"]["mean"]
                    close_std = norm_json["CLOSE"]["std"]
                    return (data * close_std) + close_mean
                else:
                    print(f"WARN: Missing 'mean' or 'std' in norm JSON for CLOSE")
                    return data
            except KeyError as e: 
                print(f"WARN: Missing key in norm JSON: {e}")
                return data
            except Exception as e: 
                print(f"WARN: Error during denormalize: {e}")
                return data
    return data

def denormalize_target_returns(data, target_returns_mean, target_returns_std, horizon_idx):
    """Denormalizes target returns using preprocessor-calculated normalization stats."""
    data = np.asarray(data)
    try:
        if horizon_idx < len(target_returns_mean) and horizon_idx < len(target_returns_std):
            mean_h = target_returns_mean[horizon_idx]
            std_h = target_returns_std[horizon_idx]
            return (data * std_h) + mean_h
        else:
            print(f"WARN: Horizon index {horizon_idx} out of range for target normalization stats")
            return data
    except Exception as e:
        print(f"WARN: Error during denormalize_target_returns: {e}")
        return data

# --- End Denormalization Functions ---


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
        config = self.params
        # Example post-update logic (if needed):
        # if config.get("stl_period") is not None and config.get("stl_period") > 1: ...

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
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")
        test_close_prices = datasets.get("test_close_prices")  # Future prices for target calculation
        
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
                        
                        # Denormalize predictions and targets using target normalization stats
                        train_preds_denorm = denormalize_target_returns(train_preds_h, target_returns_means, target_returns_stds, idx)
                        train_target_denorm = denormalize_target_returns(train_target_h, target_returns_means, target_returns_stds, idx)
                        val_preds_denorm = denormalize_target_returns(val_preds_h, target_returns_means, target_returns_stds, idx)
                        val_target_denorm = denormalize_target_returns(val_target_h, target_returns_means, target_returns_stds, idx)
                        
                        # Denormalize baseline using JSON normalization
                        baseline_train_denorm = denormalize(baseline_train_h, config)
                        baseline_val_denorm = denormalize(baseline_val_h, config)
                        
                        # Calculate final prices (baseline + returns if use_returns)
                        if use_returns:
                            train_target_price = baseline_train_denorm + train_target_denorm
                            train_pred_price = baseline_train_denorm + train_preds_denorm
                            val_target_price = baseline_val_denorm + val_target_denorm
                            val_pred_price = baseline_val_denorm + val_preds_denorm
                        else:
                            train_target_price = train_target_denorm
                            train_pred_price = train_preds_denorm
                            val_target_price = val_target_denorm
                            val_pred_price = val_preds_denorm
                        
                        # Denormalize uncertainties using target normalization stats
                        train_unc_denorm = denormalize_target_returns(train_unc_h, [0.0] * len(target_returns_means), target_returns_stds, idx)
                        val_unc_denorm = denormalize_target_returns(val_unc_h, [0.0] * len(target_returns_means), target_returns_stds, idx)
                        
                        # Metrics
                        train_mae_h = np.mean(np.abs(train_preds_denorm - train_target_denorm))
                        train_r2_h = r2_score(train_target_price, train_pred_price)
                        train_unc_mean_h = np.mean(np.abs(train_unc_denorm))
                        train_snr_h = np.mean(train_pred_price) / (train_unc_mean_h + 1e-9)
                        
                        val_mae_h = np.mean(np.abs(val_preds_denorm - val_target_denorm))
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
                    
                    # Denormalize predictions and targets using target normalization stats
                    test_preds_denorm = denormalize_target_returns(test_preds_h, target_returns_means, target_returns_stds, idx)
                    test_target_denorm = denormalize_target_returns(test_target_h, target_returns_means, target_returns_stds, idx)
                    
                    # Baseline is already denormalized, use directly
                    baseline_test_denorm = baseline_test_h
                    
                    # Calculate final prices (baseline + returns if use_returns)
                    if use_returns:
                        test_target_price = baseline_test_denorm + test_target_denorm
                        test_pred_price = baseline_test_denorm + test_preds_denorm
                    else:
                        test_target_price = test_target_denorm
                        test_pred_price = test_preds_denorm
                    
                    # Denormalize uncertainties using target normalization stats
                    test_unc_denorm = denormalize_target_returns(test_unc_h, [0.0] * len(target_returns_means), target_returns_stds, idx)
                    
                    # Metrics
                    test_mae_h = np.mean(np.abs(test_preds_denorm - test_target_denorm))
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
            arrays_to_check_len = [final_predictions[0], baseline_test, test_dates]
            num_test_points = min(len(arr) for arr in arrays_to_check_len if arr is not None)
            print(f"Determined consistent output length: {num_test_points}")
            
            final_dates = list(test_dates[:num_test_points]) if test_dates is not None else list(range(num_test_points))
            final_baseline = baseline_test[:num_test_points].flatten() if baseline_test is not None else None  # Flatten baseline here

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
                    # Denormalize predictions and targets using target normalization stats
                    preds_denorm = denormalize_target_returns(preds_raw, target_returns_means, target_returns_stds, idx)
                    target_denorm = denormalize_target_returns(target_raw, target_returns_means, target_returns_stds, idx)
                    
                    # --- Apply FIX: Ensure baseline and denormalized returns are 1D before adding ---
                    if use_returns:
                        if final_baseline is None: 
                            raise ValueError("Baseline missing.")
                        # Baseline is already denormalized from target calculation, use directly
                        pred_price_denorm = final_baseline + preds_denorm  # (N,) + (N,) -> (N,)
                        target_price_denorm = final_baseline + target_denorm  # (N,) + (N,) -> (N,)
                    else:
                        pred_price_denorm = preds_denorm
                        target_price_denorm = target_denorm

                    # Denormalize uncertainty (no mean shift for uncertainty)
                    unc_denorm = denormalize_target_returns(unc_raw, [0.0] * len(target_returns_means), target_returns_stds, idx)
                    
                except Exception as e: 
                    print(f"WARN: Error denorm H={h}: {e}")
                
                # Add flattened results
                output_data[f"Target_H{h}"] = target_price_denorm
                output_data[f"Prediction_H{h}"] = pred_price_denorm
                uncertainty_data[f"Uncertainty_H{h}"] = unc_denorm

            # --- Save Predictions DataFrame (output_file) ---
            output_file = config.get("output_file", self.params["output_file"])
            try:
                print("\nChecking final lengths for Predictions DataFrame:")
                [print(f"  - {k}: {len(v)}") for k, v in output_data.items()]
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
                    print("\nChecking final lengths for Uncertainty DataFrame:")
                    [print(f"  - {k}: {len(v)}") for k, v in uncertainty_data.items()]
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

        # --- Plot Predictions for 'plotted_horizon' (CORRECTED - Flattening & Variable Names) ---
        print(f"\nGenerating prediction plot for H={plotted_horizon}...")
        try:
            # Use CORRECT variable names from last iteration, sliced
            preds_plot_raw = list_test_preds[plotted_index][:num_test_points]  # Shape (num_test_points,) or (num_test_points, 1)
            target_plot_raw = y_test_list[plotted_index][:num_test_points]  # Shape (num_test_points,) or (num_test_points, 1)
            unc_plot_raw = list_test_unc[plotted_index][:num_test_points]  # Shape (num_test_points,) or (num_test_points, 1)
            baseline_plot = final_baseline  # Already sliced, shape (num_test_points,)

            # CRITICAL FIX: Calculate true target returns mathematically
            # Instead of denormalizing potentially incorrect normalized targets,
            # calculate the true returns directly from baseline and future prices
            
            baseline_plot_values = baseline_plot.flatten()
            
            # Denormalize predictions using current normalization stats
            preds_plot_denorm = denormalize_target_returns(preds_plot_raw.flatten(), target_returns_means, target_returns_stds, plotted_index)
            
            # Calculate true target returns if we have access to future prices
            if test_close_prices is not None and len(test_close_prices) > num_test_points + plotted_horizon:
                # Calculate true returns: future_price - baseline_price
                future_prices_slice = test_close_prices[plotted_horizon:num_test_points + plotted_horizon]
                true_target_returns = future_prices_slice - baseline_plot_values
                target_plot_denorm = true_target_returns
                print(f"Using calculated true returns for plotting (length: {len(target_plot_denorm)})")
            else:
                # Fallback: denormalize the normalized targets (may have bias)
                target_plot_denorm = denormalize_target_returns(target_plot_raw.flatten(), target_returns_mean, target_returns_std, plotted_index)
                print(f"Using denormalized targets for plotting (may have bias)")
            
            unc_plot_denorm = denormalize_target_returns(unc_plot_raw.flatten(), [0.0] * len(target_returns_mean), target_returns_std, plotted_index)
            
            if use_returns:
                # --- Baseline is already denormalized, use directly ---
                baseline_plot_denorm = baseline_plot.flatten()
                pred_plot_price_flat = (baseline_plot_denorm + preds_plot_denorm).flatten()
                target_plot_price_flat = (baseline_plot_denorm + target_plot_denorm).flatten()
                # Show the actual baseline prices (current prices) for comparison
                baseline_plot_price_flat = baseline_plot_denorm.copy()
            else:
                pred_plot_price_flat = preds_plot_denorm.flatten()
                target_plot_price_flat = target_plot_denorm.flatten()
                # When not using returns, show baseline as the reference
                baseline_plot_price_flat = baseline_plot.flatten() if baseline_plot is not None else np.zeros_like(pred_plot_price_flat)
            
            unc_plot_denorm_flat = unc_plot_denorm.flatten()

            # Determine plot points and slice FLATTENED arrays
            n_plot = config.get("plot_points", self.params["plot_points"])
            num_avail_plot = len(pred_plot_price_flat)  # Length of data available for plot
            plot_slice = slice(max(0, num_avail_plot - n_plot), num_avail_plot)

            dates_plot_final = final_dates[plot_slice]
            pred_plot_final = pred_plot_price_flat[plot_slice]
            target_plot_final = target_plot_price_flat[plot_slice]
            baseline_plot_final = baseline_plot_price_flat[plot_slice]
            unc_plot_final = unc_plot_denorm_flat[plot_slice]  # This is now 1D

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(dates_plot_final, pred_plot_final, label=f"Predicted Future Price H{plotted_horizon}", 
                    color=config.get("plot_color_predicted", "red"), lw=1.5, zorder=3)
            plt.plot(dates_plot_final, target_plot_final, label=f"Target Future Price H{plotted_horizon}", 
                    color=config.get("plot_color_target", "orange"), lw=1.5, zorder=2)
            plt.plot(dates_plot_final, baseline_plot_final, label="Current Price (Baseline)", 
                    color=config.get("plot_color_true", "blue"), lw=1, ls='--', alpha=0.7, zorder=1)
            plt.fill_between(dates_plot_final, pred_plot_final - abs(unc_plot_final), pred_plot_final + abs(unc_plot_final),
                            color=config.get("plot_color_uncertainty", "green"), alpha=0.2, 
                            label=f"Uncertainty H{plotted_horizon}", zorder=0)
            plt.title(f"Predicted vs Target Future Prices (H={plotted_horizon})")
            plt.xlabel("Time")
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
        if "target_returns_mean" not in preprocessor_params or "target_returns_std" not in preprocessor_params:
            raise ValueError("Preprocessor did not return 'target_returns_mean' or 'target_returns_std' for evaluation. Check preprocessor configuration and execution.")
        target_returns_mean = preprocessor_params["target_returns_mean"]
        target_returns_std = preprocessor_params["target_returns_std"]
        
        print(f"Validation data X shape: {x_val.shape}")
        print("Making predictions on validation data...")
        try: 
            mc_samples = config.get("mc_samples", 100)
            list_predictions, _ = predictor_plugin.predict_with_uncertainty(x_val, mc_samples=mc_samples)
            print(f"Preds list length: {len(list_predictions)}")
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
            
            baseline_val_eval_sliced = baseline_val_eval[:num_val_points].flatten() if baseline_val_eval is not None else None  # Flatten baseline
            
            for idx, h in enumerate(config['predicted_horizons']):
                preds_raw = list_predictions[idx][:num_val_points].flatten()  # Flatten preds
                
                # Denormalize predictions using target normalization stats
                preds_denorm = denormalize_target_returns(preds_raw, target_returns_mean, target_returns_std, idx)
                
                if use_returns_eval:
                    # Denormalize baseline and add to returns
                    baseline_denorm = denormalize(baseline_val_eval_sliced, config)
                    pred_price = baseline_denorm + preds_denorm
                else:
                    pred_price = preds_denorm
                
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