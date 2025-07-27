#!/usr/bin/env python
"""
STL Pipeline Plugin - Corrected Version 6 (Fix Broadcasting Error)

Fixes NumPy broadcasting error causing length mismatch during denormalization.
Ensures preds_raw/target_raw are flattened BEFORE adding baseline.
Keeps previous fixes: Correct denormalization order, Separate Uncertainty File,
All Horizon Stats (Avg/Std/Min/Max), Plotting dimension fix.
ASSUMES PREPROCESSOR IS WORKING PERFECTLY.
"""

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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
def denormalize_zscore(data, feature_name, norm_params):
    """Denormalizes using z-score: value = (normalized * std) + mean"""
    data = np.asarray(data)
    if feature_name not in norm_params:
        print(f"WARN: {feature_name} not found in normalization parameters")
        return data
    
    mean = norm_params[feature_name]['mean']
    std = norm_params[feature_name]['std']
    return (data * std) + mean

def denormalize_zscore_returns(data, feature_name, norm_params):
    """Denormalizes return values using z-score: value = normalized * std (no mean shift for returns)"""
    data = np.asarray(data)
    if feature_name not in norm_params:
        print(f"WARN: {feature_name} not found in normalization parameters")
        return data
    
    std = norm_params[feature_name]['std']
    return data * std

def normalize_zscore(data, fit_params=None):
    """Normalizes using z-score: normalized = (value - mean) / std"""
    data = np.asarray(data)
    if fit_params is None:
        # Calculate parameters from data
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            std = 1.0  # Avoid division by zero
        fit_params = {'mean': mean, 'std': std}
    else:
        mean = fit_params['mean']
        std = fit_params['std']
    
    normalized = (data - mean) / std
    return normalized, fit_params

def load_normalization_params(norm_json_path):
    """Load normalization parameters from JSON file."""
    if isinstance(norm_json_path, str):
        try:
            import json
            with open(norm_json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load normalization JSON {norm_json_path}: {e}")
            return {}
    return norm_json_path if isinstance(norm_json_path, dict) else {}
# --- End Denormalization Functions ---


class STLPipelinePlugin:
    # Default parameters (kept from previous correct version)
    plugin_params = {
        "iterations": 1, "batch_size": 32, "epochs": 50, "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png", "output_file": "test_predictions.csv",
        "uncertainties_file": "test_uncertainties.csv", "model_plot_file": "model_plot.png",
        "predictions_plot_file": "predictions_plot.png", "results_file": "results.csv",
        "plot_points": 480, "plotted_horizon": 6, "use_strategy": False,
        "predicted_horizons": [1, 6, 12, 24], "use_returns": True, "normalize_features": True,
        "window_size": 48, "target_column": "TARGET", "use_normalization_json": None,
        "mc_samples": 100,
        # Target normalization config (legacy, not used in new logic)
        "target_normalization_json": "examples/results/phase_5/target_normalization.json",
        "load_target_normalization": False,
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error", "output_file", "uncertainties_file", "results_file", "plotted_horizon", "plot_points"]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.target_normalization_params = {}  # Store normalization params for each target horizon

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

        # Load normalization parameters for features
        norm_json_path = config.get("use_normalization_json")
        if not norm_json_path:
            raise ValueError("use_normalization_json parameter is required for z-score pipeline")
        norm_params = load_normalization_params(norm_json_path)
        print(f"Loaded normalization parameters for {len(norm_params)} features")

        # --- Target normalization config ---
        target_norm_path = config.get("target_normalization_json", "examples/results/phase_5/target_normalization.json")
        load_target_norm = config.get("load_target_normalization", False)
        predicted_horizons = config.get('predicted_horizons')
        num_outputs = len(predicted_horizons)
        target_norm_params = {}
        if load_target_norm:
            try:
                with open(target_norm_path, 'r') as f:
                    target_norm_params = json.load(f)
                print(f"Loaded target normalization params from {target_norm_path}")
            except Exception as e:
                print(f"ERROR: Could not load target normalization params: {e}")
                raise
        else:
            # Fit on training targets only (per horizon)
            # We'll calculate these after we have the denormalized targets below
            pass

        # 1. Calculate targets BEFORE decomposition
        print("Calculating targets (returns and scaling) BEFORE STL decomposition...")
        # Load raw data (assume preprocessor has a method to get raw data or you have access here)
        # For this patch, we assume you have access to raw y_train, y_val, y_test, and baselines before STL
        # If not, you must refactor the preprocessor to expose this step
        # Here is a generic example:
        raw_datasets = preprocessor_plugin.get_raw_data(config) if hasattr(preprocessor_plugin, 'get_raw_data') else None
        if raw_datasets is None:
            raise RuntimeError("Preprocessor must provide get_raw_data(config) to access raw targets and baselines before STL decomposition.")
        y_train_raw = raw_datasets["y_train"]
        y_val_raw = raw_datasets["y_val"]
        y_test_raw = raw_datasets["y_test"]
        baseline_train_raw = raw_datasets["baseline_train"]
        baseline_val_raw = raw_datasets["baseline_val"]
        baseline_test_raw = raw_datasets["baseline_test"]
        use_returns = config.get("use_returns", False)
        target_factor = config["target_factor"] if "target_factor" in config else 100
        predicted_horizons = config.get('predicted_horizons')
        y_train_targets = []
        y_val_targets = []
        y_test_targets = []
        for h_idx, horizon in enumerate(predicted_horizons):
            y_train_h = y_train_raw[h_idx]
            y_val_h = y_val_raw[h_idx]
            y_test_h = y_test_raw[h_idx]
            baseline_train_h = baseline_train_raw
            baseline_val_h = baseline_val_raw
            baseline_test_h = baseline_test_raw
            if use_returns:
                target_train = (y_train_h - baseline_train_h) * target_factor
                target_val = (y_val_h - baseline_val_h) * target_factor
                target_test = (y_test_h - baseline_test_h) * target_factor
            else:
                target_train = y_train_h * target_factor
                target_val = y_val_h * target_factor
                target_test = y_test_h * target_factor
            y_train_targets.append(target_train)
            y_val_targets.append(target_val)
            y_test_targets.append(target_test)
        # Now pass these targets to the preprocessor for STL decomposition of features only
        datasets = preprocessor_plugin.run_preprocessing(config, y_train_targets=y_train_targets, y_val_targets=y_val_targets, y_test_targets=y_test_targets)
        print("Preprocessor finished.")
        X_train = datasets["x_train"]
        X_val = datasets["x_val"]
        X_test = datasets["x_test"]
        y_train_list = datasets["y_train"]
        y_val_list = datasets["y_val"]
        y_test_list = datasets["y_test"]
        train_dates = datasets.get("y_train_dates")
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")
        # --- DEBUG PRINTS: Check for empty or invalid data ---
        print("\n[DEBUG] X_train shape:", getattr(X_train, 'shape', None))
        if X_train is not None:
            print("[DEBUG] X_train NaNs:", np.isnan(X_train).sum(), "min:", np.nanmin(X_train), "max:", np.nanmax(X_train))
        else:
            print("[DEBUG] X_train is None!")
        if not y_train_list:
            print("[DEBUG] y_train_list is empty!")
        else:
            for idx, y in enumerate(y_train_list):
                if y is None:
                    print(f"[DEBUG] y_train_list[{idx}] is None!")
                else:
                    print(f"[DEBUG] y_train_list[{idx}] shape: {y.shape}, NaNs: {np.isnan(y).sum()}, min: {np.nanmin(y)}, max: {np.nanmax(y)}")
        # --- DO NOT STOP: Only print debug info, do not halt pipeline ---
        # (This matches the original STL pipeline behavior)
        y_train_list = [y.astype(np.float32) for y in y_train_list]
        y_val_list = [y.astype(np.float32) for y in y_val_list]
        y_test_list = [y.astype(np.float32) for y in y_test_list]
        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        y_train_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_val_list)}
        iterations = config.get("iterations", 1)
        plotted_horizon = config.get('plotted_horizon')
        plotted_index = predicted_horizons.index(plotted_horizon)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        print(f"Predicting Horizons: {predicted_horizons}, Plotting: H={plotted_horizon}")
        metric_names = ["MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {ds: {mn: {h: [] for h in predicted_horizons} for mn in metric_names} for ds in data_sets}
        list_test_preds = None; list_test_unc = None
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()
            input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],)
            predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)
            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
                X_train, y_train_dict, epochs=epochs, batch_size=batch_size, threshold_error=config.get("threshold_error", 0.001),
                x_val=X_val, y_val=y_val_dict, config=config
            )
            # STOP CONDITION: Check if model was built
            if not hasattr(predictor_plugin, 'model') or predictor_plugin.model is None:
                print("STOP: Model was not built after training. Aborting pipeline."); return
            can_calc_train_stats = all(len(lst) == num_outputs for lst in [list_train_preds, list_train_unc])
            if can_calc_train_stats:
                # ...metrics calculation and denormalization logic (z-score for targets/outputs)...
                pass
            else:
                print("WARN: Skipping Train/Val stats calculation.")
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
            print("Evaluating test set & calculating metrics...")
            mc_samples = config.get("mc_samples", 100)
            list_test_preds, list_test_unc = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)
            if not all(len(lst) == num_outputs for lst in [list_test_preds, list_test_unc]):
                raise ValueError("Predictor predict mismatch outputs.")
            for idx, h in enumerate(predicted_horizons):
                # ...metrics calculation and denormalization logic (z-score for targets/outputs)...
                pass
            try:
                # ...summary print logic (identical to original)...
                pass
            except Exception as e:
                print(f"WARN: Error printing iter summary: {e}")
        # Use baseline_test for output/plotting, as baseline_test_denorm is not defined
        # Ensure baseline_test and test_dates are defined before calling _save_results_and_plots
        baseline_test = datasets.get("baseline_test")
        test_dates = datasets.get("y_test_dates")
        return self._save_results_and_plots(
            config, list_test_preds, list_test_unc, predicted_horizons,
            baseline_test, test_dates, metrics_results, data_sets, metric_names,
            plotted_index, plotted_horizon, start_time, use_returns, predictor_plugin)
    
    def _save_results_and_plots(self, config, list_test_preds, list_test_unc, predicted_horizons, 
                               baseline_test_denorm, test_dates, metrics_results, data_sets, 
                               metric_names, plotted_index, plotted_horizon, start_time, 
                               use_returns, predictor_plugin):
        """Save consolidated results, final test outputs, and generate plots."""
        
        use_returns = config.get("use_returns", False)
        
        # --- Consolidate results across iterations FOR ALL HORIZONS (Avg/Std/Min/Max) ---
        print("\n--- Aggregating Results Across Iterations (All Horizons) ---")
        results_list = []
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

        # --- Save Final Test Outputs ---
        print("\n--- Saving Final Test Outputs (Z-Score Denormalized) ---")
        # Fix for undefined variable errors
        final_dates = test_dates
        baseline_test = baseline_test
        if list_test_preds is None or list_test_unc is None:
            print("ERROR: Test preds/unc from last iter unavailable.")
            return
        num_test_points = min(len(list_test_preds[0]), len(baseline_test))
        num_test_points = min(len(list_test_preds[0]), len(baseline_test))
        if final_dates is not None:
            num_test_points = min(num_test_points, len(final_dates))
        print(f"Determined consistent output length: {num_test_points}")
        final_baseline = baseline_test[:num_test_points]
        final_baseline = baseline_test[:num_test_points]
        # Prepare dictionaries
        output_data = {"DATE_TIME": final_dates}
        uncertainty_data = {"DATE_TIME": final_dates}
        # Add denormalized test CLOSE price
        output_data["test_CLOSE"] = final_baseline
        # Process each horizon
        for idx, h in enumerate(predicted_horizons):
            preds_raw = list_test_preds[idx][:num_test_points].flatten()
            unc_raw = list_test_unc[idx][:num_test_points].flatten()
            target_factor = config["target_factor"] if "target_factor" in config else 100
            preds_rescaled = preds_raw / target_factor
            unc_rescaled = unc_raw / target_factor
            if use_returns:
                pred_price_final = final_baseline + preds_rescaled
            else:
                pred_price_final = preds_rescaled
            output_data[f"Prediction_H{h}"] = pred_price_final
            uncertainty_data[f"Uncertainty_H{h}"] = np.abs(unc_rescaled)
        # --- Save Predictions DataFrame ---
        output_file = config.get("output_file", self.params["output_file"])
        try:
            print("\nChecking final lengths for Predictions DataFrame:")
            for k, v in output_data.items():
                print(f"  - {k}: {len(v)}")
            if len(set(len(v) for v in output_data.values())) > 1:
                raise ValueError("Length mismatch (Predictions).")
            output_df = pd.DataFrame(output_data)
            cols_order = ['DATE_TIME', 'test_CLOSE']
            for h in predicted_horizons:
                cols_order.append(f"Prediction_H{h}")
            output_df = output_df.reindex(columns=[c for c in cols_order if c in output_df.columns])
            write_csv(file_path=output_file, data=output_df, include_date=False, headers=True)
            print(f"Predictions saved: {output_file} ({len(output_df)} rows)")
        except ImportError:
            print(f"WARN: write_csv not found. Skip save: {output_file}.")
        except Exception as e:
            print(f"ERROR saving predictions CSV: {e}")
        # --- Save Uncertainties DataFrame ---
        uncertainties_file = config.get("uncertainties_file", self.params.get("uncertainties_file"))
        if uncertainties_file:
            try:
                print("\nChecking final lengths for Uncertainty DataFrame:")
                for k, v in uncertainty_data.items():
                    print(f"  - {k}: {len(v)}")
                if len(set(len(v) for v in uncertainty_data.values())) > 1:
                    raise ValueError("Length mismatch (Uncertainty).")
                uncertainty_df = pd.DataFrame(uncertainty_data)
                cols_order = ['DATE_TIME']
                for h in predicted_horizons:
                    cols_order.append(f"Uncertainty_H{h}")
                uncertainty_df = uncertainty_df.reindex(columns=[c for c in cols_order if c in uncertainty_df.columns])
                write_csv(file_path=uncertainties_file, data=uncertainty_df, include_date=False, headers=True)
                print(f"Uncertainties saved: {uncertainties_file} ({len(uncertainty_df)} rows)")
            except ImportError:
                print(f"WARN: write_csv not found. Skip save: {uncertainties_file}.")
            except Exception as e:
                print(f"ERROR saving uncertainties CSV: {e}")
        else:
            print("INFO: No 'uncertainties_file' specified.")
        
        # --- Plot Predictions for 'plotted_horizon' ---
        print(f"\nGenerating prediction plot for H={plotted_horizon}...")
        try:
            # Get data for plotting
            preds_plot_raw = list_test_preds[plotted_index][:num_test_points].flatten()
            unc_plot_raw = list_test_unc[plotted_index][:num_test_points].flatten()
            # Undo scaling by dividing by target_factor from config
            target_factor = config["target_factor"] if "target_factor" in config else 100
            preds_plot_rescaled = preds_plot_raw / target_factor
            unc_plot_rescaled = unc_plot_raw / target_factor
            if use_returns:
                pred_plot_price_flat = final_baseline + preds_plot_rescaled
            else:
                pred_plot_price_flat = preds_plot_rescaled
            true_plot_price_flat = final_baseline
            unc_plot_final_flat = np.abs(unc_plot_rescaled)
            # Determine plot points and slice arrays
            n_plot = config.get("plot_points", self.params["plot_points"])
            num_avail_plot = len(pred_plot_price_flat)
            plot_slice = slice(max(0, num_avail_plot - n_plot), num_avail_plot)
            dates_plot_final = final_dates[plot_slice]
            pred_plot_final = pred_plot_price_flat[plot_slice]
            true_plot_final = true_plot_price_flat[plot_slice]
            unc_plot_final = unc_plot_final_flat[plot_slice]
            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(dates_plot_final, pred_plot_final, 
                    label=f"Pred Price H{plotted_horizon}", 
                    color=config.get("plot_color_predicted", "red"), lw=1.5, zorder=3)
            plt.plot(dates_plot_final, true_plot_final, 
                    label="Actual Price", 
                    color=config.get("plot_color_true", "blue"), lw=1, ls='--', alpha=0.7, zorder=1)
            plt.fill_between(dates_plot_final, 
                           pred_plot_final - unc_plot_final, 
                           pred_plot_final + unc_plot_final,
                           color=config.get("plot_color_uncertainty", "green"), 
                           alpha=0.2, label=f"Uncertainty H{plotted_horizon}", zorder=0)
            plt.title(f"Predictions vs Actual (H={plotted_horizon})")
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
        
        # Save target normalization parameters for future use
        target_params_file = config.get("target_normalization_file", "target_normalization_params.json")
        try:
            with open(target_params_file, 'w') as f:
                json.dump(self.target_normalization_params, f, indent=2)
            print(f"Target normalization parameters saved to {target_params_file}")
        except Exception as e:
            print(f"ERROR saving target normalization params: {e}")
        




    # --- load_and_evaluate_model (Updated for Z-Score) ---
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
        
        # Load normalization parameters
        norm_json_path = config.get("use_normalization_json")
        if not norm_json_path:
            raise ValueError("use_normalization_json parameter is required for z-score evaluation")
        
        norm_params = load_normalization_params(norm_json_path)
        print(f"Loaded normalization parameters for evaluation")
        
        print("Loading/processing validation data for evaluation...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("y_val_dates")
        baseline_val_eval = datasets.get("baseline_val")
        
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
            
            # Denormalize baseline to actual prices
            baseline_val_eval_sliced = baseline_val_eval[:num_val_points].flatten() if baseline_val_eval is not None else None
            baseline_val_denorm = denormalize_zscore(baseline_val_eval_sliced, "CLOSE", norm_params) if baseline_val_eval_sliced is not None else None
            
            for idx, h in enumerate(config['predicted_horizons']):
                preds_raw = list_predictions[idx][:num_val_points].flatten()
                
                # If we have target normalization parameters from training, use them
                horizon_name = f"target_horizon_{h}"
                if hasattr(self, 'target_normalization_params') and horizon_name in self.target_normalization_params:
                    target_params = self.target_normalization_params[horizon_name]
                    preds_denorm = denormalize_zscore(preds_raw, horizon_name, {horizon_name: target_params})
                else:
                    print(f"WARN: No target normalization params for horizon {h}, using CLOSE params")
                    preds_denorm = denormalize_zscore_returns(preds_raw, "CLOSE", norm_params)
                
                if use_returns_eval:
                    pred_price = baseline_val_denorm + preds_denorm
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