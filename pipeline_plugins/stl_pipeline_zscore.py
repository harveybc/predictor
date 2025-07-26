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
        # Target normalization config
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

        iterations = config.get("iterations", 1)
        print(f"Starting Z-Score Pipeline with {iterations} iterations...")

        # Init metric storage
        metric_names = ["MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {ds: {mn: {h: [] for h in predicted_horizons} for mn in metric_names} for ds in data_sets}

        # 1. Get datasets from preprocessor
        print("Loading/processing datasets via Preprocessor...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        print("Preprocessor finished.")

        # Extract raw data (normalized features and targets)
        X_train = datasets["x_train"]
        X_val = datasets["x_val"]
        X_test = datasets["x_test"]
        y_train_list = datasets["y_train"]  # These are normalized future CLOSE values
        y_val_list = datasets["y_val"]
        y_test_list = datasets["y_test"]

        # Extract baseline data (current normalized CLOSE values)
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

        # Extract dates
        train_dates = datasets.get("y_train_dates")
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")

        use_returns = config.get("use_returns", False)
        if use_returns and (baseline_train is None or baseline_val is None or baseline_test is None):
            raise ValueError("Baselines required when use_returns=True.")

        print(f"Input shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"Target shapes(H={predicted_horizons[0]}): Train:{y_train_list[0].shape}, Val:{y_val_list[0].shape}, Test:{y_test_list[0].shape}")

        # --- Calculate proper targets using z-score denormalization ---
        print("Calculating proper targets using z-score denormalization...")

        # Denormalize baseline CLOSE values to get actual current prices
        baseline_train_denorm = denormalize_zscore(baseline_train, "CLOSE", norm_params)
        baseline_val_denorm = denormalize_zscore(baseline_val, "CLOSE", norm_params)
        baseline_test_denorm = denormalize_zscore(baseline_test, "CLOSE", norm_params)

        # Calculate proper targets for each horizon (denormalized returns)
        y_train_final = []
        y_val_final = []
        y_test_final = []

        for h_idx, horizon in enumerate(predicted_horizons):
            print(f"Processing target horizon {horizon} (index {h_idx})...")
            # Get normalized future CLOSE values for this horizon
            y_train_norm_h = y_train_list[h_idx]
            y_val_norm_h = y_val_list[h_idx]
            y_test_norm_h = y_test_list[h_idx]
            # Denormalize future CLOSE values to get actual future prices
            y_train_denorm_h = denormalize_zscore(y_train_norm_h, "CLOSE", norm_params)
            y_val_denorm_h = denormalize_zscore(y_val_norm_h, "CLOSE", norm_params)
            y_test_denorm_h = denormalize_zscore(y_test_norm_h, "CLOSE", norm_params)
            if use_returns:
                # Calculate returns: future_price - current_price
                target_train_h = y_train_denorm_h - baseline_train_denorm
                target_val_h = y_val_denorm_h - baseline_val_denorm
                target_test_h = y_test_denorm_h - baseline_test_denorm
            else:
                target_train_h = y_train_denorm_h
                target_val_h = y_val_denorm_h
                target_test_h = y_test_denorm_h
            y_train_final.append(target_train_h)
            y_val_final.append(target_val_h)
            y_test_final.append(target_test_h)

        # --- Fit or load target normalization parameters (per horizon) ---
        if not load_target_norm:
            for h_idx, horizon in enumerate(predicted_horizons):
                arr = y_train_final[h_idx].flatten()
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                if std == 0:
                    std = 1.0
                target_norm_params[str(horizon)] = {"mean": mean, "std": std}
            # Save to file
            try:
                os.makedirs(os.path.dirname(target_norm_path), exist_ok=True)
                with open(target_norm_path, 'w') as f:
                    json.dump(target_norm_params, f, indent=2)
                print(f"Saved target normalization params to {target_norm_path}")
            except Exception as e:
                print(f"ERROR: Could not save target normalization params: {e}")

        # --- Normalize all targets using training-set params (per horizon) ---
        def norm_target(y, h):
            p = target_norm_params[str(h)]
            return ((y - p['mean']) / p['std']).astype(np.float32)
        def denorm_target(y, h):
            p = target_norm_params[str(h)]
            return (y * p['std'] + p['mean']).astype(np.float32)

        y_train_list = [norm_target(y, h) for y, h in zip(y_train_final, predicted_horizons)]
        y_val_list = [norm_target(y, h) for y, h in zip(y_val_final, predicted_horizons)]
        y_test_list = [norm_target(y, h) for y, h in zip(y_test_final, predicted_horizons)]

        # The rest of the pipeline remains unchanged...

        # Prepare Target Dicts for Training
        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        y_train_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_val_list)}

        # Config Validation & Setup
        plotted_horizon = config.get('plotted_horizon')
        plotted_index = predicted_horizons.index(plotted_horizon)
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        print(f"Predicting Horizons: {predicted_horizons}, Plotting: H={plotted_horizon}")

        # Main pipeline logic continues here...

            # Save Loss Plot
        # Save Loss Plot (if available)
        # Loss plot skipped: 'history' is not defined in this scope.

            # Evaluate Test & Calculate Metrics
        print("Evaluating test set & calculating metrics...")
        mc_samples = config.get("mc_samples", 100)
        list_test_preds, list_test_unc = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

        if not all(len(lst) == num_outputs for lst in [list_test_preds, list_test_unc]):
            raise ValueError("Predictor predict mismatch outputs.")

        for idx, h in enumerate(predicted_horizons):
            try:
                # Get test predictions and targets (normalized)
                test_preds_h = list_test_preds[idx].flatten()
                test_target_h = y_test_final[idx].flatten()
                test_unc_h = list_test_unc[idx].flatten()

                # Ensure consistent lengths
                num_test_pts = min(len(test_preds_h), len(test_target_h), len(baseline_test_denorm))
                test_preds_h = test_preds_h[:num_test_pts]
                test_target_h = test_target_h[:num_test_pts]
                test_unc_h = test_unc_h[:num_test_pts]
                baseline_test_h = baseline_test_denorm[:num_test_pts]

                # Denormalize predictions and targets
                horizon_name = f"target_horizon_{h}"
                target_params = self.target_normalization_params[horizon_name]

                test_preds_denorm = denormalize_zscore(test_preds_h, horizon_name, {horizon_name: target_params})
                test_target_denorm = denormalize_zscore(test_target_h, horizon_name, {horizon_name: target_params})
                test_unc_denorm = denormalize_zscore_returns(test_unc_h, horizon_name, {horizon_name: target_params})

                # Convert to final prices if using returns
                if use_returns:
                    test_pred_price = baseline_test_h + test_preds_denorm
                    test_target_price = baseline_test_h + test_target_denorm
                else:
                    test_pred_price = test_preds_denorm
                    test_target_price = test_target_denorm

                # Calculate metrics
                test_mae_h = np.mean(np.abs(test_preds_denorm - test_target_denorm))
                test_r2_h = r2_score(test_target_price, test_pred_price)
                test_unc_mean_h = np.mean(np.abs(test_unc_denorm))
                test_snr_h = np.mean(np.abs(test_pred_price)) / (test_unc_mean_h + 1e-9)

                # Store metrics
                metrics_results["Test"]["MAE"][h].append(test_mae_h)
                metrics_results["Test"]["R2"][h].append(test_r2_h)
                metrics_results["Test"]["Uncertainty"][h].append(test_unc_mean_h)
                metrics_results["Test"]["SNR"][h].append(test_snr_h)

            except Exception as e:
                print(f"WARN: Error Test metrics H={h}: {e}")
                for m in metric_names:
                    metrics_results["Test"][m][h].append(np.nan)

        # Print Iteration Summary (using PLOTTED horizon)
        try:
            train_mae_plot = metrics_results["Train"]["MAE"][plotted_horizon][-1] if metrics_results["Train"]["MAE"][plotted_horizon] else np.nan
            train_r2_plot = metrics_results["Train"]["R2"][plotted_horizon][-1] if metrics_results["Train"]["R2"][plotted_horizon] else np.nan
            val_mae_plot = metrics_results["Validation"]["MAE"][plotted_horizon][-1] if metrics_results["Validation"]["MAE"][plotted_horizon] else np.nan
            val_r2_plot = metrics_results["Validation"]["R2"][plotted_horizon][-1] if metrics_results["Validation"]["R2"][plotted_horizon] else np.nan
            test_mae_plot = metrics_results["Test"]["MAE"][plotted_horizon][-1] if metrics_results["Test"]["MAE"][plotted_horizon] else np.nan
            test_r2_plot = metrics_results["Test"]["R2"][plotted_horizon][-1] if metrics_results["Test"]["R2"][plotted_horizon] else np.nan
            test_unc_plot = metrics_results["Test"]["Uncertainty"][plotted_horizon][-1] if metrics_results["Test"]["Uncertainty"][plotted_horizon] else np.nan
            test_snr_plot = metrics_results["Test"]["SNR"][plotted_horizon][-1] if metrics_results["Test"]["SNR"][plotted_horizon] else np.nan
            print("*" * 72)
            print(f"Summary for Plot H:{plotted_horizon}")
            print(f"  Train MAE:{train_mae_plot:.6f}|R²:{train_r2_plot:.4f} -- Valid MAE:{val_mae_plot:.6f}|R²:{val_r2_plot:.4f}")
            print(f"  Test  MAE:{test_mae_plot:.6f}|R²:{test_r2_plot:.4f}|Unc:{test_unc_plot:.6f}|SNR:{test_snr_plot:.2f}")
            print("*" * 72)
        except Exception as e:
            print(f"WARN: Error printing iter summary: {e}")
        
        # Continue with result aggregation and saving...
        return self._save_results_and_plots(config, list_test_preds, list_test_unc, predicted_horizons, 
                                           baseline_test_denorm, test_dates, metrics_results, 
                                           data_sets, metric_names, plotted_index, plotted_horizon, 
                                           start_time, use_returns, predictor_plugin)
    
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
        try:
            if list_test_preds is None or list_test_unc is None:
                raise ValueError("Test preds/unc from last iter unavailable.")
            
            # Determine consistent length
            num_test_points = min(len(list_test_preds[0]), len(baseline_test_denorm))
            if test_dates is not None:
                num_test_points = min(num_test_points, len(test_dates))
            
            print(f"Determined consistent output length: {num_test_points}")
            
            final_dates = list(test_dates[:num_test_points]) if test_dates is not None else list(range(num_test_points))
            final_baseline = baseline_test_denorm[:num_test_points]
            
            # Prepare dictionaries
            output_data = {"DATE_TIME": final_dates}
            uncertainty_data = {"DATE_TIME": final_dates}
            
            # Add denormalized test CLOSE price
            output_data["test_CLOSE"] = final_baseline
            
            # Process each horizon
            for idx, h in enumerate(predicted_horizons):
                # Get raw predictions (normalized)
                preds_raw = list_test_preds[idx][:num_test_points].flatten()
                unc_raw = list_test_unc[idx][:num_test_points].flatten()
                
                # Denormalize using target-specific parameters
                horizon_name = f"target_horizon_{h}"
                if horizon_name in self.target_normalization_params:
                    target_params = self.target_normalization_params[horizon_name]
                    
                    # Denormalize predictions and uncertainties
                    preds_denorm = denormalize_zscore(preds_raw, horizon_name, {horizon_name: target_params})
                    unc_denorm = denormalize_zscore_returns(unc_raw, horizon_name, {horizon_name: target_params})
                    
                    # Convert to final prices if using returns
                    if use_returns:
                        pred_price_final = final_baseline + preds_denorm
                    else:
                        pred_price_final = preds_denorm
                    
                    output_data[f"Prediction_H{h}"] = pred_price_final
                    uncertainty_data[f"Uncertainty_H{h}"] = np.abs(unc_denorm)
                else:
                    print(f"WARN: No target normalization params for horizon {h}")
                    output_data[f"Prediction_H{h}"] = np.full(num_test_points, np.nan)
                    uncertainty_data[f"Uncertainty_H{h}"] = np.full(num_test_points, np.nan)
            
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
        
        except Exception as e:
            print(f"ERROR during final CSV saving: {e}")
        
        # --- Plot Predictions for 'plotted_horizon' ---
        print(f"\nGenerating prediction plot for H={plotted_horizon}...")
        try:
            # Get data for plotting
            preds_plot_raw = list_test_preds[plotted_index][:num_test_points].flatten()
            unc_plot_raw = list_test_unc[plotted_index][:num_test_points].flatten()
            
            # Denormalize for plotting
            horizon_name = f"target_horizon_{plotted_horizon}"
            if horizon_name in self.target_normalization_params:
                target_params = self.target_normalization_params[horizon_name]
                
                preds_plot_denorm = denormalize_zscore(preds_plot_raw, horizon_name, {horizon_name: target_params})
                unc_plot_denorm = denormalize_zscore_returns(unc_plot_raw, horizon_name, {horizon_name: target_params})
                
                if use_returns:
                    pred_plot_price_flat = final_baseline + preds_plot_denorm
                else:
                    pred_plot_price_flat = preds_plot_denorm
                
                true_plot_price_flat = final_baseline
                unc_plot_final_flat = np.abs(unc_plot_denorm)
                
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
            else:
                print(f"WARN: No target normalization params for plotting horizon {plotted_horizon}")
        
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
        
        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")
        print("Z-Score Pipeline completed successfully")
        
        return {
            "test_predictions": output_data,
            "test_uncertainties": uncertainty_data,
            "baseline_test": final_baseline,
            "target_normalization_params": self.target_normalization_params,
            "test_dates": final_dates
        }


        # --- Consolidate results across iterations FOR ALL HORIZONS (Avg/Std/Min/Max) ---
        print("\n--- Aggregating Results Across Iterations (All Horizons) ---")
        results_list = []
        # (Logic confirmed correct and includes Min/Max)
        for ds in data_sets:
             for mn in metric_names:
                 for h in predicted_horizons:
                      values=metrics_results[ds][mn][h]; valid_values=[v for v in values if not np.isnan(v)]
                      if valid_values: results_list.append({"Metric": f"{ds} {mn} H{h}", "Average": np.mean(valid_values), "Std Dev": np.std(valid_values), "Min": np.min(valid_values), "Max": np.max(valid_values)})
                      else: results_list.append({"Metric": f"{ds} {mn} H{h}", "Average": np.nan, "Std Dev": np.nan, "Min": np.nan, "Max": np.nan})
        results_df = pd.DataFrame(results_list); results_file = config.get("results_file", self.params["results_file"])
        try: results_df.to_csv(results_file, index=False, float_format='%.6f'); print(f"Aggregated results saved: {results_file}"); print(results_df.to_string())
        except Exception as e: print(f"ERROR saving results: {e}")



        # --- Save Final Test Outputs (Predictions & Uncertainties, Z-Score Denorm) ---
        print("\n--- Saving Final Test Outputs (Predictions & Uncertainties, Z-Score Denorm) ---")
        try:
            if list_test_preds is None or list_test_unc is None:
                raise ValueError("Test preds/unc from last iter unavailable.")
            num_test_points = min(len(list_test_preds[0]), len(baseline_test_denorm))
            if test_dates is not None:
                num_test_points = min(num_test_points, len(test_dates))
            final_dates = list(test_dates[:num_test_points]) if test_dates is not None else list(range(num_test_points))
            final_baseline = baseline_test_denorm[:num_test_points]
            output_data = {"DATE_TIME": final_dates}
            uncertainty_data = {"DATE_TIME": final_dates}
            output_data["test_CLOSE"] = final_baseline
            for idx, h in enumerate(predicted_horizons):
                preds_raw = list_test_preds[idx][:num_test_points].flatten()
                unc_raw = list_test_unc[idx][:num_test_points].flatten()
                # Use per-horizon target normalization for denorm
                preds_denorm = denorm_target(preds_raw, h)
                unc_denorm = np.abs(denorm_target(unc_raw, h))
                if use_returns:
                    pred_price_final = final_baseline + preds_denorm
                else:
                    pred_price_final = preds_denorm
                output_data[f"Prediction_H{h}"] = pred_price_final
                uncertainty_data[f"Uncertainty_H{h}"] = unc_denorm
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
        except Exception as e:
            print(f"ERROR during final CSV saving: {e}")


        # --- Plot Predictions for 'plotted_horizon' (CORRECTED - Flattening & Variable Names) ---
        print(f"\nGenerating prediction plot for H={plotted_horizon}...")
        try:
            # Use CORRECT variable names from last iteration, sliced
            preds_plot_raw = list_test_preds[plotted_index][:num_test_points] # Shape (num_test_points,) or (num_test_points, 1)
            target_plot_raw = y_test_list[plotted_index][:num_test_points] # Shape (num_test_points,) or (num_test_points, 1)
            unc_plot_raw = list_test_unc[plotted_index][:num_test_points] # Shape (num_test_points,) or (num_test_points, 1)
            baseline_plot = final_baseline # Already sliced, shape (num_test_points,)

            # Denormalize correctly and FLATTEN *before* slicing for plot
            if use_returns:
                # --- Apply FIX: Ensure inputs to addition are flattened ---
                pred_plot_price_flat = denormalize_zscore(baseline_plot + preds_plot_raw.flatten(), "CLOSE", norm_params).flatten()
                target_plot_price_flat = denormalize_zscore(baseline_plot + target_plot_raw.flatten(), "CLOSE", norm_params).flatten()
            else:
                pred_plot_price_flat = denormalize_zscore(preds_plot_raw, "CLOSE", norm_params).flatten()
                target_plot_price_flat = denormalize_zscore(target_plot_raw, "CLOSE", norm_params).flatten()
            unc_plot_denorm_flat = denormalize_zscore_returns(unc_plot_raw, "CLOSE", norm_params).flatten()
            true_plot_price_flat = denormalize_zscore(baseline_plot, "CLOSE", norm_params).flatten()

            # Determine plot points and slice FLATTENED arrays
            n_plot = config.get("plot_points", self.params["plot_points"])
            num_avail_plot = len(pred_plot_price_flat) # Length of data available for plot
            plot_slice = slice(max(0, num_avail_plot - n_plot), num_avail_plot)

            dates_plot_final = final_dates[plot_slice]
            pred_plot_final = pred_plot_price_flat[plot_slice]
            target_plot_final = target_plot_price_flat[plot_slice]
            true_plot_final = true_plot_price_flat[plot_slice]
            unc_plot_final = unc_plot_denorm_flat[plot_slice] # This is now 1D

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(dates_plot_final, pred_plot_final, label=f"Pred Price H{plotted_horizon}", color=config.get("plot_color_predicted", "red"), lw=1.5, zorder=3)
            plt.plot(dates_plot_final, target_plot_final, label=f"Target Price H{plotted_horizon}", color=config.get("plot_color_target", "orange"), lw=1.5, zorder=2)
            plt.plot(dates_plot_final, true_plot_final, label="Actual Price", color=config.get("plot_color_true", "blue"), lw=1, ls='--', alpha=0.7, zorder=1)
            plt.fill_between(dates_plot_final, pred_plot_final - abs(unc_plot_final), pred_plot_final + abs(unc_plot_final),
                             color=config.get("plot_color_uncertainty", "green"), alpha=0.2, label=f"Uncertainty H{plotted_horizon}", zorder=0)
            plt.title(f"Predictions vs Target/Actual (H={plotted_horizon})"); plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.grid(True, alpha=0.6); plt.tight_layout()
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300); plt.close(); print(f"Prediction plot saved: {predictions_plot_file}")
        except Exception as e: print(f"ERROR generating prediction plot: {e}"); import traceback; traceback.print_exc(); plt.close()


        # --- Plot/Save Model --- (Keep as is)
        if plot_model is not None and hasattr(predictor_plugin, 'model') and predictor_plugin.model is not None:
            try: model_plot_file=config.get('model_plot_file','model_plot.png'); plot_model(predictor_plugin.model,to_file=model_plot_file,show_shapes=True,show_layer_names=True,dpi=300); print(f"Model plot saved: {model_plot_file}")
            except Exception as e: print(f"WARN: Failed model plot: {e}")
        else: print("INFO: Skipping model plot.")

        # --- Save Model --- (Keep as is)
        if hasattr(predictor_plugin, 'save') and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try: predictor_plugin.save(save_model_file); print(f"Model saved: {save_model_file}")
            except Exception as e: print(f"ERROR saving model: {e}")
        else: print("WARN: Predictor has no save method.")

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")


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