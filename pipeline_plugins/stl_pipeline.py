#!/usr/bin/env python
"""
STL Pipeline Plugin - Corrected Version

This plugin orchestrates the complete forecasting flow:
  1. Obtains datasets from a (now assumed perfect) Preprocessor Plugin.
  2. Executes iterations of training, validation, and evaluation using the Predictor Plugin.
  3. Calculates and prints metrics (MAE, R², uncertainty, SNR).
  4. Generates and saves loss plot and prediction plot (for plotted_horizon).
  5. Saves consolidated results (Avg, Std, Min, Max for ALL horizons) to results_file.
  6. Saves denormalized predictions/targets/close to output_file.
  7. Saves denormalized uncertainties to a separate uncertainties_file.
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
# Conditional import for plot_model
try:
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None # Handle optional dependency
# Assume necessary TF/Keras imports happen elsewhere or within predictor/callbacks
import tensorflow as tf
import tensorflow.keras.backend as K
# --- Assuming write_csv is correctly imported ---
from app.data_handler import write_csv


# --- Denormalization Functions (Keep as provided) ---
def denormalize(data, config):
    """
    Denormalizes the data using the provided configuration. Assumes data is price or price delta.
    """
    # Check if data needs conversion (e.g., from list/scalar)
    data = np.asarray(data) # Ensure it's a numpy array

    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f: norm_json = json.load(f)
            except Exception as e: print(f"WARN: Failed to load norm JSON {norm_json}: {e}"); return data # Return original if fail
        if isinstance(norm_json, dict) and "CLOSE" in norm_json:
            try:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                if diff == 0: return data + close_min # Avoid division by zero if max=min
                # Assuming input 'data' is normalized between 0 and 1 if min/max scaling was used
                # Or assumes input 'data' is Z-score normalized if mean/std scaling was used (adjust if needed)
                # The provided logic seems to assume min-max scaling was used: data * (max-min) + min
                return data * diff + close_min
            except KeyError as e: print(f"WARN: Missing key in norm JSON: {e}"); return data
            except Exception as e: print(f"WARN: Error during denormalize: {e}"); return data
    # print("WARN: No normalization JSON found or 'CLOSE' key missing. Returning original data.") # Can be noisy
    return data # Return original if no normalization info

def denormalize_returns(data, config):
    """
    Denormalizes return values (deltas) using the provided configuration.
    Only scales by the range, does not add the minimum.
    """
    data = np.asarray(data) # Ensure numpy array

    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                 with open(norm_json, 'r') as f: norm_json = json.load(f)
            except Exception as e: print(f"WARN: Failed to load norm JSON {norm_json}: {e}"); return data
        if isinstance(norm_json, dict) and "CLOSE" in norm_json:
            try:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                if diff == 0: return data # Return 0 if range is 0
                # Assuming input 'data' represents a normalized difference/return
                return data * diff
            except KeyError as e: print(f"WARN: Missing key in norm JSON: {e}"); return data
            except Exception as e: print(f"WARN: Error during denormalize_returns: {e}"); return data
    return data # Return original if no normalization info
# --- End Denormalization Functions ---


class STLPipelinePlugin:
    # Default pipeline parameters - REINSTATED uncertainties_file
    plugin_params = {
        "iterations": 1,
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv", # Predictions/Targets/Close
        "uncertainties_file": "test_uncertainties.csv", # Separate file for uncertainties
        "model_plot_file": "model_plot.png",
        "predictions_plot_file": "predictions_plot.png",
        "results_file": "results.csv", # Aggregated stats for ALL horizons
        "plot_points": 1575,
        "plotted_horizon": 6, # Horizon used for iteration summary plots/prints
        "use_strategy": False # Keep as in provided code
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items(): self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin):
        """
        Executes the complete forecasting pipeline for multi-output models.
        FIXED: Denormalization, Separate Uncertainty File, All Horizon Stats (Avg/Std/Min/Max).
        """
        start_time = time.time()
        run_config = self.params.copy(); run_config.update(config); config = run_config

        iterations = config.get("iterations", 1); print(f"Number of iterations: {iterations}")

        # --- Initialize metric storage FOR ALL HORIZONS ---
        predicted_horizons = config.get('predicted_horizons')
        if not predicted_horizons: raise ValueError("'predicted_horizons' is missing or empty.")
        num_outputs = len(predicted_horizons)

        # Use dictionaries to store lists of metrics per horizon per set
        metric_names = ["MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {
            ds: {
                mn: {h: [] for h in predicted_horizons} for mn in metric_names
            } for ds in data_sets
        }
        # Example access: metrics_results["Test"]["MAE"][12].append(mae_value)

        # 1. Get datasets from Preprocessor Plugin (ASSUMED PERFECT)
        print("Loading and processing datasets via Preprocessor Plugin...")
        try:
             datasets = preprocessor_plugin.run_preprocessing(config)
             print("Preprocessor finished.")
        except Exception as e:
             print(f"\nCRITICAL ERROR during preprocessing: {e}")
             import traceback; traceback.print_exc(); raise

        # Unpack datasets
        X_train=datasets["x_train"]; X_val=datasets["x_val"]; X_test=datasets["x_test"]
        y_train_list=datasets["y_train"]; y_val_list=datasets["y_val"]; y_test_list=datasets["y_test"]
        train_dates=datasets.get("y_train_dates"); val_dates=datasets.get("y_val_dates"); test_dates=datasets.get("y_test_dates")
        # test_close_prices = datasets.get("test_close_prices") # Original seems to use baseline_test instead
        baseline_train=datasets.get("baseline_train"); baseline_val=datasets.get("baseline_val"); baseline_test=datasets.get("baseline_test")
        # Ensure baselines are available if use_returns is True
        use_returns = config.get("use_returns", False)
        if use_returns and (baseline_train is None or baseline_val is None or baseline_test is None):
            raise ValueError("Baselines (train/val/test) are required when use_returns=True but are missing.")


        # --- Config Validation & Setup ---
        plotted_horizon = config.get('plotted_horizon')
        if plotted_horizon is None or plotted_horizon not in predicted_horizons:
            raise ValueError(f"'{plotted_horizon=}' invalid or not in {predicted_horizons=}.")
        plotted_index = predicted_horizons.index(plotted_horizon)
        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        plotted_output_name = output_names[plotted_index]

        # --- Prepare Target Data Dictionaries for Training ---
        if not all(len(lst)==num_outputs for lst in [y_train_list, y_val_list, y_test_list]): raise ValueError("Y lists length mismatch.")
        y_train_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: y.reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_val_list)}

        # Get raw target for the specific plotted horizon (for iteration summary)
        y_test_plot_target_raw = y_test_list[plotted_index].reshape(-1, 1).astype(np.float32)


        print(f"Input shapes: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Target shapes (e.g., H={predicted_horizons[0]}): Train: {y_train_list[0].shape}, Val: {y_val_list[0].shape}, Test: {y_test_list[0].shape}")
        window_size = config.get("window_size"); batch_size = config.get("batch_size", 32); epochs = config.get("epochs", 50)
        print(f"Predicting Horizons: {predicted_horizons}, Plotting Horizon: {plotted_horizon}")


        # --- Iteration Loop ---
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()

            # --- Build & Train Model ---
            input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim==3 else (X_train.shape[1],)
            predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)
            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
                X_train, y_train_dict, epochs=epochs, batch_size=batch_size,
                threshold_error=config.get("threshold_error", 0.001),
                x_val=X_val, y_val=y_val_dict, config=config
            ) # Assuming train returns lists for all horizons

            # --- Check returned prediction/uncertainty list lengths ---
            if not all(len(lst) == num_outputs for lst in [list_train_preds, list_train_unc, list_val_preds, list_val_unc]):
                 print("WARN: Predictor train method did not return lists matching the number of output horizons. Cannot calculate full stats.")
                 # Fallback: Only calculate for plotted horizon? Or raise error?
                 # For now, we will proceed, but subsequent loops might fail.
                 # Consider modifying predictor plugin's train method.

            # --- Calculate & Store Train/Val Metrics FOR ALL HORIZONS ---
            print("Calculating Train/Validation metrics for all horizons...")
            for idx, h in enumerate(predicted_horizons):
                try:
                    # Ensure lists have enough elements before accessing index `idx`
                    if idx >= len(list_train_preds) or idx >= len(list_train_unc) or \
                       idx >= len(list_val_preds) or idx >= len(list_val_unc) or \
                       idx >= len(y_train_list) or idx >= len(y_val_list):
                        print(f"WARN: Skipping Train/Val metrics for H={h} - insufficient data returned from train.")
                        continue

                    # Get data for horizon h
                    train_preds_h = list_train_preds[idx].flatten()
                    train_target_h = y_train_list[idx].flatten()
                    train_unc_h = list_train_unc[idx].flatten()
                    val_preds_h = list_val_preds[idx].flatten()
                    val_target_h = y_val_list[idx].flatten()
                    val_unc_h = list_val_unc[idx].flatten()

                    # Align lengths (use validation length as reference, usually shortest if early stopping)
                    num_train_pts = min(len(train_preds_h), len(train_target_h), len(baseline_train))
                    num_val_pts = min(len(val_preds_h), len(val_target_h), len(baseline_val))

                    train_preds_h = train_preds_h[:num_train_pts]
                    train_target_h = train_target_h[:num_train_pts]
                    train_unc_h = train_unc_h[:num_train_pts]
                    val_preds_h = val_preds_h[:num_val_pts]
                    val_target_h = val_target_h[:num_val_pts]
                    val_unc_h = val_unc_h[:num_val_pts]
                    baseline_train_h = baseline_train[:num_train_pts]
                    baseline_val_h = baseline_val[:num_val_pts]

                    # Calculate denormalized prices
                    if use_returns:
                        train_target_price = denormalize(baseline_train_h + train_target_h, config)
                        train_pred_price = denormalize(baseline_train_h + train_preds_h, config)
                        val_target_price = denormalize(baseline_val_h + val_target_h, config)
                        val_pred_price = denormalize(baseline_val_h + val_preds_h, config)
                    else:
                        train_target_price = denormalize(train_target_h, config)
                        train_pred_price = denormalize(train_preds_h, config)
                        val_target_price = denormalize(val_target_h, config)
                        val_pred_price = denormalize(val_preds_h, config)

                    # Calculate metrics for horizon h
                    train_mae_h = np.mean(np.abs(denormalize_returns(train_preds_h - train_target_h, config)))
                    train_r2_h = r2_score(train_target_price, train_pred_price)
                    train_unc_mean_h = np.mean(np.abs(denormalize_returns(train_unc_h, config))) # Avg uncertainty magnitude
                    train_pred_mean = np.mean(train_pred_price)
                    train_snr_h = train_pred_mean / (train_unc_mean_h + 1e-9) if train_unc_mean_h > 0 else np.inf

                    val_mae_h = np.mean(np.abs(denormalize_returns(val_preds_h - val_target_h, config)))
                    val_r2_h = r2_score(val_target_price, val_pred_price)
                    val_unc_mean_h = np.mean(np.abs(denormalize_returns(val_unc_h, config)))
                    val_pred_mean = np.mean(val_pred_price)
                    val_snr_h = val_pred_mean / (val_unc_mean_h + 1e-9) if val_unc_mean_h > 0 else np.inf

                    # Store metrics
                    metrics_results["Train"]["MAE"][h].append(train_mae_h)
                    metrics_results["Train"]["R2"][h].append(train_r2_h)
                    metrics_results["Train"]["Uncertainty"][h].append(train_unc_mean_h)
                    metrics_results["Train"]["SNR"][h].append(train_snr_h)
                    metrics_results["Validation"]["MAE"][h].append(val_mae_h)
                    metrics_results["Validation"]["R2"][h].append(val_r2_h)
                    metrics_results["Validation"]["Uncertainty"][h].append(val_unc_mean_h)
                    metrics_results["Validation"]["SNR"][h].append(val_snr_h)

                except Exception as e:
                    print(f"WARN: Error calculating Train/Val metrics for H={h}: {e}")
                    # Append NaN if calculation failed
                    for metric in metric_names:
                         metrics_results["Train"][metric][h].append(np.nan)
                         metrics_results["Validation"][metric][h].append(np.nan)


            # --- Save Loss Plot ---
            loss_plot_file = config.get("loss_plot_file", self.params["loss_plot_file"])
            try:
                plt.figure(figsize=(10, 5)); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
                plt.title(f"Loss - Iter {iteration}"); plt.ylabel("Loss"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True, alpha=0.6)
                plt.savefig(loss_plot_file); plt.close(); print(f"Loss plot saved: {loss_plot_file}")
            except Exception as e: print(f"WARN: Failed loss plot: {e}"); plt.close()


            # --- Evaluate on Test Dataset & Calculate Test Metrics FOR ALL HORIZONS ---
            print("Evaluating on test dataset (MC sampling) & calculating metrics...")
            mc_samples = config.get("mc_samples", 100)
            list_test_predictions, list_uncertainty_estimates = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

            # Check lengths
            if not all(len(lst) == num_outputs for lst in [list_test_predictions, list_uncertainty_estimates]):
                 raise ValueError("Predictor predict method did not return lists matching number of outputs.")

            for idx, h in enumerate(predicted_horizons):
                 try:
                     # Get data for horizon h
                     test_preds_h = list_test_predictions[idx].flatten()
                     test_target_h = y_test_list[idx].flatten()
                     test_unc_h = list_uncertainty_estimates[idx].flatten()

                     # Align lengths
                     num_test_pts = min(len(test_preds_h), len(test_target_h), len(baseline_test))
                     test_preds_h = test_preds_h[:num_test_pts]
                     test_target_h = test_target_h[:num_test_pts]
                     test_unc_h = test_unc_h[:num_test_pts]
                     baseline_test_h = baseline_test[:num_test_pts]

                     # Calculate denormalized prices (CORRECTED: add baseline BEFORE denormalize)
                     if use_returns:
                         test_target_price = denormalize(baseline_test_h + test_target_h, config)
                         test_pred_price = denormalize(baseline_test_h + test_preds_h, config)
                     else:
                         test_target_price = denormalize(test_target_h, config)
                         test_pred_price = denormalize(test_preds_h, config)

                     # Calculate metrics for horizon h
                     test_mae_h = np.mean(np.abs(denormalize_returns(test_preds_h - test_target_h, config)))
                     test_r2_h = r2_score(test_target_price, test_pred_price)
                     test_unc_mean_h = np.mean(np.abs(denormalize_returns(test_unc_h, config))) # Avg uncertainty magnitude
                     test_pred_mean = np.mean(test_pred_price)
                     test_snr_h = test_pred_mean / (test_unc_mean_h + 1e-9) if test_unc_mean_h > 0 else np.inf

                     # Store metrics
                     metrics_results["Test"]["MAE"][h].append(test_mae_h)
                     metrics_results["Test"]["R2"][h].append(test_r2_h)
                     metrics_results["Test"]["Uncertainty"][h].append(test_unc_mean_h)
                     metrics_results["Test"]["SNR"][h].append(test_snr_h)

                 except Exception as e:
                     print(f"WARN: Error calculating Test metrics for H={h}: {e}")
                     for metric in metric_names: metrics_results["Test"][metric][h].append(np.nan)


            # --- Print Iteration Summary (using PLOTTED horizon) ---
            # Retrieve metrics calculated above for the plotted horizon
            try:
                train_mae_plot = metrics_results["Train"]["MAE"][plotted_horizon][-1] # Last iteration's value
                train_r2_plot = metrics_results["Train"]["R2"][plotted_horizon][-1]
                val_mae_plot = metrics_results["Validation"]["MAE"][plotted_horizon][-1]
                val_r2_plot = metrics_results["Validation"]["R2"][plotted_horizon][-1]
                test_mae_plot = metrics_results["Test"]["MAE"][plotted_horizon][-1]
                test_r2_plot = metrics_results["Test"]["R2"][plotted_horizon][-1]
                test_unc_plot = metrics_results["Test"]["Uncertainty"][plotted_horizon][-1]
                test_snr_plot = metrics_results["Test"]["SNR"][plotted_horizon][-1]

                print("************************************************************************")
                print(f"Iteration {iteration} Completed | Time: {time.time() - iter_start:.2f} sec | Plotted Horizon: {plotted_horizon}")
                print(f"  Train MAE: {train_mae_plot:.6f} | Train R²: {train_r2_plot:.4f}")
                print(f"  Valid MAE: {val_mae_plot:.6f} | Valid R²: {val_r2_plot:.4f}")
                print(f"  Test  MAE: {test_mae_plot:.6f} | Test  R²: {test_r2_plot:.4f} | Test Unc: {test_unc_plot:.6f} | Test SNR: {test_snr_plot:.2f}")
                print("************************************************************************")
            except IndexError:
                 print(f"WARN: Could not retrieve metrics for plotted horizon {plotted_horizon} for iteration summary.")
            except Exception as e:
                 print(f"WARN: Error printing iteration summary: {e}")
            # --- End of Iteration Loop ---


        # --- Consolidate results across iterations FOR ALL HORIZONS ---
        print("\n--- Aggregating Results Across Iterations (All Horizons) ---")
        results_list = []
        for ds in data_sets:
             for mn in metric_names:
                 for h in predicted_horizons:
                      values = metrics_results[ds][mn][h]
                      if values: # Check if list is not empty
                           results_list.append({
                               "Metric": f"{ds} {mn} H{h}",
                               "Average": np.nanmean(values), # Use nanmean etc. to handle potential NaNs
                               "Std Dev": np.nanstd(values),
                               "Min": np.nanmin(values),
                               "Max": np.nanmax(values)
                           })
                      else: # Handle case where no metrics were collected for this combo
                           results_list.append({
                                "Metric": f"{ds} {mn} H{h}", "Average": np.nan, "Std Dev": np.nan, "Min": np.nan, "Max": np.nan
                           })

        results_df = pd.DataFrame(results_list)
        results_file = config.get("results_file", self.params["results_file"])
        try:
            results_df.to_csv(results_file, index=False, float_format='%.6f')
            print(f"Aggregated results (Avg/Std/Min/Max for ALL horizons) saved to {results_file}")
            # Optionally print the results table
            print(results_df.to_string())
        except Exception as e:
            print(f"ERROR saving aggregated results: {e}")


        # --- Save Final Test Predictions/Targets/Close and Uncertainties (Separate Files) ---
        print("\n--- Saving Final Test Outputs (Predictions & Uncertainties Separately) ---")
        try:
            # Determine consistent length for output (using last iteration's test data)
            num_test_points = min(len(d) for d in [list_test_predictions[0], baseline_test, test_dates] if d is not None) # Use length of predictions, baseline, dates

            # Prepare common data
            final_dates = list(test_dates[:num_test_points]) if test_dates is not None else list(range(num_test_points))
            final_baseline = baseline_test[:num_test_points] if baseline_test is not None else None

            # Prepare dictionaries for the two files
            output_data = {"DATE_TIME": final_dates}
            uncertainty_data = {"DATE_TIME": final_dates}

            # Denormalize and add test CLOSE price (actual price at time t)
            try:
                # baseline_test *is* the aligned test close price before denormalization
                denorm_test_close_price = denormalize(final_baseline, config) if final_baseline is not None else np.full(num_test_points, np.nan)
                output_data["test_CLOSE"] = denorm_test_close_price
            except Exception as e: print(f"WARN: Error denormalizing test_CLOSE for output: {e}")

            # Process each horizon
            for idx, h in enumerate(predicted_horizons):
                # Get raw results (sliced)
                preds_raw = list_test_predictions[idx][:num_test_points]
                target_raw = y_test_list[idx][:num_test_points]
                unc_raw = list_uncertainty_estimates[idx][:num_test_points]

                # Denormalize Predictions and Targets (Add baseline BEFORE denormalizing price)
                if use_returns:
                     if final_baseline is None: raise ValueError("Baseline needed for return denormalization but is None.")
                     pred_price = final_baseline + preds_raw
                     target_price = final_baseline + target_raw
                else:
                     pred_price = preds_raw
                     target_price = target_raw

                denorm_pred_price = denormalize(pred_price, config)
                denorm_target_price = denormalize(target_price, config)

                # Denormalize Uncertainty (Relative)
                denorm_unc = denormalize_returns(unc_raw, config)

                # Add to dictionaries
                output_data[f"Target_H{h}"] = denorm_target_price.flatten()
                output_data[f"Prediction_H{h}"] = denorm_pred_price.flatten()
                uncertainty_data[f"Uncertainty_H{h}"] = denorm_unc.flatten()

            # Create and Save Predictions DataFrame
            output_file = config.get("output_file", self.params["output_file"])
            try:
                output_df = pd.DataFrame(output_data)
                # Reorder columns
                cols_order = ['DATE_TIME', 'test_CLOSE'] if 'test_CLOSE' in output_df else ['DATE_TIME']
                for h in predicted_horizons: cols_order.extend([f"Target_H{h}", f"Prediction_H{h}"])
                output_df = output_df.reindex(columns=[c for c in cols_order if c in output_df.columns]) # Keep only existing columns
                write_csv(file_path=output_file, data=output_df, include_date=False, headers=True)
                print(f"Final test predictions/targets ({num_test_points} rows) saved to {output_file}")
            except ImportError: print(f"WARN: write_csv not found. Skipping save to {output_file}.")
            except Exception as e: print(f"ERROR saving predictions CSV: {e}")

            # Create and Save Uncertainties DataFrame
            uncertainties_file = config.get("uncertainties_file", self.params.get("uncertainties_file")) # Get from params
            if uncertainties_file:
                try:
                    uncertainty_df = pd.DataFrame(uncertainty_data)
                    # Reorder columns
                    cols_order = ['DATE_TIME']
                    for h in predicted_horizons: cols_order.append(f"Uncertainty_H{h}")
                    uncertainty_df = uncertainty_df.reindex(columns=[c for c in cols_order if c in uncertainty_df.columns])
                    write_csv(file_path=uncertainties_file, data=uncertainty_df, include_date=False, headers=True)
                    print(f"Final test uncertainties ({num_test_points} rows) saved to {uncertainties_file}")
                except ImportError: print(f"WARN: write_csv not found. Skipping save to {uncertainties_file}.")
                except Exception as e: print(f"ERROR saving uncertainties CSV: {e}")
            else:
                 print("INFO: No 'uncertainties_file' specified in config. Skipping uncertainty CSV.")

        except Exception as e:
            print(f"ERROR during final CSV saving block: {e}")
        # --- End of CSV Saving Block ---


        # --- Plot Predictions for the Configured 'plotted_horizon' ---
        print(f"\nGenerating prediction plot for plotted horizon: {plotted_horizon}...")
        try:
            # Get data for plotted horizon (using last iteration results)
            preds_plot_raw = list_test_predictions[plotted_index][:num_test_points]
            target_plot_raw = y_test_list[plotted_index][:num_test_points]
            unc_plot_raw = list_uncertainty_estimates[plotted_index][:num_test_points]
            baseline_plot = final_baseline # Already sliced baseline

            # CORRECT Denormalization for plotting
            if use_returns:
                pred_plot_price = denormalize(baseline_plot + preds_plot_raw, config)
                target_plot_price = denormalize(baseline_plot + target_plot_raw, config)
            else:
                pred_plot_price = denormalize(preds_plot_raw, config)
                target_plot_price = denormalize(target_plot_raw, config)
            unc_plot_denorm = denormalize_returns(unc_plot_raw, config) # Uncertainty is relative
            true_plot_price = denormalize(baseline_plot, config) # Denormalize baseline to get actual price

            # Determine plot points and slice
            n_plot = config.get("plot_points", self.params["plot_points"])
            plot_slice = slice(max(0, num_test_points - n_plot), num_test_points)

            dates_plot_final = final_dates[plot_slice]
            pred_plot_final = pred_plot_price[plot_slice]
            target_plot_final = target_plot_price[plot_slice]
            true_plot_final = true_plot_price[plot_slice]
            unc_plot_final = unc_plot_denorm[plot_slice]

            # Plotting
            plt.figure(figsize=(14, 7))
            plt.plot(dates_plot_final, pred_plot_final, label=f"Predicted Price H{plotted_horizon}", color=config.get("plot_color_predicted", "red"), lw=1.5, zorder=3)
            plt.plot(dates_plot_final, target_plot_final, label=f"Target Price H{plotted_horizon}", color=config.get("plot_color_target", "orange"), lw=1.5, zorder=2)
            plt.plot(dates_plot_final, true_plot_final, label="Actual Price", color=config.get("plot_color_true", "blue"), lw=1, ls='--', alpha=0.7, zorder=1)
            plt.fill_between(dates_plot_final, pred_plot_final - abs(unc_plot_final), pred_plot_final + abs(unc_plot_final), color=config.get("plot_color_uncertainty", "green"), alpha=0.2, label=f"Uncertainty H{plotted_horizon}", zorder=0)
            plt.title(f"Predictions vs Target/Actual (H={plotted_horizon})")
            plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.grid(True, alpha=0.6); plt.tight_layout()
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300); plt.close()
            print(f"Prediction plot saved to {predictions_plot_file}")
        except Exception as e: print(f"ERROR generating prediction plot: {e}"); plt.close()

        # --- Plot and save the model diagram ---
        if plot_model is not None and hasattr(predictor_plugin, 'model') and predictor_plugin.model is not None:
            try:
                model_plot_file = config.get('model_plot_file', self.params.get('model_plot_file', 'model_plot.png'))
                plot_model(predictor_plugin.model, to_file=model_plot_file, show_shapes=True, show_layer_names=True, dpi=300)
                print(f"Model plot saved to {model_plot_file}")
            except Exception as e: print(f"WARN: Failed to generate model plot: {e}")
        else: print("INFO: Skipping model plot generation.")

        # --- Save the trained model ---
        if hasattr(predictor_plugin, 'save') and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try: predictor_plugin.save(save_model_file); print(f"Model saved to {save_model_file}")
            except Exception as e: print(f"ERROR saving model: {e}")
        else: print("WARN: Predictor does not have save method.")

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")

    # --- load_and_evaluate_model (Keep as provided by user) ---
    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin):
        """
        Loads a pre-trained model and evaluates it using validation data.
        The predictions are denormalized and saved to a CSV file along with DATE_TIME.
        """
        # NOTE: This function was not modified per user request to only fix pipeline issues.
        # It might need similar updates regarding denormalization and multi-horizon handling
        # if it's intended to be used with the current multi-output structure.
        from tensorflow.keras.models import load_model
        print(f"Loading pre-trained model from {config['load_model']}...")
        try:
            # !!! Need to define or import custom objects if used during saving !!!
            custom_objects = {} # Add {"combined_loss": combined_loss, "mmd": mmd_metric, "huber": huber_metric} if needed
            predictor_plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
            print("Model loaded successfully.")
        except Exception as e: print(f"Failed to load model: {e}"); return

        print("Loading/processing validation data for evaluation...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("y_val_dates") # Use y_val_dates as it aligns with X
        print(f"Validation data: X shape: {x_val.shape}")

        print("Making predictions on validation data...")
        try:
            mc_samples = config.get("mc_samples", 100)
            list_predictions, _ = predictor_plugin.predict_with_uncertainty(x_val, mc_samples=mc_samples)
            print(f"Raw predictions list length: {len(list_predictions)}")
        except Exception as e: print(f"Failed predictions: {e}"); return

        # --- Adapt saving for multi-output ---
        try:
            num_val_points = len(list_predictions[0])
            final_dates = list(val_dates[:num_val_points]) if val_dates is not None else list(range(num_val_points))
            output_data = {"DATE_TIME": final_dates}
            baseline_val_eval = datasets.get("baseline_val")[:num_val_points] if datasets.get("baseline_val") is not None else None

            for idx, h in enumerate(config['predicted_horizons']):
                preds_raw = list_predictions[idx][:num_val_points]
                if config.get("use_returns"):
                     if baseline_val_eval is None: raise ValueError("Baseline needed for eval denorm.")
                     pred_price = baseline_val_eval + preds_raw
                else: pred_price = preds_raw
                denorm_pred_price = denormalize(pred_price, config)
                output_data[f"Prediction_H{h}"] = denorm_pred_price.flatten()

            evaluate_df = pd.DataFrame(output_data)
            evaluate_filename = config.get('output_file', 'eval_predictions.csv') # Use output_file or specific eval file
            # from app.data_handler import write_csv # Already imported
            write_csv(file_path=evaluate_filename, data=evaluate_df, include_date=False, headers=True)
            print(f"Validation predictions saved to {evaluate_filename}")
        except ImportError: print(f"WARN: write_csv not found. Cannot save eval predictions.")
        except Exception as e: print(f"Failed to save validation predictions: {e}")