#!/usr/bin/env python
"""
STL Pipeline Plugin

This plugin orchestrates the complete forecasting flow when using the STL Preprocessor Plugin:
  1. It obtains datasets produced by the STL Preprocessor Plugin which includes decomposed channels.
  2. It combines the decomposed channels (trend, seasonal, and noise) into a multi-channel input.
  3. It then executes iterations of training, validation, and evaluation using the Predictor Plugin.
  4. It calculates and prints metrics (MAE, R², uncertainty, SNR), generates and saves loss and prediction plots,
     and saves consolidated results to CSV files.
  5. It also supports loading and evaluating a pre-trained model.

All printed messages, error checks, and statistics are computed with respect to the target (Y) dataset.
"""

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model
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
from app.data_handler import write_csv


def denormalize(data, config):
    """
    Denormalizes the data using the provided configuration.
    """
    if "use_normalization_json" in config:
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_min = norm_json["CLOSE"]["min"]
            close_max = norm_json["CLOSE"]["max"]
            diff = close_max - close_min
            return data * diff + close_min
    return data

def denormalize_returns(data, config):
    """
    Denormalizes the data using the provided configuration.
    """
    if "use_normalization_json" in config:
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_min = norm_json["CLOSE"]["min"]
            close_max = norm_json["CLOSE"]["max"]
            diff = close_max - close_min
            return data * diff
    return data


"""
STL Pipeline Plugin orchestrator using multi-output predictor.
"""


# Assume these utility functions are defined elsewhere
# def denormalize(data, config): ...
# def denormalize_returns(data, config): ...

# Assume these custom callbacks are defined/imported
# class EarlyStoppingWithPatienceCounter(tf.keras.callbacks.EarlyStopping): ...
# class ReduceLROnPlateauWithCounter(tf.keras.callbacks.ReduceLROnPlateau): ...
# class ClearMemoryCallback(tf.keras.callbacks.Callback): ...

class STLPipelinePlugin:
    # Default pipeline parameters
    plugin_params = {
        "iterations": 1,
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001, # Note: threshold_error currently unused in train snippet
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv",
        "model_plot_file": "model_plot.png",
        # "uncertainties_file": "test_uncertainties.csv", # Removed, merged into output_file
        "predictions_plot_file": "predictions_plot.png",
        "results_file": "results.csv",
        "plot_points": 1575,
        "plotted_horizon": 6, # Default value if not in config
        "use_strategy": False
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error"]

    def __init__(self):
        self.params = self.plugin_params.copy()
        # Note: Predictor specific attributes like feedback lists are managed within predictor instance

    def set_params(self, **kwargs):
        """
        Updates pipeline parameters.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Returns pipeline debug information.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds pipeline debug information to the provided dictionary.
        """
        debug_info.update(self.get_debug_info())

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin):
        """
        Executes the complete forecasting pipeline for multi-output models.
        """
        start_time = time.time()
        run_config = self.params.copy()
        run_config.update(config)
        config = run_config

        iterations = config.get("iterations", 1)
        print(f"Number of iterations: {iterations}")

        # Initialize metric lists for the 'plotted_horizon'.
        training_mae_list, training_r2_list, training_unc_list, training_snr_list = [], [], [], []
        validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list = [], [], [], []
        test_mae_list, test_r2_list, test_unc_list, test_snr_list = [], [], [], []

        # 1. Get datasets from Preprocessor Plugin.
        print("Loading and processing datasets using Preprocessor Plugin...")
        datasets = preprocessor_plugin.run_preprocessing(config)

        X_train = datasets["x_train"]
        X_val   = datasets["x_val"]
        X_test  = datasets["x_test"]
        y_train_list = datasets["y_train"] # Assumed List[np.ndarray]
        y_val_list = datasets["y_val"]     # Assumed List[np.ndarray]
        y_test_list = datasets["y_test"]     # Assumed List[np.ndarray]
        train_dates = datasets.get("y_train_dates")
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")
        test_close_prices = datasets.get("test_close_prices")
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

        # --- Config Validation ---
        if 'predicted_horizons' not in config: raise ValueError("'predicted_horizons' missing.")
        predicted_horizons = config['predicted_horizons']
        num_outputs = len(predicted_horizons)
        plotted_horizon = config.get('plotted_horizon');
        if plotted_horizon is None: raise ValueError("'plotted_horizon' missing.")
        if plotted_horizon not in predicted_horizons: raise ValueError(f"'{plotted_horizon=}' not in {predicted_horizons=}.")
        try:
            plotted_index = predicted_horizons.index(plotted_horizon)
            output_names = [f"output_horizon_{h}" for h in predicted_horizons] # Needed for dict keys
            plotted_output_name = output_names[plotted_index]
        except ValueError: raise ValueError(f"Index error for {plotted_horizon=}.")

        # --- Prepare Target Data Dictionaries ---
        if not all(len(lst) == num_outputs for lst in [y_train_list, y_val_list, y_test_list]):
             raise ValueError("Length mismatch: predicted_horizons vs y_train/y_val/y_test lists.")
        y_train_dict = {name: np.reshape(y, (-1, 1)).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: np.reshape(y, (-1, 1)).astype(np.float32) for name, y in zip(output_names, y_val_list)}
        y_test_plot_target_raw = np.reshape(y_test_list[plotted_index], (-1, 1)).astype(np.float32)

        # --- Print Shapes ---
        print(f"Input shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"Target shapes (using first horizon): y_train: {y_train_list[0].shape if y_train_list else 'N/A'}, y_val: {y_val_list[0].shape if y_val_list else 'N/A'}, y_test: {y_test_list[0].shape if y_test_list else 'N/A'}")

        # --- Other Params ---
        window_size = config.get("window_size")
        if window_size is None and config.get("plugin", "") in ["lstm", "cnn", "transformer", "ann"]:
             raise ValueError("`window_size` needed for sequence models.")
        print(f"Predicted Horizons: {predicted_horizons}, Plotted Horizon: {plotted_horizon}")
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        threshold_error = config.get("threshold_error", 0.001)

        # --- Iteration Loop ---
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()

            # --- Build Model ---
            input_shape_for_build = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],)
            if window_size is not None and input_shape_for_build[0] != window_size:
                print(f"WARN: Config window_size {window_size} != actual data window size {input_shape_for_build[0]}. Using actual.")
            predictor_plugin.build_model(input_shape=input_shape_for_build, x_train=X_train, config=config)

            # --- Train Model ---
            # Assumes predictor_plugin.train defined correctly for multi-output
            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
                X_train, y_train_dict, epochs=epochs, batch_size=batch_size,
                threshold_error=threshold_error, x_val=X_val, y_val=y_val_dict, config=config
            )

            # --- Select Data for Plotted Horizon ---
            train_preds_plot = np.reshape(list_train_preds[plotted_index], (-1, 1))
            val_preds_plot = np.reshape(list_val_preds[plotted_index], (-1, 1))
            y_train_plot_target = y_train_dict[plotted_output_name]
            y_val_plot_target = y_val_dict[plotted_output_name]
            train_unc_plot = np.reshape(list_train_unc[plotted_index], (-1, 1))
            val_unc_plot = np.reshape(list_val_unc[plotted_index], (-1, 1))

            # --- Calculate Train/Val Metrics (Plotted Horizon) ---
            use_returns = config.get("use_returns", False)
            try:
                if use_returns:
                     baseline_train_plot = baseline_train[:len(y_train_plot_target)]
                     baseline_val_plot = baseline_val[:len(y_val_plot_target)]
                     train_r2 = r2_score(denormalize((baseline_train_plot + y_train_plot_target), config).flatten(),
                                         denormalize((baseline_train_plot + train_preds_plot), config).flatten())
                     val_r2 = r2_score(denormalize((baseline_val_plot + y_val_plot_target), config).flatten(),
                                       denormalize((baseline_val_plot + val_preds_plot[:len(baseline_val_plot)]), config).flatten())
                     train_mae = np.mean(np.abs(denormalize_returns(train_preds_plot - y_train_plot_target, config)))
                     val_mae = np.mean(np.abs(denormalize_returns(val_preds_plot[:len(y_val_plot_target)] - y_val_plot_target, config)))
                else:
                     train_r2 = r2_score(denormalize(y_train_plot_target, config).flatten(),
                                         denormalize(train_preds_plot, config).flatten())
                     val_r2 = r2_score(denormalize(y_val_plot_target, config).flatten(),
                                       denormalize(val_preds_plot[:len(y_val_plot_target)], config).flatten())
                     train_mae = np.mean(np.abs(denormalize_returns(train_preds_plot - y_train_plot_target, config)))
                     val_mae = np.mean(np.abs(denormalize_returns(val_preds_plot[:len(y_val_plot_target)] - y_val_plot_target, config)))
            except Exception as e:
                print(f"WARN: Error calculating Train/Val MAE/R2: {e}")
                train_mae, train_r2, val_mae, val_r2 = np.nan, np.nan, np.nan, np.nan

            # --- Save Loss Plot ---
            loss_plot_file = config.get("loss_plot_file", self.params["loss_plot_file"])
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.title(f"Model Loss - Iteration {iteration}")
                plt.ylabel("Loss"); plt.xlabel("Epoch")
                plt.legend(loc="upper right"); plt.grid(True, linestyle='--', alpha=0.6)
                plt.savefig(loss_plot_file); plt.close()
                print(f"Loss plot saved to {loss_plot_file}")
            except Exception as e: print(f"WARN: Failed to save loss plot: {e}")

            # --- Evaluate on Test Dataset ---
            print("\nEvaluating on test dataset using MC sampling...")
            mc_samples = config.get("mc_samples", 100)
            # Assumes predictor_plugin.predict_with_uncertainty is updated for multi-output
            list_test_predictions, list_uncertainty_estimates = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

            # Select test data for the plotted horizon
            test_preds_plot = np.reshape(list_test_predictions[plotted_index], (-1, 1))
            test_unc_plot = np.reshape(list_uncertainty_estimates[plotted_index], (-1, 1))

            # Calculate Test Metrics (Plotted Horizon)
            try:
                if use_returns:
                     baseline_test_plot = baseline_test[:len(y_test_plot_target_raw)]
                     test_r2 = r2_score(denormalize((baseline_test_plot + y_test_plot_target_raw), config).flatten(),
                                        denormalize((baseline_test_plot + test_preds_plot), config).flatten())
                     test_mae = np.mean(np.abs(denormalize_returns(test_preds_plot[:len(y_test_plot_target_raw)] - y_test_plot_target_raw, config)))
                else:
                     test_r2 = r2_score(denormalize(y_test_plot_target_raw, config).flatten(),
                                        denormalize(test_preds_plot, config).flatten())
                     test_mae = np.mean(np.abs(denormalize_returns(test_preds_plot[:len(y_test_plot_target_raw)] - y_test_plot_target_raw, config)))

                # Calculate Uncertainty and SNR (Plotted Horizon)
                test_unc_last = np.mean(test_unc_plot)
                if use_returns:
                     test_mean = np.mean(denormalize((baseline_test_plot + test_preds_plot), config))
                else:
                     test_mean = np.mean(denormalize(test_preds_plot, config))
                test_snr = test_mean / (test_unc_last + 1e-9) if test_unc_last > 1e-9 else np.inf
            except Exception as e:
                 print(f"WARN: Error calculating Test MAE/R2/Unc/SNR: {e}")
                 test_mae, test_r2, test_unc_last, test_snr = np.nan, np.nan, np.nan, np.nan

            # Use Test uncertainty/SNR as proxy for Train/Val summary stats
            train_unc_last = test_unc_last
            val_unc_last = test_unc_last
            train_snr = test_snr
            val_snr = test_snr

            # Append metrics (calculated only for the plotted horizon)
            training_mae_list.append(train_mae); training_r2_list.append(train_r2)
            training_unc_list.append(train_unc_last); training_snr_list.append(train_snr)
            validation_mae_list.append(val_mae); validation_r2_list.append(val_r2)
            validation_unc_list.append(val_unc_last); validation_snr_list.append(val_snr)
            test_mae_list.append(test_mae); test_r2_list.append(test_r2)
            test_unc_list.append(test_unc_last); test_snr_list.append(test_snr)

            # Print metrics summary for the iteration (plotted horizon only)
            print("************************************************************************")
            print(f"Iteration {iteration} Completed | Time: {time.time() - iter_start:.2f} sec | Plotted Horizon: {plotted_horizon}")
            print(f"  Train MAE: {train_mae:.6f} | Train R²: {train_r2:.4f}")
            print(f"  Valid MAE: {val_mae:.6f} | Valid R²: {val_r2:.4f}")
            print(f"  Test  MAE: {test_mae:.6f} | Test  R²: {test_r2:.4f} | Test Unc: {test_unc_last:.6f} | Test SNR: {test_snr:.2f}")
            print("************************************************************************")
            # End of Iteration Loop

        # --- Consolidate results across iterations ---
        # Note: Average/StdDev calculated over iterations, using metrics from the 'plotted_horizon'.
        results_metrics = ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR",
                           "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                           "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"]
        results_avg = [np.mean(l) if len(l)>0 else np.nan for l in [training_mae_list, training_r2_list, training_unc_list, training_snr_list,
                                            validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list,
                                            test_mae_list, test_r2_list, test_unc_list, test_snr_list]]
        results_std = [np.std(l) if len(l)>0 else np.nan for l in [training_mae_list, training_r2_list, training_unc_list, training_snr_list,
                                           validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list,
                                           test_mae_list, test_r2_list, test_unc_list, test_snr_list]]

        results = {"Metric": results_metrics, "Average": results_avg, "Std Dev": results_std}
        if config.get("use_strategy", False):
             pass # No change needed based on original logic for this flag

        results_file = config.get("results_file", self.params["results_file"])
        try:
            pd.DataFrame(results).to_csv(results_file, index=False, float_format='%.6f')
            print(f"Aggregated results (based on plotted_horizon metrics) saved to {results_file}")
        except Exception as e:
            print(f"ERROR saving aggregated results: {e}")

        # --- Save Final Test Predictions and Uncertainties for ALL Horizons ---
        # --- Save Final Test Predictions and Uncertainties for ALL Horizons ---
        print(f"Preparing final output CSV for all {num_outputs} predicted horizons...")

        # 1) Determine consistent length based on minimum available points
        try:
            if not list_test_predictions: raise ValueError("No test predictions available.")
            num_test_points = len(list_test_predictions[0])
            min_available_len = num_test_points
            sources_to_check = {'test_dates': test_dates, 'test_close_prices': test_close_prices}
            if config.get("use_returns", False): sources_to_check['baseline_test'] = baseline_test
            for idx, y_arr in enumerate(y_test_list): sources_to_check[f'y_test_{idx}'] = y_arr

            for name, data_arr in sources_to_check.items():
                if data_arr is not None: min_available_len = min(min_available_len, len(data_arr))
                elif name.startswith('y_test_'): raise ValueError(f"Target array '{name}' is None.")

            if min_available_len < num_test_points:
                print(f"WARN: Truncating output rows from {num_test_points} to {min_available_len} due to shortest input array.")
            num_test_points = min_available_len

        except Exception as e:
             print(f"ERROR determining consistent output length: {e}. Skipping CSV output.")
             num_test_points = 0

        if num_test_points > 0:
            # 2) Prepare Dates, Close Prices, Baseline using the final num_test_points
            if test_dates is not None: final_dates = list(test_dates)[:num_test_points]
            else: final_dates = list(range(num_test_points))

            # Use raw close prices before potential denormalization issue
            final_close_raw = test_close_prices[:num_test_points] if test_close_prices is not None else np.full(num_test_points, np.nan)
            # Attempt denormalization for close price separately
            try:
                 denorm_test_close_prices = denormalize(final_close_raw, config) if config.get("use_normalization_json") else final_close_raw
                 if len(denorm_test_close_prices) != num_test_points:
                      print(f"WARN: Denormalized test_CLOSE length ({len(denorm_test_close_prices)}) != expected ({num_test_points}). Using NaN.")
                      denorm_test_close_prices = np.full(num_test_points, np.nan)
            except Exception as e:
                 print(f"WARN: Error denormalizing test_CLOSE: {e}. Using NaN.")
                 denorm_test_close_prices = np.full(num_test_points, np.nan)


            final_baseline = None
            if config.get("use_returns", False) and baseline_test is not None:
                 final_baseline = baseline_test[:num_test_points]

            # 3) Build Dictionary for DataFrame (All Horizons)
            output_data = {"DATE_TIME": final_dates}
            if not np.isnan(denorm_test_close_prices).all():
                 output_data["test_CLOSE"] = denorm_test_close_prices

            print("DEBUG: Populating output_data dictionary...")
            for idx, h in enumerate(predicted_horizons):
                # Slice ALL arrays using the final num_test_points
                preds_raw = np.reshape(list_test_predictions[idx][:num_test_points], (-1, 1))
                unc_raw = np.reshape(list_uncertainty_estimates[idx][:num_test_points], (-1, 1))
                target_raw = np.reshape(y_test_list[idx][:num_test_points], (-1, 1))

                # --- DEBUGGING LENGTH ISSUE: BYPASS DENORMALIZE FOR TARGET/PRED ---
                # Denormalize only Uncertainty for now, assign raw Target/Prediction
                print(f"  H={h}: Raw shapes - Target: {target_raw.shape}, Pred: {preds_raw.shape}, Unc: {unc_raw.shape}")
                try:
                     denorm_unc = denormalize_returns(unc_raw, config)
                     # Verify uncertainty length after potential denorm
                     if len(denorm_unc) != num_test_points:
                         print(f"WARN: Denormalized Uncertainty_H{h} length ({len(denorm_unc)}) != expected ({num_test_points}). Skipping uncertainty column.")
                         denorm_unc_flat = np.full(num_test_points, np.nan) # Use NaN placeholder
                     else:
                          denorm_unc_flat = denorm_unc.flatten()

                     # Assign RAW values for Target and Prediction (BYPASSING DENORMALIZE)
                     denorm_target_flat = target_raw.flatten()
                     denorm_preds_flat = preds_raw.flatten()

                     # Print shapes before adding to dict
                     print(f"  H={h}: Final shapes - Target: {denorm_target_flat.shape}, Pred: {denorm_preds_flat.shape}, Unc: {denorm_unc_flat.shape}")

                     # Add columns using the arrays that *should* have the correct length
                     output_data[f"Target_H{h}"] = denorm_target_flat
                     output_data[f"Prediction_H{h}"] = denorm_preds_flat
                     output_data[f"Uncertainty_H{h}"] = denorm_unc_flat

                except Exception as e:
                    print(f"WARN: Error during denormalization/assignment for H={h}: {e}. Skipping columns for this horizon.")
                    continue # Skip adding columns for this horizon if errors occur

            # --- END DEBUGGING BLOCK ---


            # 4) Create and Save DataFrame
            print("DEBUG: Attempting to create DataFrame...")
            try:
                # Check final lengths before creating DataFrame
                all_lengths_ok = True
                for k, v in output_data.items():
                    current_len = len(v) if hasattr(v, '__len__') else -1
                    print(f"  - Final check - {k}: length {current_len}")
                    if current_len != num_test_points:
                        print(f"ERROR: Final length mismatch for {k}. Expected {num_test_points}, got {current_len}.")
                        all_lengths_ok = False
                        # Optionally remove the problematic column: del output_data[k]

                if not all_lengths_ok:
                     raise ValueError("Cannot create DataFrame due to final length mismatches (see logs above).")

                final_output_df = pd.DataFrame(output_data)

                cols_order = ['DATE_TIME']
                if 'test_CLOSE' in output_data: cols_order.append('test_CLOSE')
                for h in predicted_horizons:
                     # Only include columns if they likely exist (adjust if needed)
                     if f"Target_H{h}" in output_data:
                          cols_order.extend([f"Target_H{h}", f"Prediction_H{h}", f"Uncertainty_H{h}"])
                final_output_df = final_output_df.reindex(columns=cols_order)

                output_file = config.get("output_file", self.params["output_file"])
                # from app.data_handler import write_csv # Assumed imported earlier
                write_csv(file_path=output_file, data=final_output_df, include_date=False, headers=config.get('headers', True))
                print(f"Final test data ({num_test_points} rows) saved to {output_file}. NOTE: Targets/Predictions might be normalized if denormalize was bypassed.")
            except ImportError:
                 output_file = config.get("output_file", self.params["output_file"])
                 print(f"WARN: Could not import write_csv. Saving with pandas default to {output_file}.")
                 final_output_df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
            except ValueError as ve:
                 print(f"ERROR: Failed to create final output DataFrame: {ve}")
                 # Print lengths again on error
                 for k, v in output_data.items(): print(f"  - Error state - {k}: length {len(v) if hasattr(v, '__len__') else 'N/A'}")
            except Exception as e:
                 print(f"ERROR: Failed to save final predictions CSV: {e}")
        # --- End of CSV Saving Block ---


        # --- Plot Predictions for the Configured 'plotted_horizon' ---
        print(f"Generating prediction plot for plotted horizon: {plotted_horizon}...")
        try:
            # Use data selected earlier (_plot vars) and denormalize
            if use_returns:
                 baseline_test_plot = baseline_test[:len(y_test_plot_target_raw)]
                 pred_plot_denorm = denormalize((baseline_test_plot + test_preds_plot), config).flatten()
                 target_plot_denorm = denormalize((baseline_test_plot + y_test_plot_target_raw), config).flatten()
                 unc_plot_denorm = denormalize_returns(test_unc_plot, config).flatten()
            else:
                 pred_plot_denorm = denormalize(test_preds_plot, config).flatten()
                 target_plot_denorm = denormalize(y_test_plot_target_raw, config).flatten()
                 unc_plot_denorm = denormalize_returns(test_unc_plot, config).flatten()
            true_plot_denorm = denorm_test_close_prices # Already aligned and denormalized

            # Determine plot points and slice data
            n_plot = config.get("plot_points", self.params["plot_points"])
            num_available_points = len(final_dates) # Use length determined for CSV
            plot_slice = slice(max(0, num_available_points - n_plot), num_available_points)

            pred_plot_final = pred_plot_denorm[plot_slice]
            target_plot_final = target_plot_denorm[plot_slice]
            true_plot_final = true_plot_denorm[plot_slice] if true_plot_denorm is not None else None
            unc_plot_final = unc_plot_denorm[plot_slice]
            dates_plot_final = final_dates[plot_slice]

            # Plotting colors
            plot_color_predicted = config.get("plot_color_predicted", "red")
            plot_color_true = config.get("plot_color_true", "blue")
            plot_color_target = config.get("plot_color_target", "orange")
            plot_color_uncertainty = config.get("plot_color_uncertainty", "green")

            # Generate Plot
            plt.figure(figsize=(14, 7))
            plt.plot(dates_plot_final, pred_plot_final, label=f"Predicted Price (H={plotted_horizon})", color=plot_color_predicted, linewidth=1.5, zorder=3)
            plt.plot(dates_plot_final, target_plot_final, label=f"Target Price (H={plotted_horizon})", color=plot_color_target, linewidth=1.5, zorder=2)
            if true_plot_final is not None and not np.isnan(true_plot_final).all():
                 plt.plot(dates_plot_final, true_plot_final, label="Actual Price", color=plot_color_true, linewidth=1, linestyle='--', alpha=0.7, zorder=1)
            unc_plot_final_abs = np.abs(unc_plot_final)
            plt.fill_between(dates_plot_final, pred_plot_final - unc_plot_final_abs, pred_plot_final + unc_plot_final_abs,
                             color=plot_color_uncertainty, alpha=0.2, label=f"Uncertainty (H={plotted_horizon})", zorder=0)
            time_unit = "days" if config.get("use_daily", False) else "hours"
            plt.title(f"Predictions vs Target/Actual (Plotted Horizon: {plotted_horizon} {time_unit})")
            plt.xlabel("Time"); plt.ylabel("Price")
            plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300); plt.close()
            print(f"Prediction plot for horizon {plotted_horizon} saved to {predictions_plot_file}")
        except Exception as e:
            print(f"ERROR: Failed to generate prediction plot: {e}")

        # --- Plot and save the model diagram ---
        if plot_model is not None and hasattr(predictor_plugin, 'model') and predictor_plugin.model is not None:
            try:
                model_plot_file = config.get('model_plot_file', self.params.get('model_plot_file', 'model_plot.png'))
                plot_model(predictor_plugin.model, to_file=model_plot_file, show_shapes=True, show_dtype=False, show_layer_names=True, expand_nested=True, dpi=300, show_layer_activations=True)
                print(f"Model plot saved to {model_plot_file}")
            except Exception as e: print(f"WARN: Failed to generate model plot: {e}")
        else: print("INFO: Skipping model plot generation.")

        # --- Save the trained model ---
        if hasattr(predictor_plugin, 'save') and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try: predictor_plugin.save(save_model_file); print(f"Model saved to {save_model_file}")
            except Exception as e: print(f"ERROR: Failed to save model to {save_model_file}: {e}")
        else: print("WARN: Predictor plugin does not have a save method.")

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")

    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin):
        """
        Loads a pre-trained model and evaluates it using validation data.
        The predictions are denormalized and saved to a CSV file along with DATE_TIME.
        """
        from tensorflow.keras.models import load_model
        print(f"Loading pre-trained model from {config['load_model']}...")
        try:
            custom_objects = {"combined_loss": combined_loss, "mmd": mmd_metric, "huber": huber_metric}
            predictor_plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load the model from {config['load_model']}: {e}")
            return

        print("Loading and processing validation data for evaluation...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("dates_val")
        print(f"Processed validation data: X shape: {x_val.shape}")

        print("Making predictions on validation data...")
        try:
            x_val_array = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()
            mc_samples = config.get("mc_samples", 100)
            predictions, _ = predictor_plugin.predict_with_uncertainty(x_val_array, mc_samples=mc_samples)
            print(f"Predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"Failed to make predictions: {e}")
            return

        if predictions.ndim == 1 or predictions.shape[1] == 1:
            predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        else:
            num_steps = predictions.shape[1]
            pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
            predictions_df = pd.DataFrame(predictions, columns=pred_cols)
        if val_dates is not None:
            predictions_df['DATE_TIME'] = pd.Series(val_dates[:len(predictions_df)])
        else:
            predictions_df['DATE_TIME'] = pd.NaT
            print("Warning: DATE_TIME for validation predictions not captured.")
        cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
        predictions_df = predictions_df[cols]
        evaluate_filename = config['output_file']
        from app.data_handler import write_csv
        try:
            write_csv(file_path=evaluate_filename, data=predictions_df,
                      include_date=False, headers=config.get('headers', True))
            print(f"Validation predictions with DATE_TIME saved to {evaluate_filename}")
        except Exception as e:
            print(f"Failed to save validation predictions to {evaluate_filename}: {e}")

# Debugging usage example (run directly)
if __name__ == "__main__":
    pipeline_plugin = STLPipelinePlugin()
    test_config = {
        "x_train_file": "data/x_train.csv",
        "y_train_file": "data/y_train.csv",
        "x_validation_file": "data/x_val.csv",
        "y_validation_file": "data/y_val.csv",
        "x_test_file": "data/x_test.csv",
        "y_test_file": "data/y_test.csv",
        "headers": True,
        "max_steps_train": 1000,
        "max_steps_val": 500,
        "max_steps_test": 500,
        "window_size": 24,
        "time_horizon": 6,
        "use_returns": False,
        "stl_period": 24,
        "stl_window": 24,
        "stl_plot_file": "stl_plot.png",
        "plugin": "ann",
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv",
        "model_plot_file": "model_plot.png",
        "uncertainties_file": "test_uncertainties.csv",
        "predictions_plot_file": "predictions_plot.png",
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001
    }
    from app.plugin_loader import load_plugin
    predictor_class, _ = load_plugin('predictor.plugins', test_config.get('plugin', 'default_predictor'))
    predictor_plugin = predictor_class()
    predictor_plugin.set_params(**test_config)
    from app.plugin_loader import load_plugin as load_preprocessor_plugin
    preprocessor_class, _ = load_preprocessor_plugin('preprocessor.plugins', 'default_preprocessor')
    preprocessor_plugin = preprocessor_class()
    preprocessor_plugin.set_params(**test_config)
    pipeline_plugin.run_prediction_pipeline(test_config, predictor_plugin, preprocessor_plugin)
