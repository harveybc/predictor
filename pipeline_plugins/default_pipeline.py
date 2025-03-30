#!/usr/bin/env python
"""
Default Pipeline Plugin

Este plugin orquesta el flujo completo:
  1. Obtiene los datasets mediante el Preprocessor Plugin.
  2. Ejecuta iteraciones de entrenamiento, validación y evaluación utilizando el Predictor Plugin.
  3. Calcula métricas (MAE, R², incertidumbre, SNR), genera gráficos de pérdida y predicción, y guarda resultados en archivos CSV.
  4. Permite, adicionalmente, la carga y evaluación de modelos preentrenados.

La lógica de preprocesamiento (creación de ventanas deslizantes, etc.) se delega al Preprocessor Plugin,
manteniendo así la separación de preocupaciones.
"""

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model

# Se asume que las funciones de preprocesamiento y de manejo de archivos (como write_csv) se encuentran en sus respectivos plugins.

class PipelinePlugin:
    # Parámetros por defecto específicos del pipeline.
    plugin_params = {
        "iterations": 1,
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv",
        "model_plot_file": "model_plot.png",
        "uncertainties_file": "test_uncertainties.csv",
        "predictions_plot_file": "predictions_plot.png",
        "plot_points": 1575,
        "plotted_horizon": 6,
        "use_strategy": False  # Si se activa, se guardan estadísticas extendidas.
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Actualiza los parámetros del pipeline combinando los parámetros específicos con la configuración global.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Devuelve información de debug de los parámetros relevantes del pipeline.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Agrega la información de debug al diccionario proporcionado.
        """
        debug_info.update(self.get_debug_info())

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin):
        """
        Executes the complete forecasting pipeline for multi-output models.
        """
        start_time = time.time()
        # Merge config with defaults stored in self.params
        run_config = self.params.copy()
        run_config.update(config)
        config = run_config # Use merged config hereafter

        iterations = config.get("iterations", 1)
        print(f"Number of iterations: {iterations}")

        # Initialize metric lists. Values stored will be for the 'plotted_horizon'.
        training_mae_list, training_r2_list, training_unc_list, training_snr_list = [], [], [], []
        validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list = [], [], [], []
        test_mae_list, test_r2_list, test_unc_list, test_snr_list = [], [], [], []

        # 1. Get datasets from the Preprocessor Plugin.
        print("Loading and processing datasets using Preprocessor Plugin...")
        # Assume preprocessor returns dict including y_train, y_val, y_test as LISTS of arrays
        datasets = preprocessor_plugin.run_preprocessing(config)

        # Extract data components
        X_train = datasets["x_train"] # Assume shape (samples, window, features) after preproc combine_channels if used there
        X_val   = datasets["x_val"]
        X_test  = datasets["x_test"]
        # Decomposed channels might be used by predictor internally or ignored if X_* are already combined
        # x_train_trend = datasets.get("x_train_trend") ... etc.

        # Targets (Assume LIST of arrays/Series, one per horizon)
        y_train_list = datasets["y_train"]
        y_val_list = datasets["y_val"]
        y_test_list = datasets["y_test"] # Kept as list

        # Optional data
        train_dates = datasets.get("y_train_dates") # Use Y dates as they align with targets
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")
        test_close_prices = datasets.get("test_close_prices") # Aligned with y_test_dates
        baseline_train = datasets.get("baseline_train") # Aligned with y_train_dates
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

        # --- Configuration parameters & Validation ---
        if 'predicted_horizons' not in config:
            raise ValueError("Config must contain 'predicted_horizons' list.")
        predicted_horizons = config['predicted_horizons']
        num_outputs = len(predicted_horizons)

        plotted_horizon = config.get('plotted_horizon') # Get required param
        if plotted_horizon is None: raise ValueError("Config must contain 'plotted_horizon'.")
        if plotted_horizon not in predicted_horizons:
             raise ValueError(f"'plotted_horizon' ({plotted_horizon}) not in 'predicted_horizons' ({predicted_horizons}).")
        try:
            plotted_index = predicted_horizons.index(plotted_horizon)
            plotted_output_name = f"output_horizon_{plotted_horizon}" # Assumes this convention
        except ValueError:
             raise ValueError(f"Logic error finding index for 'plotted_horizon' {plotted_horizon}.")

        # --- Prepare Target Data Dictionaries for Training ---
        if len(y_train_list) != num_outputs or len(y_val_list) != num_outputs or len(y_test_list) != num_outputs:
             raise ValueError("Length mismatch: predicted_horizons vs y_train/y_val/y_test lists from preprocessor.")

        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        # Reshape each target array to (N, 1)
        y_train_dict = {name: np.reshape(y, (-1, 1)).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: np.reshape(y, (-1, 1)).astype(np.float32) for name, y in zip(output_names, y_val_list)}
        # Extract raw target for the single plotted horizon for metrics/plotting later
        y_test_plot_target_raw = np.reshape(y_test_list[plotted_index], (-1, 1)).astype(np.float32)

        # --- Print data shapes ---
        print(f"Input shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"Target shapes (using first horizon): y_train: {y_train_list[0].shape if y_train_list else 'N/A'}, y_val: {y_val_list[0].shape if y_val_list else 'N/A'}, y_test: {y_test_list[0].shape if y_test_list else 'N/A'}")

        # --- Other parameters ---
        window_size = config.get("window_size") # Still needed for build_model input shape if applicable
        if window_size is None and config.get("plugin", "") in ["lstm", "cnn", "transformer", "ann"]:
             raise ValueError("`window_size` must be defined in config for sequence models.")
        print(f"Predicted Horizons: {predicted_horizons}, Plotted Horizon: {plotted_horizon}")
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        threshold_error = config.get("threshold_error", 0.001)

        # 3. Update the Predictor Plugin configuration if needed
        # predictor_plugin.set_params(predicted_horizons=predicted_horizons) # Predictor might get this via config in build/train

        # 4. Training and evaluation iterations.
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()

            # Build the model - determine input shape correctly
            # Assumes X_train has shape (samples, window_size, num_features)
            input_shape_for_build = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],) # Handle 2D/3D input
            if window_size is not None and input_shape_for_build[0] != window_size:
                print(f"WARN: Config window_size {window_size} != actual data window size {input_shape_for_build[0]}. Using actual.")
            predictor_plugin.build_model(input_shape=input_shape_for_build, x_train=X_train, config=config)

            # Train the model - uses dictionaries for y_train/y_val
            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
                X_train, y_train_dict, epochs=epochs, batch_size=batch_size,
                threshold_error=threshold_error, x_val=X_val, y_val=y_val_dict, config=config
            )

            # --- Select Predictions/Targets/Uncertainty for the Plotted Horizon ---
            train_preds_plot = np.reshape(list_train_preds[plotted_index], (-1, 1))
            val_preds_plot = np.reshape(list_val_preds[plotted_index], (-1, 1))
            y_train_plot_target = y_train_dict[plotted_output_name] # Already (N, 1)
            y_val_plot_target = y_val_dict[plotted_output_name]   # Already (N, 1)
            train_unc_plot = np.reshape(list_train_unc[plotted_index], (-1, 1)) # Placeholder unc
            val_unc_plot = np.reshape(list_val_unc[plotted_index], (-1, 1))     # Placeholder unc

            # --- Calculate R² and MAE for the Plotted Horizon ---
            if config.get("use_returns", False):
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

            # --- Save loss plot ---
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f"Model Loss - Iteration {iteration}")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(loc="upper right"); plt.grid(True, linestyle='--', alpha=0.6)
            loss_plot_file = config.get("loss_plot_file", self.params["loss_plot_file"])
            try: plt.savefig(loss_plot_file); plt.close()
            except Exception as e: print(f"WARN: Failed to save loss plot: {e}")
            print(f"Loss plot saved to {loss_plot_file}")

            # --- Evaluate on Test Dataset using CORRECTED predict_with_uncertainty ---
            print("\nEvaluating on test dataset using MC sampling...")
            mc_samples = config.get("mc_samples", 100)
            # This call now uses the corrected method returning lists
            list_test_predictions, list_uncertainty_estimates = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

            # Select test data for the plotted horizon
            test_preds_plot = np.reshape(list_test_predictions[plotted_index], (-1, 1))
            test_unc_plot = np.reshape(list_uncertainty_estimates[plotted_index], (-1, 1))
            # y_test_plot_target_raw was prepared earlier

            # Calculate Test R² and MAE for the Plotted Horizon
            if config.get("use_returns", False):
                 baseline_test_plot = baseline_test[:len(y_test_plot_target_raw)]
                 test_r2 = r2_score(denormalize((baseline_test_plot + y_test_plot_target_raw), config).flatten(),
                                    denormalize((baseline_test_plot + test_preds_plot), config).flatten())
                 test_mae = np.mean(np.abs(denormalize_returns(test_preds_plot[:len(y_test_plot_target_raw)] - y_test_plot_target_raw, config)))
            else:
                 test_r2 = r2_score(denormalize(y_test_plot_target_raw, config).flatten(),
                                    denormalize(test_preds_plot, config).flatten())
                 test_mae = np.mean(np.abs(denormalize_returns(test_preds_plot[:len(y_test_plot_target_raw)] - y_test_plot_target_raw, config)))

            # Calculate Uncertainty and SNR for the Plotted Horizon
            test_unc_last = np.mean(test_unc_plot) # Use actual MC uncertainty from test

            # Denormalize predictions for SNR calculation
            if config.get("use_returns", False):
                 test_mean = np.mean(denormalize((baseline_test_plot + test_preds_plot), config))
            else:
                 test_mean = np.mean(denormalize(test_preds_plot, config))
            test_snr = test_mean / (test_unc_last + 1e-9) if test_unc_last > 1e-9 else np.inf

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
        results_avg = [np.mean(l) for l in [training_mae_list, training_r2_list, training_unc_list, training_snr_list,
                                            validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list,
                                            test_mae_list, test_r2_list, test_unc_list, test_snr_list]]
        results_std = [np.std(l) for l in [training_mae_list, training_r2_list, training_unc_list, training_snr_list,
                                           validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list,
                                           test_mae_list, test_r2_list, test_unc_list, test_snr_list]]

        results = {"Metric": results_metrics, "Average": results_avg, "Std Dev": results_std}
        # use_strategy check remains as original, potentially modifying 'results' if needed
        if config.get("use_strategy", False):
             pass # No change implemented here, results dict remains as calculated

        results_file = config.get("results_file", self.params["results_file"])
        pd.DataFrame(results).to_csv(results_file, index=False, float_format='%.6f')
        print(f"Aggregated results (based on plotted_horizon metrics) saved to {results_file}")

        # --- Save Final Test Predictions and Uncertainties for ALL Horizons ---
        # Uses the results from the LAST iteration's MC sampling
        print(f"Preparing final output CSV for all {num_outputs} predicted horizons...")
        num_test_points = len(list_test_predictions[0]) # Length based on first head's predictions

        # Prepare dates, ensuring alignment and correct length
        if test_dates is not None and len(test_dates) >= num_test_points:
            final_dates = list(test_dates)[:num_test_points]
        else:
            print("WARN: Test dates not available or length mismatch. Using range index.")
            final_dates = np.arange(num_test_points)

        # Prepare aligned close prices and baseline
        final_close_raw = test_close_prices[:num_test_points] if test_close_prices is not None and len(test_close_prices) >= num_test_points else np.full(num_test_points, np.nan)
        denorm_test_close_prices = denormalize(final_close_raw, config) if config.get("use_normalization_json") else final_close_raw

        final_baseline = None
        if config.get("use_returns", False) and baseline_test is not None and len(baseline_test) >= num_test_points:
             final_baseline = baseline_test[:num_test_points]

        # Build Dictionary for DataFrame (All Horizons)
        output_data = {"DATE_TIME": final_dates}
        if not np.isnan(denorm_test_close_prices).all():
             output_data["test_CLOSE"] = denorm_test_close_prices

        for idx, h in enumerate(predicted_horizons):
            # Select and ensure correct length for this horizon's data
            preds_raw = np.reshape(list_test_predictions[idx][:num_test_points], (-1, 1))
            unc_raw = np.reshape(list_uncertainty_estimates[idx][:num_test_points], (-1, 1))
            target_raw = np.reshape(y_test_list[idx][:num_test_points], (-1, 1))

            # Denormalize
            if config.get("use_returns", False):
                 baseline_h = final_baseline if final_baseline is not None else np.zeros_like(target_raw)
                 # Verify baseline length again before use
                 if len(baseline_h) != len(target_raw):
                     print(f"WARN: Final baseline length mismatch for H={h}. Using zeros.")
                     baseline_h = np.zeros_like(target_raw)
                 denorm_preds = denormalize((baseline_h + preds_raw), config)
                 denorm_target = denormalize((baseline_h + target_raw), config)
                 denorm_unc = denormalize_returns(unc_raw, config)
            else:
                 denorm_preds = denormalize(preds_raw, config)
                 denorm_target = denormalize(target_raw, config)
                 denorm_unc = denormalize_returns(unc_raw, config)

            # Add columns
            output_data[f"Target_H{h}"] = denorm_target.flatten()
            output_data[f"Prediction_H{h}"] = denorm_preds.flatten()
            output_data[f"Uncertainty_H{h}"] = denorm_unc.flatten()

        # Create and Save DataFrame
        final_output_df = pd.DataFrame(output_data)
        # Define column order
        cols_order = ['DATE_TIME']
        if 'test_CLOSE' in output_data: cols_order.append('test_CLOSE')
        for h in predicted_horizons:
             cols_order.extend([f"Target_H{h}", f"Prediction_H{h}", f"Uncertainty_H{h}"])
        # Reorder if all columns exist
        try:
            final_output_df = final_output_df[cols_order]
        except KeyError as e:
            print(f"WARN: Could not reorder output CSV columns. Error: {e}")

        output_file = config.get("output_file", self.params["output_file"])
        try:
             # Ensure data_handler and write_csv are correctly imported/accessible
             from app.data_handler import write_csv
             write_csv(file_path=output_file, data=final_output_df, include_date=False, headers=config.get('headers', True))
             print(f"Final test predictions and uncertainties for all horizons saved to {output_file}")
        except ImportError:
             print(f"WARN: Could not import write_csv from app.data_handler. Saving with pandas default.")
             final_output_df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S.%f') # Include microseconds
             print(f"Final test predictions and uncertainties saved to {output_file} (pandas default).")
        except Exception as e:
             print(f"ERROR: Failed to save final predictions CSV: {e}")


        # --- Plot Predictions for the Configured 'plotted_horizon' ---
        print(f"Generating prediction plot for plotted horizon: {plotted_horizon}...")
        try:
            # Use the data selected earlier for the plotted horizon (_plot variables)
            # Denormalize the specific data needed for the plot
            if config.get("use_returns", False):
                 baseline_test_plot = baseline_test[:len(y_test_plot_target_raw)]
                 pred_plot_denorm = denormalize((baseline_test_plot + test_preds_plot), config).flatten()
                 target_plot_denorm = denormalize((baseline_test_plot + y_test_plot_target_raw), config).flatten()
                 unc_plot_denorm = denormalize_returns(test_unc_plot, config).flatten()
            else:
                 pred_plot_denorm = denormalize(test_preds_plot, config).flatten()
                 target_plot_denorm = denormalize(y_test_plot_target_raw, config).flatten()
                 unc_plot_denorm = denormalize_returns(test_unc_plot, config).flatten()

            # Use the already prepared denormalized close prices aligned with test set
            true_plot_denorm = denorm_test_close_prices

            # Determine plot points and slice data correctly using final_dates
            n_plot = config.get("plot_points", self.params["plot_points"])
            num_available_points = len(final_dates) # Use length of dates
            plot_slice = slice(max(0, num_available_points - n_plot), num_available_points)

            # Ensure all arrays are sliced consistently
            pred_plot_final = pred_plot_denorm[plot_slice]
            target_plot_final = target_plot_denorm[plot_slice]
            true_plot_final = true_plot_denorm[plot_slice] if true_plot_denorm is not None else None
            unc_plot_final = unc_plot_denorm[plot_slice]
            dates_plot_final = final_dates[plot_slice]

            # Check lengths after slicing
            if not (len(pred_plot_final) == len(target_plot_final) == len(unc_plot_final) == len(dates_plot_final)):
                 print("WARN: Length mismatch after slicing for plot. Plot may be inaccurate.")

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

            # Save the plot
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300)
            plt.close()
            print(f"Prediction plot for horizon {plotted_horizon} saved to {predictions_plot_file}")
        except Exception as e:
            print(f"ERROR: Failed to generate prediction plot: {e}")

        # --- Plot and save the model diagram ---
        if plot_model is not None and hasattr(predictor_plugin, 'model') and predictor_plugin.model is not None:
            try:
                model_plot_file = config.get('model_plot_file', self.params.get('model_plot_file', 'model_plot.png'))
                plot_model(predictor_plugin.model, to_file=model_plot_file,
                           show_shapes=True, show_dtype=False, show_layer_names=True,
                           expand_nested=True, dpi=300, show_layer_activations=True)
                print(f"Model plot saved to {model_plot_file}")
            except Exception as e:
                print(f"WARN: Failed to generate model plot: {e}")
        else:
             print("INFO: Skipping model plot generation (plot_model not imported or model not built).")

        # --- Save the trained model ---
        if hasattr(predictor_plugin, 'save') and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try:
                predictor_plugin.save(save_model_file)
                print(f"Model saved to {save_model_file}")
            except Exception as e:
                print(f"ERROR: Failed to save model to {save_model_file}: {e}")
        else:
             print("WARN: Predictor plugin does not have a save method.")

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")



    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin):
        """
        Carga un modelo preentrenado y lo evalúa usando datos de validación.
        Las predicciones se denormalizan y se guardan en un CSV junto con DATE_TIME.
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

# Debugging usage example: ejecutar el plugin directamente.
if __name__ == "__main__":
    pipeline_plugin = PipelinePlugin()
    # Configuración de prueba (ajustar según los datos disponibles)
    test_config = {
        "x_train_file": "data/train.csv",
        "x_validation_file": "data/val.csv",
        "x_test_file": "data/test.csv",
        "headers": True,
        "max_steps_train": 1000,
        "max_steps_val": 500,
        "max_steps_test": 500,
        "window_size": 24,
        "time_horizon": 1,
        "use_returns": False,
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
