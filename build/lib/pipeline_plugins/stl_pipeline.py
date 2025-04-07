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

class STLPipelinePlugin:
    # Default pipeline parameters
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
        "use_strategy": False  # If True, extended statistics are saved.
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Update the pipeline parameters by merging plugin-specific settings with global configuration.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Returns debug information for the pipeline.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds the pipeline debug information to the provided dictionary.
        """
        debug_info.update(self.get_debug_info())

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin):
        """
        Executes the complete forecasting pipeline:
          1. Obtains datasets using the STL Preprocessor Plugin.
          2. Combines decomposed channels (trend, seasonal, noise) into a multi-channel input.
          3. Runs iterations of training and evaluation using the Predictor Plugin.
          4. Calculates and saves metrics, loss and prediction plots, and consolidated results.

        Args:
            config (dict): Global configuration.
            predictor_plugin: Plugin responsible for model building, training, and prediction.
            preprocessor_plugin: Plugin responsible for data preprocessing and STL decomposition.
        """
        start_time = time.time()
        iterations = config.get("iterations", self.params["iterations"])
        print(f"Number of iterations: {iterations}")

        # Initialize metric lists.
        training_mae_list, training_r2_list, training_unc_list, training_snr_list = [], [], [], []
        validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list = [], [], [], []
        test_mae_list, test_r2_list, test_unc_list, test_snr_list = [], [], [], []

        # 1. Get datasets from the Preprocessor Plugin.
        print("Loading and processing datasets using Preprocessor Plugin...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        # Standard keys from preprocessor:
        x_train_raw = datasets["x_train"]   # Log-transformed raw series windows.
        x_val_raw   = datasets["x_val"]
        x_test_raw  = datasets["x_test"]

        # Decomposed channels:
        x_train_trend = datasets.get("x_train_trend")
        x_train_seasonal = datasets.get("x_train_seasonal")
        x_train_noise = datasets.get("x_train_noise")
        x_val_trend = datasets.get("x_val_trend")
        x_val_seasonal = datasets.get("x_val_seasonal")
        x_val_noise = datasets.get("x_val_noise")
        x_test_trend = datasets.get("x_test_trend")
        x_test_seasonal = datasets.get("x_test_seasonal")
        x_test_noise = datasets.get("x_test_noise")

        # Targets (as provided by preprocessor from Y files).
        y_train = datasets["y_train"]
        y_val = datasets["y_val"]
        y_test = datasets["y_test"]

        # Convert target lists to 2D arrays.
        y_train_array = y_train[0] if isinstance(y_train, list) and len(y_train)==1 else np.stack(y_train, axis=1)
        y_val_array = y_val[0] if isinstance(y_val, list) and len(y_val)==1 else np.stack(y_val, axis=1)
        y_test_array = y_test[0] if isinstance(y_test, list) and len(y_test)==1 else np.stack(y_test, axis=1)

        train_dates = datasets.get("dates_train")
        val_dates = datasets.get("dates_val")
        test_dates = datasets.get("dates_test")
        test_close_prices = datasets.get("test_close_prices")
        if config.get("use_returns", False):
            baseline_train = datasets.get("baseline_train")
            baseline_val = datasets.get("baseline_val")
            baseline_test = datasets.get("baseline_test")

        # Print data shapes.
        print(f"Training data shapes: x_train: {x_train_raw.shape}, y_train: {y_train_array.shape}")
        print(f"Validation data shapes: x_val: {x_val_raw.shape}, y_val: {y_val_array.shape}")
        print(f"Test data shapes: x_test: {x_test_raw.shape}, y_test: {y_test_array.shape}")

        time_horizon = config.get("time_horizon")
        window_size = config.get("window_size")
        if time_horizon is None:
            raise ValueError("`time_horizon` is not defined in the configuration.")
        if config["plugin"] in ["lstm", "cnn", "transformer", "ann"] and window_size is None:
            raise ValueError("`window_size` must be defined for CNN, Transformer and LSTM plugins.")
        print(f"Time Horizon: {time_horizon}")
        batch_size = config.get("batch_size", self.params["batch_size"])
        epochs = config.get("epochs", self.params["epochs"])
        threshold_error = config.get("threshold_error", self.params["threshold_error"])

        # 2. Combine decomposed channels if available.
        # If decomposed channels exist, combine them into a multi-channel input.
        # Otherwise, use the raw x_train.
        def combine_channels(raw, trend, seasonal, noise):
            if trend is not None and seasonal is not None and noise is not None:
                # Each channel is expected to be of shape (samples, window_size, 1).
                return np.concatenate((trend, seasonal, noise), axis=2)
            return raw

        X_train = combine_channels(x_train_raw, x_train_trend, x_train_seasonal, x_train_noise)
        X_val = combine_channels(x_val_raw, x_val_trend, x_val_seasonal, x_val_noise)
        X_test = combine_channels(x_test_raw, x_test_trend, x_test_seasonal, x_test_noise)

        # 3. Update the Predictor Plugin configuration (e.g., time horizon).
        predictor_plugin.set_params(time_horizon=time_horizon)

        # 4. Training and evaluation iterations.
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()
            # Build the model using the combined multi-channel input.
            if config["plugin"] in ["lstm", "cnn", "transformer", "ann"]:
                # X_train shape: (samples, window_size, num_channels)
                predictor_plugin.build_model(input_shape=(window_size, X_train.shape[2]), x_train=X_train, config=config)
            else:
                predictor_plugin.build_model(input_shape=X_train.shape[1], x_train=X_train, config=config)

            history, train_preds, train_unc, val_preds, val_unc = predictor_plugin.train(
                X_train, y_train, epochs=epochs, batch_size=batch_size,
                threshold_error=threshold_error, x_val=X_val, y_val=y_val, config=config
            )

            # If using returns, perform inverse scaling.
            if config.get("use_returns", False):
                inv_scale_factor = 1.0 / config.get("target_scaling_factor", 100.0)
                print(f"DEBUG: Inversely scaling predictions by factor {inv_scale_factor}.")
                train_preds = train_preds * inv_scale_factor
                val_preds = val_preds * inv_scale_factor

            # Calculate R² for training and validation.
            if config.get("use_returns", False):
                train_r2 = r2_score(
                    (baseline_train[:, -1] + np.stack(y_train, axis=1)[:, -1]).flatten(),
                    (baseline_train[:, -1] + train_preds[:, 0]).flatten()
                )
                val_r2 = r2_score(
                    (baseline_val[:, -1] + np.stack(y_val, axis=1)[:, -1]).flatten(),
                    (baseline_val[:, -1] + val_preds[:, 0]).flatten()
                )
            else:
                train_r2 = r2_score(np.stack(y_train, axis=1)[:, -1].flatten(), train_preds[:, 0].flatten())
                val_r2 = r2_score(np.stack(y_val, axis=1)[:, -1].flatten(), val_preds[:, 0].flatten())

            # Calculate MAE.
            n_train = train_preds.shape[0]
            n_val = val_preds.shape[0]
            train_mae = np.mean(np.abs(train_preds[:, -1] - np.stack(y_train, axis=1)[:n_train, -1]))
            val_mae = np.mean(np.abs(val_preds[:, -1] - np.stack(y_val, axis=1)[:n_val, -1]))

            # Save loss plot.
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f"Model Loss for {config['plugin'].upper()} - Iteration {iteration}")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Val"], loc="upper left")
            loss_plot_file = config.get("loss_plot_file", self.params["loss_plot_file"])
            plt.savefig(loss_plot_file)
            plt.close()
            print(f"Loss plot saved to {loss_plot_file}")

            print("\nEvaluating on test dataset...")
            mc_samples = config.get("mc_samples", 100)
            test_predictions, uncertainty_estimates = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)
            n_test = test_predictions.shape[0]
            y_test_array = np.stack(y_test, axis=1)

            if config.get("use_returns", False) and "baseline_test" in datasets:
                print("DEBUG: baseline_test shape:", datasets["baseline_test"].shape)
            else:
                print("DEBUG: Not using returns or baseline_test not available")
            print("DEBUG: y_test_array shape:", y_test_array.shape)

            test_mae = np.mean(np.abs(test_predictions[:, -1] - y_test_array[:n_test, -1]))
            test_r2 = r2_score(y_test_array.flatten(), test_predictions[:, 0].flatten())

            # Calculate uncertainty (mean of the last column).
            train_unc_last = np.mean(train_unc[:, -1])
            val_unc_last = np.mean(val_unc[:, -1])
            test_unc_last = np.mean(uncertainty_estimates[:, -1])

            # Calculate SNR (signal-to-noise ratio).
            if config.get("use_returns", False):
                train_mean = np.mean(baseline_train[:, -1] + train_preds[:, -1])
                val_mean = np.mean(baseline_val[:, -1] + val_preds[:, -1])
                test_mean = np.mean(baseline_test[:, -1] + test_predictions[:, -1])
            else:
                train_mean = np.mean(train_preds[:, -1])
                val_mean = np.mean(val_preds[:, -1])
                test_mean = np.mean(test_predictions[:, -1])
            train_snr = 1 / (train_unc_last / train_mean)
            val_snr = 1 / (val_unc_last / val_mean)
            test_snr = 1 / (test_unc_last / test_mean)

            # Append metrics.
            training_mae_list.append(train_mae)
            training_r2_list.append(train_r2)
            training_unc_list.append(train_unc_last)
            training_snr_list.append(train_snr)
            validation_mae_list.append(val_mae)
            validation_r2_list.append(val_r2)
            validation_unc_list.append(val_unc_last)
            validation_snr_list.append(val_snr)
            test_mae_list.append(test_mae)
            test_r2_list.append(test_r2)
            test_unc_list.append(test_unc_last)
            test_snr_list.append(test_snr)

            print("************************************************************************")
            print(f"Iteration {iteration} completed in {time.time() - iter_start:.2f} seconds")
            print(f"Training MAE: {train_mae}, Training R²: {train_r2}, Training Uncertainty: {train_unc_last}, Training SNR: {train_snr}")
            print(f"Validation MAE: {val_mae}, Validation R²: {val_r2}, Validation Uncertainty: {val_unc_last}, Validation SNR: {val_snr}")
            print(f"Test MAE: {test_mae}, Test R²: {test_r2}, Test Uncertainty: {test_unc_last}, Test SNR: {test_snr}")
            print("************************************************************************")

        # Consolidate results.
        if config.get("use_strategy", False):
            results = {
                "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR",
                           "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                           "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"],
                "Average": [np.mean(training_mae_list), np.mean(training_r2_list),
                            np.mean(training_unc_list), np.mean(training_snr_list),
                            np.mean(validation_mae_list), np.mean(validation_r2_list),
                            np.mean(validation_unc_list), np.mean(validation_snr_list),
                            np.mean(test_mae_list), np.mean(test_r2_list),
                            np.mean(test_unc_list), np.mean(test_snr_list)],
                "Std Dev": [np.std(training_mae_list), np.std(training_r2_list),
                            np.std(training_unc_list), np.std(training_snr_list),
                            np.std(validation_mae_list), np.std(validation_r2_list),
                            np.std(validation_unc_list), np.std(validation_snr_list),
                            np.std(test_mae_list), np.std(test_r2_list),
                            np.std(test_unc_list), np.std(test_snr_list)]
            }
        else:
            results = {
                "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR",
                           "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                           "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"],
                "Average": [np.mean(training_mae_list), np.mean(training_r2_list),
                            np.mean(training_unc_list), np.mean(training_snr_list),
                            np.mean(validation_mae_list), np.mean(validation_r2_list),
                            np.mean(validation_unc_list), np.mean(validation_snr_list),
                            np.mean(test_mae_list), np.mean(test_r2_list),
                            np.mean(test_unc_list), np.mean(test_snr_list)],
                "Std Dev": [np.std(training_mae_list), np.std(training_r2_list),
                            np.std(training_unc_list), np.std(training_snr_list),
                            np.std(validation_mae_list), np.std(validation_r2_list),
                            np.std(validation_unc_list), np.std(validation_snr_list),
                            np.std(test_mae_list), np.std(test_r2_list),
                            np.std(test_unc_list), np.std(test_snr_list)]
            }
        results_file = config.get("results_file", "results.csv")
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Denormalize final test predictions if normalization configuration is provided.
        norm_json = config.get("use_normalization_json")
        if norm_json is None:
            norm_json = {}
        elif isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_min = norm_json["CLOSE"]["min"]
            close_max = norm_json["CLOSE"]["max"]
            denorm_test_close_prices = test_close_prices * (close_max - close_min) + close_min
        else:
            denorm_test_close_prices = test_close_prices

        if config.get("use_normalization_json") is not None:
            norm_json = config.get("use_normalization_json")
            if isinstance(norm_json, str):
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
            if config.get("use_returns", False):
                if "CLOSE" in norm_json:
                    close_min = norm_json["CLOSE"]["min"]
                    close_max = norm_json["CLOSE"]["max"]
                    diff = close_max - close_min
                    if baseline_test is not None:
                        test_predictions = (test_predictions + baseline_test) * diff + close_min
                        y_test_array = np.stack(y_test, axis=1)
                        denorm_y_test = (y_test_array + baseline_test) * diff + close_min
                    else:
                        print("Warning: Baseline test values not found; skipping returns denormalization.")
                        denorm_y_test = np.stack(y_test, axis=1)
                else:
                    print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
                    denorm_y_test = np.stack(y_test, axis=1)
            else:
                if "CLOSE" in norm_json:
                    close_min = norm_json["CLOSE"]["min"]
                    close_max = norm_json["CLOSE"]["max"]
                    test_predictions = test_predictions * (close_max - close_min) + close_min
                    denorm_y_test = np.stack(y_test, axis=1) * (close_max - close_min) + close_min
                else:
                    print("Warning: 'CLOSE' not found; skipping denormalization for non-returns mode.")
                    denorm_y_test = np.stack(y_test, axis=1)
        else:
            denorm_y_test = np.stack(y_test, axis=1)

        denorm_test_close_prices = test_close_prices * (close_max - close_min) + close_min

        # Save final test predictions to CSV.
        final_test_file = config.get("output_file", "test_predictions.csv")
        num_pred_steps = test_predictions.shape[1]
        pred_cols = [f"Prediction_{i+1}" for i in range(num_pred_steps)]
        test_predictions_df = pd.DataFrame(test_predictions, columns=pred_cols)
        if test_dates is not None:
            test_predictions_df['DATE_TIME'] = pd.Series(test_dates[:len(test_predictions_df)])
        else:
            test_predictions_df['DATE_TIME'] = pd.NaT
        cols = ['DATE_TIME'] + [col for col in test_predictions_df.columns if col != 'DATE_TIME']
        test_predictions_df = test_predictions_df[cols]
        denorm_y_test_df = pd.DataFrame(denorm_y_test, columns=[f"Target_{i+1}" for i in range(denorm_y_test.shape[1])])
        test_predictions_df = pd.concat([test_predictions_df, denorm_y_test_df], axis=1)
        test_predictions_df['test_CLOSE'] = denorm_test_close_prices
        from app.data_handler import write_csv
        write_csv(file_path=final_test_file, data=test_predictions_df, include_date=False, headers=config.get('headers', True))
        print(f"Final validation predictions saved to {final_test_file}")

        # Compute and save uncertainty estimates (denormalized).
        print("Computing uncertainty estimates using MC sampling...")
        try:
            mc_samples = config.get("mc_samples", 100)
            _, uncertainty_estimates = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)
            if config.get("use_normalization_json") is not None:
                norm_json = config.get("use_normalization_json")
                if isinstance(norm_json, str):
                    with open(norm_json, 'r') as f:
                        norm_json = json.load(f)
                if "CLOSE" in norm_json:
                    diff = norm_json["CLOSE"]["max"] - norm_json["CLOSE"]["min"]
                    denorm_uncertainty = uncertainty_estimates * diff
                else:
                    print("Warning: 'CLOSE' not found; uncertainties remain normalized.")
                    denorm_uncertainty = uncertainty_estimates
            else:
                denorm_uncertainty = uncertainty_estimates
            uncertainty_df = pd.DataFrame(denorm_uncertainty, columns=[f"Uncertainty_{i+1}" for i in range(denorm_uncertainty.shape[1])])
            if test_dates is not None:
                uncertainty_df['DATE_TIME'] = pd.Series(test_dates[:len(uncertainty_df)])
            else:
                uncertainty_df['DATE_TIME'] = pd.NaT
            cols = ['DATE_TIME'] + [col for col in uncertainty_df.columns if col != 'DATE_TIME']
            uncertainty_df = uncertainty_df[cols]
            uncertainties_file = config.get("uncertainties_file", "test_uncertainties.csv")
            uncertainty_df.to_csv(uncertainties_file, index=False)
            print(f"Uncertainty predictions saved to {uncertainties_file}")
        except Exception as e:
            print(f"Failed to compute or save uncertainty predictions: {e}")

        # Plot predictions for the configured horizon.
        plotted_horizon = config.get("plotted_horizon", 6)
        plotted_idx = plotted_horizon - 1
        if plotted_idx >= test_predictions.shape[1]:
            raise ValueError(f"Plotted horizon index {plotted_idx} is out of bounds for predictions shape {test_predictions.shape}")
        pred_plot = test_predictions[:, plotted_idx]
        n_plot = config.get("plot_points", 1575)
        if len(pred_plot) > n_plot:
            pred_plot = pred_plot[-n_plot:]
            test_dates_plot = test_dates[-n_plot:] if test_dates is not None else np.arange(len(pred_plot))
        else:
            test_dates_plot = test_dates if test_dates is not None else np.arange(len(pred_plot))
        true_plot = denorm_test_close_prices
        if len(true_plot) > len(test_dates_plot):
            true_plot = true_plot[-len(test_dates_plot):]
        uncertainty_plot = denorm_uncertainty[:, plotted_idx]
        if len(uncertainty_plot) > n_plot:
            uncertainty_plot = uncertainty_plot[-n_plot:]
        plot_color_predicted = config.get("plot_color_predicted", "blue")
        plot_color_true = config.get("plot_color_true", "red")
        plot_color_uncertainty = config.get("plot_color_uncertainty", "green")
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates_plot, pred_plot, label="Predicted Price", color=plot_color_predicted, linewidth=2)
        plt.plot(test_dates_plot, true_plot, label="True Price", color=plot_color_true, linewidth=2)
        plt.fill_between(test_dates_plot, pred_plot - uncertainty_plot, pred_plot + uncertainty_plot,
                         color=plot_color_uncertainty, alpha=0.15, label="Uncertainty")
        if config.get("use_daily", False):
            plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} days)")
        else:
            plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} hours)")
        plt.xlabel("Close Time")
        plt.ylabel("EUR Price [USD]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try:
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300)
            plt.close()
            print(f"Prediction plot saved to {predictions_plot_file}")
        except Exception as e:
            print(f"Failed to generate prediction plot: {e}")

        # Plot and save the model diagram.
        try:
            plot_model(predictor_plugin.model, to_file=config['model_plot_file'],
                       show_shapes=True, show_dtype=False, show_layer_names=True,
                       expand_nested=True, dpi=300, show_layer_activations=True)
            print(f"Model plot saved to {config['model_plot_file']}")
        except Exception as e:
            print(f"Failed to generate model plot: {e}")
            print("Download Graphviz from https://graphviz.org/download/")
        save_model_file = config.get("save_model", "pretrained_model.keras")
        try:
            predictor_plugin.save(save_model_file)
            print(f"Model saved to {save_model_file}")
        except Exception as e:
            print(f"Failed to save model to {save_model_file}: {e}")

        print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

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
