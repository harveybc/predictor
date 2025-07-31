#!/usr/bin/env python
"""
STL Pipeline Plugin - Per-Horizon Z-Score Denormalization
Uses per-horizon target normalization parameters from preprocessor.
Handles separate mean/std for each prediction horizon during denormalization.
"""

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

try:
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None

try:
    from app.data_handler import write_csv
except ImportError:
    print("CRITICAL ERROR: Could not import 'write_csv' from 'app.data_handler'.")
    raise

def denormalize(data, config):
    """Denormalizes price or price delta using JSON normalization parameters."""
    data = np.asarray(data)
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f: norm_json = json.load(f)
            except Exception as e: 
                print(f"WARN: Failed load norm JSON {norm_json}: {e}")
                return data
        if isinstance(norm_json, dict) and "CLOSE" in norm_json:
            try:
                if "mean" in norm_json["CLOSE"] and "std" in norm_json["CLOSE"]:
                    close_mean = norm_json["CLOSE"]["mean"]
                    close_std = norm_json["CLOSE"]["std"]
                    return (data * close_std) + close_mean
                else:
                    print(f"WARN: Missing 'mean' or 'std' in norm JSON")
                    return data
            except Exception as e: 
                print(f"WARN: Error during denormalize: {e}")
                return data
    return data

def denormalize_returns(data, config, horizon_idx=None, target_returns_mean=None, target_returns_std=None):
    """Denormalizes return values using per-horizon stats and JSON parameters."""
    data = np.asarray(data)
    if horizon_idx is not None and target_returns_mean is not None and target_returns_std is not None:
        try:
            h_mean = target_returns_mean[horizon_idx]
            h_std = target_returns_std[horizon_idx]
            # Denormalize to normalized close price scale
            denorm_return = (data * h_std) + h_mean
            # Further denormalize to real-life scale using JSON parameters
            return denormalize(denorm_return, config)
        except Exception as e: 
            print(f"WARN: Error in denormalize_returns for horizon {horizon_idx}: {e}")
            return data
    return data

class STLPipelinePlugin:
    plugin_params = {
        "iterations": 1, "batch_size": 32, "epochs": 50, "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png", "output_file": "test_predictions.csv",
        "uncertainties_file": "test_uncertainties.csv", "model_plot_file": "model_plot.png",
        "predictions_plot_file": "predictions_plot.png", "results_file": "results.csv",
        "plot_points": 480, "plotted_horizon": 6, "use_strategy": False,
        "predicted_horizons": [1, 6, 12, 24], "use_returns": True, "normalize_features": True,
        "window_size": 48, "target_column": "TARGET", "use_normalization_json": None,
        "mc_samples": 100,
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error", "output_file", 
                        "uncertainties_file", "results_file", "plotted_horizon", "plot_points"]

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

        predicted_horizons = config.get('predicted_horizons')
        num_outputs = len(predicted_horizons)
        metric_names = ["MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {ds: {mn: {h: [] for h in predicted_horizons} for mn in metric_names} for ds in data_sets}

        print("Loading/processing datasets via Preprocessor...")
        datasets, plugin_debug_vars = preprocessor_plugin.run_preprocessing(config)
        print("Preprocessor finished.")
        X_train = datasets["x_train"]
        X_val = datasets["x_val"]
        X_test = datasets["x_test"]
        y_train_dict = datasets["y_train"]
        y_val_dict = datasets["y_val"]
        y_test_dict = datasets["y_test"]
        train_dates = datasets.get("y_train_dates")
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")
        test_close_prices = datasets.get("test_close_prices")
        use_returns = config.get("use_returns", False)
        if use_returns and (baseline_train is None or baseline_val is None or baseline_test is None):
            raise ValueError("Baselines required when use_returns=True.")

        target_returns_mean = plugin_debug_vars.get('target_returns_mean', [0.0] * len(predicted_horizons))
        target_returns_std = plugin_debug_vars.get('target_returns_std', [1.0] * len(predicted_horizons))
        if not isinstance(target_returns_mean, list):
            print(f"WARN: target_returns_mean not a list. Converting: {target_returns_mean}")
            target_returns_mean = [float(target_returns_mean)] * len(predicted_horizons) if isinstance(target_returns_mean, (int, float)) else [0.0] * len(predicted_horizons)
        if not isinstance(target_returns_std, list):
            print(f"WARN: target_returns_std not a list. Converting: {target_returns_std}")
            target_returns_std = [float(target_returns_std)] * len(predicted_horizons) if isinstance(target_returns_std, (int, float)) else [1.0] * len(predicted_horizons)
        if len(target_returns_mean) != len(predicted_horizons):
            print(f"WARN: target_returns_mean length mismatch. Expected {len(predicted_horizons)}, got {len(target_returns_mean)}")
            target_returns_mean = target_returns_mean[:len(predicted_horizons)] + [0.0] * max(0, len(predicted_horizons) - len(target_returns_mean))
        if len(target_returns_std) != len(predicted_horizons):
            print(f"WARN: target_returns_std length mismatch. Expected {len(predicted_horizons)}, got {len(target_returns_std)}")
            target_returns_std = target_returns_std[:len(predicted_horizons)] + [1.0] * max(0, len(predicted_horizons) - len(target_returns_std))

        if use_returns:
            print(f"Per-horizon target normalization stats loaded:")
            for i, h in enumerate(predicted_horizons):
                print(f"  Horizon {h}: Mean={target_returns_mean[i]:.6f}, Std={target_returns_std[i]:.6f}")

        plotted_horizon = config.get('plotted_horizon')
        try:
            plotted_index = predicted_horizons.index(plotted_horizon)
        except ValueError:
            print(f"WARN: Plotted horizon {plotted_horizon} not in {predicted_horizons}. Using first horizon.")
            plotted_horizon = predicted_horizons[0]
            plotted_index = 0
        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        print("Available y_train_dict keys:", list(y_train_dict.keys()))
        print("Expected output_names:", output_names)

        def truncate_targets(target_dict, output_names):
            missing = [name for name in output_names if name not in target_dict]
            if missing:
                raise KeyError(f"Missing target horizons in preprocessor output: {missing}.")
            lengths = [len(target_dict[name]) for name in output_names]
            min_len = min(lengths)
            truncated = {}
            for name in output_names:
                arr = target_dict[name][:min_len]
                truncated[name] = arr.reshape(-1, 1).astype(np.float32)
            return truncated

        y_train_dict = truncate_targets(y_train_dict, output_names)
        y_val_dict = truncate_targets(y_val_dict, output_names)
        y_test_dict = truncate_targets(y_test_dict, output_names)
        print(f"Input shapes: Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")
        print(f"Target shapes(H={predicted_horizons[0]}): Train:{y_train_dict[output_names[0]].shape}, Val:{y_val_dict[output_names[0]].shape}, Test:{y_test_dict[output_names[0]].shape}")
        batch_size = config.get("batch_size", 32)
        epochs = config.get("epochs", 50)
        print(f"Predicting Horizons: {predicted_horizons}, Plotting: H={plotted_horizon}")

        list_test_preds = []
        list_test_unc = []
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()
            input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],)
            predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)
            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
                X_train, y_train_dict, epochs=epochs, batch_size=batch_size, threshold_error=config.get("threshold_error", 0.001),
                x_val=X_val, y_val=y_val_dict, config=config
            )

            can_calc_train_stats = all(len(lst) == num_outputs for lst in [list_train_preds, list_train_unc])
            if can_calc_train_stats:
                print("Calculating Train/Validation metrics (all horizons)...")
                for idx, h in enumerate(predicted_horizons):
                    try:
                        h_mean = target_returns_mean[idx]
                        h_std = target_returns_std[idx]
                        train_preds_h = list_train_preds[idx].flatten()
                        train_target_h = y_train_dict[output_names[idx]].flatten()
                        train_unc_h = list_train_unc[idx].flatten()
                        val_preds_h = list_val_preds[idx].flatten()
                        val_target_h = y_val_dict[output_names[idx]].flatten()
                        val_unc_h = list_val_unc[idx].flatten()
                        num_train_pts = min(len(train_preds_h), len(train_target_h), len(baseline_train))
                        num_val_pts = min(len(val_preds_h), len(val_target_h), len(baseline_val))
                        train_preds_h = train_preds_h[:num_train_pts]
                        train_target_h = train_target_h[:num_train_pts]
                        train_unc_h = train_unc_h[:num_train_pts]
                        baseline_train_h = baseline_train[:num_train_pts].flatten()
                        val_preds_h = val_preds_h[:num_val_pts]
                        val_target_h = val_target_h[:num_val_pts]
                        val_unc_h = val_unc_h[:num_val_pts]
                        baseline_val_h = baseline_val[:num_val_pts].flatten()

                        if use_returns:
                            baseline_train_denorm = denormalize(baseline_train_h, config)
                            baseline_val_denorm = denormalize(baseline_val_h, config)
                            train_preds_denorm_return = denormalize_returns(train_preds_h, config, idx, target_returns_mean, target_returns_std)
                            train_target_denorm_return = denormalize_returns(train_target_h, config, idx, target_returns_mean, target_returns_std)
                            val_preds_denorm_return = denormalize_returns(val_preds_h, config, idx, target_returns_mean, target_returns_std)
                            val_target_denorm_return = denormalize_returns(val_target_h, config, idx, target_returns_mean, target_returns_std)
                            train_pred_price = baseline_train_denorm + train_preds_denorm_return
                            train_target_price = baseline_train_denorm + train_target_denorm_return
                            val_pred_price = baseline_val_denorm + val_preds_denorm_return
                            val_target_price = baseline_val_denorm + val_target_denorm_return
                            train_unc_final = denormalize_returns(train_unc_h, config, idx, target_returns_mean, target_returns_std)
                            val_unc_final = denormalize_returns(val_unc_h, config, idx, target_returns_mean, target_returns_std)
                        else:
                            train_pred_price = denormalize(train_preds_h, config)
                            train_target_price = denormalize(train_target_h, config)
                            val_pred_price = denormalize(val_preds_h, config)
                            val_target_price = denormalize(val_target_h, config)
                            train_unc_final = denormalize_returns(train_unc_h, config)
                            val_unc_final = denormalize_returns(val_unc_h, config)

                        train_mae_h = np.mean(np.abs(train_pred_price - train_target_price))
                        train_r2_h = r2_score(train_target_price, train_pred_price)
                        train_unc_mean_h = np.mean(np.abs(train_unc_final))
                        train_snr_h = np.mean(np.abs(train_pred_price)) / (train_unc_mean_h + 1e-9)
                        val_mae_h = np.mean(np.abs(val_pred_price - val_target_price))
                        val_r2_h = r2_score(val_target_price, val_pred_price)
                        val_unc_mean_h = np.mean(np.abs(val_unc_final))
                        val_snr_h = np.mean(np.abs(val_pred_price)) / (val_unc_mean_h + 1e-9)
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
                print("WARN: Skipping Train/Val metrics - incomplete predictions/uncertainties.")
                [metrics_results[ds][m][h].append(np.nan) for ds in ["Train", "Validation"] for m in metric_names for h in predicted_horizons]

            print("Predicting on test set...")
            try:
                test_pred, test_unc = predictor_plugin.predict(X_test, config)
                list_test_preds.append(test_pred)
                list_test_unc.append(test_unc)
                if len(test_pred) != num_outputs or len(test_unc) != num_outputs:
                    print(f"WARN: Test prediction incomplete. Got {len(test_pred)}/{len(test_unc)} outputs, expected {num_outputs}.")
                    test_pred = [np.zeros_like(y_test_dict[output_names[0]].flatten()) for _ in range(num_outputs)]
                    test_unc = [np.zeros_like(y_test_dict[output_names[0]].flatten()) for _ in range(num_outputs)]
                    list_test_preds[-1] = test_pred
                    list_test_unc[-1] = test_unc
            except Exception as e: 
                print(f"WARN: Error in test prediction: {e}")
                test_pred = [np.zeros_like(y_test_dict[output_names[0]].flatten()) for _ in range(num_outputs)]
                test_unc = [np.zeros_like(y_test_dict[output_names[0]].flatten()) for _ in range(num_outputs)]
                list_test_preds.append(test_pred)
                list_test_unc.append(test_unc)

            print(f"Iteration {iteration} finished in {time.time() - iter_start:.2f} seconds.")

        print("\nCalculating Test metrics...")
        for idx, h in enumerate(predicted_horizons):
            try:
                test_preds_h = np.mean([test_pred[idx].flatten() for test_pred in list_test_preds], axis=0)
                test_unc_h = np.mean([test_unc[idx].flatten() for test_unc in list_test_unc], axis=0)
                test_target_h = y_test_dict[output_names[idx]].flatten()
                num_test_pts = min(len(test_preds_h), len(test_target_h), len(baseline_test))
                test_preds_h = test_preds_h[:num_test_pts]
                test_unc_h = test_unc_h[:num_test_pts]
                test_target_h = test_target_h[:num_test_pts]
                baseline_test_h = baseline_test[:num_test_pts].flatten()

                if use_returns:
                    baseline_test_denorm = denormalize(baseline_test_h, config)
                    test_preds_denorm_return = denormalize_returns(test_preds_h, config, idx, target_returns_mean, target_returns_std)
                    test_target_denorm_return = denormalize_returns(test_target_h, config, idx, target_returns_mean, target_returns_std)
                    test_pred_price = baseline_test_denorm + test_preds_denorm_return
                    test_target_price = baseline_test_denorm + test_target_denorm_return
                    test_unc_final = denormalize_returns(test_unc_h, config, idx, target_returns_mean, target_returns_std)
                else:
                    test_pred_price = denormalize(test_preds_h, config)
                    test_target_price = denormalize(test_target_h, config)
                    test_unc_final = denormalize_returns(test_unc_h, config)

                test_mae_h = np.mean(np.abs(test_pred_price - test_target_price))
                test_r2_h = r2_score(test_target_price, test_pred_price)
                test_unc_mean_h = np.mean(np.abs(test_unc_final))
                test_snr_h = np.mean(np.abs(test_pred_price)) / (test_unc_mean_h + 1e-9)
                metrics_results["Test"]["MAE"][h].append(test_mae_h)
                metrics_results["Test"]["R2"][h].append(test_r2_h)
                metrics_results["Test"]["Uncertainty"][h].append(test_unc_mean_h)
                metrics_results["Test"]["SNR"][h].append(test_snr_h)
            except Exception as e: 
                print(f"WARN: Error Test metrics H={h}: {e}")
                [metrics_results["Test"][m][h].append(np.nan) for m in metric_names]

        print("\nSaving metrics to results file...")
        results_data = []
        for ds in data_sets:
            for h in predicted_horizons:
                for m in metric_names:
                    values = metrics_results[ds][m][h]
                    mean_val = np.nanmean(values) if values else np.nan
                    results_data.append({
                        "Dataset": ds, "Horizon": h, "Metric": m, "Value": mean_val
                    })
        results_df = pd.DataFrame(results_data)
        try:
            write_csv(config.get("results_file", "results.csv"), results_df, headers=True)
            print(f"Metrics saved to {config.get('results_file', 'results.csv')}")
        except Exception as e:
            print(f"ERROR saving results: {e}")

        print("\nGenerating plots...")
        try:
            plt.figure(figsize=(12, 6))
            plot_points = min(config.get("plot_points", 480), len(test_pred_price))
            start_idx = max(0, len(test_pred_price) - plot_points)
            target_plot = test_target_price[start_idx:]
            pred_plot = test_pred_price[start_idx:]
            baseline_plot = baseline_test_h[start_idx:start_idx + plot_points]
            baseline_plot_denorm = denormalize(baseline_plot, config)
            target_plot_raw = y_test_dict[output_names[plotted_index]][start_idx:start_idx + plot_points].flatten()
            plot_h_mean = target_returns_mean[plotted_index]
            plot_h_std = target_returns_std[plotted_index]
            target_denorm_return = denormalize_returns(target_plot_raw, config, plotted_index, target_returns_mean, target_returns_std)
            target_plot_price_flat = (baseline_plot_denorm + target_denorm_return).flatten()
            true_plot_price_flat = baseline_plot_denorm.flatten()
            pred_plot_flat = pred_plot.flatten()
            test_dates_plot = test_dates[start_idx:start_idx + plot_points] if test_dates is not None else range(plot_points)
            plt.plot(test_dates_plot, true_plot_price_flat, label="True Price", color="blue")
            plt.plot(test_dates_plot, pred_plot_flat, label="Predicted Price", color="red", linestyle="--")
            plt.plot(test_dates_plot, target_plot_price_flat, label="Target Price", color="green", linestyle=":")
            plt.title(f"Predictions vs True Prices (Horizon={plotted_horizon})")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.savefig(config.get("predictions_plot_file", "predictions_plot.png"))
            plt.close()
            print(f"Predictions plot saved to {config.get('predictions_plot_file', 'predictions_plot.png')}")
        except Exception as e:
            print(f"ERROR generating predictions plot: {e}")

        print("\nSaving test predictions...")
        test_pred_dict = {}
        for idx, h in enumerate(predicted_horizons):
            test_pred_h = np.mean([test_pred[idx].flatten() for test_pred in list_test_preds], axis=0)
            test_unc_h = np.mean([test_unc[idx].flatten() for test_unc in list_test_unc], axis=0)
            test_target_h = y_test_dict[output_names[idx]].flatten()
            num_test_pts = min(len(test_pred_h), len(test_target_h), len(baseline_test))
            test_pred_h = test_pred_h[:num_test_pts]
            test_unc_h = test_unc_h[:num_test_pts]
            test_target_h = test_target_h[:num_test_pts]
            baseline_test_h = baseline_test[:num_test_pts].flatten()
            if use_returns:
                baseline_test_denorm = denormalize(baseline_test_h, config)
                test_pred_price = baseline_test_denorm + denormalize_returns(test_pred_h, config, idx, target_returns_mean, target_returns_std)
                test_target_price = baseline_test_denorm + denormalize_returns(test_target_h, config, idx, target_returns_mean, target_returns_std)
                test_unc_final = denormalize_returns(test_unc_h, config, idx, target_returns_mean, target_returns_std)
            else:
                test_pred_price = denormalize(test_pred_h, config)
                test_target_price = denormalize(test_target_h, config)
                test_unc_final = denormalize_returns(test_unc_h, config)
            test_pred_dict[f"pred_h{h}"] = test_pred_price
            test_pred_dict[f"unc_h{h}"] = test_unc_final
            test_pred_dict[f"target_h{h}"] = test_target_price
        test_pred_dict["baseline"] = denormalize(baseline_test[:num_test_pts], config)
        if test_dates is not None and len(test_dates) >= num_test_pts:
            test_pred_dict["dates"] = test_dates[:num_test_pts]
        try:
            test_pred_df = pd.DataFrame(test_pred_dict)
            write_csv(config.get("output_file", "test_predictions.csv"), test_pred_df, headers=True)
            print(f"Test predictions saved to {config.get('output_file', 'test_predictions.csv')}")
        except Exception as e:
            print(f"ERROR saving test predictions: {e}")

        print(f"\nPipeline completed in {time.time() - start_time:.2f} seconds.")
        return metrics_results, list_test_preds, list_test_unc