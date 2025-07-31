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

def load_normalization_json(config):
    """Loads normalization parameters from JSON file."""
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
                return norm_json
            except Exception as e:
                print(f"WARN: Failed to load norm JSON {norm_json}: {e}")
                return {}
        return norm_json
    return {}

def denormalize(data, config):
    """Denormalizes price or price delta using JSON normalization parameters."""
    data = np.asarray(data)
    norm_json = load_normalization_json(config)
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
    """Denormalizes return values using per-horizon stats."""
    data = np.asarray(data)
    if horizon_idx is not None and target_returns_mean is not None and target_returns_std is not None:
        try:
            h_mean = target_returns_mean[horizon_idx]
            h_std = target_returns_std[horizon_idx]
            # Denormalize to real-world return scale
            return (data * h_std) + h_mean
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
                        baseline_train_h = baseline_train[:num_train_pts]
                        val_preds_h = val_preds_h[:num_val_pts]
                        val_target_h = val_target_h[:num_val_pts]
                        val_unc_h = val_unc_h[:num_val_pts]
                        baseline_val_h = baseline_val[:num_val_pts]
                        if use_returns:
                            train_preds_denorm = denormalize_returns(train_preds_h, config, idx, target_returns_mean, target_returns_std)
                            train_target_denorm = denormalize_returns(train_target_h, config, idx, target_returns_mean, target_returns_std)
                            baseline_train_denorm = baseline_train_h  # Already in real-world scale
                            train_pred_price = baseline_train_denorm + train_preds_denorm
                            train_target_price = baseline_train_denorm + train_target_denorm
                            val_preds_denorm = denormalize_returns(val_preds_h, config, idx, target_returns_mean, target_returns_std)
                            val_target_denorm = denormalize_returns(val_target_h, config, idx, target_returns_mean, target_returns_std)
                            baseline_val_denorm = baseline_val_h  # Already in real-world scale
                            val_pred_price = baseline_val_denorm + val_preds_denorm
                            val_target_price = baseline_val_denorm + val_target_denorm
                            train_mae = np.abs(train_pred_price - train_target_price).mean()
                            val_mae = np.abs(val_pred_price - val_target_price).mean()
                            train_r2 = r2_score(train_target_price, train_pred_price)
                            val_r2 = r2_score(val_target_price, val_pred_price)
                        else:
                            train_pred_price = denormalize(train_preds_h, config)
                            train_target_price = denormalize(train_target_h, config)
                            val_pred_price = denormalize(val_preds_h, config)
                            val_target_price = denormalize(val_target_h, config)
                            train_mae = np.abs(train_pred_price - train_target_price).mean()
                            val_mae = np.abs(val_pred_price - val_target_price).mean()
                            train_r2 = r2_score(train_target_price, train_pred_price)
                            val_r2 = r2_score(val_target_price, val_pred_price)
                        train_unc_avg = train_unc_h.mean()
                        val_unc_avg = val_unc_h.mean()
                        train_snr = np.abs(train_pred_price.mean() / (train_unc_h.std() + 1e-8))
                        val_snr = np.abs(val_pred_price.mean() / (val_unc_h.std() + 1e-8))
                        metrics_results["Train"]["MAE"][h].append(train_mae)
                        metrics_results["Train"]["R2"][h].append(train_r2)
                        metrics_results["Train"]["Uncertainty"][h].append(train_unc_avg)
                        metrics_results["Train"]["SNR"][h].append(train_snr)
                        metrics_results["Validation"]["MAE"][h].append(val_mae)
                        metrics_results["Validation"]["R2"][h].append(val_r2)
                        metrics_results["Validation"]["Uncertainty"][h].append(val_unc_avg)
                        metrics_results["Validation"]["SNR"][h].append(val_snr)
                        print(f"  H={h}: Train MAE={train_mae:.6f}, R2={train_r2:.6f}, Unc={train_unc_avg:.6f}, SNR={train_snr:.6f}")
                        print(f"  H={h}: Val MAE={val_mae:.6f}, R2={val_r2:.6f}, Unc={val_unc_avg:.6f}, SNR={val_snr:.6f}")
                    except Exception as e:
                        print(f"WARN: Failed metrics for H={h}: {e}")

            print(f"\nPredicting on Test set...")
            list_test_preds_h, list_test_unc_h = predictor_plugin.predict(X_test, config=config)
            if len(list_test_preds_h) != num_outputs or len(list_test_unc_h) != num_outputs:
                raise ValueError(f"Test predictions ({len(list_test_preds_h)}) or uncertainties ({len(list_test_unc_h)}) mismatch with {num_outputs} outputs")
            list_test_preds.append(list_test_preds_h)
            list_test_unc.append(list_test_unc_h)

            print(f"Calculating Test metrics (all horizons)...")
            for idx, h in enumerate(predicted_horizons):
                try:
                    test_preds_h = list_test_preds_h[idx].flatten()
                    test_target_h = y_test_dict[output_names[idx]].flatten()
                    test_unc_h = list_test_unc_h[idx].flatten()
                    num_test_pts = min(len(test_preds_h), len(test_target_h), len(baseline_test))
                    test_preds_h = test_preds_h[:num_test_pts]
                    test_target_h = test_target_h[:num_test_pts]
                    test_unc_h = test_unc_h[:num_test_pts]
                    baseline_test_h = baseline_test[:num_test_pts]
                    if use_returns:
                        test_preds_denorm = denormalize_returns(test_preds_h, config, idx, target_returns_mean, target_returns_std)
                        test_target_denorm = denormalize_returns(test_target_h, config, idx, target_returns_mean, target_returns_std)
                        baseline_test_denorm = baseline_test_h  # Already in real-world scale
                        test_pred_price = baseline_test_denorm + test_preds_denorm
                        test_target_price = baseline_test_denorm + test_target_denorm
                        test_mae = np.abs(test_pred_price - test_target_price).mean()
                        test_r2 = r2_score(test_target_price, test_pred_price)
                    else:
                        test_pred_price = denormalize(test_preds_h, config)
                        test_target_price = denormalize(test_target_h, config)
                        test_mae = np.abs(test_pred_price - test_target_price).mean()
                        test_r2 = r2_score(test_target_price, test_pred_price)
                    test_unc_avg = test_unc_h.mean()
                    test_snr = np.abs(test_pred_price.mean() / (test_unc_h.std() + 1e-8))
                    metrics_results["Test"]["MAE"][h].append(test_mae)
                    metrics_results["Test"]["R2"][h].append(test_r2)
                    metrics_results["Test"]["Uncertainty"][h].append(test_unc_avg)
                    metrics_results["Test"]["SNR"][h].append(test_snr)
                    print(f"  H={h}: Test MAE={test_mae:.6f}, R2={test_r2:.6f}, Unc={test_unc_avg:.6f}, SNR={test_snr:.6f}")
                except Exception as e:
                    print(f"WARN: Failed test metrics for H={h}: {e}")

            iter_time = time.time() - iter_start
            print(f"Iteration {iteration} completed in {iter_time:.2f}s.")

        print("\n=== Aggregating Results ===")
        results_df = pd.DataFrame()
        for ds in data_sets:
            for metric in metric_names:
                for h in predicted_horizons:
                    values = metrics_results[ds][metric][h]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values) if len(values) > 1 else 0.0
                        results_df.loc[f"{ds}_{metric}", f"H={h}_mean"] = mean_val
                        results_df.loc[f"{ds}_{metric}", f"H={h}_std"] = std_val
        try:
            write_csv(results_df, config.get("results_file", "results.csv"))
            print(f"Results saved to {config.get('results_file', 'results.csv')}")
        except Exception as e:
            print(f"WARN: Failed to save results: {e}")

        print("\n=== Saving Test Predictions ===")
        test_pred_price = []
        for idx, h in enumerate(predicted_horizons):
            try:
                test_preds_h = np.mean([list_test_preds[i][idx] for i in range(len(list_test_preds))], axis=0).flatten()
                test_unc_h = np.mean([list_test_unc[i][idx] for i in range(len(list_test_unc))], axis=0).flatten()
                num_test_pts = min(len(test_preds_h), len(baseline_test))
                test_preds_h = test_preds_h[:num_test_pts]
                test_unc_h = test_unc_h[:num_test_pts]
                baseline_test_h = baseline_test[:num_test_pts]
                if use_returns:
                    test_preds_denorm = denormalize_returns(test_preds_h, config, idx, target_returns_mean, target_returns_std)
                    baseline_test_denorm = baseline_test_h  # Already in real-world scale
                    test_pred_price_h = baseline_test_denorm + test_preds_denorm
                else:
                    test_pred_price_h = denormalize(test_preds_h, config)
                test_pred_price.append(test_pred_price_h)
                output_df = pd.DataFrame({
                    'Date': test_dates[:num_test_pts] if test_dates is not None else range(num_test_pts),
                    f'Prediction_H{h}': test_pred_price_h,
                    f'Uncertainty_H{h}': test_unc_h
                })
                try:
                    write_csv(output_df, f"{config.get('output_file', 'test_predictions')}_h{h}.csv")
                    print(f"Saved predictions for H={h} to {config.get('output_file', 'test_predictions')}_h{h}.csv")
                except Exception as e:
                    print(f"WARN: Failed to save predictions for H={h}: {e}")
            except Exception as e:
                print(f"WARN: Failed processing predictions for H={h}: {e}")

        print("\n=== Plotting Loss ===")
        if history:
            plt.figure(figsize=(10, 6))
            for metric in history.history.keys():
                if not metric.startswith('val_'):
                    val_metric = f'val_{metric}'
                    if val_metric in history.history:
                        plt.plot(history.history[metric], label=f'Train {metric}')
                        plt.plot(history.history[val_metric], label=f'Val {metric}', linestyle='--')
            plt.xlabel('Epoch')
            plt.ylabel('Loss/Metric')
            plt.legend()
            plt.title('Training and Validation Loss')
            try:
                plt.savefig(config.get("loss_plot_file", "loss_plot.png"))
                plt.close()
                print(f"Loss plot saved to {config.get('loss_plot_file', 'loss_plot.png')}")
            except Exception as e:
                print(f"WARN: Failed to save loss plot: {e}")

        print("\n=== Plotting Model ===")
        if plot_model and hasattr(predictor_plugin, 'model'):
            try:
                plot_model(predictor_plugin.model, to_file=config.get("model_plot_file", "model_plot.png"), show_shapes=True)
                print(f"Model plot saved to {config.get('model_plot_file', 'model_plot.png')}")
            except Exception as e:
                print(f"WARN: Failed to save model plot: {e}")

        print("\n=== Plotting Predictions ===")
        try:
            plt.figure(figsize=(12, 6))
            plot_points = min(config.get("plot_points", 480), len(test_pred_price[plotted_index]))
            start_idx = max(0, len(test_pred_price[plotted_index]) - plot_points)
            target_plot = test_pred_price[plotted_index][start_idx:]
            pred_plot = test_pred_price[plotted_index][start_idx:]
            baseline_plot = baseline_test[start_idx:start_idx + plot_points]
            target_plot_raw = y_test_dict[output_names[plotted_index]][start_idx:start_idx + plot_points].flatten()
            plot_h_mean = target_returns_mean[plotted_index]
            plot_h_std = target_returns_std[plotted_index]
            target_denorm_return = denormalize_returns(target_plot_raw, config, plotted_index, target_returns_mean, target_returns_std)
            target_plot_price_flat = (baseline_plot + target_denorm_return).flatten()
            true_plot_raw = datasets["y_test_raw"][start_idx + plotted_horizon:start_idx + plotted_horizon + plot_points]
            true_plot_price_flat = true_plot_raw.flatten()  # Already in real-world scale
            pred_plot_flat = pred_plot.flatten()
            test_dates_plot = test_dates[start_idx:start_idx + plot_points] if test_dates is not None else range(plot_points)
            plt.plot(test_dates_plot, true_plot_price_flat, label="True Price", color="blue")
            plt.plot(test_dates_plot, pred_plot_flat, label="Predicted Price", color="red", linestyle="--")
            plt.plot(test_dates_plot, target_plot_price_flat, label="Target Price", color="green", linestyle=":")
            plt.xlabel("Date" if test_dates is not None else "Index")
            plt.ylabel("Price")
            plt.title(f"Predictions vs True Prices (H={plotted_horizon})")
            plt.legend()
            try:
                plt.savefig(config.get("predictions_plot_file", "predictions_plot.png"))
                plt.close()
                print(f"Predictions plot saved to {config.get('predictions_plot_file', 'predictions_plot.png')}")
            except Exception as e:
                print(f"WARN: Failed to save predictions plot: {e}")
        except Exception as e:
            print(f"WARN: Failed to generate predictions plot: {e}")

        total_time = time.time() - start_time
        print(f"\n=== Pipeline completed in {total_time:.2f}s ===")
        ret = {
            "predictions": test_pred_price,
            "uncertainties": [np.mean([list_test_unc[i][j] for i in range(len(list_test_unc))], axis=0).flatten() for j in range(num_outputs)],
            "metrics": metrics_results,
            "history": history
        }
        return ret