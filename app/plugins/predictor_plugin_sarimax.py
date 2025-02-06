import numpy as np
import logging
import os
import pickle
import pandas as pd

from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Plugin:
    """
    SARIMAX Predictor Plugin using 6 separate 1-step models, each forecasting
    one of the 6 ticks. This avoids iterative .append() and is usually faster.
    Preserves the same interface (ANN plugin style).
    """

    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.001,
        'activation': 'tanh',
        'patience': 5,
        'l2_reg': 1e-3,
        'time_horizon': 6  # we fix 6-step horizon
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        # We'll store 6 separate SARIMAX results:
        self.results_list = [None]*6  
        self.model = None  # Unused, but we keep it for interface consistency

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        self.params['input_dim'] = input_shape
        print(f"Parallel Single-Step SARIMAX => input_shape: {input_shape}, time_horizon: {self.params['time_horizon']}")
        # We won't actually build anything yet; done in train() for each step
        self.model = None

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        horizon = self.params['time_horizon']
        if y_train.shape[1] != horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected 2D with {horizon} steps.")

        # We'll train 6 separate models: each uses y_train[:,k] as endog
        exog_train = x_train if self.params['input_dim'] > 0 else None

        print(f"Building {horizon} separate SARIMAX(0,1,0) models for demonstration.")

        for k in range(horizon):
            print(f"Fitting model for horizon step {k} ...")
            endog_k = y_train[:, k]

            # Basic example: (p,d,q)=(0,1,0), no seasonality
            model_k = SARIMAX(
                endog=endog_k,
                exog=exog_train,
                order=(0, 1, 0),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results_k = model_k.fit(disp=False)
            self.results_list[k] = results_k

        # Mock history object
        class MockHistory:
            def __init__(self):
                self.history = {'loss': []}

        history = MockHistory()

        # We'll define a "final_loss" as the average MAE of all 6 one-step fits
        mae_list = []
        for k in range(horizon):
            endog_k = y_train[:, k]
            exog_k = exog_train
            one_step_pred_k = self.results_list[k].predict(start=0, end=len(endog_k)-1, exog=exog_k)
            mae_k = np.mean(np.abs(one_step_pred_k - endog_k))
            mae_list.append(mae_k)

        final_loss = np.mean(mae_list)
        history.history['loss'].append(final_loss)
        print(f"Average 1-step training MAE across 6 horizons: {final_loss}")
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Evaluate on train data
        train_predictions = self.predict(x_train)
        print(f"train_predictions shape: {train_predictions.shape}, y_train shape: {y_train.shape}")
        train_mae = self.calculate_mae(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        # Evaluate on val if provided
        if x_val is not None and y_val is not None:
            val_predictions = self.predict(x_val)
            print(f"val_predictions shape: {val_predictions.shape}, y_val shape: {y_val.shape}")
            val_mae = self.calculate_mae(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
        else:
            val_predictions = np.array([])
            val_mae = None
            val_r2 = None

        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions

    def predict(self, data):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        horizon = self.params['time_horizon']
        N = len(data)
        preds = np.zeros((N, horizon))

        exog_data = data if self.params['input_dim'] > 0 else None

        # For each horizon k, we do a single-step forecast for all N rows at once.
        # This is typically vectorized => fast
        for k in range(horizon):
            if self.results_list[k] is None:
                raise ValueError(f"Model for horizon {k} not trained.")
            preds[:, k] = self.results_list[k].predict(start=0, end=N-1, exog=exog_data)

        print(f"Parallel Single-Step predict => input: {data.shape}, output: {preds.shape}")
        return preds

    def calculate_mse(self, y_true, y_pred):
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        mse = np.mean((y_true.flatten() - y_pred.flatten())**2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        if any(r is None for r in self.results_list):
            raise ValueError("One or more sub-models not trained. Train first.")
        with open(file_path, 'wb') as f:
            pickle.dump(self.results_list, f)
        print(f"Predictor model(s) saved to {file_path}")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.results_list = pickle.load(f)
        print(f"Model(s) loaded from {file_path}")
