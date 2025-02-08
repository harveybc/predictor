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
    one of the 6 ticks. This avoids the iterative .append() overhead and is usually faster.
    It preserves the exact interface of the ANN plugin.
    """

    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.001,
        'activation': 'tanh',
        'patience': 5,
        'l2_reg': 1e-3
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        """
        Initialize plugin with defaults.
        """
        self.params = self.plugin_params.copy()
        # We'll store 6 separate SARIMAX results (one per horizon/tick)
        self.results_list = [None]*6
        # We'll also store each submodel's training length:
        self.train_lengths = [0]*6

        self.model = None  # Unused, but kept for interface consistency

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided kwargs.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return debug info for external usage.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Build the 'model' placeholders. We'll do actual instantiations in train().
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        self.params['input_dim'] = input_shape
        print(f"Parallel Single-Step SARIMAX => input_shape: {input_shape}, "
              f"time_horizon: {self.params['time_horizon']}")
        self.model = None

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train 6 separate submodels, one for each horizon step. Each is a standard
        SARIMAX(0,1,0) (basic differencing) for demonstration, but you can adjust.
        
        Returns:
            (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        horizon = self.params['time_horizon']
        if y_train.shape[1] != horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected 2D with {horizon} steps.")

        exog_train = x_train if self.params['input_dim'] > 0 else None
        N_train = len(x_train)

        print(f"Building {horizon} separate SARIMAX(0,1,0) models.")
        
        # Fit each of the 6 sub-models
        for k in range(horizon):
            print(f"Fitting model for horizon step {k} ...")
            endog_k = y_train[:, k]
            self.train_lengths[k] = N_train  # store length for out-of-sample predictions

            # Basic example: (p,d,q) = (0,1,0)
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

        # Keras-like history object => we store 'loss' and 'val_loss'
        class MockHistory:
            def __init__(self):
                self.history = {'loss': [], 'val_loss': []}
        history = MockHistory()

        # Compute average 1-step MAE across all 6 submodels
        mae_list = []
        for k in range(horizon):
            endog_k = y_train[:, k]
            one_step_pred_k = self.results_list[k].predict(
                start=0, end=N_train - 1, exog=exog_train
            )
            mae_k = np.mean(np.abs(one_step_pred_k - endog_k))
            mae_list.append(mae_k)
        final_loss = np.mean(mae_list)
        history.history['loss'].append(final_loss)
        print(f"Average 1-step training MAE across 6 horizons: {final_loss}")

        # Compare final_loss to threshold
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Evaluate multi-step predictions on training data
        train_predictions = self.predict(x_train)
        print(f"train_predictions shape: {train_predictions.shape}, y_train shape: {y_train.shape}")
        train_mae = self.calculate_mae(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        # Evaluate on validation data
        if x_val is not None and y_val is not None:
            val_predictions = self.predict(x_val)
            print(f"val_predictions shape: {val_predictions.shape}, y_val shape: {y_val.shape}")
            val_mae = self.calculate_mae(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
            # store val_mae in 'val_loss'
            history.history['val_loss'].append(val_mae)
        else:
            val_predictions = np.array([])
            val_mae = None
            val_r2 = None
            history.history['val_loss'].append(None)

        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions

    def predict(self, data):
        """
        Predict multi-step shape => (N, horizon).
        For each submodel k, do a vectorized predict using out-of-sample indices:
            start = self.train_lengths[k]
            end   = self.train_lengths[k] + N - 1

        So statsmodels knows we want N out-of-sample predictions with shape (N, input_dim).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        horizon = self.params['time_horizon']
        N = len(data)
        preds = np.zeros((N, horizon))

        exog_data = data if self.params['input_dim'] > 0 else None

        # We forecast from index=[train_length..train_length+N-1], so we pass exog of shape (N, input_dim)
        for k in range(horizon):
            if self.results_list[k] is None:
                raise ValueError(f"Model for horizon {k} not trained.")
            start_idx = self.train_lengths[k]
            end_idx   = self.train_lengths[k] + N - 1
            preds[:, k] = self.results_list[k].predict(
                start=start_idx, end=end_idx, exog=exog_data
            )

        print(f"Parallel Single-Step predict => input: {data.shape}, output: {preds.shape}")
        return preds

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with (N, horizon).
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        mse = np.mean((y_true.flatten() - y_pred.flatten())**2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Flatten-based MAE => consistent with (N, horizon).
        """
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        """
        Save the 6 submodels.
        """
        # ensure submodels are trained
        if any(r is None for r in self.results_list):
            raise ValueError("One or more sub-models not trained. Train first.")
        # Save both the results_list and the train_lengths
        save_data = {
            'results_list': self.results_list,
            'train_lengths': self.train_lengths
        }
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Predictor model(s) saved to {file_path}")

    def load(self, file_path):
        """
        Load previously saved submodels.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.results_list = data['results_list']
            self.train_lengths = data['train_lengths']
        print(f"Model(s) loaded from {file_path}")
