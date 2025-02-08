import numpy as np
import logging
import os
import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

class Plugin:
    """
    SARIMAX Predictor Plugin for multi-step forecasting using a rolling window approach.
    
    This plugin performs the following steps:
      1. Runs auto_arima on the entire training set (first column of y_train) to obtain fixed ARIMA parameters.
      2. For both training and validation datasets, it performs a rolling forecast:
         - For each time tick t, fit a SARIMAX model on the previous min(rolling_window, t+1) observations
           (from the underlying univariate series given by y[:,0]) and the corresponding exogenous data.
         - Use that fitted model to forecast the next 'time_horizon' ticks.
         - This rolling procedure is repeated tick-by-tick (stride=1), yielding an output prediction array
           with the same shape as y (i.e. (N, time_horizon)).
      3. Detailed progress is printed (with a tqdm progress bar) and input/output shapes are reported.
    
    The plugin’s interface (class name, methods, parameters, return values) is identical to the provided ANN plugin.
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
        # Additional parameters for rolling evaluation:
        'rolling_window': 48,   # Use the last 48 ticks for model fitting
        'time_horizon': 6       # Forecast next 6 ticks
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        """
        Initialize plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        # These will store the ARIMA order parameters found by auto_arima:
        self.order = None
        self.seasonal_order = None
        # Stored training/validation data (for rolling forecast evaluation)
        self._x_train = None
        self._y_train = None
        self._x_val = None
        self._y_val = None
        # Rolling forecast predictions
        self.train_predictions = None
        self.val_predictions = None
        self.model = None  # Unused placeholder (kept for interface consistency)

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided kwargs.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return a dict of debug info.
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
        Build the model placeholder.
        
        Args:
            input_shape (int): Number of exogenous features.
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        self.params['input_dim'] = input_shape
        print(f"SARIMAX building - input_shape: {input_shape}")
        print(f"SARIMAX final output dimension (time_horizon): {self.params['time_horizon']}")
        self.model = None

    def _rolling_forecast(self, x, y):
        """
        Internal method to perform rolling window forecasting on dataset (x, y).
        For each time index t (from 0 to N - time_horizon), it fits a SARIMAX model
        on the last min(rolling_window, t+1) observations (using the first column of y)
        and forecasts the next 'time_horizon' ticks using the corresponding future exogenous data.
        
        Args:
            x: np.ndarray of shape (N, input_dim) - exogenous data.
            y: np.ndarray of shape (N, time_horizon) - target multi-step values.
               The first column of y is used as the underlying univariate time series.
               
        Returns:
            preds: np.ndarray of shape (N, time_horizon), where row t contains the forecast
                   for time steps t+1 to t+time_horizon. For indices where forecast cannot be made,
                   the row is filled with np.nan.
        """
        N = len(x)
        horizon = self.params['time_horizon']
        window_size = self.params['rolling_window']
        preds = np.empty((N, horizon))
        preds[:] = np.nan  # initialize with NaNs

        # Use the first column of y as the underlying time series
        ts = y[:, 0]
        print(f"Starting rolling forecast evaluation on data with shape: {x.shape}")
        # Loop over all indices where we have enough future exogenous data.
        # We forecast for t from 0 to N - horizon.
        for t in tqdm(range(N - horizon), desc="Rolling Forecast"):
            # Determine the training window for this forecast: use all available data if t < window_size, else last 'window_size' observations.
            start_idx = 0 if t < window_size else t - window_size + 1
            end_idx = t + 1  # observations up to time t (inclusive)
            y_window = ts[start_idx:end_idx]
            x_window = x[start_idx:end_idx] if x is not None else None
            # For the forecast, we need the exogenous data for the next 'horizon' ticks.
            x_forecast = x[t+1:t+horizon+1] if x is not None else None

            try:
                model = SARIMAX(
                    endog=y_window,
                    exog=x_window,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                forecast = results.get_forecast(steps=horizon, exog=x_forecast)
                preds[t, :] = forecast.predicted_mean.values
            except Exception as e:
                print(f"Rolling forecast at index {t} failed: {e}")
                preds[t, :] = np.nan

        print(f"Rolling forecast completed. Prediction shape: {preds.shape}")
        return preds

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the SARIMAX plugin using a rolling window evaluation.
        
        Procedure:
          1. Run auto_arima on the entire training set (first column of y_train) to obtain ARIMA parameters.
          2. Using a rolling window (of size 48, stride=1), fit a SARIMAX model at each tick and forecast the next 'time_horizon' ticks.
          3. Perform this rolling forecast on both training and validation datasets.
        
        Returns:
            (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{horizon}).")
        
        # Store training (and validation) data for later use in predict()
        self._x_train = x_train
        self._y_train = y_train
        if x_val is not None and y_val is not None:
            self._x_val = x_val
            self._y_val = y_val

        N_train = len(x_train)
        # Run auto_arima on the entire training set's first column to fix model order.
        print("Running auto_arima on training data (first column) to determine model order...")
        auto_model = auto_arima(y_train[:, 0], exogenous=x_train, seasonal=False,
                                trace=True, error_action='ignore', suppress_warnings=True)
        self.order = auto_model.order
        self.seasonal_order = auto_model.seasonal_order  # e.g., (0,0,0,0)
        print(f"Auto_arima determined order: {self.order}, seasonal_order: {self.seasonal_order}")

        # Perform rolling forecast evaluation on training data.
        print("Starting rolling forecast evaluation on training data...")
        self.train_predictions = self._rolling_forecast(x_train, y_train)
        
        # Compute metrics only on the indices where predictions are valid (not NaN)
        valid_train = ~np.isnan(self.train_predictions).any(axis=1)
        if np.sum(valid_train) == 0:
            raise ValueError("No valid rolling forecasts obtained on training data.")
        train_mae = self.calculate_mae(y_train[valid_train], self.train_predictions[valid_train])
        train_r2 = r2_score(y_train[valid_train], self.train_predictions[valid_train])
        print(f"Training MAE: {train_mae}, Training R2: {train_r2}")

        # Perform rolling forecast evaluation on validation data, if provided.
        if self._x_val is not None and self._y_val is not None:
            print("Starting rolling forecast evaluation on validation data...")
            self.val_predictions = self._rolling_forecast(self._x_val, self._y_val)
            valid_val = ~np.isnan(self.val_predictions).any(axis=1)
            if np.sum(valid_val) == 0:
                raise ValueError("No valid rolling forecasts obtained on validation data.")
            val_mae = self.calculate_mae(self._y_val[valid_val], self.val_predictions[valid_val])
            val_r2 = r2_score(self._y_val[valid_val], self.val_predictions[valid_val])
        else:
            self.val_predictions = np.array([])
            val_mae = None
            val_r2 = None

        # Create a Keras-like history object with both 'loss' and 'val_loss'
        class MockHistory:
            def __init__(self):
                self.history = {'loss': [], 'val_loss': []}
        history = MockHistory()
        history.history['loss'].append(train_mae)
        history.history['val_loss'].append(val_mae)

        return history, train_mae, train_r2, val_mae, val_r2, self.train_predictions, self.val_predictions

    def predict(self, data):
        """
        Predict method.
        If the provided data exactly matches the stored training or validation exogenous data,
        the stored rolling forecast predictions are returned.
        Otherwise, a new rolling forecast is performed on the provided data.
        
        Returns:
            np.ndarray of shape (N, time_horizon)
        """
        if self._x_train is not None and np.array_equal(data, self._x_train):
            print("Returning stored training predictions.")
            return self.train_predictions
        elif self._x_val is not None and np.array_equal(data, self._x_val):
            print("Returning stored validation predictions.")
            return self.val_predictions
        else:
            print("Data does not match stored training/validation sets; running new rolling forecast.")
            # For new data, we cannot use the ground-truth target series; here we use zeros as placeholder.
            dummy_y = np.zeros((len(data), self.params['time_horizon']))
            return self._rolling_forecast(data, dummy_y)

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE (consistent with shape (N, time_horizon)).
        """
        print(f"Calculating MSE => y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch: y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        mse = np.mean((y_true.flatten() - y_pred.flatten())**2)
        print(f"Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Flatten-based MAE (consistent with shape (N, time_horizon)).
        """
        print(f"y_true sample: {y_true.flatten()[:5]}")
        print(f"y_pred sample: {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        """
        Save the model configuration and stored data.
        """
        save_data = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            '_x_train': self._x_train,
            '_y_train': self._y_train,
            '_x_val': self._x_val,
            '_y_val': self._y_val,
            'train_predictions': self.train_predictions,
            'val_predictions': self.val_predictions
        }
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        """
        Load a saved model configuration.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.order = data['order']
            self.seasonal_order = data['seasonal_order']
            self._x_train = data['_x_train']
            self._y_train = data['_y_train']
            self._x_val = data['_x_val']
            self._y_val = data['_y_val']
            self.train_predictions = data['train_predictions']
            self.val_predictions = data['val_predictions']
        print(f"Model loaded from {file_path}")
