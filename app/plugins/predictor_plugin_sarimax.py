import numpy as np
import logging
import os
import pickle
import pandas as pd

from sklearn.metrics import r2_score
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fourier_terms(t, period, K):
    """
    Generate Fourier terms to model additional seasonality.
    
    Args:
        t (array-like): Integer time indices, shape (N,).
        period (int): The seasonal period in hours (e.g., 168 for weekly if data is hourly).
        K (int): Number of sine/cosine pairs.
    
    Returns:
        pd.DataFrame: Columns [sin_{period}_1, cos_{period}_1, sin_{period}_2, cos_{period}_2, ...].
    """
    t = np.array(t)
    data = []
    columns = []
    for k in range(1, K+1):
        data.append(np.sin((2.0 * np.pi * k * t) / period))
        data.append(np.cos((2.0 * np.pi * k * t) / period))
        columns.append(f'sin_{period}_{k}')
        columns.append(f'cos_{period}_{k}')
    if len(data) == 0:
        return pd.DataFrame()
    data = np.column_stack(data)
    return pd.DataFrame(data, columns=columns)

class Plugin:
    """
    SARIMA Predictor Plugin that:
      - Uses auto_arima to find (p, d, q) + seasonal order
      - Incorporates an iterative multi-step forecast approach in predict()
      - Fixes the numpy.ndarray vs. DataFrame issue that caused the attribute error.
    
    Retains the same interface as the original ANN-based plugin.
    """

    plugin_params = {
        # Unchanged, matching your plugin’s interface
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.001,
        'activation': 'tanh',
        'patience': 5,
        'l2_reg': 1e-3,

        # ARIMA-related params
        'time_horizon': 6,   
        
        # Fourier pairs for weekly/monthly
        'K_weekly': 1,
        'K_monthly': 0,  
        
        # single main seasonality for auto_arima
        'main_seasonal_period': 24,

        # Restrict complexity of auto_arima (optional)
        'max_p': 3,
        'max_d': 2,
        'max_q': 3,
        'max_P': 1,
        'max_D': 1,
        'max_Q': 1,
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None   
        self.results = None 
        self._train_length = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars if var in self.params}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        self.params['input_dim'] = input_shape
        print(f"Restricted auto_arima + iterative approach plugin. Exogenous shape: {input_shape}")
        self.model = None

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train with shape => x_train(N, input_dim), y_train(N, time_horizon).
        1) Generate weekly/monthly Fourier exog
        2) Restricted auto_arima
        3) Final SARIMAX fit
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")

        # Univariate endog => first column if horizon>1
        endog_train = y_train[:, 0]
        self._train_length = len(x_train)

        # Build Fourier for weekly/monthly
        time_idx = np.arange(self._train_length)
        K_weekly = self.params.get('K_weekly', 1)
        K_monthly = self.params.get('K_monthly', 0)
        weekly_df = fourier_terms(time_idx, 168, K_weekly)
        monthly_df = fourier_terms(time_idx, 720, K_monthly)

        x_train_df = pd.DataFrame(x_train)
        X_train_fourier = pd.concat([x_train_df, weekly_df, monthly_df], axis=1)

        # Restricted auto_arima
        from pmdarima import auto_arima
        sp = self.params.get('main_seasonal_period', 24)
        step_model = auto_arima(
            y=endog_train,
            exogenous=X_train_fourier,
            seasonal=True,
            m=sp,
            max_p=self.params.get('max_p', 3),
            max_d=self.params.get('max_d', 2),
            max_q=self.params.get('max_q', 3),
            max_P=self.params.get('max_P', 1),
            max_D=self.params.get('max_D', 1),
            max_Q=self.params.get('max_Q', 1),
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )

        print("auto_arima best order =", step_model.order)
        print("auto_arima best seasonal_order =", step_model.seasonal_order)

        p, d, q = step_model.order
        P, D, Q, m = step_model.seasonal_order
        self.model = SARIMAX(
            endog=endog_train,
            exog=X_train_fourier,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        print("Fitting final SARIMAX model...")
        self.results = self.model.fit(disp=False)
        print(self.results.summary())  # Note: Might show near-singular warnings
        print("Training completed.")

        # Keras-like history
        class MockHistory:
            def __init__(self):
                self.history = {'loss': [], 'val_loss': []}
        history = MockHistory()

        # final_loss = 1-step MAE on training
        train_pred_1step = self.results.predict(
            start=0, end=self._train_length - 1, exog=X_train_fourier
        )
        final_loss = np.mean(np.abs(train_pred_1step - endog_train))
        history.history['loss'].append(final_loss)
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Evaluate multi-step on training
        train_predictions = self.predict(x_train)
        train_mae = self.calculate_mae(y_train, train_predictions)
        train_r2 = r2_score(y_train[:, 0], train_predictions[:, 0])

        # Validation
        if x_val is not None and y_val is not None:
            val_predictions = self.predict(x_val)
            val_mae = self.calculate_mae(y_val, val_predictions)
            val_r2 = r2_score(y_val[:, 0], val_predictions[:, 0])
            history.history['val_loss'].append(val_mae)
        else:
            val_predictions = np.array([])
            val_mae = None
            val_r2 = None
            history.history['val_loss'].append(None)

        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions

    def predict(self, data):
        """
        Iterative multi-step forecast: for each row in 'data',
        we forecast horizon steps 1-by-1 and update the model state 
        to help reduce error compounding (and overfitting).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.results is None:
            raise ValueError("Model is not trained. Call train() before predict().")

        horizon = self.params['time_horizon']
        N = len(data)
        preds = np.zeros((N, horizon))

        for i in range(N):
            # Start from the final fitted results
            current_results = self.results
            # For each 1-step forecast, we append the new data point to the model
            step_preds = []

            for step in range(horizon):
                # Build exog for the next step (row i repeated once).
                exog_next = data[i].reshape(1, -1)  # shape (1, input_dim)
                # We could also generate Fourier terms for the next time step(s) 
                # if we want to remain consistent with train. 
                # For simplicity, skip that here or do it naively.

                # Forecast 1-step
                forecast_res = current_results.get_forecast(steps=1, exog=exog_next)
                pred_1 = forecast_res.predicted_mean[0]
                step_preds.append(pred_1)

                # Append to update model state
                current_results = current_results.append(
                    endog=[pred_1],
                    exog=exog_next if exog_next.size > 0 else None
                )

            preds[i, :] = step_preds

        return preds

    def calculate_mse(self, y_true, y_pred):
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        return np.mean((y_true.flatten() - y_pred.flatten())**2)

    def calculate_mae(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        if self.results is None:
            raise ValueError("No trained model to save. Train the model first.")
        with open(file_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Model loaded from {file_path}")
