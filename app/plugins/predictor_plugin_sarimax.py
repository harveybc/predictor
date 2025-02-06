import numpy as np
import logging
import os
import pickle

from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Plugin:
    """
    SARIMA Predictor Plugin using statsmodels for multi-step forecasting.

    This plugin builds, trains, and evaluates a SARIMAX model that outputs (N, time_horizon).
    It preserves exactly the same structure and interface (methods, parameters, return values)
    as the original ANN-based plugin example, but implements SARIMAX instead.
    """

    # Default parameters (identical to the ANN example for interface consistency)
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.001,
        'activation': 'tanh',
        'patience': 5,
        'l2_reg': 1e-3,

        # Typical SARIMA-related parameters stored here for convenience
        'order': (1, 1, 1),
        'seasonal_order': (0, 0, 0, 0),
        'time_horizon': 6  # For multi-step forecasting
    }

    # Variables for debugging (identical to the ANN example)
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        """
        Initialize the plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.model = None   # Will store the SARIMAX model specification
        self.results = None # Will store the fitted SARIMAX results object

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided kwargs.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return a dict of debug info from plugin params.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars if var in self.params}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Build (prepare) the SARIMAX model specification. 
        For interface consistency, we receive `input_shape` (int),
        though SARIMAX does not strictly need it like an ANN does.

        Args:
            input_shape (int): Number of input features for exogenous data (x_train).
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        
        # Store input_shape in params for debugging consistency
        self.params['input_dim'] = input_shape

        print(f"Preparing SARIMA model with order={self.params['order']} "
              f"and seasonal_order={self.params['seasonal_order']}.")
        print(f"Exogenous input shape: {input_shape}")

        # We only define the specification here (no direct instantiation with data).
        # Actual instantiation of SARIMAX will happen during train().
        self.model = None  # Just a placeholder to mimic the ANN approach

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the model with shape => x_train(N, input_dim), y_train(N, time_horizon).

        Although SARIMAX typically handles 1D endog, we keep the multi-step
        shape for interface compatibility. If time_horizon > 1, we will
        use y_train[:, 0] as the main series for fitting. Multi-step predictions
        will be handled in `predict()` by forecasting 'time_horizon' steps.

        Args:
            x_train (np.ndarray): Training exogenous data, shape (N, input_dim).
            y_train (np.ndarray): Training target data, shape (N, time_horizon).
            epochs (int): Number of epochs (not used in SARIMAX, but maintained for interface).
            batch_size (int): Batch size (not used directly in SARIMAX, kept for interface).
            threshold_error (float): Threshold for printing a warning about final_loss.
            x_val (np.ndarray, optional): Validation exogenous data, shape (M, input_dim).
            y_val (np.ndarray, optional): Validation target data, shape (M, time_horizon).

        Returns:
            tuple: (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")

        # Check the multi-step dimension
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(
                f"y_train shape {y_train.shape}, expected (N,{exp_horizon})."
            )

        # For simplicity, we fit only the first step (column) if horizon > 1
        endog_train = y_train[:, 0] if exp_horizon > 1 else y_train.ravel()
        exog_train = x_train if self.params['input_dim'] > 0 else None

        # Instantiate SARIMAX now with the training data
        self.model = SARIMAX(
            endog=endog_train,
            exog=exog_train,
            order=self.params['order'],
            seasonal_order=self.params['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Fit the model (equivalent to "training" in the ANN sense)
        print("Fitting SARIMAX model...")
        self.results = self.model.fit(disp=False)
        print("Training completed.")

        # Create a minimal history-like object to mimic Keras usage
        class MockHistory:
            def __init__(self):
                self.history = {'loss': []}

        history = MockHistory()

        # We define "final_loss" as a simple train error metric to mimic the ANN example
        # We'll calculate the MAE on the training set for the single-step approach
        train_predictions_1step = self.results.predict(
            start=0, 
            end=len(endog_train)-1, 
            exog=exog_train
        )
        final_loss = np.mean(np.abs(train_predictions_1step - endog_train))  # MAE
        history.history['loss'].append(final_loss)

        # Compare final_loss to threshold
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Evaluate on training data (multi-step style for consistency):
        # We'll produce an array (N, exp_horizon) for train predictions
        train_predictions = self.predict(x_train)
        train_mae = self.calculate_mae(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions) if exp_horizon == 1 \
            else r2_score(y_train[:, 0], train_predictions[:, 0])  # R² for first step if multi-step

        # Evaluate on validation data if provided
        if x_val is not None and y_val is not None:
            val_predictions = self.predict(x_val)
            val_mae = self.calculate_mae(y_val, val_predictions)
            if exp_horizon == 1:
                val_r2 = r2_score(y_val, val_predictions)
            else:
                # For multi-step, compare only the first step for R²
                val_r2 = r2_score(y_val[:, 0], val_predictions[:, 0])
        else:
            # If no validation data is provided, set placeholders
            val_predictions = np.array([])
            val_mae = None
            val_r2 = None

        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions

    def predict(self, data):
        """
        Produce multi-step forecasts of shape (N, time_horizon).
        For each row in `data` (exogenous input), we forecast time_horizon steps.

        Args:
            data (np.ndarray): Exogenous data, shape (N, input_dim).

        Returns:
            np.ndarray: Predictions array of shape (N, time_horizon).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # If there's no fitted model, raise an error
        if self.results is None:
            raise ValueError("Model is not trained. Call train() before predict().")

        # Prepare output array
        N = len(data)
        horizon = self.params['time_horizon']
        preds = np.zeros((N, horizon))

        # We do a simple rolling-like approach: for each row i in data, 
        # we forecast 'horizon' steps ahead. The model state is not updated
        # with each row's new information (this is a simplified approach).
        for i in range(N):
            # Single row as exogenous for the next horizon steps
            exog_i = np.tile(data[i], (horizon, 1)) if data.ndim == 2 else None
            # Forecast future horizon steps
            forecast_result = self.results.get_forecast(steps=horizon, exog=exog_i)
            forecast_mean = forecast_result.predicted_mean
            # Store the multi-step forecast in preds[i]
            preds[i, :] = forecast_mean

        return preds

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with multi-step shape (N, time_horizon).
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        mse = np.mean((y_true_f - y_pred_f) ** 2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Flatten-based MAE => consistent with multi-step shape (N, time_horizon).
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        """
        Save the trained SARIMAX model to file.
        """
        if self.results is None:
            raise ValueError("No trained model to save. Train the model first.")
        with open(file_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        """
        Load a trained SARIMAX model from file.
        """
        with open(file_path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Model loaded from {file_path}")
