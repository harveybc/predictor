import numpy as np
import logging
import os
import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

class Plugin:
    """
    Statistical Regressor Plugin (Single-pass) for multi-step forecasting.

    This plugin mimics the ML evaluation workflow:
      1. Train a single multi-output regressor (once) on (x_train, y_train).
      2. Predict on the entire training set to get training metrics.
      3. (Optionally) predict on the validation set to get validation metrics.

    It preserves the same interface (class name, methods, parameters, return values)
    as the previous SARIMAX rolling plugin, but uses a single-pass regression approach
    for a fair comparison with machine learning models that do not re-train at each step.
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
        'rolling_window': 48,   # Kept for interface consistency, not used here
        'time_horizon': 6       # Number of forecast steps per sample
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        """
        Initialize the plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        # We store references to x, y for predict()
        self._x_train = None
        self._y_train = None
        self._x_val = None
        self._y_val = None

        self.train_predictions = None
        self.val_predictions = None

        # The single-pass regression model
        self.model = None

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

    def build_model(self, input_shape, x_train=None, config=None):
        """
        Build the model placeholder.

        Args:
            input_shape (int): Number of exogenous features.
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int.")
        self.params['input_dim'] = input_shape
        print(f"Statistical Regressor building - input_shape: {input_shape}")
        print(f"Final output dimension (time_horizon): {self.params['time_horizon']}")

        # We'll use a random forest for multi-output regression
        # You can switch to another model if desired, e.g. LinearRegression, etc.
        self.model = RandomForestRegressor(n_estimators=500, random_state=42)

    def _rolling_forecast(self, x, y):
        """
        Single-pass forecast method (name is kept for interface consistency).

        Since we want to mimic ML evaluation (train-once, predict all),
        we simply run self.model.predict(x).

        Args:
            x: np.ndarray of shape (N, input_dim).
            y: np.ndarray of shape (N, time_horizon). Not directly used here
               except to confirm shape.

        Returns:
            preds: np.ndarray of shape (N, time_horizon).
        """
        horizon = self.params['time_horizon']
        N = len(x)

        if self.model is None:
            raise ValueError("Model not found. Did you forget to call train() or build_model()?")

        # Predict using the trained multi-output regressor
        #preds = self.model.predict(x)  # shape: (N, horizon)
        preds,uncertainties = self.predict_with_uncertainty(x,self.params.get('mc_samples', 20))  # shape: (N, horizon)

        # Ensure it has the expected shape
        if preds.shape != (N, horizon):
            raise ValueError(f"Expected predictions shape {(N, horizon)}, got {preds.shape}")

        return preds

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the plugin with a single-pass multi-output regression.

        Args:
            x_train: np.ndarray, shape (N, input_dim).
            y_train: np.ndarray, shape (N, time_horizon).
            epochs: int (kept for interface, unused here).
            batch_size: int (kept for interface, unused).
            threshold_error: float (kept for interface, unused).
            x_val: np.ndarray or None.
            y_val: np.ndarray or None.

        Returns:
            (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions).
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{horizon}).")

        # Store training and validation data for predict()
        self._x_train = x_train
        self._y_train = y_train
        if x_val is not None and y_val is not None:
            self._x_val = x_val
            self._y_val = y_val

        # Build or ensure the model is built
        if self.model is None:
            self.build_model(x_train.shape[1])

        print("Fitting the multi-output regressor on the entire training set...")
        self.model.fit(x_train, y_train)

        # Generate predictions on training set
        self.train_predictions = self._rolling_forecast(x_train, y_train)
        train_mae = self.calculate_mae(y_train, self.train_predictions)
        train_r2 = r2_score(y_train, self.train_predictions)
        print(f"Training MAE: {train_mae}, Training R2: {train_r2}")

        # Generate predictions on validation set (if provided)
        if x_val is not None and y_val is not None:
            self.val_predictions = self._rolling_forecast(x_val, y_val)
            val_mae = self.calculate_mae(y_val, self.val_predictions)
            val_r2 = r2_score(y_val, self.val_predictions)
        else:
            self.val_predictions = np.array([])
            val_mae = None
            val_r2 = None

        class MockHistory:
            def __init__(self):
                self.history = {'loss': [], 'val_loss': []}
        history = MockHistory()
        history.history['loss'].append(train_mae)
        history.history['val_loss'].append(val_mae)

        return history, self.train_predictions, self.val_predictions
    

    def predict_with_uncertainty(self, data, mc_samples=100):
        """
        Perform prediction and uncertainty estimation using predictions from individual trees
        of RandomForestRegressor.

        For each input sample, the function computes the mean prediction and the standard 
        deviation (as an uncertainty estimate) across a subset of trees.

        Args:
            data (np.ndarray): Input data for prediction.
            mc_samples (int): Number of trees to sample for uncertainty estimation. If greater
                              than the available trees, all trees are used.
        
        Returns:
            tuple: (mean_predictions, uncertainty_estimates) where both are np.ndarray with shape 
                   (n_samples, time_horizon)
        """
        if self.model is None or not hasattr(self.model, 'estimators_'):
            raise ValueError("Model not built or doesn't support uncertainty estimation.")

        estimators = self.model.estimators_
        n_estimators = len(estimators)
        # Use all trees if mc_samples exceeds the available number of trees
        if mc_samples > n_estimators:
            mc_samples = n_estimators
        
        # Randomly select a subset of trees from the forest
        selected_estimators = np.random.choice(estimators, size=mc_samples, replace=False)
        
        # Gather predictions from each selected tree
        preds = np.array([est.predict(data) for est in selected_estimators])  # shape: (mc_samples, n_samples, time_horizon)
        
        # Calculate mean and std deviation across the trees
        mean_predictions = np.mean(preds, axis=0)          # shape: (n_samples, time_horizon)
        uncertainty_estimates = np.std(preds, axis=0)        # shape: (n_samples, time_horizon)
                
        return mean_predictions, uncertainty_estimates


    def predict(self, data):
        """
        Predict method.

        If the provided data exactly matches stored training or validation exogenous data,
        the stored predictions are returned. Otherwise, a new single-pass inference is performed.

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
            print("Data does not match stored training/validation sets; performing single-pass inference.")
            dummy_y = np.zeros((len(data), self.params['time_horizon']))
            return self._rolling_forecast(data, dummy_y)

    def calculate_mse(self, y_true, y_pred):
        """
        Compute the flatten-based MSE, consistent with shape (N, time_horizon).
        """
        print(f"Calculating MSE => y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch: y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        mse = np.mean((y_true.flatten() - y_pred.flatten())**2)
        print(f"Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        # Convert to numpy array if inputs are DataFrames
        if isinstance(y_true, pd.DataFrame):
            # Use to_numpy() if available, otherwise use .values
            y_true = y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true.values
        else:
            # Ensure conversion in case y_true is not an ndarray already
            y_true = np.array(y_true)
        
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.to_numpy() if hasattr(y_pred, "to_numpy") else y_pred.values
        else:
            y_pred = np.array(y_pred)

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
            '_x_train': self._x_train,
            '_y_train': self._y_train,
            '_x_val': self._x_val,
            '_y_val': self._y_val,
            'train_predictions': self.train_predictions,
            'val_predictions': self.val_predictions,
            'model': self.model
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
            self._x_train = data['_x_train']
            self._y_train = data['_y_train']
            self._x_val = data['_x_val']
            self._y_val = data['_y_val']
            self.train_predictions = data['train_predictions']
            self.val_predictions = data['val_predictions']
            self.model = data['model']
        print(f"Model loaded from {file_path}")
