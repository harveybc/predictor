import numpy as np
import logging
import os
import pickle
import pandas as pd

from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Plugin:
    """
    SARIMAX Predictor Plugin for multi-step forecasting.
    
    This plugin builds, trains, and evaluates a SARIMAX model that outputs (N, time_horizon),
    preserving the exact interface of the provided ANN plugin.
    """

    # We keep the same default parameters for interface consistency.
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
        Initialize the plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.model = None        # Will store the SARIMAX specification
        self.results = None      # Will store the fitted SARIMAX results

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided kwargs.
        (Kept identical in signature to the ANN version.)
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return a dict of debug info from plugin params.
        (Identical to ANN.)
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        (Identical to ANN.)
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Build the 'model'. For SARIMAX, we just store input_shape and
        prepare placeholders. We must print shapes as per the ANN style.

        Args:
            input_shape (int): Number of input features (exogenous dimension).
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        
        # We mimic storing input_shape in params and print debug info like the ANN.
        self.params['input_dim'] = input_shape
        time_horizon = self.params['time_horizon'] if 'time_horizon' in self.params else 1

        print(f"SARIMAX building - input_shape: {input_shape}")
        print(f"SARIMAX final output dimension (time_horizon): {time_horizon}")

        # We won't fully instantiate SARIMAX here because we need actual data in train().
        self.model = None
        self.results = None

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the SARIMAX model.

        We keep the same signature and logic style:
        - Print shapes
        - Fit the model
        - Return (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)

        Args:
            x_train (np.ndarray): shape (N, input_dim)
            y_train (np.ndarray): shape (N, time_horizon)
            epochs (int): not directly used by SARIMAX, but kept for interface
            batch_size (int): not used directly, but kept for interface
            threshold_error (float): used for a final warning check
            x_val (np.ndarray, optional): shape (M, input_dim)
            y_val (np.ndarray, optional): shape (M, time_horizon)
        
        Returns:
            A tuple: (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params.get('time_horizon', 1)
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")
        
        # SARIMAX typically fits a single column endog, so we use y_train[:,0].
        # We'll handle multi-step in predict().
        endog_train = y_train[:, 0] if exp_horizon > 1 else y_train.ravel()
        exog_train = x_train if self.params['input_dim'] > 0 else None

        # Build and fit a simple SARIMAX(0,1,0) or let user specify more sophisticated order if desired.
        # For demonstration, we'll just do a minimal example that differencing might handle a trend.
        # You could add param logic here if you want to pass in (p,d,q).
        print("Fitting a SARIMAX(0,1,0) model for demonstration. Modify if needed.")
        self.model = SARIMAX(
            endog=endog_train,
            exog=exog_train,
            order=(0, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        print("Fitting SARIMAX model ...")
        self.results = self.model.fit(disp=False)
        print(self.results.summary())
        print("Training completed.")
        
        # Create a minimal 'history' object to mimic the ANN's Keras history
        class MockHistory:
            def __init__(self):
                self.history = {'loss': []}  # We'll store a 'loss' for consistency

        history = MockHistory()

        # We'll define final_loss as the MAE on the training set for 1-step predictions
        train_pred_1step = self.results.predict(start=0, end=len(endog_train)-1, exog=exog_train)
        final_loss = np.mean(np.abs(train_pred_1step - endog_train))
        history.history['loss'].append(final_loss)
        print(f"Final training loss (MAE on 1-step): {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Evaluate multi-step predictions on training data
        train_predictions = self.predict(x_train)
        print(f"train_predictions shape: {train_predictions.shape}")
        print(f"y_train shape: {y_train.shape}")

        train_mae = self.calculate_mae(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        # Evaluate on validation if provided
        if x_val is not None and y_val is not None:
            val_predictions = self.predict(x_val)
            print(f"val_predictions shape: {val_predictions.shape}")
            print(f"y_val shape: {y_val.shape}")

            val_mae = self.calculate_mae(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
        else:
            val_predictions = np.array([])
            val_mae = None
            val_r2 = None

        # Return the same structure
        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions

    import numpy as np
import logging
import os
import pickle
import pandas as pd

from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Plugin:
    """
    SARIMAX Predictor Plugin for multi-step forecasting.
    
    This plugin builds, trains, and evaluates a SARIMAX model that outputs (N, time_horizon),
    preserving the exact interface of the provided ANN plugin.
    """

    # We keep the same default parameters for interface consistency.
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
        Initialize the plugin with default parameters.
        """
        self.params = self.plugin_params.copy()
        self.model = None        # Will store the SARIMAX specification
        self.results = None      # Will store the fitted SARIMAX results

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided kwargs.
        (Kept identical in signature to the ANN version.)
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return a dict of debug info from plugin params.
        (Identical to ANN.)
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        (Identical to ANN.)
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Build the 'model'. For SARIMAX, we just store input_shape and
        prepare placeholders. We must print shapes as per the ANN style.

        Args:
            input_shape (int): Number of input features (exogenous dimension).
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for SARIMAX.")
        
        # We mimic storing input_shape in params and print debug info like the ANN.
        self.params['input_dim'] = input_shape
        time_horizon = self.params['time_horizon'] if 'time_horizon' in self.params else 1

        print(f"SARIMAX building - input_shape: {input_shape}")
        print(f"SARIMAX final output dimension (time_horizon): {time_horizon}")

        # We won't fully instantiate SARIMAX here because we need actual data in train().
        self.model = None
        self.results = None

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the SARIMAX model.

        We keep the same signature and logic style:
        - Print shapes
        - Fit the model
        - Return (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)

        Args:
            x_train (np.ndarray): shape (N, input_dim)
            y_train (np.ndarray): shape (N, time_horizon)
            epochs (int): not directly used by SARIMAX, but kept for interface
            batch_size (int): not used directly, but kept for interface
            threshold_error (float): used for a final warning check
            x_val (np.ndarray, optional): shape (M, input_dim)
            y_val (np.ndarray, optional): shape (M, time_horizon)
        
        Returns:
            A tuple: (history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions)
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params.get('time_horizon', 1)
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")
        
        # SARIMAX typically fits a single column endog, so we use y_train[:,0].
        # We'll handle multi-step in predict().
        endog_train = y_train[:, 0] if exp_horizon > 1 else y_train.ravel()
        exog_train = x_train if self.params['input_dim'] > 0 else None

        # Build and fit a simple SARIMAX(0,1,0) or let user specify more sophisticated order if desired.
        # For demonstration, we'll just do a minimal example that differencing might handle a trend.
        # You could add param logic here if you want to pass in (p,d,q).
        print("Fitting a SARIMAX(0,1,0) model for demonstration. Modify if needed.")
        self.model = SARIMAX(
            endog=endog_train,
            exog=exog_train,
            order=(0, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        print("Fitting SARIMAX model ...")
        self.results = self.model.fit(disp=False)
        print(self.results.summary())
        print("Training completed.")
        
        # Create a minimal 'history' object to mimic the ANN's Keras history
        class MockHistory:
            def __init__(self):
                self.history = {'loss': []}  # We'll store a 'loss' for consistency

        history = MockHistory()

        # We'll define final_loss as the MAE on the training set for 1-step predictions
        train_pred_1step = self.results.predict(start=0, end=len(endog_train)-1, exog=exog_train)
        final_loss = np.mean(np.abs(train_pred_1step - endog_train))
        history.history['loss'].append(final_loss)
        print(f"Final training loss (MAE on 1-step): {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Evaluate multi-step predictions on training data
        train_predictions = self.predict(x_train)
        print(f"train_predictions shape: {train_predictions.shape}")
        print(f"y_train shape: {y_train.shape}")

        train_mae = self.calculate_mae(y_train, train_predictions)
        train_r2 = r2_score(y_train, train_predictions)

        # Evaluate on validation if provided
        if x_val is not None and y_val is not None:
            val_predictions = self.predict(x_val)
            print(f"val_predictions shape: {val_predictions.shape}")
            print(f"y_val shape: {y_val.shape}")

            val_mae = self.calculate_mae(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
        else:
            val_predictions = np.array([])
            val_mae = None
            val_r2 = None

        # Return the same structure
        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions

    def predict(self, data):
        """
        Predict method: 
        Produce multi-step forecasts (N, time_horizon) from SARIMAX.
        We'll do an iterative approach for multi-step (time_horizon>1).

        We must ensure the output shape is (N, time_horizon), same as the ANN.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.results is None:
            raise ValueError("Model is not trained. Call train() before predict().")

        N = len(data)
        horizon = self.params.get('time_horizon', 1)
        preds = np.zeros((N, horizon))
        
        # If input_dim > 0, we treat 'data' as exog. Otherwise, no exogenous used.
        exog_data = data if self.params['input_dim'] > 0 else None

        # Iterative approach: for each row i, forecast horizon steps 1 by 1.
        # This ensures (N, horizon) output.
        for i in range(N):
            # Make a copy of the fitted results to do iterative appends
            current_results = self.results
            row_preds = []
            for step in range(horizon):
                # Single row as exog
                exog_next = exog_data[i].reshape(1, -1) if exog_data is not None else None
                forecast_res = current_results.get_forecast(steps=1, exog=exog_next)
                pred_1 = forecast_res.predicted_mean[0]
                row_preds.append(pred_1)

                # Update the model's state with the predicted value
                current_results = current_results.append(
                    endog=[pred_1],
                    exog=exog_next
                )

            preds[i, :] = row_preds

        print(f"SARIMAX predict => input data shape: {data.shape}, output preds shape: {preds.shape}")
        return preds

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with multi-step shape (N, time_horizon).
        Matches the ANN signature/behavior.
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        mse = np.mean((y_true_f - y_pred_f)**2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Flatten-based MAE => consistent with multi-step shape (N, time_horizon).
        Matches the ANN signature/behavior.
        """
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        """
        Save the trained model to file.
        We mimic the ANN approach, but we use pickle for SARIMAX results.
        """
        if self.results is None:
            raise ValueError("No trained model to save. Train the model first.")
        with open(file_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        """
        Load a trained model from file, matching ANN signature.
        """
        with open(file_path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Model loaded from {file_path}")


    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with multi-step shape (N, time_horizon).
        Matches the ANN signature/behavior.
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        mse = np.mean((y_true_f - y_pred_f)**2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Flatten-based MAE => consistent with multi-step shape (N, time_horizon).
        Matches the ANN signature/behavior.
        """
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        """
        Save the trained model to file.
        We mimic the ANN approach, but we use pickle for SARIMAX results.
        """
        if self.results is None:
            raise ValueError("No trained model to save. Train the model first.")
        with open(file_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        """
        Load a trained model from file, matching ANN signature.
        """
        with open(file_path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Model loaded from {file_path}")
