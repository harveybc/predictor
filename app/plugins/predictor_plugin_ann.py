import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianNoise
from keras import backend as K
from sklearn.metrics import r2_score 
import logging
import os
import gc
import tensorflow.keras.backend as K

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
#Flatten
from tensorflow.keras.layers import Flatten
#Add
from tensorflow.keras.layers import Add
#tfp_layers
#tensor flow probability densrflipout
tfp_layers = tfp.layers


class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """
    Custom ReduceLROnPlateau callback that prints the patience counter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0  # Track the patience counter

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: ReduceLROnPlateau patience counter: {self.patience_counter}")


class EarlyStoppingWithPatienceCounter(EarlyStopping):
    """
    Custom EarlyStopping callback that prints the patience counter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0  # Track the patience counter

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: EarlyStopping patience counter: {self.patience_counter}")


class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()


# --- REMOVE THIS MONKEY-PATCH ---
# def _patched_add_variable(self, name, shape, dtype, initializer, trainable, **kwargs):
#     return self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
# tfp.layers.DenseFlipout.add_variable = _patched_add_variable


# --- Named initializers to avoid lambda serialization warnings ---
def random_normal_initializer_42(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=42)

def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)


class Plugin:
    """
    ANN Predictor Plugin using Keras for multi-step forecasting.
    
    This plugin builds, trains, and evaluates an ANN that outputs (N, time_horizon).
    """

    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3
    }
    
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']
    
    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)



    tfp_layers = tfp.layers

    # Construcción explícita del modelo con depuración detallada
    def build_model(self, input_shape, x_train, config):
        """
        Builds a simplified N-BEATS model.
        
        Args:
            input_shape (tuple): Expected shape (window_size, 1).
            x_train (np.ndarray): Training data (for shape inference if needed).
            config (dict): Configuration parameters which may include:
                - "nbeats_num_blocks": number of blocks (default=3)
                - "nbeats_units": number of neurons per dense layer (default=64)
                - "nbeats_layers": number of layers per block (default=3)
        
        The model flattens the input and passes it through several blocks.
        Each block outputs a forecast which is summed to produce the final prediction.
        """
        window_size = input_shape[0]
        num_blocks = config.get("nbeats_num_blocks", 3)
        block_units = config.get("nbeats_units", 64)
        block_layers = config.get("nbeats_layers", 3)
        
        # Input layer accepts shape (window_size, 1)
        inputs = Input(shape=input_shape, name='input_layer')
        x = Flatten(name='flatten_layer')(inputs)  # shape: (window_size,)
        
        # Initialize residual as the flattened input.
        residual = x
        forecasts = []
        
        # Define a helper function to build a single block.
        def nbeats_block(res, block_id):
            r = res
            for i in range(block_layers):
                r = Dense(block_units, activation='relu', name=f'block{block_id}_dense_{i+1}')(r)
            # Forecast branch outputs a single value.
            forecast = Dense(1, activation='linear', name=f'block{block_id}_forecast')(r)
            # Backcast branch estimates the part of the input explained by this block.
            backcast = Dense(int(res.shape[-1]), activation='linear', name=f'block{block_id}_backcast')(r)
            # Update residual: subtract the backcast.
            updated_res = Add(name=f'block{block_id}_residual')([res, -backcast])
            return updated_res, forecast
        
        # Build blocks sequentially.
        for b in range(1, num_blocks + 1):
            residual, forecast = nbeats_block(residual, b)
            forecasts.append(forecast)
        
        # Sum forecasts from all blocks.
        if len(forecasts) > 1:
            final_forecast = Add(name='forecast_sum')(forecasts)
        else:
            final_forecast = forecasts[0]
        
        # Define and compile the model.
        self.model = Model(inputs=inputs, outputs=final_forecast, name='NBeatsModel')
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("learning_rate", 0.001))
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print("N-BEATS model built successfully.")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the N-BEATS model.
        
        Args:
            x_train (np.ndarray): Training input with shape (samples, window_size, 1).
            y_train (list): List containing a single array of targets (shape (samples,)).
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size.
            threshold_error (float): Not used here (for compatibility).
            x_val (np.ndarray): Validation input.
            y_val (list): List containing a single array of validation targets.
            config (dict): Additional configuration if needed.
        
        Returns:
            history: Training history.
            train_preds: Predictions on training data.
            train_unc: Uncertainty estimates (zeros array).
            val_preds: Predictions on validation data.
            val_unc: Uncertainty estimates (zeros array).
        """
        history = self.model.fit(x_train, y_train[0],
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val[0]),
                                 verbose=1)
        # Obtain predictions.
        train_preds = self.model.predict(x_train, batch_size=batch_size)
        val_preds = self.model.predict(x_val, batch_size=batch_size)
        # Set uncertainties to zero (as per instruction).
        train_unc = np.zeros_like(train_preds)
        val_unc = np.zeros_like(val_preds)
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Generates predictions with uncertainty estimates.
        For N-BEATS, we simply return the predictions and zeros as uncertainty.
        
        Args:
            x_test (np.ndarray): Test input data.
            mc_samples (int): Number of Monte Carlo samples (unused here).
        
        Returns:
            tuple: (predictions, uncertainty_estimates)
        """
        predictions = self.model.predict(x_test)
        uncertainty_estimates = np.zeros_like(predictions)
        return predictions, uncertainty_estimates

    def save(self, file_path):
        """
        Saves the current model to the specified file.
        
        Args:
            file_path (str): Path to save the model.
        """
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}")


