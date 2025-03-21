#!/usr/bin/env python
"""
Enhanced N-BEATS Predictor Plugin (Forecasting Seasonal Component without Bayesian Output)

This module implements an enhanced version of an N-BEATS forecasting plugin without using a Bayesian output layer.
It is designed to predict the seasonal component (or its returns) extracted via STL decomposition.
The target data is assumed to be a 2‑column array where:
  - Column 0 is the target seasonal magnitude (normalized).
  - Column 1 is the target phase (extracted via a Hilbert transform).

The model architecture uses an N‑BEATS–style backbone and then splits into two branches:
  - A deterministic branch (using a standard Dense layer) for predicting the magnitude.
  - A deterministic branch for predicting the phase.
Their outputs are concatenated to form a final 2‑D output.

The composite loss function combines:
  - Huber loss on the magnitude.
  - MMD loss on the magnitude (weighted by mmd_lambda).
  - Phase loss computed as 1 - cos(predicted_phase - true_phase) (weighted by lambda_phase).

Custom metrics (MAE and R²) are computed on the magnitude only.
Additional callbacks print key training statistics (learning rate, patience counters, etc.).

Note: The input x (and target y) are assumed to be the seasonal component (or its returns) extracted from the close prices,
shifted by a given forecast horizon.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # still imported if needed for loss functions, etc.
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
from tensorflow.keras.losses import Huber
import gc
import os
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score

# ---------------------------
# Custom Callbacks (copied exactly from transformer plugin)
# ---------------------------
class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """
    Custom ReduceLROnPlateau callback that prints the patience counter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

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
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: EarlyStopping patience counter: {self.patience_counter}")

class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()

# ---------------------------
# Custom Metrics and Loss Functions
# ---------------------------
def mae_magnitude(y_true, y_pred):
    """
    Computes Mean Absolute Error on the magnitude (first column) only.
    """
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """
    Custom R² metric computed on the magnitude component only.
    """
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    SS_res = tf.reduce_sum(tf.square(mag_true - mag_pred))
    SS_tot = tf.reduce_sum(tf.square(mag_true - tf.reduce_mean(mag_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

def compute_mmd(x, y, sigma=1.0, sample_size=256):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples.
    """
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:sample_size]
    x_sample = tf.gather(x, idx)
    y_sample = tf.gather(y, idx)
    def gaussian_kernel(x, y, sigma):
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 0)
        dist = tf.reduce_sum(tf.square(x - y), axis=-1)
        return tf.exp(-dist / (2.0 * sigma ** 2))
    K_xx = gaussian_kernel(x_sample, x_sample, sigma)
    K_yy = gaussian_kernel(y_sample, y_sample, sigma)
    K_xy = gaussian_kernel(x_sample, y_sample, sigma)
    return tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2 * tf.reduce_mean(K_xy)

def composite_loss(y_true, y_pred, mmd_lambda, lambda_phase, sigma=1.0):
    """
    Composite loss that combines:
      - Huber loss on the magnitude (first column) between prediction and target.
      - MMD loss on the magnitude (first column).
      - Phase loss computed as 1 - cos(predicted_phase - true_phase).
    
    Args:
        y_true: Tensor of shape (batch_size, 2). Column 0 is target magnitude,
                column 1 is target phase.
        y_pred: Tensor of shape (batch_size, 2). Column 0 is predicted magnitude,
                column 1 is predicted phase.
        mmd_lambda: Weight for the MMD loss.
        lambda_phase: Weight for the phase loss.
        sigma: Parameter for the MMD loss.
    
    Returns:
        Total loss value.
    """
    mag_true = y_true[:, 0:1]
    phase_true = y_true[:, 1:2]
    mag_pred = y_pred[:, 0:1]
    phase_pred = y_pred[:, 1:2]
    
    huber_loss_val = Huber()(mag_true, mag_pred)
    mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
    phase_loss = 1 - tf.cos(phase_pred - phase_true)
    total_loss = huber_loss_val + (mmd_lambda * mmd_loss_val) + (lambda_phase * phase_loss)
    return total_loss

# ---------------------------
# Enhanced N-BEATS Plugin Definition (Deterministic Version Without Bayesian Output)
# ---------------------------
class Plugin:
    """
    Enhanced N-BEATS Predictor Plugin using Keras for forecasting the seasonal component
    (or its returns) extracted via STL decomposition.
    
    The model is designed to learn both the magnitude and the phase of the seasonal pattern.
    It outputs a 2-dimensional vector per sample:
      - Column 0: Predicted magnitude (e.g. normalized seasonal value).
      - Column 1: Predicted phase.
    
    The composite loss function includes:
      - Huber loss on the magnitude.
      - MMD loss on the magnitude (weighted by mmd_lambda).
      - Phase loss computed as 1 - cos(predicted_phase - true_phase) (weighted by lambda_phase).
    
    Custom metrics (MAE and R²) are computed on the magnitude only.
    
    Additional callbacks print training statistics (learning rate, patience counters, etc.).
    
    It is assumed that the input x (and corresponding target y) are the seasonal component
    (or its returns) extracted from the close prices, shifted by a given forecast horizon.
    """
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3,        # Not used in this deterministic version.
        'mmd_lambda': 1e-3,       # Weight for MMD loss.
        'lambda_phase': 1e-3,     # Weight for phase loss.
        # N-BEATS parameters (the model always produces one forecast per window,
        # regardless of how many steps ahead the target is shifted).
        'nbeats_num_blocks': 3,
        'nbeats_units': 64,
        'nbeats_layers': 3,
        'time_horizon': 1  
    }
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size', 'time_horizon']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape, x_train, config):
        """
        Builds an enhanced N-BEATS model for forecasting the seasonal component.
        
        Args:
            input_shape (tuple): Expected shape (window_size, 1).
            x_train (np.ndarray): Training data (for shape inference if needed).
            config (dict): Configuration parameters.
        
        The model:
          - Flattens the input.
          - Passes data through several blocks; each block produces a forecast.
          - Aggregates forecasts by summing them.
          - Splits into two branches:
              • A deterministic branch for predicting the magnitude.
              • A deterministic branch for predicting the phase.
          - Concatenates the two outputs to yield a final 2D output.
        
        Note: The target y is assumed to be a 2-column array where column 0 is the
              target seasonal magnitude (normalized) and column 1 is the target phase.
        """
        window_size = input_shape[0]
        num_blocks = config.get("nbeats_num_blocks", 3)
        block_units = config.get("nbeats_units", 64)
        block_layers = config.get("nbeats_layers", 3)
        
        # Input layer: shape (window_size, 1)
        inputs = Input(shape=input_shape, name='input_layer')
        x = Flatten(name='flatten_layer')(inputs)  # shape: (window_size,)
        
        # Initialize residual as the flattened input.
        residual = x
        forecasts = []
        
        def nbeats_block(res, block_id):
            """
            Constructs a single N-BEATS block.
            
            Args:
                res: Input tensor from the previous block.
                block_id (int): Identifier for naming.
            
            Returns:
                tuple: (updated_residual, forecast)
            """
            r = res
            for i in range(block_layers):
                r = Dense(block_units, activation='relu', name=f'block{block_id}_dense_{i+1}')(r)
            # Forecast branch outputs a single value.
            forecast = Dense(1, activation='linear', name=f'block{block_id}_forecast')(r)
            # Use static shape for units.
            units = int(res.shape[-1])
            backcast = Dense(units, activation='linear', name=f'block{block_id}_backcast')(r)
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
        
        # Branch 1: Deterministic output for magnitude.
        deterministic_mag_output = Dense(1, activation='linear', name='deterministic_mag_output')(final_forecast)
        
        # Branch 2: Deterministic output for phase.
        phase_output = Dense(1, activation='linear', name='phase_output')(final_forecast)
        
        # Concatenate outputs to form final output (shape: (batch_size, 2)).
        final_output = Concatenate(name='final_output')([deterministic_mag_output, phase_output])
        
        self.model = Model(inputs=inputs, outputs=final_output, name='NBeatsModel')
        
        optimizer = Adam(learning_rate=config.get("learning_rate", 0.0001))
        mmd_lambda = config.get("mmd_lambda", 1e-3)
        lambda_phase = config.get("lambda_phase", 1e-3)
        # Compile with composite loss and custom metrics (computed on magnitude only).
        self.model.compile(optimizer=optimizer,
                           loss=lambda y_true, y_pred: composite_loss(y_true, y_pred, mmd_lambda, lambda_phase, sigma=1.0),
                           metrics=[mae_magnitude, r2_metric])
        print("DEBUG: MMD lambda =", mmd_lambda, "and lambda_phase =", lambda_phase)
        print("N-BEATS model built successfully.")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the N-BEATS model with enhanced callbacks.
        
        Prints training statistics for:
          - MAE on the magnitude.
          - Learning rate (at each epoch).
          - EarlyStopping and ReduceLROnPlateau patience counters.
        
        Args:
            x_train (np.ndarray): Training input with shape (samples, window_size, 1).
            y_train (list): List containing a single array of targets (shape (samples, 2)),
                             where column 0 is the seasonal magnitude target and column 1 is the phase target.
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size.
            threshold_error (float): (Unused, for compatibility).
            x_val (np.ndarray): Validation input.
            y_val (list): List containing a single array of validation targets.
            config (dict): Additional configuration (including early stopping parameters).
        
        Returns:
            tuple: (history, train_preds, train_unc, val_preds, val_unc)
        """
        # Define callbacks.
        callbacks = [
            EarlyStoppingWithPatienceCounter(monitor='val_loss',
                                             patience=config.get("early_stopping_patience", 10),
                                             verbose=1),
            ReduceLROnPlateauWithCounter(monitor='val_loss',
                                         factor=0.5,
                                         patience=config.get("reduce_lr_patience", 3),
                                         verbose=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: 
                           print(f"DEBUG: Learning Rate at epoch {epoch+1}: {K.get_value(self.model.optimizer.lr)}")),
            ClearMemoryCallback()
        ]
        
        print(f"DEBUG: Starting training for {epochs} epochs with batch size {batch_size}")
        history = self.model.fit(x_train, y_train[0],
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val[0]),
                                 callbacks=callbacks,
                                 verbose=1)
        train_preds = self.model.predict(x_train, batch_size=batch_size)
        val_preds = self.model.predict(x_val, batch_size=batch_size)
        # For compatibility, set uncertainty estimates (for magnitude) as zeros.
        train_unc = np.zeros_like(train_preds[:, 0:1])
        val_unc = np.zeros_like(val_preds[:, 0:1])
        # Calculate and print MAE and R² on training data (using magnitude only).
        self.calculate_mae(y_train[0], train_preds)
        self.calculate_r2(y_train[0], train_preds)
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Generates predictions with uncertainty estimates.
        For this model, predictions are returned with zero uncertainty.
        
        Args:
            x_test (np.ndarray): Test input data.
            mc_samples (int): Number of Monte Carlo samples (unused).
        
        Returns:
            tuple: (predictions, uncertainty_estimates)
        """
        predictions = self.model.predict(x_test)
        uncertainty_estimates = np.zeros_like(predictions[:, 0:1])
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
        """
        Loads a model from the specified file.
        
        Args:
            file_path (str): Path from which to load the model.
        """
        self.model = load_model(file_path, custom_objects={'composite_loss': composite_loss, 
                                                             'compute_mmd': compute_mmd, 
                                                             'r2_metric': r2_metric,
                                                             'mae_magnitude': mae_magnitude})
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        """
        Calculates and prints the Mean Absolute Error for the magnitude component.
        """
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"DEBUG: y_true (magnitude sample): {mag_true.flatten()[:5]}")
        print(f"DEBUG: y_pred (magnitude sample): {mag_pred.flatten()[:5]}")
        mae = np.mean(np.abs(mag_true.flatten() - mag_pred.flatten()))
        print(f"Calculated MAE (magnitude): {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        """
        Calculates and prints the R² score for the magnitude component.
        """
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"Calculating R² for magnitude: y_true shape={mag_true.shape}, y_pred shape={mag_pred.shape}")
        SS_res = np.sum((mag_true - mag_pred) ** 2, axis=0)
        SS_tot = np.sum((mag_true - np.mean(mag_true, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (SS_res / (SS_tot + np.finfo(float).eps))
        r2 = np.mean(r2_scores)
        print(f"Calculated R² (magnitude): {r2}")
        return r2

# ---------------------------
# Debugging usage example (if run as main)
# ---------------------------
if __name__ == "__main__":
    plugin = Plugin()
    # For example: window_size=24, input shape (24,1)
    plugin.build_model(input_shape=(24, 1), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
