#!/usr/bin/env python
"""
Enhanced N-BEATS Predictor Plugin (Single-Step Forecast with Bayesian Output)

This module implements an enhanced version of an N-BEATS forecasting plugin.
It includes:
  - A Bayesian output layer using tfp.layers.DenseFlipout.
  - Custom loss function combining Huber loss and MMD loss.
  - Custom metrics: Mean Absolute Error (MAE) and a custom R² metric.
  - Callbacks that print key training statistics:
      • Current learning rate at each epoch end.
      • EarlyStopping and ReduceLROnPlateau patience counters.
      • MMD lambda value is printed before training.
      
Note: This plugin produces a single output per input window.

References:
- Bayesian layers in TensorFlow Probability: :contentReference[oaicite:0]{index=0}
- N-BEATS architecture (simplified): :contentReference[oaicite:1]{index=1}
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
from tensorflow.keras.losses import Huber
import gc
import os
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score

# Shortcut for TFP layers.
tfp_layers = tfp.layers

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
def r2_metric(y_true, y_pred):
    """
    Custom R² metric.
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
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

def custom_loss(y_true, y_pred, mmd_lambda, sigma=1.0):
    """
    Custom loss: Huber loss plus MMD loss weighted by mmd_lambda.
    """
    huber_loss = Huber()(y_true, y_pred)
    mmd_loss = compute_mmd(y_pred, y_true, sigma=sigma)
    total_loss = huber_loss + (mmd_lambda * mmd_loss)
    return total_loss

# ---------------------------
# Bayesian Layer Functions (copied exactly from transformer plugin)
# ---------------------------
def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
    print("DEBUG: In posterior_mean_field_custom:")
    print("       dtype =", dtype, "kernel_shape =", kernel_shape)
    print("       Received bias_size =", bias_size, "; overriding to 0")
    print("DEBUG: trainable =", trainable, "name =", name)
    if not isinstance(name, str):
        print("DEBUG: 'name' is not a string; setting to None")
        name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.))
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42),
                      dtype=dtype, trainable=trainable, name="posterior_loc")
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43),
                        dtype=dtype, trainable=trainable, name="posterior_scale")
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)
    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
    except Exception as e:
        print("DEBUG: Exception during reshape in posterior:", e)
        raise e
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape)
    )

def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    print("DEBUG: In prior_fn:")
    print("       dtype =", dtype, "kernel_shape =", kernel_shape)
    print("       Received bias_size =", bias_size, "; overriding to 0")
    print("DEBUG: trainable =", trainable, "name =", name)
    if not isinstance(name, str):
        print("DEBUG: 'name' is not a string in prior_fn; setting to None")
        name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
    except Exception as e:
        print("DEBUG: Exception during reshape in prior_fn:", e)
        raise e
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape)
    )

# ---------------------------
# Enhanced N-BEATS Plugin Definition (Single-Step Output with Bayesian Output)
# ---------------------------
class Plugin:
    """
    Enhanced N-BEATS Predictor Plugin using Keras for single-step forecasting.
    
    Enhancements include:
      - A Bayesian output layer via tfp.layers.DenseFlipout.
      - Custom loss: Huber loss + MMD loss.
      - Custom metrics: MAE and R².
      - Callbacks: EarlyStoppingWithPatienceCounter, ReduceLROnPlateauWithCounter,
                   LambdaCallback to print current learning rate, and ClearMemoryCallback.
    
    This version produces a single output.
    """
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3,
        'mmd_lambda': 1e-3,  # Hyperparameter for MMD loss weighting
        # N-BEATS parameters
        'nbeats_num_blocks': 3,
        'nbeats_units': 64,
        'nbeats_layers': 3,
        'time_horizon': 1  # single-step forecasting
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
        Builds an enhanced N-BEATS model with Bayesian output.
        
        Args:
            input_shape (tuple): Expected shape (window_size, 1).
            x_train (np.ndarray): Training data (for shape inference if needed).
            config (dict): Configuration parameters.
        
        The model:
          - Flattens the input.
          - Passes data through several blocks; each block produces a forecast.
          - Aggregates forecasts by summing them.
          - Applies a Bayesian output layer (DenseFlipout) to produce the final prediction.
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
        
        # Apply Bayesian output layer:
        bayesian_output = tfp_layers.DenseFlipout(
            1,
            activation='linear',
            posterior_fn=posterior_mean_field_custom,
            prior_fn=prior_fn,
            name='bayesian_output'
        )(final_forecast)
        
        self.model = Model(inputs=inputs, outputs=bayesian_output, name='NBeatsModel')
        
        optimizer = Adam(learning_rate=config.get("learning_rate", 0.0001))
        mmd_lambda = config.get("mmd_lambda", 1e-3)
        # Compile with custom loss and metrics (MAE and R²).
        self.model.compile(optimizer=optimizer,
                           loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, mmd_lambda=mmd_lambda),
                           metrics=['mae', r2_metric])
        print("DEBUG: MMD lambda =", mmd_lambda)
        print("N-BEATS model built successfully.")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the N-BEATS model with enhanced callbacks.
        
        Prints training statistics for:
          - MAE
          - Learning rate (at each epoch)
          - EarlyStopping patience counter
          - ReduceLROnPlateau patience counter
        
        Args:
            x_train (np.ndarray): Training input with shape (samples, window_size, 1).
            y_train (list): List containing a single array of targets (shape (samples,)).
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
        # Uncertainty estimates set to zero for compatibility.
        train_unc = np.zeros_like(train_preds)
        val_unc = np.zeros_like(val_preds)
        # Calculate and print MAE and R² on training data.
        self.calculate_mae(y_train[0], train_preds)
        self.calculate_r2(y_train[0], train_preds)
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Generates predictions with uncertainty estimates.
        For N-BEATS, predictions are returned with zero uncertainty.
        
        Args:
            x_test (np.ndarray): Test input data.
            mc_samples (int): Number of Monte Carlo samples (unused).
        
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
        """
        Loads a model from the specified file.
        
        Args:
            file_path (str): Path from which to load the model.
        """
        self.model = load_model(file_path, custom_objects={'custom_loss': custom_loss, 'compute_mmd': compute_mmd, 'r2_metric': r2_metric})
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        """
        Calculates and prints the Mean Absolute Error.
        """
        print(f"DEBUG: y_true (sample): {y_true.flatten()[:5]}")
        print(f"DEBUG: y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        """
        Calculates and prints the R² score.
        """
        print(f"Calculating R² for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch in calculate_r2: y_true={y_true.shape}, y_pred={y_pred.shape}")
        SS_res = np.sum((y_true - y_pred) ** 2, axis=0)
        SS_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (SS_res / (SS_tot + np.finfo(float).eps))
        r2 = np.mean(r2_scores)
        print(f"Calculated R²: {r2}")
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
