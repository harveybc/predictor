#!/usr/bin/env python
"""
Enhanced N-BEATS Predictor Plugin (Single-Step Forecast with Bayesian Output and Phase Loss)

This module implements an enhanced version of an N-BEATS forecasting plugin.
It includes:
  - A Bayesian output layer (via tfp.layers.DenseFlipout) that predicts the magnitude.
  - A parallel branch that predicts the phase.
  - A composite loss function combining:
       • Huber loss (for magnitude)
       • MMD loss (for magnitude) weighted by mmd_lambda
       • Phase loss (1 - cos(predicted_phase - true_phase)) weighted by lambda_phase
  - Custom metrics: MAE and a custom R² metric computed on the magnitude only.
  - Callbacks that print key training statistics:
      • Current learning rate at each epoch end.
      • EarlyStopping and ReduceLROnPlateau patience counters.
      • MMD lambda value is printed before training.
      
Note: The model now outputs a 2-dimensional vector per sample. At inference time,
you can use only the magnitude forecast if desired.

References:
- Bayesian layers in TensorFlow Probability: :contentReference[oaicite:0]{index=0}
- N-BEATS architecture (simplified): :contentReference[oaicite:1]{index=1}
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Add, Concatenate
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
def mae_magnitude(y_true, y_pred):
    """
    Computes MAE on the magnitude component (first column).
    """
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """
    Custom R² metric computed on the magnitude component.
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
    Composite loss combining:
      - Huber loss on magnitude (first column)
      - MMD loss on magnitude (first column)
      - Phase loss computed as 1 - cos(predicted_phase - true_phase)
    
    Args:
        y_true: Tensor with shape (batch_size, 2). Column 0 is magnitude target,
                column 1 is phase target.
        y_pred: Tensor with shape (batch_size, 2). Column 0 is predicted magnitude,
                column 1 is predicted phase.
        mmd_lambda: Weight for MMD loss.
        lambda_phase: Weight for phase loss.
        sigma: Standard deviation for MMD loss.
    
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
# Enhanced N-BEATS Plugin Definition (Single-Step Output with Bayesian Output and Phase Loss)
# ---------------------------
class Plugin:
    """
    Enhanced N-BEATS Predictor Plugin using Keras for single-step forecasting.
    
    Enhancements include:
      - A Bayesian output branch (via tfp.layers.DenseFlipout) predicting magnitude.
      - A parallel branch predicting phase.
      - Composite loss: Huber + MMD (on magnitude) and phase loss (1 - cos(diff)).
      - Custom metrics: MAE and R² (computed on magnitude).
      - Callbacks: EarlyStoppingWithPatienceCounter, ReduceLROnPlateauWithCounter,
                   LambdaCallback (printing learning rate), and ClearMemoryCallback.
    
    This version produces a 2-dimensional output per sample.
    """
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3,
        'mmd_lambda': 1e-3,       # Weight for MMD loss
        'lambda_phase': 1e-3,     # Weight for phase loss
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
        Builds an enhanced N-BEATS model with Bayesian output for magnitude and a branch for phase.
        
        Args:
            input_shape (tuple): Expected shape (window_size, 1).
            x_train (np.ndarray): Training data (for shape inference if needed).
            config (dict): Configuration parameters.
        
        The model:
          - Flattens the input.
          - Passes data through several blocks; each block produces a forecast.
          - Aggregates forecasts by summing them.
          - Splits into two branches:
              • Bayesian branch for magnitude (using DenseFlipout).
              • Deterministic branch for phase.
          - Concatenates the two outputs into a final 2D output.
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
        
        # Branch 1: Bayesian output for magnitude.
        bayesian_mag_output = tfp_layers.DenseFlipout(
            1,
            activation='linear',
            kernel_posterior_fn=posterior_mean_field_custom,
            kernel_prior_fn=prior_fn,
            name='bayesian_mag_output'
        )(final_forecast)
        
        # Branch 2: Deterministic output for phase.
        phase_output = Dense(1, activation='linear', name='phase_output')(final_forecast)
        
        # Concatenate outputs to form final output (shape: (batch_size, 2))
        final_output = Concatenate(name='final_output')([bayesian_mag_output, phase_output])
        
        self.model = Model(inputs=inputs, outputs=final_output, name='NBeatsModel')
        
        optimizer = Adam(learning_rate=config.get("learning_rate", 0.0001))
        mmd_lambda = config.get("mmd_lambda", 1e-3)
        lambda_phase = config.get("lambda_phase", 1e-3)
        # Compile with composite loss and custom metrics (MAE and R² on magnitude).
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
          - MAE (magnitude)
          - Learning rate (at each epoch)
          - EarlyStopping patience counter
          - ReduceLROnPlateau patience counter
        
        Args:
            x_train (np.ndarray): Training input with shape (samples, window_size, 1).
            y_train (list): List containing a single array of targets (shape (samples, 2)),
                             where column 0 is magnitude and column 1 is phase.
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
        train_unc = np.zeros_like(train_preds[:, 0:1])
        val_unc = np.zeros_like(val_preds[:, 0:1])
        # Calculate and print MAE and R² on training data (using magnitude only).
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
        # Return only magnitude uncertainty as zeros.
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
        Calculates and prints the Mean Absolute Error for magnitude.
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
        Calculates and prints the R² score for magnitude.
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
    # Example: window_size=24, input shape (24,1)
    plugin.build_model(input_shape=(24, 1), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
