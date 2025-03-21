#!/usr/bin/env python
"""
Enhanced N-BEATS Predictor Plugin using Keras for forecasting the seasonal component
(without Bayesian output).

This plugin is designed to learn both the magnitude of the seasonal pattern.
It outputs a 2-dimensional vector per sample:
  - Column 0: Predicted magnitude (e.g. normalized seasonal value).

The composite loss function combines:
  - Huber loss on the magnitude.
  - MMD loss on the magnitude (weighted by mmd_lambda).


Custom metrics (MAE and R²) are computed on the magnitude only.
If the target y is provided as a one-dimensional tensor, it is automatically expanded


It is assumed that the input x (and corresponding target y) are the seasonal component
(or its returns) extracted from the close prices, shifted by a given forecast horizon.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # Imported for loss functions if needed.
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
from tensorflow.keras.losses import Huber
import gc
import os
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform

# ---------------------------
# Custom Callbacks (copied exactly from transformer plugin)
# ---------------------------
class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """Custom ReduceLROnPlateau callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        print(f"DEBUG: ReduceLROnPlateau patience counter: {self.patience_counter}")

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    """Custom EarlyStopping callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
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
    If y_true is one-dimensional or has only one column, it is expanded to 2 columns

    """
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """
    Computes the R² metric on the magnitude (first column) only.
    If y_true is one-dimensional or has only one column
    """
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
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

def composite_loss(y_true, y_pred, mmd_lambda, sigma=1.0):
    """
    Composite loss combining Huber loss on the magnitude and MMD loss on the magnitude.
    Assumes y_true and y_pred are either 1D tensors or 2D with a single column.
    """
    if y_true.shape.ndims == 1 or (y_true.shape.ndims == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    
    huber_loss_val = Huber()(mag_true, mag_pred)
    mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
    total_loss = huber_loss_val + (mmd_lambda * mmd_loss_val)
    return total_loss


# ---------------------------
# Enhanced N-BEATS Plugin Definition (Deterministic Version Without Bayesian Output)
# ---------------------------
class Plugin:
    """
    Enhanced N-BEATS Predictor Plugin using Keras for forecasting the seasonal component
    (or its returns) extracted via STL decomposition.

    

    The composite loss function includes:
      - Huber loss on the magnitude.
      - MMD loss on the magnitude (weighted by mmd_lambda).


    Custom metrics (MAE and R²) are computed on the magnitude only.
    Additional callbacks print training statistics (learning rate, patience counters, etc.).

    It is assumed that the input x (and target y) are the seasonal component (or returns)
    extracted from the close prices, shifted by a given forecast horizon.
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
        # N-BEATS parameters.
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

          - Concatenates the two outputs to yield a final 2D output.

        """
        window_size = input_shape[0]
        num_blocks = config.get("nbeats_num_blocks", 3)
        block_units = config.get("nbeats_units", 64)
        block_layers = config.get("nbeats_layers", 3)
        l2_reg = config.get("l2_reg", 1e-5)
        inputs = Input(shape=input_shape, name='input_layer')
        x = Flatten(name='flatten_layer')(inputs)
        
        residual = x
        forecasts = []
        
        def nbeats_block(res, block_id):
            r = res
            for i in range(block_layers):
                r = Dense(block_units, activation='relu', name=f'block{block_id}_dense_{i+1}')(r)
            forecast = Dense(1, activation='linear', name=f'block{block_id}_forecast')(r)
            units = int(res.shape[-1])
            backcast = Dense(units, activation='linear', name=f'block{block_id}_backcast')(r)
            updated_res = Add(name=f'block{block_id}_residual')([res, -backcast])
            return updated_res, forecast
        
        for b in range(1, num_blocks + 1):
            residual, forecast = nbeats_block(residual, b)
            forecasts.append(forecast)
        
        if len(forecasts) > 1:
            final_forecast = tf.keras.layers.add(forecasts, name='forecast_sum')
        else:
            final_forecast = forecasts[0]
        
        # Single branch: Deterministic output for magnitude.
        final_output = Dense(1, 
                            activation='linear', 
                            name='final_output',
                            kernel_initializer=GlorotUniform(),
                            kernel_regularizer=l2(l2_reg),
                        )(final_forecast)

        
        self.model = Model(inputs=inputs, outputs=final_output, name='NBeatsModel')
        
        optimizer = Adam(learning_rate=config.get("learning_rate", 0.0001))
        mmd_lambda = config.get("mmd_lambda", 1e-3)
        self.model.compile(optimizer=optimizer,
                   loss=lambda y_true, y_pred: composite_loss(y_true, y_pred, mmd_lambda, sigma=1.0),
                   metrics=[mae_magnitude, r2_metric])
        print("DEBUG: MMD lambda =", mmd_lambda)
        print("N-BEATS model built successfully.")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        callbacks = [
            EarlyStoppingWithPatienceCounter(monitor='val_loss',
                                             patience=config.get("early_patience", 60),
                                             verbose=1),
            ReduceLROnPlateauWithCounter(monitor='val_loss',
                                         factor=0.5,
                                         patience=config.get("early_patience", 20)/3,
                                         verbose=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: 
               print(f"DEBUG: Learning Rate at epoch {epoch+1}: {K.get_value(self.model.optimizer.learning_rate)}")),

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
        train_unc = np.zeros_like(train_preds[:, 0:1])
        val_unc = np.zeros_like(val_preds[:, 0:1])
        self.calculate_mae(y_train[0], train_preds)
        self.calculate_r2(y_train[0], train_preds)
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        predictions = self.model.predict(x_test)
        # Do not reshape; assume predictions are already of correct shape (n_samples, 1)
        uncertainty_estimates = np.zeros_like(predictions)
        return predictions, uncertainty_estimates



    def save(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={'composite_loss': composite_loss, 
                                                             'compute_mmd': compute_mmd, 
                                                             'r2_metric': r2_metric,
                                                             'mae_magnitude': mae_magnitude})
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"DEBUG: y_true (magnitude sample): {mag_true.flatten()[:5]}")
        print(f"DEBUG: y_pred (magnitude sample): {mag_pred.flatten()[:5]}")
        mae = np.mean(np.abs(mag_true.flatten() - mag_pred.flatten()))
        print(f"Calculated MAE (magnitude): {mae}")
        return mae


    def calculate_r2(self, y_true, y_pred):
        
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
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
    plugin.build_model(input_shape=(24, 1), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
