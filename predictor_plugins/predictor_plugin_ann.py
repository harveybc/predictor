#!/usr/bin/env python
"""
Enhanced Multi-Branch Predictor Plugin using Keras for forecasting EUR/USD returns.

This plugin is designed to use the decomposed signals produced by the STL Preprocessor Plugin.
It assumes the input is a multi-channel time window where each channel corresponds to a decomposed component:
  - Trend component
  - Seasonal component
  - Noise (residual) component

The architecture is composed of three branches—each processing one channel through its own Dense sub-network.
The outputs of these branches are concatenated and fed to a final set of layers to produce the predicted return.
The loss is computed using a composite loss (Huber + MMD), and custom metrics (MAE and R²) are calculated
on the predicted return. This implementation is intended for the case when use_returns is True.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K
import gc
import os
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
# Replace your previous Python global with a TensorFlow variable:
last_mae = tf.Variable(1.0, trainable=False, dtype=tf.float32)
last_std = tf.Variable(0.0, trainable=False, dtype=tf.float32)
# ---------------------------
# Custom Callbacks (same as before)
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
    """Compute MAE on the first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """Compute R² metric on the first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    SS_res = tf.reduce_sum(tf.square(mag_true - mag_pred))
    SS_tot = tf.reduce_sum(tf.square(mag_true - tf.reduce_mean(mag_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

def compute_mmd(x, y, sigma=1.0, sample_size=256):
    """Compute the Maximum Mean Discrepancy (MMD) between two samples."""
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
    if y_true.shape.ndims == 1 or (y_true.shape.ndims == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    
    huber_loss_val = Huber()(mag_true, mag_pred)
    mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)

    # Example penalty usage (unchanged)
    #average = tf.reduce_mean(tf.abs(mag_pred))
    #penalty = 0.001 - average
    #penalty = (1 / 0.001) * tf.maximum(penalty, 0.0)

    #calcualte the level correction, if the signed average is positive and the true value is less than the prediction, penalize
    # if the signed average is negative and the true value is greater than the prediction, penalize 
    signed_average_pred = tf.reduce_mean(mag_pred)
    signed_avg_error = tf.reduce_mean(mag_true - mag_pred)
    abs_avg_pred = tf.abs(signed_average_pred)
    return_error = tf.cond(tf.greater(abs_avg_pred, 1e-8),
                           lambda: (signed_avg_error - signed_average_pred) / abs_avg_pred,
                           lambda: (signed_avg_error - signed_average_pred) / (abs_avg_pred + 1e-8))
    
    penalty = tf.cond(tf.greater(abs_avg_pred, 1e-8),
                           lambda: (signed_avg_error/signed_average_pred),
                           lambda: (signed_avg_error/1e-8))


    # penalize a quantity proportional to the sum of the abs(signed_error) and the abs of (difference between the true value and the prediction)
    penalty =  0.001*tf.abs(penalty) #best 0.001
    

    # Compute the batch signed error to use as feedback
    batch_signed_error =1*return_error # best 1
    batch_std = 1*tf.math.reduce_mean(mag_true - mag_pred)
    #print(f"DEBUG: Batch signed error: {batch_signed_error}, Batch std: {batch_std}")

    # Update the global tf.Variable 'last_mae' using assign.
    with tf.control_dependencies([last_mae.assign(batch_signed_error)]):
        #total_loss = (penalty + 1.0) * (huber_loss_val + (mmd_lambda * mmd_loss_val))
        total_loss = (penalty + 1.0)*(huber_loss_val + mmd_lambda * mmd_loss_val)
    # Update the global tf.Variable 'last_std' using assign.
    with tf.control_dependencies([last_std.assign(batch_std)]):
        #total_loss = (penalty + 1.0) * (huber_loss_val + (mmd_lambda * mmd_loss_val))
        total_loss = (penalty + 1.0)*(huber_loss_val + mmd_lambda * mmd_loss_val)

    return total_loss



# ---------------------------
# Multi-Branch Predictor Plugin Definition
# ---------------------------
class Plugin:
    """
    Enhanced Multi-Branch Predictor Plugin.

    This plugin builds a multi-branch model to process STL-decomposed input channels:
      - One branch processes the trend channel.
      - One branch processes the seasonal channel.
      - One branch processes the noise (residual) channel.

    Each branch passes its input through dedicated Dense layers.
    Their outputs are concatenated and passed to further Dense layers to produce the final prediction,
    which represents the return (price variation) for the forecast horizon.

    The loss is computed as a composite of Huber and MMD losses.
    Custom metrics (MAE and R²) are computed on the predicted return.
    """
    plugin_params = {
        'batch_size': 32,
        'num_branch_layers': 2,      # Number of Dense layers in each branch
        'branch_units': 32,          # Units in each branch layer
        'merged_units': 64,          # Units in the merged network
        'learning_rate': 0.0001,
        'activation': 'relu',
        'l2_reg': 1e-5,
        'mmd_lambda': 1e-3,
        'time_horizon': 6           # Forecast horizon (in hours)
    }
    plugin_debug_vars = ['batch_size', 'num_branch_layers', 'branch_units', 'merged_units', 'learning_rate', 'l2_reg', 'time_horizon']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """Update predictor plugin parameters with provided configuration."""
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """Return debug information for the predictor plugin."""
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add predictor plugin debug information to the given dictionary."""
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape, x_train, config):
        """
        Build a multi-branch model for forecasting returns using STL-decomposed data.
        An extra error and std feedback channels are added and fed directly (without additional Dense processing)
        into the merged dense layer.
        
        Args:
            input_shape (tuple): Expected input shape (window_size, num_channels).
                                  Originally, num_channels is 3 (trend, seasonal, noise).
                                  With the error and std channels, the raw input is augmented to shape
                                  (window_size, 3) then two extra channels are added via Lambda layers, resulting in shape (window_size, 5).
            x_train (np.ndarray): Training data (for shape inference).
            config (dict): Configuration parameters.
        """
        window_size, num_channels = input_shape  # Expecting num_channels=3
        l2_reg = config.get("l2_reg", self.params["l2_reg"])
        activation = config.get("activation", self.params["activation"])
        num_branch_layers = config.get("num_branch_layers", self.params["num_branch_layers"])
        branch_units = config.get("branch_units", self.params["branch_units"])
        merged_units = config.get("merged_units", self.params["merged_units"])
        time_horizon = config.get("time_horizon", self.params["time_horizon"])

        # Define input layer with original 3 channels.
        inputs = Input(shape=(window_size, num_channels), name="input_layer")
        
        # Add an error feedback channel by using a Lambda layer that outputs a tensor of shape (batch_size, window_size, 1)
        # filled with the current value of the global variable 'last_mae'.
        def get_error_channel(x):
            batch_size = tf.shape(x)[0]
            win = tf.shape(x)[1]
            return tf.fill([batch_size, win, 1], last_mae)
        error_channel = Lambda(get_error_channel, output_shape=lambda input_shape: (input_shape[0], input_shape[1], 1), name="error_channel")(inputs)
        
        # Add a standard deviation feedback channel by using a Lambda layer that outputs a tensor of shape (batch_size, window_size, 1)
        # filled with the current value of the global variable 'last_std'.
        def get_std_channel(x):
            batch_size = tf.shape(x)[0]
            win = tf.shape(x)[1]
            return tf.fill([batch_size, win, 1], last_std)
        std_channel = Lambda(get_std_channel, output_shape=lambda input_shape: (input_shape[0], input_shape[1], 1), name="std_channel")(inputs)
        
        # Concatenate the original input with the error and std channels.
        # The augmented input now has shape (window_size, 3+2=5).
        augmented_input = Concatenate(axis=2, name="augmented_input")([inputs, error_channel, std_channel])
        
        # Split the augmented input into separate channels.
        # Channel 0: Trend, Channel 1: Seasonal, Channel 2: Noise, Channel 3: Error, Channel 4: Std.
        trend_input = Lambda(lambda x: x[:, :, 0:1], name="trend_input")(augmented_input)
        seasonal_input = Lambda(lambda x: x[:, :, 1:2], name="seasonal_input")(augmented_input)
        noise_input = Lambda(lambda x: x[:, :, 2:3], name="noise_input")(augmented_input)
        error_input = Lambda(lambda x: x[:, :, 3:4], name="error_input")(augmented_input)
        std_input = Lambda(lambda x: x[:, :, 4:5], name="std_input")(augmented_input)
        
        # Define a function to build a branch for a given channel.
        def build_branch(branch_input, branch_name):
            x = Flatten(name=f"{branch_name}_flatten")(branch_input)
            for i in range(num_branch_layers):
                x = Dense(branch_units, activation=activation,
                          #kernel_regularizer=l2(l2_reg),
                          name=f"{branch_name}_dense_{i+1}")(x)
            return x
        
        # Build branches for trend, seasonal, and noise channels.
        trend_branch = build_branch(trend_input, "trend")
        seasonal_branch = build_branch(seasonal_input, "seasonal")
        noise_branch = build_branch(noise_input, "noise")
        
        # For error and std channels, simply flatten them (no Dense processing).
        error_flat = Flatten(name="error_flatten")(error_input)
        std_flat = Flatten(name="std_flatten")(std_input)
        
        # Concatenate all branch outputs.
        merged = Concatenate(name="merged_branches")([trend_branch, seasonal_branch, noise_branch, error_flat, std_flat])
        
        # Further process merged features.
        merged_dense = Dense(merged_units, activation=activation,
                             #kernel_regularizer=l2(l2_reg),
                             name="merged_dense")(merged)
        # Final prediction layer.
        final_output = Dense(1, activation="linear", name="final_output")(merged_dense)
        
        self.model = Model(inputs=inputs, outputs=final_output, name="MultiBranchPredictor")
        optimizer = Adam(learning_rate=config.get("learning_rate", self.params["learning_rate"]))
        mmd_lambda = config.get("mmd_lambda", self.params["mmd_lambda"])
        
        self.model.compile(optimizer=optimizer,
                           loss=lambda y_true, y_pred: composite_loss(y_true, y_pred, mmd_lambda, sigma=1.0),
                           metrics=[mae_magnitude, r2_metric])
        print("DEBUG: MMD lambda =", mmd_lambda)
        print("Multi-Branch Predictor model built successfully.")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Train the model using the provided training and validation datasets.
        Uses early stopping, learning rate reduction, and memory cleanup callbacks.

        Args:
            x_train (np.ndarray): Training inputs.
            y_train (list): List containing target arrays.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            threshold_error (float): Threshold error for early stopping (unused here, but kept for interface consistency).
            x_val (np.ndarray): Validation inputs.
            y_val (list): List containing validation target arrays.
            config (dict): Configuration parameters.

        Returns:
            tuple: (history, train_predictions, train_uncertainty, val_predictions, val_uncertainty)
        """
        callbacks = [
            EarlyStoppingWithPatienceCounter(monitor="val_loss",
                                             patience=config.get("early_patience", 60),
                                             verbose=1),
            ReduceLROnPlateauWithCounter(monitor="val_loss",
                                         factor=0.5,
                                         patience=config.get("early_patience", 20) / 3,
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
        # For now, uncertainty estimation is not implemented (set to zeros).
        train_uncertainty = np.zeros_like(train_preds)
        val_uncertainty = np.zeros_like(val_preds)
        self.calculate_mae(y_train[0], train_preds)
        self.calculate_r2(y_train[0], train_preds)
        return history, train_preds, train_uncertainty, val_preds, val_uncertainty

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Predicts on the test data.
        Currently, returns zeros for uncertainty estimates.
        
        Args:
            x_test (np.ndarray): Test inputs.
            mc_samples (int): Number of Monte Carlo samples (unused here).

        Returns:
            tuple: (predictions, uncertainty_estimates)
        """
        predictions = self.model.predict(x_test)
        uncertainty_estimates = np.zeros_like(predictions)
        return predictions, uncertainty_estimates

    def save(self, file_path):
        """Saves the trained model to the specified file path."""
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        """Loads a model from the specified file path."""
        self.model = load_model(file_path, custom_objects={
            "composite_loss": composite_loss,
            "compute_mmd": compute_mmd,
            "r2_metric": r2_metric,
            "mae_magnitude": mae_magnitude
        })
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        """Calculates and prints the Mean Absolute Error (MAE) for the first column."""
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"DEBUG: y_true (sample): {mag_true.flatten()[:5]}")
        print(f"DEBUG: y_pred (sample): {mag_pred.flatten()[:5]}")
        mae = np.mean(np.abs(mag_true.flatten() - mag_pred.flatten()))
        print(f"Calculated MAE (magnitude): {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        """Calculates and prints the R² metric for the first column."""
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"Calculating R²: y_true shape={mag_true.shape}, y_pred shape={mag_pred.shape}")
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
    # For debugging, assume input shape (window_size, num_channels) where num_channels=3.
    # Example: window_size=24, 3 channels (trend, seasonal, noise).
    plugin.build_model(input_shape=(24, 3), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
