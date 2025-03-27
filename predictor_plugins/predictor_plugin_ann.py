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
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K
import gc
import os
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform

# Denine TensorFlow global variables(used from the composite loss function):
last_mae = tf.Variable(1.0, trainable=False, dtype=tf.float32)
last_std = tf.Variable(0.0, trainable=False, dtype=tf.float32)
intercept=tf.Variable(1e-8, trainable=False, dtype=tf.float32)# best 1e-8
p_control=tf.Variable(0, trainable=False, dtype=tf.float32) #best 0.1
d_control=tf.Variable(1, trainable=False, dtype=tf.float32)
i_control=tf.Variable(1, trainable=False, dtype=tf.float32)
peak_reward = tf.constant(-1500, dtype=tf.float32)             # Peak value (can be negative)
peak_penalty = tf.constant(1500, dtype=tf.float32)             # Peak value (can be negative)


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
    """
    Composite loss function that computes a Mean Squared Error (MSE) loss and an MMD loss along with
    custom reward and penalty components based on a Gaussian-like profile.
    
    The reward and penalty are computed as follows:
      - Reward: Uses a Gaussian-like function with a peak of -1 and width of 8e-4,
                centered on abs_avg_true, evaluated at abs_avg_error.
      - Penalty: Uses a Gaussian-like function with a peak of 1 and width of 8e-4,
                centered on 0, evaluated at signed_avg_error.
    
    Parameters:
      y_true   : Ground truth tensor.
      y_pred   : Predicted values tensor.
      mmd_lambda: Weight for the MMD loss.
      sigma    : Sigma parameter for compute_mmd.
      
    Returns:
      total_loss: The aggregated loss value.
    """
    # Ensure y_true has shape [batch_size, 1]
    if y_true.shape.ndims == 1 or (y_true.shape.ndims == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
    
    # Extract magnitude components (first column)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    
    # Compute primary losses:
    # Replace Huber loss with Mean Squared Error (MSE) loss.
    mse_loss_val = tf.keras.losses.MeanSquaredError()(mag_true, mag_pred)
    mae_loss_val = tf.keras.losses.MeanAbsoluteError()(mag_true, mag_pred)
    mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
    
    # Calculate summary statistics for use in reward and penalty.
    signed_avg_pred  = tf.reduce_mean(mag_pred)             # Average predicted value
    signed_avg_true  = tf.reduce_mean(mag_true)               # Average true value
    signed_avg_error = tf.reduce_mean(mag_true - mag_pred)    # Average error (true minus predicted)
    #abs_avg_pred     = tf.abs(signed_avg_pred)               # Absolute average of prediction
    abs_avg_true     = tf.abs(signed_avg_true)               # Absolute average of true values
    #abs_avg_error    = tf.abs(signed_avg_error - signed_avg_true)  # Absolute difference from signed average true
    
    # Define the Gaussian-like function.
    def gaussian_like(value, center, peak, width):
        """
        Gaussian-like function: Computes peak * exp(-k * (value - center)^2),
        where k = 4 * ln(2) / (width^2).
        
        Parameters:
          value : The arbitrary input value.
          center: The center (optimal value) for the function.
          peak  : The function's peak value (can be negative).
          width : The width at half the peak value.
          
        Returns:
          A Tensor representing the Gaussian-like output.
        """
        # Compute k = 4*ln(2)/(width^2)
        k = (4.0 * tf.math.log(tf.constant(2.0, dtype=tf.float32))) / (tf.math.square(width))
        # Compute and return the Gaussian-like value.
        return peak * tf.math.exp(-k * tf.math.square(value - center))
        # k=4332169.878499658
    
    def vertical_left_asymptote(value, center):
        res = tf.cond(tf.greater_equal(value, center),
            lambda: mse_loss_val*1000 - 1,
            lambda: 1e1*tf.math.log(tf.abs(value - center))+7
        )   
        return res


    # --- Compute custom reward and penalty using the Gaussian-like function ---
    #verify that the abs_avg_true is not zero
    abs_avg_true = tf.cond(
        tf.greater(abs_avg_true, 1e-10),
        lambda: abs_avg_true,
        lambda: 1e-10
    )   
    # Reward: Peak of -1, width 8e-4, centered on abs_avg_true.
    # Here, the arbitrary value is abs_avg_error.
    reward = gaussian_like(
        value=signed_avg_pred,
        center=abs_avg_true,
        peak=peak_reward,
        width=abs_avg_true/2
    )
    
    # Penalty: Peak of 1, width 8e-4, centered on 0.
    # Here, the arbitrary value is signed_avg_error.
    penalty = gaussian_like(
        value=signed_avg_pred,
        center=0.0,
        peak=peak_penalty,
        width=abs_avg_true/2
    )
    
    # --- Additional feedback values ---
    # Compute a divisor to avoid division by very small numbers.
    divisor = tf.cond(
        tf.greater(abs_avg_true, tf.constant(1e-10, dtype=tf.float32)),
        lambda: abs_avg_true,
        lambda: tf.constant(1e-10, dtype=tf.float32)
    )
    
    # Calculate batch-level feedback values.
    batch_signed_error = p_control * signed_avg_error / divisor
    batch_std = p_control * tf.math.reduce_mean(tf.abs(mag_true - mag_pred)) / divisor
    
    #calcualte the vertical left asymptote
    asymptote = vertical_left_asymptote(signed_avg_true, signed_avg_pred)


    # Update global variables last_mae and last_std with control dependencies.
    with tf.control_dependencies([last_mae.assign(batch_signed_error)]):
        #total_loss = reward + penalty + 3e6*mae_loss_val+ 3e8*mse_loss_val + mmd_lambda * mmd_loss_val
        total_loss = 1e1*mse_loss_val + asymptote
    with tf.control_dependencies([last_std.assign(batch_std)]):
        #total_loss = reward + penalty + 3e6*mae_loss_val+ 3e8*mse_loss_val + mmd_lambda * mmd_loss_val
        total_loss = 1e1*mse_loss_val + asymptote
    return total_loss



# --- Named initializer to avoid lambda serialization warnings ---
def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

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
    Their outputs are concatenated and then combined with the flattened error and std feedback channels
    (which bypass further Dense processing) and passed to a merged Dense layer.
    The final output is produced by a Bayesian layer (tfp.layers.DenseFlipout) plus a deterministic bias layer,
    so that uncertainty estimates can be derived.
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
        The architecture processes the trend, seasonal, and noise channels via Dense branches,
        while the error and std feedback channels are simply flattened and then concatenated with the branch outputs.
        The merged features are passed to a Dense layer, and finally to a Bayesian output layer (DenseFlipout)
        plus a deterministic bias layer.
        
        Args:
            input_shape (tuple): Expected input shape (window_size, num_channels).
                                  Originally, num_channels is 3 (trend, seasonal, noise).
                                  With error and std channels added, the raw input is augmented to shape (window_size, 5).
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
        
        # Add error and std feedback channels.
        def get_error_channel(x):
            batch_size = tf.shape(x)[0]
            win = tf.shape(x)[1]
            return tf.fill([batch_size, win, 1], last_mae)
        error_channel = Lambda(get_error_channel, output_shape=lambda inp: (inp[0], inp[1], 1), name="error_channel")(inputs)
        
        def get_std_channel(x):
            batch_size = tf.shape(x)[0]
            win = tf.shape(x)[1]
            return tf.fill([batch_size, win, 1], last_std)
        std_channel = Lambda(get_std_channel, output_shape=lambda inp: (inp[0], inp[1], 1), name="std_channel")(inputs)
        
        # Concatenate the original input with error and std channels.
        augmented_input = Concatenate(axis=2, name="augmented_input")([inputs, error_channel, std_channel])
        # Now the augmented input has shape (window_size, 3+2=5).
        
        # Split the augmented input into separate channels.
        trend_input = Lambda(lambda x: x[:, :, 0:1], name="trend_input")(augmented_input)
        seasonal_input = Lambda(lambda x: x[:, :, 1:2], name="seasonal_input")(augmented_input)
        noise_input = Lambda(lambda x: x[:, :, 2:3], name="noise_input")(augmented_input)
        error_input = Lambda(lambda x: x[:, :, 3:4], name="error_input")(augmented_input)
        std_input = Lambda(lambda x: x[:, :, 4:5], name="std_input")(augmented_input)
        
        # Build Dense branches for trend, seasonal, and noise channels.
        def build_branch(branch_input, branch_name):
            x = Flatten(name=f"{branch_name}_flatten")(branch_input)
            for i in range(num_branch_layers):
                x = Dense(branch_units, activation=activation,
                          kernel_regularizer=l2(l2_reg),
                          name=f"{branch_name}_dense_{i+1}")(x)
            return x
        
        trend_branch = build_branch(trend_input, "trend")
        seasonal_branch = build_branch(seasonal_input, "seasonal")
        noise_branch = build_branch(noise_input, "noise")
        
        # For error and std channels, simply flatten them (no extra Dense layers).
        error_flat = Flatten(name="error_flatten")(error_input)
        std_flat = Flatten(name="std_flatten")(std_input)
        
        # Concatenate all branch outputs.
        merged = Concatenate(name="merged_branches")([trend_branch, seasonal_branch, noise_branch, error_flat, std_flat])
        
        # Further process merged features.
        merged_dense = Dense(merged_units, activation=activation,
                             kernel_regularizer=l2(l2_reg),
                             name="merged_dense")(merged)
        
        # Further process merged features.
        merged_dense = Dense(merged_units, activation=activation,
                             kernel_regularizer=l2(l2_reg),
                             name="merged_dense_last")(merged_dense)
        
        # --- Bayesian Output Layer Implementation (copied from ANN plugin) ---
        KL_WEIGHT = self.params.get('kl_weight', 1e-3)

        # Monkey-patch DenseFlipout to use add_weight instead of deprecated add_variable
        def _patched_add_variable(self, name, shape, dtype, initializer, trainable, **kwargs):
            return self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
        tfp.layers.DenseFlipout.add_variable = _patched_add_variable

        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0; target kl_weight:", KL_WEIGHT)

        def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
            print("DEBUG: In posterior_mean_field_custom:")
            print("       dtype =", dtype, "kernel_shape =", kernel_shape)
            print("       Received bias_size =", bias_size, "; overriding to 0")
            print("       trainable =", trainable, "name =", name)
            if not isinstance(name, str):
                print("DEBUG: 'name' is not a string; setting to None")
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            print("DEBUG: posterior: computed n =", n)
            c = np.log(np.expm1(1.))
            print("DEBUG: posterior: computed c =", c)
            loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42), dtype=dtype, trainable=trainable, name="posterior_loc")
            scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43), dtype=dtype, trainable=trainable, name="posterior_scale")
            scale = 1e-3 + tf.nn.softplus(scale + c)
            scale = tf.clip_by_value(scale, 1e-3, 1.0)
            print("DEBUG: posterior: created loc shape:", loc.shape, "scale shape:", scale.shape)
            try:
                loc_reshaped = tf.reshape(loc, kernel_shape)
                scale_reshaped = tf.reshape(scale, kernel_shape)
                print("DEBUG: posterior: reshaped loc to", loc_reshaped.shape, "and scale to", scale_reshaped.shape)
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
            print("       trainable =", trainable, "name =", name)
            if not isinstance(name, str):
                print("DEBUG: 'name' is not a string in prior_fn; setting to None")
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            print("DEBUG: prior_fn: computed n =", n)
            loc = tf.zeros([n], dtype=dtype)
            scale = tf.ones([n], dtype=dtype)
            try:
                loc_reshaped = tf.reshape(loc, kernel_shape)
                scale_reshaped = tf.reshape(scale, kernel_shape)
                print("DEBUG: prior_fn: reshaped loc to", loc_reshaped.shape, "and scale to", scale_reshaped.shape)
            except Exception as e:
                print("DEBUG: Exception during reshape in prior_fn:", e)
                raise e
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
                reinterpreted_batch_ndims=len(kernel_shape)
            )

        DenseFlipout = tfp.layers.DenseFlipout
        print("DEBUG: Creating DenseFlipout final layer with units:", 1)
        flipout_layer = DenseFlipout(
            units=1,
            activation='linear',
            kernel_posterior_fn=posterior_mean_field_custom,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
            name="output_layer"
        )
        bayesian_output = tf.keras.layers.Lambda(
            lambda t: flipout_layer(t),
            output_shape=lambda s: (s[0], 1),
            name="bayesian_dense_flipout"
        )(merged_dense)
        print("DEBUG: After DenseFlipout (via Lambda), bayesian_output shape:", bayesian_output.shape)

        bias_layer = tf.keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer=random_normal_initializer_44,
            name="deterministic_bias"
        )(merged_dense)
        print("DEBUG: Deterministic bias layer output shape:", bias_layer.shape)

        outputs = bayesian_output + bias_layer
        print("DEBUG: Final outputs shape after adding bias:", outputs.shape)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="MultiBranchPredictor")
        optimizer = AdamW(learning_rate=config.get("learning_rate", self.params["learning_rate"]))
        mmd_lambda = config.get("mmd_lambda", self.params["mmd_lambda"])
        
        self.model.compile(optimizer=optimizer,
                           loss=lambda y_true, y_pred: composite_loss(y_true, y_pred, mmd_lambda, sigma=1.0),
                           metrics=[mae_magnitude, r2_metric])
        print("DEBUG: MMD lambda =", mmd_lambda)
        print("Multi-Branch Predictor model built successfully.")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
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
        train_uncertainty = np.zeros_like(train_preds)
        val_uncertainty = np.zeros_like(val_preds)
        self.calculate_mae(y_train[0], train_preds)
        self.calculate_r2(y_train[0], train_preds)
        return history, train_preds, train_uncertainty, val_preds, val_uncertainty

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        predictions = np.array([self.model(x_test, training=True).numpy() for _ in range(mc_samples)])
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        return mean_predictions, uncertainty_estimates

    def save(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={
            "composite_loss": composite_loss,
            "compute_mmd": compute_mmd,
            "r2_metric": r2_metric,
            "mae_magnitude": mae_magnitude
        })
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
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