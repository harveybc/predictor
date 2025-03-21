import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras import backend as K
from sklearn.metrics import r2_score
import gc
import os
import tensorflow.keras.backend as K

# Shortcut for tfp layers
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

class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()

# ---------------------------
# Named initializers to avoid lambda serialization warnings
# ---------------------------
def random_normal_initializer_42(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=42)

def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

# ---------------------------
# Bayesian layer functions (copied exactly from transformer plugin)
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
    print("DEBUG: posterior: computed n =", n)
    c = np.log(np.expm1(1.))
    print("DEBUG: posterior: computed c =", c)
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42),
                      dtype=dtype, trainable=trainable, name="posterior_loc")
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43),
                        dtype=dtype, trainable=trainable, name="posterior_scale")
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
    print("DEBUG: trainable =", trainable, "name =", name)
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

# ---------------------------
# MMD loss and custom loss (copied exactly)
# ---------------------------
def compute_mmd(x, y, sigma=1.0, sample_size=256):
    with tf.device('/CPU:0'):
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
    huber_loss = Huber()(y_true, y_pred)
    mmd_loss = compute_mmd(y_pred, y_true, sigma=sigma)
    total_loss = huber_loss + (mmd_lambda * mmd_loss)
    return total_loss

# ---------------------------
# Enhanced N-BEATS Plugin Definition (single-step output)
# ---------------------------
class Plugin:
    """
    Enhanced N-BEATS Predictor Plugin using Keras for single-step forecasting.
    Enhancements include:
      - A Bayesian output layer (using DenseFlipout exactly as in the transformer plugin)
      - Custom loss: Huber loss + MMD loss.
      - Callbacks: EarlyStoppingWithPatienceCounter, ReduceLROnPlateauWithCounter, KL annealing, MMD logging, and ClearMemoryCallback.
    
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
        self.kl_weight_var = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

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
                - "learning_rate": learning rate for the Adam optimizer (default=0.001)
        
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
            # Use the static shape attribute for units instead of tf.shape.
            units = int(res.shape[-1])
            backcast = Dense(units, activation='linear', name=f'block{block_id}_backcast')(r)
            # Update residual: subtract the backcast from the input residual.
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
        self.model.compile(optimizer=optimizer, loss='mse')
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
        train_preds = self.model.predict(x_train, batch_size=batch_size)
        val_preds = self.model.predict(x_val, batch_size=batch_size)
        # Uncertainty is set to zero for compatibility.
        train_unc = np.zeros_like(train_preds)
        val_unc = np.zeros_like(val_preds)
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Generates predictions with uncertainty estimates.
        For N-BEATS, predictions are returned with zero uncertainty.
        
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
        self.model = load_model(file_path, custom_objects={'custom_loss': custom_loss, 'compute_mmd': compute_mmd})
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        print(f"Calculating R² for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch in calculate_r2: y_true={y_true.shape}, y_pred={y_pred.shape}")
        ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
        r2_scores = 1 - (ss_res / ss_tot)
        r2_scores = np.where(ss_tot == 0, 0, r2_scores)
        r2 = np.mean(r2_scores)
        print(f"Calculated R²: {r2}")
        return r2

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# ---------------------------
# Debugging usage example
# ---------------------------
if __name__ == "__main__":
    plugin = Plugin()
    # Example: window_size=24, input has 1 feature
    plugin.build_model(input_shape=(24, 1), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
