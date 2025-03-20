import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score
import gc
import os

# --- Custom Callbacks (copied exactly from the transformer plugin) ---

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

# --- Named initializer to avoid lambda serialization warnings ---
def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

# --- Bayesian layer functions (copied exactly from transformer plugin) ---

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

# --- MMD compute function (copied exactly) ---
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

# --- Custom Loss: Huber loss + MMD ---
def custom_loss(y_true, y_pred, mmd_lambda, sigma=1.0):
    huber_loss = Huber()(y_true, y_pred)
    mmd_loss = compute_mmd(y_pred, y_true, sigma=sigma)
    total_loss = huber_loss + (mmd_lambda * mmd_loss)
    return total_loss

# ---------------------------
# Enhanced N-BEATS Plugin Definition
# ---------------------------
class Plugin:
    """
    Enhanced N-BEATS Predictor Plugin using Keras for multi-step forecasting.
    Enhancements include:
      - A Bayesian output layer (using tfp.layers.DenseFlipout)
      - Custom loss combining Huber loss and MMD loss
      - Callbacks: EarlyStoppingWithPatienceCounter, ReduceLROnPlateauWithCounter, KL annealing, and memory clearing.
    
    The core N-BEATS architecture remains unchanged.
    """
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3,
        'nbeats_num_blocks': 3,
        'nbeats_units': 64,
        'nbeats_layers': 3,
        'time_horizon': 1  # For single-step forecasting
    }
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size', 'time_horizon']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.kl_weight_var = None  # Will be initialized in build_model

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    # --- Build Model with Enhanced Bayesian Output and MMD loss ---
    def build_model(self, input_shape, x_train, config):
        """
        Builds the N-BEATS model with Bayesian output layer.
        The architecture uses the same block structure as before.
        The final output layer is replaced by a Bayesian DenseFlipout layer
        plus a deterministic bias, and the model is compiled with a custom loss.
        """
        window_size = input_shape[0]
        num_blocks = config.get("nbeats_num_blocks", self.params['nbeats_num_blocks'])
        block_units = config.get("nbeats_units", self.params['nbeats_units'])
        block_layers = config.get("nbeats_layers", self.params['nbeats_layers'])
        
        # Input layer accepts shape (window_size, 1)
        inputs = Input(shape=input_shape, name='input_layer')
        x = Flatten(name='flatten_layer')(inputs)  # shape: (window_size,)
        
        # Initialize residual as the flattened input.
        residual = x
        forecasts = []
        
        # Helper function to build a single block.
        def nbeats_block(res, block_id):
            r = res
            for i in range(block_layers):
                r = Dense(block_units, activation='relu', name=f'block{block_id}_dense_{i+1}')(r)
            # Forecast branch outputs a single value.
            forecast = Dense(1, activation='linear', name=f'block{block_id}_forecast')(r)
            # Backcast branch estimates the part of the input explained by this block.
            backcast = Dense(int(res.shape[-1]), activation='linear', name=f'block{block_id}_backcast')(r)
            updated_res = Add(name=f'block{block_id}_residual')([res, -backcast])
            return updated_res, forecast
        
        # Build blocks sequentially.
        for b in range(1, num_blocks + 1):
            residual, forecast = nbeats_block(residual, b)
            forecasts.append(forecast)
        
        # Sum forecasts from all blocks to form a deterministic output (legacy path)
        if len(forecasts) > 1:
            final_forecast = Add(name='forecast_sum')(forecasts)
        else:
            final_forecast = forecasts[0]
        
        # --- Add Bayesian Output Layer ---
        # Instead of using the deterministic final_forecast, we now feed the last residual into a Bayesian DenseFlipout layer.
        l2_reg = self.params.get('l2_reg', 1e-5)
        KL_WEIGHT = self.params.get('kl_weight', 1e-3)
        time_horizon = self.params.get('time_horizon', 1)
        tfp_layers = tfp.layers
        
        # Patch DenseFlipout to avoid lambda serialization warnings
        def _patched_add_variable(self, name, shape, dtype, initializer, trainable, **kwargs):
            return self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
        tfp_layers.DenseFlipout.add_variable = _patched_add_variable
        
        # Initialize a KL weight variable (used in the loss via our custom loss function)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0; target kl_weight:", KL_WEIGHT)
        
        # Create Bayesian DenseFlipout layer using the exact functions from the transformer plugin.
        DenseFlipout = tfp.layers.DenseFlipout
        print("DEBUG: Creating DenseFlipout final layer with units:", time_horizon)
        bayesian_output = DenseFlipout(
            units=time_horizon,
            activation='linear',
            kernel_posterior_fn=posterior_mean_field_custom,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
            name="bayesian_output"
        )(residual)
        
        # Deterministic bias layer
        bias_layer = Dense(
            units=time_horizon,
            activation='linear',
            kernel_initializer=random_normal_initializer_44,
            name="deterministic_bias",
            kernel_regularizer=l2(l2_reg)
        )(residual)
        outputs = Add(name="final_output")([bayesian_output, bias_layer])
        print("DEBUG: Final outputs shape after adding bias:", outputs.shape)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='NBeatsModel')
        print("DEBUG: Model created. Input shape:", self.model.input_shape, "Output shape:", self.model.output_shape)
        
        # Compile with custom loss (Huber + MMD). We pass the current KL weight and set sigma=1.0.
        self.model.compile(
            optimizer=Adam(learning_rate=self.params.get("learning_rate", 0.0001)),
            loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, mmd_lambda=KL_WEIGHT),
            metrics=['mae']
        )
        print("DEBUG: Adam optimizer created with learning_rate:", self.params.get("learning_rate", 0.0001))
        print("DEBUG: Model compiled with custom loss (Huber + MMD) and metrics=['mae']")
        self.model.summary()

    # --- Training Function with Enhanced Callbacks ---
    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the N-BEATS model using the enhanced training regime.
        Incorporates KL annealing, MMD logging, early stopping, and ReduceLROnPlateau.
        """
        # Ensure x_train and x_val are in the proper shape.
        if isinstance(x_train, tuple): x_train = x_train[0]
        if isinstance(x_val, tuple): x_val = x_val[0]
        
        # Set up KL annealing and MMD logging callbacks.
        mmd_lambda = self.params.get('kl_weight', 1e-3)
        self.kl_weight = tf.Variable(mmd_lambda, trainable=False, dtype=tf.float32, name='kl_weight')
        
        class KLAnnealingCallback(tf.keras.callbacks.Callback):
            def __init__(self, plugin, target_kl, anneal_epochs):
                super().__init__()
                self.plugin = plugin
                self.target_kl = target_kl
                self.anneal_epochs = anneal_epochs
            def on_epoch_begin(self, epoch, logs=None):
                new_kl = self.target_kl * min(1.0, (epoch + 1) / self.anneal_epochs)
                self.plugin.kl_weight_var.assign(new_kl)
                print(f"DEBUG: Epoch {epoch+1}: KL weight updated to {new_kl}")
        
        class MMDLoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, plugin, x_train, y_train):
                super().__init__()
                self.plugin = plugin
                self.x_train = x_train
                self.y_train = y_train
            def on_epoch_end(self, epoch, logs=None):
                preds = self.plugin.model(self.x_train, training=True)
                mmd_value = compute_mmd(preds, self.y_train)
                print(f"                                        MMD Lambda = {self.plugin.kl_weight.numpy():.6f}, MMD Loss = {mmd_value.numpy():.6f}")
        
        anneal_epochs = config.get("kl_anneal_epochs", 10)
        target_kl = self.params.get('kl_weight', 1e-3)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        mmd_logging_callback = MMDLoggingCallback(self, x_train, y_train)
        
        min_delta = config.get("min_delta", 1e-4)
        early_stopping_monitor = EarlyStoppingWithPatienceCounter(
            monitor='val_loss',
            patience=config.get('early_patience', 10),
            restore_best_weights=True,
            verbose=2,
            min_delta=min_delta
        )
        reduce_lr_patience = max(1, config.get('early_patience', 10) // 3)
        reduce_lr_monitor = ReduceLROnPlateauWithCounter(
            monitor='val_loss',
            factor=0.1,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
        callbacks = [kl_callback, mmd_logging_callback, early_stopping_monitor, reduce_lr_monitor, ClearMemoryCallback()]
        
        history = self.model.fit(
            x_train, y_train[0],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val[0]),
            shuffle=True,
            callbacks=callbacks,
            verbose=1
        )
        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")
        return history, self.model.predict(x_train, batch_size=batch_size), np.zeros_like(self.model.predict(x_train, batch_size=batch_size)), self.model.predict(x_val, batch_size=batch_size), np.zeros_like(self.model.predict(x_val, batch_size=batch_size))

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Generates predictions with uncertainty estimates using Monte Carlo sampling.
        """
        predictions = np.array([self.model(x_test, training=True).numpy() for _ in range(mc_samples)])
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        print("DEBUG: Mean predictions shape:", mean_predictions.shape)
        print("DEBUG: Uncertainty estimates shape:", uncertainty_estimates.shape)
        return mean_predictions, uncertainty_estimates

    def predict(self, x):
        return self.model.predict(x)

    def save(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={
            'custom_loss': custom_loss,
            'compute_mmd': compute_mmd
        })
        print(f"Model loaded from {file_path}")
