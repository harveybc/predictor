#!/usr/bin/env python
"""
Enhanced Multi-Branch Predictor Plugin using Keras for forecasting EUR/USD returns.

This plugin uses decomposed signals (Trend, Seasonal, Noise) as input channels.
The architecture has parallel branches for each channel, concatenated results,
and multiple output heads using Bayesian layers for uncertainty estimation.
NO feedback loops are implemented in this version.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Lambda, Add, Identity
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LambdaCallback
# from tensorflow.keras.losses import Huber # Original mentioned Huber, but code uses MSE
from tensorflow.keras.losses import MeanSquaredError # Use MSE as implemented
import tensorflow.keras.backend as K
import gc
import os
from sklearn.metrics import r2_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform # Not used in current implementation, but kept import

# Removed global feedback/control variable definitions

# ---------------------------
# Custom Callbacks (Modified for informative prints)
# ---------------------------

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    """Custom ReduceLROnPlateau callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        # Store current LR before super call modifies it
        current_lr = K.get_value(self.model.optimizer.learning_rate)
        super().on_epoch_end(epoch, logs) # Let superclass handle LR reduction logic
        self.patience_counter = self.wait # Superclass updates 'wait' counter
        # Print counter *after* super call updates it
        print(f"Epoch {epoch+1}: ReduceLROnPlateau patience counter: {self.patience_counter}/{self.patience}")
        new_lr = K.get_value(self.model.optimizer.learning_rate)
        if new_lr < current_lr:
             print(f"Epoch {epoch+1}: ReduceLROnPlateau reduced learning rate to {new_lr:.7f}.")


class EarlyStoppingWithPatienceCounter(EarlyStopping):
    """Custom EarlyStopping callback that prints the patience counter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs) # Let superclass handle stopping logic
        self.patience_counter = self.wait # Superclass updates 'wait' counter
        # Print counter *after* super call updates it
        print(f"Epoch {epoch+1}: EarlyStopping patience counter: {self.patience_counter}/{self.patience}")


# ---------------------------
# Custom Metrics and Helper Functions
# ---------------------------
def mae_magnitude(y_true, y_pred):
    """Compute MAE on the prediction output (assumed single value)."""
    # Reshape based on typical Keras usage during training/evaluation
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, 1])
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def r2_metric(y_true, y_pred):
    """Compute R² metric on the prediction output."""
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, 1])
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

@tf.function # Decorate MMD for potential graph optimization
def compute_mmd(x, y, sigma=1.0, sample_size=256):
    """Compute the Maximum Mean Discrepancy (MMD) between two samples."""
    # Ensure inputs are float32
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # Sample efficiently if needed
    x_len = tf.shape(x)[0]
    y_len = tf.shape(y)[0]
    sample_size_tf = tf.cast(sample_size, tf.int32) # Ensure sample_size is tensor/int

    idx_x = tf.random.shuffle(tf.range(x_len))[:sample_size_tf]
    idx_y = tf.random.shuffle(tf.range(y_len))[:sample_size_tf]
    x_sample = tf.gather(x, idx_x)
    y_sample = tf.gather(y, idx_y)

    # Define Gaussian kernel using TensorFlow operations
    @tf.function
    def gaussian_kernel(a, b, sigma):
        a = tf.expand_dims(a, 1) # (N, 1, D)
        b = tf.expand_dims(b, 0) # (1, M, D)
        # Ensure dimensions match for subtraction if D > 1
        dist_sq = tf.reduce_sum(tf.square(a - b), axis=-1) # (N, M)
        # Ensure sigma is float32 and non-zero
        sigma_sq = tf.square(tf.cast(sigma, tf.float32)) + tf.keras.backend.epsilon()
        return tf.exp(-dist_sq / (2.0 * sigma_sq))

    # Calculate kernel matrices
    K_xx = gaussian_kernel(x_sample, x_sample, sigma)
    K_yy = gaussian_kernel(y_sample, y_sample, sigma)
    K_xy = gaussian_kernel(x_sample, y_sample, sigma)

    # Calculate MMD squared statistic
    mmd2 = tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2 * tf.reduce_mean(K_xy)
    # Return non-negative MMD value
    return tf.maximum(mmd2, 0.0)


# --- Composite Loss Function (Simplified - No Feedback Updates/Control) ---
# Defined globally as requested
def composite_loss(y_true, y_pred,
                   # Required arguments:
                   head_index, # Still needed for potential future use or logging, but not for updates
                   mmd_lambda,
                   sigma
                   # Removed args: p, i, d, list_last_signed_error, ... list_local_feedback
                   ):
    """
    Global composite loss function for a specific head. NO feedback updates.
    Returns the scalar loss value (MSE + Asymptote + MMD).
    """
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, 1])
    mag_true = y_true
    mag_pred = y_pred

    # --- Calculate Primary Losses ---
    mse_loss_val = tf.keras.losses.MeanSquaredError()(mag_true, mag_pred)
    mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
    mse_min = tf.maximum(mse_loss_val, 1e-10)

    # --- Calculate Summary Statistics ---
    signed_avg_pred = tf.reduce_mean(mag_pred)
    signed_avg_true = tf.reduce_mean(mag_true)

    # --- Calculate Dynamic Asymptote Penalty (Original User Logic) ---
    def vertical_dynamic_asymptote(value, center):
        # This nested function needs access to mse_loss_val from the outer scope
        res = tf.cond(tf.greater_equal(value, center),
            lambda: 3*tf.math.log(tf.abs(value - center) + 1e-9)+20,
            lambda: mse_loss_val*1e3 - 1)
        res = tf.cond(tf.greater_equal(center, value),
            lambda: mse_loss_val*1e3 - 1,
            lambda: 3*tf.math.log(tf.abs(value - center) + 1e-9)+20)
        return res
    asymptote = vertical_dynamic_asymptote(signed_avg_pred, signed_avg_true)
    # asymptote = 0.0 # Option to disable asymptote if needed

    # --- REMOVED Feedback Metrics Calculation ---
    # --- REMOVED Control Function Call ---
    # --- REMOVED Update Feedback Variables ---

    # Calculate final loss term directly
    total_loss = 1e4 * mse_min + asymptote + mmd_lambda * mmd_loss_val

    return total_loss


# --- Removed global dummy_feedback_control function ---


# --- Named initializer ---
def random_normal_initializer_44(shape, dtype=None):
    """Initializes with random normal distribution."""
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

# --- Bayesian Helper Functions (Defined globally as requested) ---
def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
    """Custom posterior distribution function for DenseFlipout kernel."""
    if not isinstance(name, str): name = None
    bias_size = 0 # No bias in kernel posterior
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.)) # Constant for softplus transformation
    loc_name = f"{name}_loc" if name else "posterior_loc"
    scale_name = f"{name}_scale" if name else "posterior_scale"
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42), dtype=dtype, trainable=trainable, name=loc_name)
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43), dtype=dtype, trainable=trainable, name=scale_name)
    scale = 1e-3 + tf.nn.softplus(scale + c) # Softplus transformation with offset and floor
    scale = tf.clip_by_value(scale, 1e-3, 1.0) # Clip scale

    # Debug prints (optional, removed .name access)
    # print(f"DEBUG: posterior: created loc shape: {loc.shape}")
    # print(f"DEBUG: posterior: created scale shape: {scale.shape}")

    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
    except Exception as e:
        print(f"ERROR: Exception during reshape in posterior (name={name}):", e)
        raise e
    # Return Independent Normal distribution
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape) # Match kernel rank
    )

def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    """Custom prior distribution function for DenseFlipout kernel."""
    # print(f"DEBUG: prior_fn (name={name}):") # Optional debug
    # print(f"       dtype={dtype}, kernel_shape={kernel_shape}, bias_size={bias_size} (overridden to 0), trainable={trainable}") # Optional debug
    if not isinstance(name, str): name = None
    bias_size = 0 # No bias in kernel prior
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype) # Prior mean is zero
    scale = tf.ones([n], dtype=dtype) # Prior scale is one
    # print(f"DEBUG: prior_fn: computed n={n}") # Optional debug
    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
        # print(f"DEBUG: prior_fn: reshaped loc to {loc_reshaped.shape} and scale to {scale_reshaped.shape}") # Optional debug
    except Exception as e:
        print(f"ERROR: Exception during reshape in prior_fn (name={name}):", e)
        raise e
    # Return Independent Normal distribution
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape) # Match kernel rank
    )


# ---------------------------
# Predictor Plugin Class Definition (Simplified)
# ---------------------------
class Plugin:
    """
    Multi-Branch Bayesian Predictor Plugin without feedback loops.
    Uses STL-decomposed features, processes channels in parallel, merges,
    and uses multiple Bayesian output heads.
    """
    # Default parameters (can be overridden by config)
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 2, # Renamed from num_branch_layers
        'intermediate': 2,      # Renamed from num_head_intermediate_layers
        'branch_units': 32,
        'merged_units': 64,
        'learning_rate': 0.0001,
        'activation': 'relu',
        'l2_reg': 1e-5,
        'mmd_lambda': 1e-3,
        'sigma_mmd': 1.0,
        'predicted_horizons': [1, 6, 12], # Example
        'kl_weight': 1e-3, # Default target KL weight
        'kl_anneal_epochs': 10,
        # Parameters for callbacks
        'min_delta': 1e-4,
        'early_patience': 10,
        'start_from_epoch': 10,
        'reduce_lr_patience': None # Default calculated from early_patience
    }
    # Debug vars can be adjusted if needed
    plugin_debug_vars = ['batch_size', 'intermediate_layers', 'intermediate', 'branch_units', 'merged_units', 'learning_rate', 'l2_reg', 'mmd_lambda']

    def __init__(self, config=None):
        """Initialize the predictor plugin."""
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config) # Update defaults with provided config

        # Basic checks for essential config items
        if 'predicted_horizons' not in self.params:
             raise ValueError("Config must contain 'predicted_horizons' list.")
        if not isinstance(self.params['predicted_horizons'], list) or not self.params['predicted_horizons']:
             raise ValueError("'predicted_horizons' must be a non-empty list.")

        self.model = None
        self.output_names = [] # Will be populated in build_model
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')

        # REMOVED initialization of feedback/control lists (local_p_control, local_feedback, last_xxx)

        print("Predictor Plugin Initialized (No feedback state).")

        # --- Apply DenseFlipout Patch (Essential) ---
        # Applying here ensures it's done once per instance creation
        if not hasattr(tfp.layers.DenseFlipout, '_already_patched_add_variable'):
            def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
                return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
            tfp.layers.DenseFlipout.add_variable = _patched_add_variable
            tfp.layers.DenseFlipout._already_patched_add_variable = True
            # print("DenseFlipout patched successfully in __init__.") # Informative print
        # else:
            # print("DenseFlipout already patched.") # Informative print


    def set_params(self, **kwargs):
        """Update predictor plugin parameters."""
        self.params.update(kwargs)

    def get_debug_info(self):
        """Return debug information for the predictor plugin."""
        # Return relevant parameters currently set
        debug_keys = ['batch_size', 'intermediate_layers', 'intermediate', 'branch_units',
                      'merged_units', 'learning_rate', 'l2_reg', 'mmd_lambda', 'predicted_horizons']
        return {k: self.params.get(k, 'N/A') for k in debug_keys}

    def add_debug_info(self, debug_info):
        """Add predictor plugin debug information to the given dictionary."""
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape, x_train, config):
        """
        Build the multi-output model without feedback inputs to heads.
        """
        # Update instance params with potentially new config from pipeline
        self.params.update(config)
        config = self.params # Use merged config

        window_size, num_channels = input_shape
        predicted_horizons = config['predicted_horizons']
        num_outputs = len(predicted_horizons)

        # --- Get Parameters ---
        l2_reg = config.get("l2_reg", 0.001)
        activation = config.get("activation", "relu")
        num_intermediate_layers = config['intermediate_layers']
        num_head_intermediate_layers = config['intermediate'] # Use 'intermediate' key
        branch_units = config.get("branch_units", 64)
        merged_units = config.get("merged_units", 128)

        # --- Input Layer ---
        inputs = Input(shape=(window_size, num_channels), name="input_layer")

        # --- Parallel Feature Processing Branches ---
        feature_branch_outputs = []
        for c in range(num_channels):
            feature_input = Lambda(lambda x, channel=c: x[:, :, channel:channel+1],
                                   name=f"feature_{c+1}_input")(inputs)
            x = Flatten(name=f"feature_{c+1}_flatten")(feature_input)
            for i in range(num_intermediate_layers):
                x = Dense(branch_units, activation=activation, kernel_regularizer=l2(l2_reg),
                          name=f"feature_{c+1}_dense_{i+1}")(x)
            feature_branch_outputs.append(x)

        # --- Merging Feature Branches ONLY ---
        if len(feature_branch_outputs) == 1:
             merged = Identity(name="merged_features")(feature_branch_outputs[0])
        elif len(feature_branch_outputs) > 1:
             merged = Concatenate(name="merged_features")(feature_branch_outputs)
        else:
             raise ValueError("Model requires at least one input feature channel.")

        # --- Define Bayesian Layer Components ---
        KL_WEIGHT = self.kl_weight_var
        DenseFlipout = tfp.layers.DenseFlipout

        # --- Build Multiple Output Heads ---
        outputs_list = []
        self.output_names = [] # Reset output names list

        for i, horizon in enumerate(predicted_horizons):
            branch_suffix = f"_h{horizon}"

            # --- REMOVED Get Control Action Feedback ---
            # --- REMOVED Concatenate Merged Features + Feedback ---

            # --- Head Intermediate Dense Layers ---
            # Input is now directly the 'merged' tensor
            head_dense_output = merged
            for j in range(num_head_intermediate_layers):
                 head_dense_output = Dense(merged_units, activation=activation, kernel_regularizer=l2(l2_reg),
                                           name=f"head_dense_{j+1}{branch_suffix}")(head_dense_output)

            # --- Bayesian / Bias Layers ---
            flipout_layer_name = f"bayesian_flipout_layer{branch_suffix}"
            flipout_layer_branch = DenseFlipout(
                units=1, activation='linear',
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flipout_layer_name: posterior_mean_field_custom(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flipout_layer_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT, name=flipout_layer_name
            )
            bayesian_output_branch = Lambda(
                lambda t: flipout_layer_branch(t),
                output_shape=lambda s: (s[0], 1), # Specify output shape
                name=f"bayesian_output{branch_suffix}"
            )(head_dense_output)

            bias_layer_branch = Dense(units=1, activation='linear', kernel_initializer=random_normal_initializer_44,
                                      name=f"deterministic_bias{branch_suffix}")(head_dense_output)

            # --- Final Head Output ---
            output_name = f"output_horizon_{horizon}"
            final_branch_output = Add(name=output_name)([bayesian_output_branch, bias_layer_branch])

            outputs_list.append(final_branch_output)
            self.output_names.append(output_name) # Store the generated name
            # --- End of Head ---

        # --- Model Definition ---
        self.model = Model(inputs=inputs, outputs=outputs_list, name=f"NoFeedbackPredictor_{len(predicted_horizons)}H")

        # --- Compilation (Using simplified GLOBAL composite_loss) ---
        optimizer = AdamW(learning_rate=config.get("learning_rate", 0.001))
        mmd_lambda = config.get("mmd_lambda", 0.1)
        sigma_mmd = config.get("sigma_mmd", 1.0)

        # Prepare loss dictionary calling the simplified global composite_loss
        loss_dict = {}
        for i, name in enumerate(self.output_names):
            # Lambda only needs to capture index and params for loss calculation itself
            loss_fn_for_head = (
                lambda index=i:
                    lambda y_true, y_pred: composite_loss( # Call GLOBAL simplified func
                        y_true, y_pred,
                        head_index=index,       # Pass index
                        mmd_lambda=mmd_lambda,  # Pass config value
                        sigma=sigma_mmd        # Pass config value
                        # REMOVED: p, i, d, and all list arguments
                    )
            )(i)
            loss_dict[name] = loss_fn_for_head

        # Prepare metrics dictionary
        metrics_dict = {name: [mae_magnitude, r2_metric] for name in self.output_names}

        self.model.compile(optimizer=optimizer,
                           loss=loss_dict,
                           metrics=metrics_dict)

        print(f"No-Feedback Predictor model built successfully for {num_outputs} horizons.")
        self.model.summary(line_length=150)


    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the multi-output model (no feedback version).
        Expects y_train/y_val dictionaries. Calculates metrics for plotted_horizon.
        """
        # Update instance params with potentially new config from pipeline
        self.params.update(config)
        config = self.params # Use merged config

        # --- Configuration Validation ---
        if 'predicted_horizons' not in config: raise ValueError("'predicted_horizons' missing.")
        if 'plotted_horizon' not in config: raise ValueError("'plotted_horizon' missing.")
        predicted_horizons = config['predicted_horizons']
        plotted_horizon = config['plotted_horizon']
        if plotted_horizon not in predicted_horizons: raise ValueError(f"'{plotted_horizon=}' not in {predicted_horizons=}.")
        try:
            plotted_index = predicted_horizons.index(plotted_horizon)
        except ValueError: raise ValueError(f"Index error for {plotted_horizon=}.")

        # --- KL Annealing Callback Definition ---
        class KLAnnealingCallback(tf.keras.callbacks.Callback):
            def __init__(self, plugin, target_kl, anneal_epochs):
                super().__init__()
                self.plugin = plugin; self.target_kl = target_kl; self.anneal_epochs = anneal_epochs
            def on_epoch_begin(self, epoch, logs=None):
                new_kl = self.target_kl * min(1.0, (epoch + 1) / self.anneal_epochs)
                self.plugin.kl_weight_var.assign(new_kl)

        # --- Setup Callbacks ---
        anneal_epochs = config.get("kl_anneal_epochs", 10)
        target_kl = self.params.get('kl_weight', 1e-3) # Use kl_weight from params
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        min_delta_es = config.get("min_delta", 1e-4)
        patience_es = config.get('early_patience', 10)
        start_epoch_es = config.get('start_from_epoch', 10)
        patience_lr = config.get("reduce_lr_patience", max(1, int(patience_es / 4)))

        # Assumes relevant Callback classes are imported/defined
        callbacks = [
            EarlyStoppingWithPatienceCounter(monitor='val_loss', patience=patience_es, restore_best_weights=True, verbose=1, start_from_epoch=start_epoch_es, min_delta=min_delta_es),
            ReduceLROnPlateauWithCounter(monitor="val_loss", factor=0.5, patience=patience_lr, verbose=1),
            # Ensure K is imported if using LambdaCallback with K
            LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}") if K else None),
            # REMOVED ClearMemoryCallback
            kl_callback
        ]

        # --- Input Data Verification ---
        if not isinstance(y_train, dict) or not isinstance(y_val, dict): raise TypeError("y_train/y_val must be dictionaries.")
        if not hasattr(self, 'output_names') or not self.output_names: raise AttributeError("self.output_names not set.")
        plotted_output_name = f"output_horizon_{plotted_horizon}"
        if plotted_output_name not in y_train or plotted_output_name not in y_val: raise ValueError(f"Target dicts missing key: '{plotted_output_name}'")
        if set(y_train.keys()) != set(self.output_names) or set(y_val.keys()) != set(self.output_names): print("WARN: Target keys may not match all output names.")

        # --- Model Training ---
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)

        # --- Post-Training Predictions ---
        list_train_preds = self.model.predict(x_train, batch_size=batch_size)
        list_val_preds = self.model.predict(x_val, batch_size=batch_size)
        # Ensure predictions are lists
        if not isinstance(list_train_preds, list): list_train_preds = [list_train_preds]
        if not isinstance(list_val_preds, list): list_val_preds = [list_val_preds]


        # Placeholder uncertainties
        list_train_uncertainty = [np.zeros_like(preds) for preds in list_train_preds]
        list_val_uncertainty = [np.zeros_like(preds) for preds in list_val_preds]

        # --- Post-Training Metrics (for 'plotted_horizon') ---
        # Assumes self.calculate_mae/r2 methods exist
        try:
            y_train_plotted = y_train[plotted_output_name]
            # Check if list_train_preds index exists before accessing
            if plotted_index < len(list_train_preds):
                 train_preds_plotted = list_train_preds[plotted_index]
                 # print(f"Calculating final MAE/R2 for plotted horizon: {plotted_horizon} (Index: {plotted_index})") # Informative print
                 if hasattr(self, 'calculate_mae') and callable(self.calculate_mae):
                     self.calculate_mae(y_train_plotted, train_preds_plotted)
                 if hasattr(self, 'calculate_r2') and callable(self.calculate_r2):
                     self.calculate_r2(y_train_plotted, train_preds_plotted)
            else:
                 print(f"ERROR: Plotted index {plotted_index} out of bounds for train predictions list (len {len(list_train_preds)}).")

        except Exception as e:
             print(f"ERROR during post-training metric calculation: {e}")

        return history, list_train_preds, list_train_uncertainty, list_val_preds, list_val_uncertainty


    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Performs Monte Carlo dropout predictions (incremental calculation).
        """
        if self.model is None: raise ValueError("Model not built/loaded.")
        if mc_samples <= 0: return [], []
        try:
            first_run_output_tf = self.model(x_test[:1], training=True)
            if not isinstance(first_run_output_tf, list): first_run_output_tf = [first_run_output_tf]
            num_heads = len(first_run_output_tf)
            if num_heads == 0: return [], []
            first_head_output = first_run_output_tf[0].numpy()
            num_test_samples = x_test.shape[0]
            output_dim = first_head_output.shape[1] if first_head_output.ndim > 1 else 1
        except Exception as e: raise ValueError("Could not determine model output structure.") from e

        means = [np.zeros((num_test_samples, output_dim), dtype=np.float32) for _ in range(num_heads)]
        m2s = [np.zeros((num_test_samples, output_dim), dtype=np.float32) for _ in range(num_heads)]
        counts = [0] * num_heads

        for i in range(mc_samples):
            head_outputs_tf = self.model(x_test, training=True)
            if not isinstance(head_outputs_tf, list): head_outputs_tf = [head_outputs_tf]
            if len(head_outputs_tf) != num_heads: continue # Skip if output structure inconsistent

            for h in range(num_heads):
                head_output_np = head_outputs_tf[h].numpy()
                if head_output_np.ndim == 1: head_output_np = np.expand_dims(head_output_np, axis=-1)
                if head_output_np.shape != (num_test_samples, output_dim): continue # Skip inconsistent shape

                counts[h] += 1
                delta = head_output_np - means[h]
                means[h] += delta / counts[h]
                delta2 = head_output_np - means[h]
                m2s[h] += delta * delta2

        list_mean_predictions = means
        list_uncertainty_estimates = []
        for h in range(num_heads):
             variance = m2s[h] / (counts[h] - 1) if counts[h] >= 2 else np.full((num_test_samples, output_dim), np.nan, dtype=np.float32)
             stddev = np.sqrt(np.maximum(variance, 0))
             list_uncertainty_estimates.append(stddev.astype(np.float32))

        return list_mean_predictions, list_uncertainty_estimates


    def save(self, file_path):
        """Saves the Keras model."""
        if self.model:
            self.model.save(file_path)
            # print(f"Model saved to {file_path}") # Informative print
        else:
            print("WARN: No model available to save.")

    def load(self, file_path):
        """Loads a Keras model with custom objects."""
        # Define custom objects needed for loading
        custom_objects = {
            # Note: Need to provide the GLOBAL composite_loss correctly, possibly via wrapper
            # Keras might struggle to deserialize the lambda used during compile.
            # Simplest is often to re-compile after loading if loss args changed.
            # "composite_loss": composite_loss, # May cause issues due to arg changes
            "compute_mmd": compute_mmd,
            "r2_metric": r2_metric,
            "mae_magnitude": mae_magnitude,
            # Add Bayesian layers if they are custom classes, TFP layers usually handled
            # 'DenseFlipout': tfp.layers.DenseFlipout # Example if needed explicitly
        }
        # It's often safer to load structure and weights separately, or re-compile
        try:
            self.model = load_model(file_path, custom_objects=custom_objects, compile=False) # Load without compiling optimizer/loss
            print(f"Predictor model structure loaded from {file_path}. Re-compile if needed.")
            # Optionally re-compile here if config is available
            # self.compile_model(self.params) # Assuming a helper method compile_model exists
        except Exception as e:
             print(f"ERROR loading model from {file_path}: {e}")
             self.model = None


    # Keep calculate_mae and calculate_r2 as they were (with corrected reshape logic)
    def calculate_mae(self, y_true, y_pred):
        """Calculates MAE between true and predicted values."""
        # Ensure inputs are numpy arrays and flattened/reshaped appropriately
        try:
             y_true_np = np.reshape(y_true, (-1, 1))
             y_pred_np = np.reshape(y_pred, (-1, 1))
             # Ensure they have same length
             min_len = min(len(y_true_np), len(y_pred_np))
             mae = np.mean(np.abs(y_true_np[:min_len] - y_pred_np[:min_len]))
             print(f"Calculated MAE: {mae:.6f}")
             return mae
        except Exception as e:
             print(f"ERROR calculating MAE: {e}")
             return np.nan

    def calculate_r2(self, y_true, y_pred):
        """Calculates R-squared between true and predicted values."""
        try:
             y_true_np = np.reshape(y_true, (-1, 1))
             y_pred_np = np.reshape(y_pred, (-1, 1))
             min_len = min(len(y_true_np), len(y_pred_np))
             y_true_np = y_true_np[:min_len]
             y_pred_np = y_pred_np[:min_len]
             # print(f"Calculating R²: y_true shape={y_true_np.shape}, y_pred shape={y_pred_np.shape}") # Informative print
             r2 = r2_score(y_true_np, y_pred_np)
             print(f"Calculated R²: {r2:.4f}")
             return r2
        except Exception as e:
             print(f"ERROR calculating R2: {e}")
             return np.nan

# Removed __main__ block for plugin file