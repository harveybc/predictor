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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add

# Denine TensorFlow global variables(used from the composite loss function):
last_mae = tf.Variable(1.0, trainable=False, dtype=tf.float32) # last mean absolute error from the composite loss function to use in global feedback
last_std = tf.Variable(0.0, trainable=False, dtype=tf.float32) # last standard deviation from the composite loss function to use in global feedback
p_control=tf.Variable(0, trainable=False, dtype=tf.float32) # proportional control for the feedback
d_control=tf.Variable(1, trainable=False, dtype=tf.float32) # derivative control for the feedback
i_control=tf.Variable(1, trainable=False, dtype=tf.float32) # integral control for the feedback
global_feedback = tf.Variable(0.0, trainable=False, dtype=tf.float32) # global feedback variable
local_feedback = [] # list of local feedback variables for each output head 


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
    mse_min = tf.cond(tf.greater(mse_loss_val,1e-10),
            lambda: mse_loss_val,
            lambda: 1e-10)
        


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
    
    def vertical_right_asymptote(value, center):
        res = tf.cond(tf.greater_equal(value, center),
            lambda: 3*tf.math.log(tf.abs(value - center))+20,
            lambda: mse_loss_val*1e3 - 1 # best 1e6
        )
        res = tf.cond(tf.greater_equal(center, value),
            lambda: mse_loss_val*1e3 - 1, # best 1e6,
            lambda: 3*tf.math.log(tf.abs(value - center))+20        
        )   
           
        return res   
    
    # --- Additional feedback values ---
    # Compute a divisor to avoid division by very small numbers.
    divisor = tf.cond(
        tf.greater(abs_avg_true, tf.constant(1e-10, dtype=tf.float32)),
        lambda: abs_avg_true,
        lambda: tf.constant(1e-10, dtype=tf.float32)
    )
    
    # Calculate batch-level feedback values.
    batch_signed_error = 0.0 * signed_avg_error / divisor
    batch_std = 0.0 * tf.math.reduce_mean(tf.abs(mag_true - mag_pred)) / divisor
    
    #calcualte the vertical left asymptote
    asymptote = vertical_right_asymptote(signed_avg_pred, signed_avg_true)

    # Update global variables last_mae and last_std with control dependencies.
    with tf.control_dependencies([last_mae.assign(batch_signed_error)]):
        #total_loss = reward + penalty + 3e6*mae_loss_val+ 3e8*mse_loss_val + mmd_lambda * mmd_loss_val
        total_loss = 1e4*mse_min + asymptote + mmd_lambda * mmd_loss_val
        #total_loss = 1e2*mse_loss_val + slope
    with tf.control_dependencies([last_std.assign(batch_std)]):
        #total_loss = reward + penalty + 3e6*mae_loss_val+ 3e8*mse_loss_val + mmd_lambda * mmd_loss_val
        total_loss = 1e4*mse_min + asymptote + mmd_lambda * mmd_loss_val#best 1e3
        #total_loss = 1e2*mse_loss_val + slope
    return total_loss

# --- Named initializer to avoid lambda serialization warnings ---
def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

# Build Dense branches for trend, seasonal, and noise channels.
def build_branch(branch_input, branch_name, num_branch_layers=2, branch_units=32, activation='relu', l2_reg=1e-5):
    x = Flatten(name=f"{branch_name}_flatten")(branch_input)
    for i in range(num_branch_layers):
        x = Dense(branch_units, activation=activation,
                kernel_regularizer=l2(l2_reg),
                name=f"{branch_name}_dense_{i+1}")(x)
    return x

def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
    """Custom posterior distribution function for DenseFlipout kernel."""
    print(f"DEBUG: posterior_mean_field_custom (name={name}):")
    print(f"       dtype={dtype}, kernel_shape={kernel_shape}, bias_size={bias_size} (overridden to 0), trainable={trainable}")
    if not isinstance(name, str): name = None # Ensure name is string or None
    bias_size = 0 # Force bias size to 0 for kernel posterior
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.))
    # Use unique variable names based on the layer name if provided
    loc_name = f"{name}_loc" if name else "posterior_loc"
    scale_name = f"{name}_scale" if name else "posterior_scale"
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42), dtype=dtype, trainable=trainable, name=loc_name)
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43), dtype=dtype, trainable=trainable, name=scale_name)
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)
    print(f"DEBUG: posterior: created loc name: {loc.name}, shape: {loc.shape}")
    print(f"DEBUG: posterior: created scale name: {scale.name}, shape: {scale.shape}")
    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
        print(f"DEBUG: posterior: reshaped loc to {loc_reshaped.shape} and scale to {scale_reshaped.shape}")
    except Exception as e:
        print(f"DEBUG: Exception during reshape in posterior (name={name}):", e)
        raise e
    # Ensure reinterpreted_batch_ndims matches the rank of the kernel shape
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape)
    )

def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    """Custom prior distribution function for DenseFlipout kernel."""
    print(f"DEBUG: prior_fn (name={name}):")
    print(f"       dtype={dtype}, kernel_shape={kernel_shape}, bias_size={bias_size} (overridden to 0), trainable={trainable}")
    if not isinstance(name, str): name = None # Ensure name is string or None
    bias_size = 0 # Force bias size to 0 for kernel prior
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    print(f"DEBUG: prior_fn: computed n={n}")
    try:
        loc_reshaped = tf.reshape(loc, kernel_shape)
        scale_reshaped = tf.reshape(scale, kernel_shape)
        print(f"DEBUG: prior_fn: reshaped loc to {loc_reshaped.shape} and scale to {scale_reshaped.shape}")
    except Exception as e:
        print(f"DEBUG: Exception during reshape in prior_fn (name={name}):", e)
        raise e
    # Ensure reinterpreted_batch_ndims matches the rank of the kernel shape
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_reshaped, scale=scale_reshaped),
        reinterpreted_batch_ndims=len(kernel_shape)
    )

# ---------------------------
# Multi-Branch Predictor Plugin Class Definition
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
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0")

    # Apply this patch once globally or within your class initialization (__init__)
    # before the build_model method is ever called.
    if not hasattr(tfp.layers.DenseFlipout, '_already_patched_add_variable'):
        def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
            # Use layer_instance instead of self
            return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
        tfp.layers.DenseFlipout.add_variable = _patched_add_variable
        tfp.layers.DenseFlipout._already_patched_add_variable = True # Mark as patched
        print("DEBUG: DenseFlipout patched successfully.")
    else:
        print("DEBUG: DenseFlipout already patched.")

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
        Build a multi-input-branch, multi-output model with global and local feedback.
        Input features and global feedback are processed in parallel branches.
        Their merged output, combined with specific local feedback for each head,
        feeds into multiple identical output heads (Dense layers + Bayesian/Bias layers).

        Args:
            input_shape (tuple): Expected input shape (window_size, num_channels).
            x_train (np.ndarray): Training data (can be None if not used for shape inference).
            config (dict): Configuration parameters including:
                        'predicted_horizons' (list): Output horizons.
                        'intermediate_layers' (int): Num dense layers in input/global feedback branches.
                        'intermediate' (int): Num dense layers within each output head before Bayesian/Bias.
                        'branch_units' (int): Units in input/global feedback branch dense layers.
                        'merged_units' (int): Units in head intermediate dense layers.
                        'activation' (str): Activation function.
                        'l2_reg' (float): L2 regularization factor.
                        'learning_rate' (float): Optimizer learning rate.
                        'mmd_lambda' (float): Lambda for MMD component in loss (if used).
        """
        # --- Assume necessary imports are done ---
        # import tensorflow as tf
        # from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Flatten, Add
        # from tensorflow.keras.models import Model
        # from tensorflow.keras.regularizers import l2
        # from tensorflow.keras.optimizers import AdamW # Or your specific optimizer
        # import tensorflow_probability as tfp
        # import numpy as np

        # --- Assume helper functions/initializers/variables are defined/accessible ---
        # - posterior_mean_field_custom, prior_fn (for Bayesian layer)
        # - composite_loss, mae_magnitude, r2_metric (for compilation)
        # - random_normal_initializer_44 (for bias layer)
        # - global_feedback (tf.Variable or Tensor, potentially multi-channel)
        # - local_feedback (Python list of tf.Variables or Tensors, one per output head)
        # - self.params (dict for default values, though direct config access is used more now)
        # - self.kl_weight_var (tf.Variable for Bayesian KL weight)
        # - DenseFlipout monkey-patch applied elsewhere

        window_size, num_channels = input_shape
        # Get parameters from config, providing defaults if necessary
        l2_reg = config.get("l2_reg", self.params.get("l2_reg", 0.001))
        activation = config.get("activation", self.params.get("activation", "relu"))
        num_intermediate_layers = config['intermediate_layers'] # Num layers for input feature/global feedback branches
        num_head_intermediate_layers = config['intermediate'] # Num layers for output head branches (before bayes/bias)
        branch_units = config.get("branch_units", self.params.get("branch_units", 64)) # Units for input/global feedback branches
        merged_units = config.get("merged_units", self.params.get("merged_units", 128)) # Units for head branches
        predicted_horizons = config['predicted_horizons']
        num_outputs = len(predicted_horizons)

        # --- Input Layer ---
        inputs = Input(shape=(window_size, num_channels), name="input_layer")

        feature_branch_outputs = []
        # --- Parallel Feature Processing Branches (Generalized) ---
        print(f"DEBUG: Building {num_channels} feature processing branches...")
        for c in range(num_channels):
            feature_input = Lambda(lambda x, channel=c: x[:, :, channel:channel+1],
                                name=f"feature_{c+1}_input")(inputs)
            x = Flatten(name=f"feature_{c+1}_flatten")(feature_input)
            for i in range(num_intermediate_layers):
                x = Dense(branch_units, activation=activation,
                        kernel_regularizer=l2(l2_reg),
                        name=f"feature_{c+1}_dense_{i+1}")(x)
            feature_branch_outputs.append(x)
            print(f"  - Branch {c+1} output shape (symbolic): {x.shape}")

        # --- Global Feedback Processing Branch ---
        print("DEBUG: Building global feedback processing branch...")
        # Helper function to safely access and flatten global feedback
        def get_flat_global_feedback(tensor_input):
            batch_size = tf.shape(tensor_input)[0]
            # Assume global_feedback is an accessible tf.Variable or Tensor
            # Use tf.identity to ensure it's treated as a tensor in graph mode
            gf = tf.identity(global_feedback)
            return tf.reshape(gf, [batch_size, -1]) # Flatten

        flat_global_feedback = Lambda(get_flat_global_feedback, name="flatten_global_feedback")(inputs)
        print(f"  - Flattened global feedback shape (symbolic): {flat_global_feedback.shape}")

        # Process flattened global feedback through dense layers
        x_gf = flat_global_feedback
        for i in range(num_intermediate_layers):
            x_gf = Dense(branch_units, activation=activation, # Using same units as feature branches
                        kernel_regularizer=l2(l2_reg),
                        name=f"global_feedback_dense_{i+1}")(x_gf)
        global_feedback_output = x_gf
        print(f"  - Processed global feedback output shape (symbolic): {global_feedback_output.shape}")

        # --- Merging Feature Branches and Global Feedback Branch ---
        all_branch_outputs = feature_branch_outputs + [global_feedback_output]
        merged = Concatenate(name="merged_all_inputs")(all_branch_outputs)
        print(f"DEBUG: Merged all input branches shape (symbolic): {merged.shape}")


        # --- Define Bayesian Layer Components (Assume functions defined elsewhere) ---
        KL_WEIGHT = self.kl_weight_var # Use the class attribute tf.Variable
        DenseFlipout = tfp.layers.DenseFlipout # Alias


        # --- Build Multiple Output Heads ---
        outputs_list = []
        output_names = []
        print(f"DEBUG: Building {num_outputs} output heads...")

        # Helper function to safely get/flatten specific local feedback tensor
        # Defined outside lambda to help with variable capture
        def get_specific_local_feedback(tensor_input, head_index):
            batch_size = tf.shape(tensor_input)[0]
            # Assume local_feedback is python list of tf.Variables/Tensors accessible here
            lf_tensor = tf.identity(local_feedback[head_index]) # Use tf.identity
            return tf.reshape(lf_tensor, [batch_size, -1]) # Flatten

        for i, horizon in enumerate(predicted_horizons):
            branch_suffix = f"_h{horizon}" # Unique suffix for layers in this head
            print(f"  - Building head for Horizon {horizon} (Index {i})...")

            # --- Local Feedback Integration for this Head ---
            # Use Lambda to capture loop index 'i' and pass it to helper
            flattened_local_feedback = Lambda(lambda x: get_specific_local_feedback(x, i),
                                            name=f"flatten_local_fb{branch_suffix}")(inputs)
            print(f"    - H{horizon}: Flattened local feedback shape (symbolic): {flattened_local_feedback.shape}")

            # Add local feedback to the merged input branch data
            head_input_with_feedback = Add(name=f"head_input_with_feedback{branch_suffix}")(
                [merged, flattened_local_feedback] # Add merged data and local feedback
            )
            print(f"    - H{horizon}: Head input combined shape (symbolic): {head_input_with_feedback.shape}")

            # --- Head Intermediate Dense Layers ---
            head_dense_output = head_input_with_feedback # Start with combined input
            print(f"    - H{horizon}: Building {num_head_intermediate_layers} intermediate head dense layers...")
            for j in range(num_head_intermediate_layers):
                head_dense_output = Dense(merged_units, activation=activation, # Using merged_units here
                                        kernel_regularizer=l2(l2_reg),
                                        name=f"head_dense_{j+1}{branch_suffix}")(head_dense_output)
            print(f"    - H{horizon}: Output shape after head dense layers (symbolic): {head_dense_output.shape}")

            # --- Bayesian Output Layer (DenseFlipout) ---
            print(f"    - H{horizon}: Creating DenseFlipout layer...")
            flipout_layer_name = f"bayesian_flipout_layer{branch_suffix}"
            # Ensure posterior_mean_field_custom and prior_fn are defined and accessible
            flipout_layer_branch = DenseFlipout(
                units=1,
                activation='linear',
                kernel_posterior_fn=lambda dtype, shape, bias_size, train, name=flipout_layer_name: posterior_mean_field_custom(dtype, shape, bias_size, train, name),
                kernel_prior_fn=lambda dtype, shape, bias_size, train, name=flipout_layer_name: prior_fn(dtype, shape, bias_size, train, name),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                name=flipout_layer_name
            )
            bayesian_output_branch = Lambda(
                lambda t: flipout_layer_branch(t), # Apply the layer instance
                name=f"bayesian_output{branch_suffix}"
            )(head_dense_output) # Input from the last intermediate head dense layer
            print(f"    - H{horizon}: Bayesian output shape (symbolic): {bayesian_output_branch.shape}")

            # --- Deterministic Bias Layer ---
            print(f"    - H{horizon}: Creating Deterministic Bias layer...")
            # Ensure random_normal_initializer_44 is defined and accessible
            bias_layer_branch = Dense(
                units=1,
                activation='linear',
                kernel_initializer=random_normal_initializer_44,
                name=f"deterministic_bias{branch_suffix}"
            )(head_dense_output) # Input from the last intermediate head dense layer
            print(f"    - H{horizon}: Deterministic bias output shape (symbolic): {bias_layer_branch.shape}")

            # --- Combine Bayesian and Bias for Final Head Output ---
            output_name = f"output_horizon_{horizon}" # Unique name for this specific output
            final_branch_output = Add(name=output_name)(
                [bayesian_output_branch, bias_layer_branch]
            )
            print(f"    - H{horizon}: Final output shape (symbolic): {final_branch_output.shape}")

            outputs_list.append(final_branch_output)
            output_names.append(output_name)
            # --- End of Head ---

        # --- Model Definition and Compilation ---
        self.model = Model(inputs=inputs, outputs=outputs_list, name=f"GeneralizedFeedbackPredictor_{len(predicted_horizons)}H")
        print("DEBUG: Model instantiated.")

        optimizer = AdamW(learning_rate=config.get("learning_rate", self.params.get("learning_rate", 0.001)))
        mmd_lambda = config.get("mmd_lambda", self.params.get("mmd_lambda", 0.1)) # If using composite_loss with MMD

        # Prepare loss and metrics for multi-output
        # Ensure composite_loss, mae_magnitude, r2_metric are defined elsewhere
        loss_dict = {name: lambda y_true, y_pred: composite_loss(y_true, y_pred, mmd_lambda, sigma=1.0)
                    for name in output_names}
        metrics_dict = {name: [mae_magnitude, r2_metric]
                        for name in output_names}

        self.model.compile(optimizer=optimizer,
                        loss=loss_dict,
                        metrics=metrics_dict)
        print("DEBUG: Model compiled.")

        print(f"Generalized Feedback Predictor model built successfully for {num_outputs} horizons: {predicted_horizons}.")
        self.model.summary(line_length=120) # Print model summary with longer lines

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
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

        anneal_epochs = config.get("kl_anneal_epochs", 10) if config is not None else 10
        target_kl = self.params.get('kl_weight', 1e-3)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        min_delta = config.get("min_delta", 1e-4) if config is not None else 1e-4
        
        callbacks = [
            EarlyStoppingWithPatienceCounter(   monitor='val_loss',
                                                patience=self.params.get('early_patience', 10),
                                                restore_best_weights=True,
                                                verbose=1,
                                                start_from_epoch=self.params.get('start_from_epoch', 10),
                                                min_delta=min_delta),
            ReduceLROnPlateauWithCounter(monitor="val_loss",
                                         factor=0.5,
                                         patience=config.get("early_patience", 20) / 4,
                                         verbose=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: 
                           print(f"DEBUG: Learning Rate at epoch {epoch+1}: {K.get_value(self.model.optimizer.learning_rate)}")),
            ClearMemoryCallback(),
            kl_callback
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