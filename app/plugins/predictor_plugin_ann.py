import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianNoise
from keras import backend as K
from sklearn.metrics import r2_score 
import logging
import os,gc
from tensorflow.keras.mixed_precision import set_global_policy


#Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Error setting GPU memory growth:", e)

# Set global mixed precision policy
set_global_policy('mixed_float16')


# --- Monkey-patch DenseFlipout to use add_weight instead of deprecated add_variable ---
def _patched_add_variable(self, name, shape, dtype, initializer, trainable, **kwargs):
    return self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
tfp.layers.DenseFlipout.add_variable = _patched_add_variable

# --- Named initializers to avoid lambda serialization warnings ---
def random_normal_initializer_42(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=42)

def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)


class Plugin:
    """
    ANN Predictor Plugin using Keras for multi-step forecasting.
    
    This plugin builds, trains, and evaluates an ANN that outputs (N, time_horizon).
    """

    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3
    }
    
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']
    
    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None
        self.overfit_penalty = None  

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape, x_train, config=None):
        """
        Builds a Bayesian ANN using DenseFlipout for uncertainty estimation.
        The final output layer is replaced by a DenseFlipout (without bias) plus a separate 
        deterministic bias layer.
        """
        from tensorflow.keras.losses import Huber

        KL_WEIGHT = self.params.get('kl_weight', 1e-3)

        print("DEBUG: tensorflow version:", tf.__version__)
        print("DEBUG: tensorflow_probability version:", tfp.__version__)
        print("DEBUG: numpy version:", np.__version__)

        x_train = np.array(x_train)
        print("DEBUG: x_train converted to numpy array. Type:", type(x_train), "Shape:", x_train.shape)
        
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for ANN.")
        print("DEBUG: input_shape is valid. Value:", input_shape)
        
        train_size = x_train.shape[0]
        print("DEBUG: Number of training samples:", train_size)
        
        layer_sizes = []
        current_size = self.params['initial_layer_size']
        print("DEBUG: Initial layer size:", current_size)
        divisor = self.params.get('layer_size_divisor', 2)
        print("DEBUG: Layer size divisor:", divisor)
        int_layers = self.params.get('intermediate_layers', 3)
        print("DEBUG: Number of intermediate layers:", int_layers)
        time_horizon = self.params['time_horizon']
        print("DEBUG: Time horizon (final layer size):", time_horizon)
        
        for i in range(int_layers):
            layer_sizes.append(current_size)
            print(f"DEBUG: Appended layer size at layer {i+1}: {current_size}")
            current_size = max(current_size // divisor, 1)
            print(f"DEBUG: Updated current_size after division at layer {i+1}: {current_size}")
        layer_sizes.append(time_horizon)
        print("DEBUG: Final layer sizes:", layer_sizes)
        
        print("DEBUG: Standard ANN input_shape:", input_shape)
        
        inputs = tf.keras.Input(shape=(input_shape,), name="model_input", dtype=tf.float32)
        print("DEBUG: Created input layer. Shape:", inputs.shape)
        x = inputs
        print("DEBUG: x tensor from inputs. Shape:", x.shape, "Type:", type(x))
        
        for idx, size in enumerate(layer_sizes[:-1]):
            print(f"DEBUG: Building Dense layer {idx+1} with size {size}")
            x = tf.keras.layers.Dense(
                units=size,
                activation=self.params.get('activation', 'tanh'),
                kernel_initializer=random_normal_initializer_42,
                name=f"dense_layer_{idx+1}"
            )(x)
        print(f"DEBUG: After Dense layer {idx+1}, x shape:", x.shape)
        x = tf.keras.layers.BatchNormalization()(x)
        print(f"DEBUG: After BatchNormalization at layer {idx+1}, x shape:", x.shape)
        
        if hasattr(x, '_keras_history'):
            print("DEBUG: x is already a KerasTensor; no conversion needed.")
        else:
            x = tf.convert_to_tensor(x)
            print("DEBUG: Converted x to tensor. New type:", type(x))
        
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0; target kl_weight:", KL_WEIGHT)
        
        default_bias_size = 0
        print("DEBUG: Using default_bias_size =", default_bias_size)
        
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
        print("DEBUG: Creating DenseFlipout final layer with units:", layer_sizes[-1])
        flipout_layer = DenseFlipout(
            units=layer_sizes[-1],
            activation='linear',
            kernel_posterior_fn=posterior_mean_field_custom,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
            name="output_layer"
        )
        bayesian_output = tf.keras.layers.Lambda(
            lambda t: flipout_layer(t),
            output_shape=lambda s: (s[0], layer_sizes[-1]),
            name="bayesian_dense_flipout"
        )(x)
        print("DEBUG: After DenseFlipout (via Lambda), bayesian_output shape:", bayesian_output.shape)
        
        bias_layer = tf.keras.layers.Dense(
            units=layer_sizes[-1],
            activation='linear',
            use_bias=True,
            kernel_initializer=random_normal_initializer_44,
            name="deterministic_bias"
        )(x)
        print("DEBUG: Deterministic bias layer output shape:", bias_layer.shape)
        
        outputs = bayesian_output + bias_layer
        print("DEBUG: Final outputs shape after adding bias:", outputs.shape)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print("DEBUG: Model created. Input shape:", self.model.input_shape, "Output shape:", self.model.output_shape)
        
        self.overfit_penalty = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.model.overfit_penalty = self.overfit_penalty
        
        initial_lr = config.get('learning_rate', 0.01)
        adam_optimizer = Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False)
        
        def combined_loss(y_true, y_pred):
            huber_loss = Huber(delta=1.0)(y_true, y_pred)
            sigma = config.get('mmd_sigma', 1.0)
            stat_weight = config.get('statistical_loss_weight', 1.0)
            mmd = mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)
            penalty_term = tf.cast(1.0, tf.float32) * tf.stop_gradient(self.overfit_penalty)
            return huber_loss + (stat_weight * mmd) + penalty_term
        
        if config.get('use_mmd', False):
            loss_fn = combined_loss
            metrics = ['mae', lambda yt, yp: mmd_metric(yt, yp, config)]
        else:
            loss_fn = Huber(delta=1.0)
            metrics = ['mae']
        
        self.model.compile(
            optimizer=adam_optimizer,
            loss=loss_fn,
            metrics=metrics   
        )
        print("DEBUG: Model compiled")
        print("Predictor Model Summary:")
        self.model.summary()
        print("✅ Standard ANN model built successfully.")


    # --------------------- Updated train Method in Plugin ---------------------
    # --------------------- Updated train Method in Plugin ---------------------
    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the model with shape => x_train (N, input_dim), y_train (N, time_horizon).
        Implements KL annealing, ReduceLROnPlateau, and uses custom callbacks to monitor learning rate,
        EarlyStopping and LRReducer wait counters, and the MMD metric on the validation set.
        This implementation follows the exact structure and prints the exact same messages as used in the autoencoder.
        """
        import tensorflow as tf
        if isinstance(x_train, tuple):
            x_train = x_train[0]
        if x_val is not None and isinstance(x_val, tuple):
            x_val = x_val[0]

        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")

        # Disable XLA JIT compilation to avoid register spillage warnings.
        tf.config.optimizer.set_jit(False)

        # KL Annealing Callback (same as before)
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

        early_patience = config.get('early_patience', 32)
        early_monitor = config.get('early_monitor', 'val_loss')
        early_stopping = EarlyStopping(monitor=early_monitor, patience=early_patience, restore_best_weights=True, verbose=1)

        update_penalty_cb = UpdateOverfitPenalty(config)

        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=early_monitor,
            factor=0.316227766,  # approx. 1/sqrt(10)
            patience=int(self.params.get('patience', 10) / 3),
            verbose=1,
            min_lr=config.get('min_lr', 1e-8)
        )

        debug_lr_cb = DebugLearningRateCallback(early_stopping, lr_reducer)
        memory_cleanup_cb = MemoryCleanupCallback()

        val_data = (x_val, y_val)

        #callbacks = [kl_callback, early_stopping, update_penalty_cb, lr_reducer, debug_lr_cb, memory_cleanup_cb]
        callbacks = [kl_callback, early_stopping, update_penalty_cb, lr_reducer, debug_lr_cb, memory_cleanup_cb]    
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_data=val_data
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        preds_training_mode = self.model(x_train, training=True)
        mae_training_mode = float(tf.reduce_mean(tf.abs(preds_training_mode - y_train)).numpy())
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        preds_eval_mode = self.model(x_train, training=False)
        mae_eval_mode = float(tf.reduce_mean(tf.abs(preds_eval_mode - y_train)).numpy())
        print(f"MAE in Evaluation Mode (manual): {mae_eval_mode:.6f}")

        train_eval_results = self.model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        train_loss, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MAE: {train_mae}")

        val_eval_results = self.model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        val_loss, val_mae = val_eval_results

        from sklearn.metrics import r2_score
        train_predictions = self.predict(x_train)
        val_predictions = self.predict(x_val)
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions)

        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions


    def predict_with_uncertainty(self, data, mc_samples=100):
        """
        Perform multiple forward passes through the model to estimate prediction uncertainty.
        
        Args:
            data (np.ndarray): Input data for prediction.
            mc_samples (int): Number of Monte Carlo samples.
        
        Returns:
            tuple: (mean_predictions, uncertainty_estimates) where both are np.ndarray with shape (n_samples, time_horizon)
        """
        import numpy as np
        print("DEBUG: Starting predict_with_uncertainty with mc_samples (expected):", mc_samples)
        predictions = []
        for i in range(mc_samples):
            preds = self.model(data, training=True)
            preds_np = preds.numpy()
            print(f"DEBUG: Sample {i+1}/{mc_samples} prediction. Expected shape: (n_samples, time_horizon), Actual shape:", preds_np.shape)
            predictions.append(preds_np)
        predictions = np.array(predictions)
        print("DEBUG: All predictions collected. Expected shape: (mc_samples, n_samples, time_horizon), Actual shape:", predictions.shape)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        print("DEBUG: Mean predictions shape:", mean_predictions.shape)
        print("DEBUG: Uncertainty estimates shape:", uncertainty_estimates.shape)
        return mean_predictions, uncertainty_estimates


    def predict(self, data):
        import os
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if isinstance(data, tuple):
            data = data[0]
        preds = self.model.predict(data)
        return preds


    def calculate_mae(self, y_true, y_pred):
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae


    def save(self, file_path):
        from tensorflow.keras.models import save_model
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")


    def load(self, file_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}")



class UpdateOverfitPenalty(Callback):
    """
    Custom Callback to update the overfit penalty value used in the loss function.
    At the end of each epoch, it computes the difference between validation MAE and training MAE,
    multiplies it by a scaling factor (0.1), and updates a TensorFlow variable in the model.
    The penalty is applied only if validation MAE is higher than training MAE.
    """
    def __init__(self, config):
        self.config = config

    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_mae = logs.get('mae')
        val_mae = logs.get('val_mae')
        if train_mae is None or val_mae is None:
            print("[UpdateOverfitPenalty] MAE metrics not available; overfit penalty not updated.")
            return
        overfitting_penalty = self.config.get('overfitting_penalty', 0.1)
        penalty = overfitting_penalty * max(0, val_mae - train_mae)
        tf.keras.backend.set_value(self.model.overfit_penalty, penalty)
        print(f"[UpdateOverfitPenalty] Epoch {epoch+1}: Updated overfit penalty to {penalty:.6f}")

# --------------------- Updated DebugLearningRateCallback.on_epoch_end ---------------------
class DebugLearningRateCallback(Callback):
    """
    Debug Callback that prints the current learning rate,
    the wait counter for EarlyStopping, and for the LR reducer.
    Additionally, updates the L2 regularization factor in layers with a kernel_regularizer of type L2,
    scaling it proportionally to the learning rate change relative to the initial learning rate.
    """
    def __init__(self, early_stopping_cb, lr_reducer_cb):
        super(DebugLearningRateCallback, self).__init__()
        self.early_stopping_cb = early_stopping_cb
        self.lr_reducer_cb = lr_reducer_cb

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        # Retrieve current learning rate from optimizer (check both 'lr' and 'learning_rate')
        if hasattr(optimizer, 'lr'):
            current_lr = tf.keras.backend.get_value(optimizer.lr)
        else:
            current_lr = tf.keras.backend.get_value(optimizer.learning_rate)
        es_wait = getattr(self.early_stopping_cb, "wait", None)
        lr_wait = getattr(self.lr_reducer_cb, "wait", None)
        best_val = getattr(self.lr_reducer_cb, "best", None)
        new_l2 = 0.0
        # Update L2 regularization if initial values are stored on the model.
        if hasattr(self.model, 'initial_lr') and self.model.initial_lr is not None:
            scaling_factor = current_lr / self.model.initial_lr
            if hasattr(self.model, 'initial_l2') and self.model.initial_l2 is not None:
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                        if isinstance(layer.kernel_regularizer, tf.keras.regularizers.L2):
                            old_l2 = layer.kernel_regularizer.l2
                            new_l2 = self.model.initial_l2 * scaling_factor
                            layer.kernel_regularizer.l2 = new_l2
                            if old_l2 != new_l2:
                                print(f"[DebugLR] Updated l2_reg in layer {layer.name} from {old_l2} to {new_l2}")
        print(f"\n[DebugLR] Epoch {epoch+1}: Learning Rate = {current_lr:.4e}, l2_reg = {new_l2:.4e}, "
              f"EarlyStopping wait = {es_wait}, LRReducer wait = {lr_wait}, LRReducer best = {best_val}")


class MemoryCleanupCallback(Callback):
    """
    Callback to force garbage collection at the end of each epoch.
    This can help free up unused memory and mitigate memory leaks.
    """
    def on_epoch_end(self, epoch, logs=None, overfit_penalty=1.0):
        gc.collect()
        #print(f"[MemoryCleanup] Epoch {epoch+1}: Garbage collection executed.")


# Updated named initializers using stateless random ops
def random_normal_initializer_42(shape, dtype=None):
    seed = (42, 0)
    return tf.random.stateless_normal(shape, seed=seed, mean=0.0, stddev=0.05, dtype=dtype)

def random_normal_initializer_44(shape, dtype=None):
    seed = (44, 0)
    return tf.random.stateless_normal(shape, seed=seed, mean=0.0, stddev=0.05, dtype=dtype)

# Updated MMD helper functions: disable XLA compilation for these functions
@tf.function(experimental_compile=False)
def gaussian_kernel_sum(x, y, sigma, chunk_size=8):
    n = tf.shape(x)[0]
    total = tf.constant(0.0, dtype=tf.float32)
    i = tf.constant(0)
    max_iter = tf.math.floordiv(n + chunk_size - 1, chunk_size)
    tf.print("DEBUG: In gaussian_kernel_sum: n =", n, "max_iter =", max_iter)
    
    def cond(i, total):
        return tf.less(i, n)
    
    def body(i, total):
        end_i = tf.minimum(i + chunk_size, n)
        x_chunk = x[i:end_i]
        tf.print("DEBUG: Processing chunk from", i, "to", end_i, "x_chunk shape =", tf.shape(x_chunk))
        diff = tf.expand_dims(x_chunk, axis=1) - tf.expand_dims(y, axis=0)
        squared_diff = tf.reduce_sum(tf.square(diff), axis=2)
        divisor = 2.0 * tf.square(sigma)
        kernel_chunk = tf.exp(-squared_diff / divisor)
        chunk_sum = tf.reduce_sum(kernel_chunk)
        tf.print("DEBUG: Chunk sum =", chunk_sum)
        total += chunk_sum
        return i + chunk_size, total

    i, total = tf.while_loop(cond, body, [i, total], maximum_iterations=max_iter)
    tf.print("DEBUG: Finished gaussian_kernel_sum; total =", total)
    return total

@tf.function(experimental_compile=False)
def mmd_loss_term(y_true, y_pred, sigma, chunk_size=16):
    tf.print("DEBUG: In mmd_loss_term: original y_true shape =", tf.shape(y_true),
             "y_pred shape =", tf.shape(y_pred))
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    tf.print("DEBUG: Reshaped y_true shape =", tf.shape(y_true), "y_pred shape =", tf.shape(y_pred))
    
    sum_K_xx = gaussian_kernel_sum(y_true, y_true, sigma, chunk_size)
    sum_K_yy = gaussian_kernel_sum(y_pred, y_pred, sigma, chunk_size)
    sum_K_xy = gaussian_kernel_sum(y_true, y_pred, sigma, chunk_size)
    
    m = tf.cast(tf.shape(y_true)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred)[0], tf.float32)
    tf.print("DEBUG: m =", m, "n =", n)
    
    mmd = sum_K_xx / (m * m) + sum_K_yy / (n * n) - 2 * sum_K_xy / (m * n)
    tf.print("DEBUG: Computed mmd =", mmd)
    return mmd

def mmd_metric(y_true, y_pred, config):
    sigma = config.get('mmd_sigma', 1.0)
    return mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)

# Updated combined loss function (wrapped with tf.function and experimental_compile=False)
@tf.function(experimental_compile=False)
def combined_loss(y_true, y_pred):
    huber_loss = Huber(delta=1.0)(y_true, y_pred)
    sigma = config.get('mmd_sigma', 1.0)
    stat_weight = config.get('statistical_loss_weight', 1.0)
    mmd = mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)
    penalty_term = tf.cast(1.0, tf.float32) * tf.stop_gradient(self.overfit_penalty)
    return huber_loss + (stat_weight * mmd) + penalty_term


# --- Updated Named initializers using stateless_random ---
def random_normal_initializer_42(shape, dtype=None):
    # Use a fixed seed vector, e.g. [42, 0]
    return tf.random.stateless_normal(shape, seed=[42, 0], mean=0.0, stddev=0.05, dtype=dtype)

def random_normal_initializer_44(shape, dtype=None):
    # Use a different fixed seed vector, e.g. [44, 0]
    return tf.random.stateless_normal(shape, seed=[44, 0], mean=0.0, stddev=0.05, dtype=dtype)

