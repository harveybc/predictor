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
import os
import gc
import tensorflow.keras.backend as K

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
        'batch_size': 32,
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

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape, x_train, config=None):
        KL_WEIGHT = self.params.get('kl_weight', 1e-3)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0; target kl_weight:", self.params.get('kl_weight', 1e-3))
        print("DEBUG: tensorflow version:", tf.__version__)
        print("DEBUG: tensorflow_probability version:", tfp.__version__)
        print("DEBUG: numpy version:", np.__version__)

        # Ensure input_shape is a tuple for Keras Input
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        
        # Convert x_train to a TensorFlow tensor if it's a NumPy array
        if isinstance(x_train, np.ndarray):
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        print("DEBUG: x_train shape after conversion:", x_train.shape)

        # Create input layer
        inputs = tf.keras.Input(shape=input_shape, name="model_input", dtype=tf.float32)
        print("DEBUG: Created input layer. Shape:", inputs.shape)

        # Common branch
        common = tf.keras.layers.Dense(
            units=self.params['initial_layer_size'],
            activation=self.params['activation'],
            kernel_initializer=random_normal_initializer_42,
            name="common_dense"
        )(inputs)
        common = tf.keras.layers.BatchNormalization(name="common_bn")(common)
        print("DEBUG: Common branch output shape:", common.shape)

        # --- Corrected Bayesian Functions ---
        def posterior_mean_field_custom(dtype, kernel_shape, bias_size, trainable, name):
            if not isinstance(name, str):
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            c = np.log(np.expm1(1.))
            loc = tf.Variable(
                tf.random.normal([n], stddev=0.05, seed=42),
                dtype=dtype, trainable=trainable, name=f"{name}_posterior_loc"
            )
            scale = tf.Variable(
                tf.random.normal([n], stddev=0.05, seed=43),
                dtype=dtype, trainable=trainable, name=f"{name}_posterior_scale"
            )
            scale = 1e-3 + tf.nn.softplus(scale + c)
            scale = tf.clip_by_value(scale, 1e-3, 1.0)
            loc = tf.reshape(loc, kernel_shape)
            scale = tf.reshape(scale, kernel_shape)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc, scale=scale),
                reinterpreted_batch_ndims=len(kernel_shape)
            )

        def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
            if not isinstance(name, str):
                name = None
            bias_size = 0
            n = int(np.prod(kernel_shape)) + bias_size
            loc = tf.zeros([n], dtype=dtype)
            scale = tf.ones([n], dtype=dtype)
            loc = tf.reshape(loc, kernel_shape)
            scale = tf.reshape(scale, kernel_shape)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc, scale=scale),
                reinterpreted_batch_ndims=len(kernel_shape)
            )
        # --- End Corrected Bayesian Functions ---

        # --- Parallel Branches for Multi-Output ---
        outputs = []
        for i in range(self.params['time_horizon']):
            # First Dense layer in branch
            branch = tf.keras.layers.Dense(
                units=self.params['initial_layer_size'] // 2,
                activation=self.params['activation'],
                kernel_initializer=random_normal_initializer_42,
                name=f"branch_{i+1}_dense"
            )(common)

            # Second Dense layer in branch
            branch = tf.keras.layers.Dense(
                units=self.params['initial_layer_size'] // 4,
                activation=self.params['activation'],
                kernel_initializer=random_normal_initializer_42,
                name=f"branch_{i+1}_hidden"
            )(branch)

            # Call DenseFlipout directly (no outer Lambda)
            branch_output = tfp.layers.DenseFlipout(
                units=1,
                activation='linear',
                kernel_posterior_fn=lambda dtype, kernel_shape, bias_size, trainable, name=None: posterior_mean_field_custom(
                    dtype, kernel_shape, bias_size, trainable, name=f"branch_{i+1}"
                ),
                kernel_prior_fn=lambda dtype, kernel_shape, bias_size, trainable, name=None: prior_fn(
                    dtype, kernel_shape, bias_size, trainable, name=f"branch_{i+1}"
                ),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                name=f"branch_{i+1}_flipout"
            )(branch)
            
            # Use a Flatten layer to ensure the output is a rank-1 tensor (vector)
            branch_output = tf.keras.layers.Flatten(name=f"branch_{i+1}_output")(branch_output)
            outputs.append(branch_output)
            print(f"DEBUG: Branch {i+1} output shape:", branch_output.shape)



        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="predictor_model")
        
        metrics = ['mae' for _ in range(self.params['time_horizon'])]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.params.get('learning_rate', 1e-4)),
            loss=[self.custom_loss for _ in range(self.params['time_horizon'])],
            metrics=metrics
        )

        print("✅ Model compiled successfully with corrected DenseFlipout layers.")
        self.model.summary()



    def compute_mmd(self, x, y, sigma=1.0, sample_size=256):
        """
        Compute Maximum Mean Discrepancy (MMD) using a Gaussian Kernel
        with a reduced sample size to avoid memory issues.
        """
        with tf.device('/CPU:0'):  # Move computation to CPU
            # Randomly sample from x and y to reduce memory usage
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


    def custom_loss(self, y_true, y_pred):
        """
        Custom loss function combining Huber loss and MMD loss.
        """
        huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
        mmd_loss = self.compute_mmd(y_pred, y_true)
        total_loss = huber_loss + self.mmd_lambda * mmd_loss
        return total_loss

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the model with MMD loss incorporated and logged at every epoch.
        """
        import tensorflow as tf

        if isinstance(x_train, tuple):
            x_train = x_train[0]
        if x_val is not None and isinstance(x_val, tuple):
            x_val = x_val[0]

        exp_horizon = self.params['time_horizon']

        # --- Transformer-style multi-output adjustment ---
        if isinstance(y_train, np.ndarray):
            y_train = [y_train[:, i] for i in range(exp_horizon)]
        if y_val is not None and isinstance(y_val, np.ndarray):
            y_val = [y_val[:, i] for i in range(exp_horizon)]

        print(f"Training with data => X: {x_train.shape}, Y: {[yt.shape for yt in y_train]}")
        # --- END adjustment ---



        # Initialize MMD lambda
        mmd_lambda = self.params.get('mmd_lambda', 0.01)
        self.mmd_lambda = tf.Variable(mmd_lambda, trainable=False, dtype=tf.float32, name='mmd_lambda')

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
                mmd_value = self.plugin.compute_mmd(preds, self.y_train)
                print(f"                                        MMD Lambda = {self.plugin.mmd_lambda.numpy():.6f}, MMD Loss = {mmd_value.numpy():.6f}")

        anneal_epochs = config.get("kl_anneal_epochs", 10) if config is not None else 10
        target_kl = self.params.get('kl_weight', 1e-3)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        mmd_logging_callback = MMDLoggingCallback(self, x_train, y_train)
        
        min_delta=config.get("min_delta", 1e-4) if config is not None else 1e-4
        early_stopping_monitor = EarlyStoppingWithPatienceCounter(
            monitor='val_loss',
            patience=self.params.get('early_patience', 10),
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=10,
            min_delta=min_delta
        )

        # ReduceLROnPlateau with patience = 1/3 of early stopping patience
        reduce_lr_patience = max(1, self.params.get('early_patience', 10) // 3)  # Ensure at least 1 patience
        reduce_lr_monitor = ReduceLROnPlateauWithCounter(
            monitor='val_loss',
            factor=0.1,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )

        callbacks = [kl_callback, mmd_logging_callback, early_stopping_monitor, reduce_lr_monitor, ClearMemoryCallback()]

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_data=(x_val, y_val)
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # --- NEW CODE for computing MAE and MMD over multi-output predictions ---

        # Get training predictions (list of tensors), stack them to shape (n_samples, time_horizon)
        preds_training_mode_list = self.model(x_train, training=True)
        preds_training_mode_array = np.stack([p.numpy() for p in preds_training_mode_list], axis=1)
        # Also stack your targets (they should already be a list of arrays) to shape (n_samples, time_horizon)
        y_train_array = np.stack(y_train, axis=1)

        # Compute MAE over the entire horizon
        mae_training_mode = np.mean(np.abs(preds_training_mode_array - y_train_array))

        # For MMD, compute per-output and take the average
        mmd_training_mode = np.mean([
            self.compute_mmd(pred, tf.convert_to_tensor(true))
            for pred, true in zip(preds_training_mode_list, y_train)
        ])
        print(f"MAE in Training Mode: {mae_training_mode:.6f}, MMD Lambda: {self.mmd_lambda.numpy():.6f}, MMD Loss: {mmd_training_mode:.6f}")

        # Repeat for evaluation predictions
        preds_eval_mode_list = self.model(x_train, training=False)
        preds_eval_mode_array = np.stack([p.numpy() for p in preds_eval_mode_list], axis=1)
        mae_eval_mode = np.mean(np.abs(preds_eval_mode_array - y_train_array))
        mmd_eval_mode = np.mean([
            self.compute_mmd(pred, tf.convert_to_tensor(true))
            for pred, true in zip(preds_eval_mode_list, y_train)
        ])
        print(f"MAE in Evaluation Mode: {mae_eval_mode:.6f}, MMD Lambda: {self.mmd_lambda.numpy():.6f}, MMD Loss: {mmd_eval_mode:.6f}")
        # --- END NEW CODE ---

        # Evaluate returns a list of loss values and then metric values (one per output)
        train_eval_results = self.model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        # Assuming that the losses are the first 'exp_horizon' elements and metrics the next 'exp_horizon'
        train_loss = np.mean(train_eval_results[:exp_horizon])
        train_mae = np.mean(train_eval_results[exp_horizon:])
        print(f"Restored Weights - Avg Loss: {train_loss}, Avg MAE: {train_mae}")

        val_eval_results = self.model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        val_loss = np.mean(val_eval_results[:exp_horizon])
        val_mae = np.mean(val_eval_results[exp_horizon:])
        print(f"Validation - Avg Loss: {val_loss}, Avg MAE: {val_mae}")


        #train_predictions = self.predict(x_train)
        mc_samples = config.get("mc_samples", 100)
        train_predictions, train_unc = self.predict_with_uncertainty(x_train, mc_samples=mc_samples)
        #val_predictions = self.predict(x_val)
        val_predictions, val_unc =  self.predict_with_uncertainty(x_val, mc_samples=mc_samples)
        return history, train_predictions, train_unc, val_predictions, val_unc


    def predict_with_uncertainty(self, data, mc_samples=100):
        import numpy as np
        print("DEBUG: Starting predict_with_uncertainty with mc_samples:", mc_samples)
        predictions = []
        for i in range(mc_samples):
            preds_list = self.model(data, training=True)
            # Convert each output tensor in the list to numpy and stack along axis=1
            preds_array = np.stack([p.numpy() for p in preds_list], axis=1)
            predictions.append(preds_array)
        predictions = np.array(predictions)  # shape: (mc_samples, n_samples, time_horizon)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        print("DEBUG: Mean predictions shape:", mean_predictions.shape)
        print("DEBUG: Uncertainty estimates shape:", uncertainty_estimates.shape)
        return mean_predictions, uncertainty_estimates



    def predict(self, data):
        import os, logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if isinstance(data, tuple):
            data = data[0]
        # self.model.predict returns a list of arrays (one per output)
        preds_list = self.model.predict(data)
        preds_array = np.stack(preds_list, axis=1)  # shape: (n_samples, time_horizon)
        return preds_array



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


