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


# --- REMOVE THIS MONKEY-PATCH ---
# def _patched_add_variable(self, name, shape, dtype, initializer, trainable, **kwargs):
#     return self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
# tfp.layers.DenseFlipout.add_variable = _patched_add_variable


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

    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input, BatchNormalization
    from tensorflow.keras.initializers import RandomNormal

    tfp_layers = tfp.layers

    def build_model(input_shape, x_train, config):
        """
        Construcción del modelo ANN Bayesian con `DenseFlipout` en una implementación multi-output.
        """
        KL_WEIGHT = config.get('kl_weight', 1e-3)
        time_horizon = config['time_horizon']
        initial_layer_size = config['initial_layer_size']
        layer_size_divisor = config['layer_size_divisor']
        intermediate_layers = config['intermediate_layers']
        activation = config['activation']
        
        # Entrada del modelo
        inputs = Input(shape=(input_shape,), name="model_input", dtype=tf.float32)
        x = inputs
        
        # Construcción de capas intermedias
        current_size = initial_layer_size
        for i in range(intermediate_layers):
            x = Dense(
                units=current_size,
                activation=activation,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                name=f"dense_layer_{i+1}"
            )(x)
            x = BatchNormalization()(x)
            current_size = max(current_size // layer_size_divisor, 1)
        
        # Crear múltiples salidas para cada horizonte de predicción
        outputs = []
        for t in range(time_horizon):
            # Bayesian DenseFlipout layer (sin `use_bias` para evitar errores)
            output_layer = tfp_layers.DenseFlipout(
                units=1,  # Salida escalar por horizonte de predicción
                activation='linear',
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                name=f"output_horizon_{t+1}"
            )(x)
            outputs.append(output_layer)
        
        # Creación del modelo con múltiples salidas
        model = Model(inputs=inputs, outputs=outputs, name="bayesian_ann_multi_output")
        
        # Compilación con pérdida independiente por salida
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss=[tf.keras.losses.Huber() for _ in range(time_horizon)],
            metrics=['mae']
        )
        
        model.summary()
        print("✅ Modelo Bayesian ANN multi-output construido exitosamente.")
        
        return model


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


