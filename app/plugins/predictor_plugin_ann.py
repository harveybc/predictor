import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from keras.layers import GaussianNoise
from keras import backend as K
from sklearn.metrics import r2_score 
from sklearn.metrics import r2_score
import gc
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber


import logging
import os

class Plugin:
    """
    ANN Predictor Plugin using Keras for multi-step forecasting.
    
    This plugin builds, trains, and evaluates an ANN that outputs (N, time_horizon).
    """

    # Default parameters
    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'activation': 'relu',
        'l2_reg': 1e-5
    }
    
    # Variables for debugging
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']
    
    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """
        Update plugin parameters with provided kwargs.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Return a dict of debug info from plugin params.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape, config=None):
        """
        Build the ANN with final layer = self.params['time_horizon'] for multi-step outputs.
        
        Args:
            input_shape (int): Number of input features for ANN.
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for ANN.")
        
        self.params['input_dim'] = input_shape
        l2_reg = self.params.get('l2_reg', 1e-4)
        time_horizon = self.params['time_horizon']  # the multi-step dimension

        # Dynamically set layer sizes
        layers = []
        current_size = self.params['initial_layer_size']
        divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // divisor, 1)
            int_layers += 1

        # Final layer => time_horizon units (N, time_horizon) output
        layers.append(time_horizon)

        print(f"ANN Layer sizes: {layers}")
        print(f"ANN input_shape: {input_shape}")

        # Build the model
        from tensorflow.keras import Model, Input
        model_input = Input(shape=(input_shape,), name="model_input")
        x = model_input
        #x = GaussianNoise(0.01)(x)  # Add noise with stddev=0.01
        # Hidden Dense layers
        idx = 0
        for size in layers[:-1]:
            idx += 1
            if idx==1:
                x = Dense(
                    units=size,
                    activation='linear',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name=f"dense_layer_{idx}"
                )(x)
            else:
                x = Dense(
                    units=size,
                    activation=self.params['activation'],
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name=f"dense_layer_{idx}"
                )(x)
        #add batch normalization
        x = BatchNormalization()(x)
        # Output layer => shape (N, time_horizon)
        model_output = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)
        #add batch normalization
        #model_output = BatchNormalization()(model_output)

        self.model = Model(inputs=model_input, outputs=model_output, name="ANN_Predictor_Model")

        print("Model Summary:")
        self.model.summary()

        # Attach an overfit_penalty variable for loss modification via callback
        self.overfit_penalty = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.model.overfit_penalty = self.overfit_penalty

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        if config is not None and config.get('use_mmd', False):
            def combined_loss(y_true, y_pred):
                huber_loss = Huber(delta=1.0)(y_true, y_pred)
                sigma = config.get('mmd_sigma', 1.0)
                stat_weight = config.get('statistical_loss_weight', 1.0)
                mmd = mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)
                penalty_term = tf.cast(1.0, tf.float32) * tf.stop_gradient(self.overfit_penalty)
                return huber_loss + stat_weight * mmd + penalty_term
            loss_fn = combined_loss
            metrics = ['mae', lambda yt, yp: mmd_metric(yt, yp, config)]
        else:
            loss_fn = Huber(delta=1.0)
            metrics = ['mae']
        if config is not None and config.get('use_mmd', False):
            print("Using combined loss with MMD and overfit penalty.")
        else:
            print("Using Huber loss.")
        self.model.compile(
            optimizer=adam_optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        print("Model compiled successfully.")
        # Store the initial learning rate and l2 regularization value for dynamic updates
        self.model.initial_lr = self.params['learning_rate']
        self.model.initial_l2 = self.params.get('l2_reg', 1e-3)



    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the predictor model using EarlyStopping, ReduceLROnPlateau, DebugLearningRate,
        UpdateOverfitPenalty, and MemoryCleanup callbacks.
        """
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N, {exp_horizon}).")


        callbacks = []
        early_patience = self.params.get('early_patience', 32)
        early_monitor = 'val_loss'
        early_stopping = EarlyStopping(monitor=early_monitor, patience=early_patience,
                                    restore_best_weights=True, verbose=1)
        callbacks.append(early_stopping)
        lr_reducer = ReduceLROnPlateau(
            monitor=early_monitor,
            factor=0.316227766,
            patience=int(early_patience / 3),
            verbose=1,
            min_lr=self.params.get('min_lr', 1e-8)
        )
        callbacks.append(lr_reducer)
        debug_lr_cb = DebugLearningRateCallback(early_stopping, lr_reducer)
        callbacks.append(debug_lr_cb)
        update_penalty_cb = UpdateOverfitPenalty()
        callbacks.append(update_penalty_cb)
        memory_cleanup_cb = MemoryCleanupCallback()
        callbacks.append(memory_cleanup_cb)

        print(f"Training CNN model with data shape: {x_train.shape}, target shape: {y_train.shape}")
        if x_val is not None and y_val is not None:
            val_data = (x_val, y_val)
        else:
            val_data = None

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_data
        )
        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")
        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        print("Forcing training mode for MAE calculation...")
        preds_training_mode = self.model(x_train, training=True).numpy()
        mae_training_mode = np.mean(np.abs(preds_training_mode - y_train[:len(preds_training_mode)]))
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        print("Forcing evaluation mode for MAE calculation...")
        preds_eval_mode = self.model(x_train, training=False).numpy()
        mae_eval_mode = np.mean(np.abs(preds_eval_mode - y_train[:len(preds_training_mode)]))
        print(f"MAE in Evaluation Mode (manual): {mae_eval_mode:.6f}")

        print("Evaluating on the full training dataset...")
        train_eval_results = self.model.evaluate(x_train, y_train[:len(preds_training_mode)], batch_size=batch_size, verbose=0)
        train_loss, train_mse, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MSE: {train_mse}, MAE: {train_mae}")

        if x_val is not None and y_val is not None:
            val_eval_results = self.model.evaluate(x_val, y_val[:x_val.shape[0]], batch_size=batch_size, verbose=0)
            _, _, val_mae = val_eval_results
            from sklearn.metrics import r2_score
            val_predictions = self.predict(x_val)
            val_r2 = r2_score(y_val[:x_val.shape[0]], val_predictions)
        else:
            val_mae = None
            val_r2 = None
            val_predictions = None

        train_predictions = self.predict(x_train)
        from sklearn.metrics import r2_score
        train_r2 = r2_score(y_train[:x_train.shape[0]], train_predictions)
        val_r2 = None if val_predictions is None else r2_score(y_val[:x_val.shape[0]], val_predictions)

        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions



    def predict(self, data):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        preds = self.model.predict(data)
        #print(f"Predictions (first 5 rows): {preds[:5]}")  # Add debug
        return preds

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with multi-step shape (N, time_horizon).
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        mse = np.mean((y_true_f - y_pred_f) ** 2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        print(f"y_true (sample): {y_true.flatten()[:5]}")
        print(f"y_pred (sample): {y_pred.flatten()[:5]}")
        mae = np.mean(np.abs(y_true.flatten() - y_pred.flatten()))
        print(f"Calculated MAE: {mae}")
        return mae


    def save(self, file_path):
        """
        Save the trained model to file.
        """
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        """
        Load a trained model from file.
        """
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}")
    

class UpdateOverfitPenalty(Callback):
    """
    Updates the overfit penalty variable (attached to the model) at the end of each epoch.
    The penalty = 0.1 * max(0, val_mae - train_mae)
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if loss is None or val_loss is None:
            print("[UpdateOverfitPenalty] MAE metrics not available; overfit penalty not updated.")
            return
        penalty = 0.02 * max(0, val_loss - loss)
        tf.keras.backend.set_value(self.model.overfit_penalty, penalty)
        print(f"[UpdateOverfitPenalty] Epoch {epoch+1}: Updated overfit penalty to {penalty:.6f}")

class DebugLearningRateCallback(Callback):
    """
    Prints the current learning rate, EarlyStopping wait counter, and ReduceLROnPlateau wait and best values.
    Additionally, updates the l2 regularization factor in layers that have a kernel_regularizer of type L2,
    scaling it proportionally to the learning rate change relative to the initial learning rate.
    """
    def __init__(self, early_stopping_cb, lr_reducer_cb):
        super(DebugLearningRateCallback, self).__init__()
        self.early_stopping_cb = early_stopping_cb
        self.lr_reducer_cb = lr_reducer_cb

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        es_wait = getattr(self.early_stopping_cb, "wait", None)
        lr_wait = getattr(self.lr_reducer_cb, "wait", None)
        best_val = getattr(self.lr_reducer_cb, "best", None)
        # Update l2_reg in all layers with a kernel_regularizer of type L2.
        if hasattr(self.model, 'initial_lr') and self.model.initial_lr is not None:
            scaling_factor = current_lr / self.model.initial_lr
            if hasattr(self.model, 'initial_l2') and self.model.initial_l2 is not None:
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                        if isinstance(layer.kernel_regularizer, tf.keras.regularizers.L2):
                            old_l2 = layer.kernel_regularizer.l2
                            new_l2 = self.model.initial_l2 * scaling_factor
                            layer.kernel_regularizer.l2 = new_l2
                            # if old!=new, print the change
                            if old_l2 != new_l2:
                                print(f"[DebugLR] Updated l2_reg in layer {layer.name} from {old_l2} to {new_l2}")
        print(f"\n[DebugLR] Epoch {epoch+1}: Learning Rate = {current_lr:.4e}, l2_reg ={new_l2:.4e} , EarlyStopping wait = {es_wait}, LRReducer wait = {lr_wait}, LRReducer best = {best_val}")
        


class MemoryCleanupCallback(Callback):
    """
    Forces garbage collection at the end of each epoch.
    """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def gaussian_kernel_sum(x, y, sigma, chunk_size=16):
    """
    Compute a chunked sum of Gaussian kernels between rows in x and y.
    """
    n = tf.shape(x)[0]
    total = tf.constant(0.0, dtype=tf.float32)
    i = tf.constant(0)
    def cond(i, total):
        return tf.less(i, n)
    def body(i, total):
        end_i = tf.minimum(i + chunk_size, n)
        x_chunk = x[i:end_i]
        diff = tf.expand_dims(x_chunk, axis=1) - tf.expand_dims(y, axis=0)
        squared_diff = tf.reduce_sum(tf.square(diff), axis=2)
        divisor = 2.0 * tf.square(sigma)
        kernel_chunk = tf.exp(-squared_diff / divisor)
        total += tf.reduce_sum(kernel_chunk)
        return i + chunk_size, total
    i, total = tf.while_loop(cond, body, [i, total])
    return total

def mmd_loss_term(y_true, y_pred, sigma, chunk_size=16):
    """
    Compute the MMD loss using a chunked Gaussian kernel sum.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    sum_K_xx = gaussian_kernel_sum(y_true, y_true, sigma, chunk_size)
    sum_K_yy = gaussian_kernel_sum(y_pred, y_pred, sigma, chunk_size)
    sum_K_xy = gaussian_kernel_sum(y_true, y_pred, sigma, chunk_size)
    m = tf.cast(tf.shape(y_true)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred)[0], tf.float32)
    mmd = sum_K_xx / (m*m) + sum_K_yy / (n*n) - 2 * sum_K_xy / (m*n)
    return mmd

def mmd_metric(y_true, y_pred, config):
    sigma = config.get('mmd_sigma', 1.0)
    return mmd_loss_term(y_true, y_pred, sigma, chunk_size=16)