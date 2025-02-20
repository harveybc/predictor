import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, BatchNormalization, LayerNormalization, Flatten, Add, Lambda, Reshape
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
# Note: using MultiHeadAttention from keras_multi_head as in the example encoder
from keras_multi_head import MultiHeadAttention
from keras import backend as K
from sklearn.metrics import r2_score, mean_absolute_error

import logging
import os
import tensorflow as tf

class Plugin:
    """
    Transformer Predictor Plugin using a Transformer Encoder architecture for multi-step forecasting.
    
    This plugin builds, trains, and evaluates a predictor model that outputs (N, time_horizon).
    The model architecture is adapted from an example encoder while preserving the expected input/output shapes.
    """

    # Default parameters
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'ff_dim_divisor': 2,       # Added for Transformer FF network
        'learning_rate': 1e-5,
        'activation': 'tanh',
        'l2_reg': 1e-2,
        'positional_encoding_dim': 16,  # For Transformer positional encoding (not used in final output)
        # For compatibility with the system, time_horizon must be set externally.
    }
    
    # Variables for debugging
    plugin_debug_vars = ['time_horizon', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']
    
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
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Add the plugin's debug info to an external dictionary.
        """
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape):
        """
        Build a Transformer Encoder model that predicts multi-step outputs.
        
        This model is adapted from the example encoder. The input shape (an int) is interpreted as the total number of features.
        We assume the data will be reshaped into (time_steps, num_channels) such that:
            time_steps = input_shape // num_channels.
        The final output dimension is set to self.params['time_horizon'].
        
        All printed messages, comments and functionalities remain unchanged except for the architectural changes and new combined loss.
        """
        # --- Helper functions for positional encoding ---
        def positional_encoding(seq_len, d_model):
            d_model_float = tf.cast(d_model, tf.float32)
            pos = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
            i = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
            angle_rates = 1 / tf.pow(10000.0, (2 * (tf.floor(i / 2)) / d_model_float))
            angle_rads = pos * angle_rates
            even_mask = tf.cast(tf.equal(tf.math.floormod(tf.range(d_model), 2), 0), tf.float32)
            even_mask = tf.reshape(even_mask, [1, d_model])
            pos_encoding = even_mask * tf.sin(angle_rads) + (1 - even_mask) * tf.cos(angle_rads)
            return pos_encoding

        def add_positional_encoding(x):
            seq_len = tf.shape(x)[1]
            d_model = tf.shape(x)[2]
            pos_enc = positional_encoding(seq_len, d_model)
            pos_enc = tf.cast(pos_enc, tf.float32)
            return x + pos_enc
        # --- End helper functions ---

        # For compatibility, derive time_steps and num_channels from input_shape.
        # We assume self.params may contain 'num_channels' (default 1).
        num_channels = self.params.get("num_channels", 1)
        if input_shape % num_channels != 0:
            raise ValueError(f"input_shape ({input_shape}) is not divisible by num_channels ({num_channels}).")
        time_steps = input_shape // num_channels
        self.params['input_dim'] = input_shape  # store original input dim

        # For multi-step forecasting, set encoding_dim equal to time_horizon.
        if 'time_horizon' not in self.params:
            raise ValueError("Parameter 'time_horizon' must be set in the plugin parameters.")
        encoding_dim = self.params['time_horizon']

        # Retrieve transformer parameters.
        inter_layers = self.params.get('intermediate_layers', 1)
        init_size = self.params.get('initial_layer_size', 128)
        layer_div = self.params.get('layer_size_divisor', 2)
        ff_div = self.params.get('ff_dim_divisor', 2)
        lr = self.params.get('learning_rate', 1e-5)

        # Compute transformer block sizes.
        sizes = []
        current = init_size
        for _ in range(inter_layers):
            sizes.append(current)
            current = max(current // layer_div, encoding_dim)
        sizes.append(encoding_dim)
        print(f"[build_model] Transformer Encoder Layer sizes: {sizes}")
        print(f"[build_model] Input sequence length: {time_steps}, Channels: {num_channels}")

        # Build the model.
        inp = Input(shape=(time_steps, num_channels), dtype=tf.float32, name="model_input")
        x = Lambda(add_positional_encoding, name="positional_encoding")(inp)
        for size in sizes[:-1]:
            ff_dim = max(size // ff_div, 1)
            # Choose number of heads depending on the layer size.
            if size < 64:
                num_heads = 2
            elif size < 128:
                num_heads = 4
            else:
                num_heads = 8
            x = Dense(size )(x)
            x = MultiHeadAttention(head_num=num_heads)(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            ffn = Dense(ff_dim, activation='tanh', kernel_initializer=HeNormal())(x)
            ffn = Dense(size)(ffn)
            x = Add()([x, ffn])
            x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten(name="flatten")(x)
        out = Dense(encoding_dim, activation='linear', kernel_initializer=GlorotUniform(), name="model_output")(x)
        # The final output shape will be (batch_size, time_horizon)
        self.model = Model(inputs=inp, outputs=out, name="Transformer_Encoder_Predictor_Model")

        # --- NEW: Define combined loss function (Huber + MMD) ---
        def gaussian_kernel_matrix(x, y, sigma):
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]
            x_expanded = tf.reshape(x, [x_size, 1, dim])
            y_expanded = tf.reshape(y, [1, y_size, dim])
            squared_diff = tf.reduce_sum(tf.square(x_expanded - y_expanded), axis=2)
            return tf.exp(-squared_diff / (2.0 * sigma**2))

        def combined_loss(y_true, y_pred):
            huber_loss = Huber(delta=1.0)(y_true, y_pred)
            sigma = 1.0
            stat_weight = 1.0
            y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
            y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
            K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
            K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
            K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
            m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
            n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
            mmd = tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
            return huber_loss + stat_weight * mmd
        # --- END NEW LOSS DEFINITION ---

        # Adam Optimizer
        adam_optimizer = Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        # Compile the model with the combined loss function
        self.model.compile(
            optimizer=adam_optimizer,
            loss=combined_loss,
            metrics=['mse', 'mae']
        )
        
        print("Predictor Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the model with shape => x_train(N, input_dim), y_train(N, time_horizon).
        """
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(
                f"y_train shape {y_train.shape}, expected (N,{exp_horizon})."
            )
        
        callbacks = []
        early_stopping_monitor = EarlyStopping(
            monitor='loss',
            patience=self.params['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)
        print(f"Training CNN model with data shape: {x_train.shape}, target shape: {y_train.shape}")
    
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            #shuffle=True,  # Enable shuffling
            callbacks=callbacks,
            #validation_data=validation_data,
            validation_split=0.2
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Force the model to run in "training mode"
        preds_training_mode = self.model(x_train, training=True)
        mae_training_mode = np.mean(np.abs(preds_training_mode.numpy() - y_train))
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        # Compare with evaluation mode
        preds_eval_mode = self.model(x_train, training=False)
        mae_eval_mode = np.mean(np.abs(preds_eval_mode.numpy() - y_train))
        print(f"MAE in Evaluation Mode (manual): {mae_eval_mode:.6f}")

        # Evaluate on the full training dataset for consistency
        train_eval_results = self.model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        train_loss, train_mse, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MSE: {train_mse}, MAE: {train_mae}")
        
        if x_val is not None and y_val is not None:
            val_eval_results = self.model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
            val_loss, val_mse, val_mae = val_eval_results
        else:
            val_eval_results = (None, None, None)
            val_loss = val_mse = val_mae = None
        
        # Predict validation data for evaluation
        train_predictions = self.predict(x_train)  # Predict train data
        val_predictions = self.predict(x_val) if x_val is not None else None  # Predict validation data
    
        # Calculate R² scores
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions) if y_val is not None else None
        
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
