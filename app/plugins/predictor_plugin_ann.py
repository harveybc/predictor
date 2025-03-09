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
import tensorflow as tf
import tensorflow_probability as tfp


import logging
import os

class Plugin:
    """
    ANN Predictor Plugin using Keras for multi-step forecasting.
    
    This plugin builds, trains, and evaluates an ANN that outputs (N, time_horizon).
    """

    # Default parameters
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'activation': 'tanh',
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

    def build_model(self, input_shape, x_train, config=None):
        """
        Builds a standard ANN (without Bayesian uncertainty estimation) using Keras Dense layers.

        Args:
            input_shape (int): Number of input features.
            x_train (np.ndarray): Training dataset to automatically determine train_size.
        """
        # Ensure x_train is a proper NumPy array.
        x_train = np.array(x_train)
        
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for ANN.")
        
        train_size = x_train.shape[0]
        print("Standard ANN: Number of training samples =", train_size)
        
        # Compute layer sizes.
        layer_sizes = []
        current_size = self.params['initial_layer_size']
        divisor = self.params.get('layer_size_divisor', 2)
        int_layers = self.params.get('intermediate_layers', 3)
        time_horizon = self.params['time_horizon']
        
        for _ in range(int_layers):
            layer_sizes.append(current_size)
            current_size = max(current_size // divisor, 1)
        layer_sizes.append(time_horizon)
        
        print("Standard ANN Layer sizes:", layer_sizes)
        print(f"Standard ANN input_shape: {input_shape}")
        
        # Build the model using standard Dense layers.
        inputs = tf.keras.Input(shape=(input_shape,), name="model_input", dtype=tf.float32)
        x = inputs
        
        # Create intermediate layers.
        for idx, size in enumerate(layer_sizes[:-1]):
            x = tf.keras.layers.Dense(units=size, 
                                    activation=self.params.get('activation', 'tanh'),
                                    kernel_initializer='glorot_uniform',
                                    name=f"dense_layer_{idx+1}")(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        # Final output layer.
        outputs = tf.keras.layers.Dense(units=layer_sizes[-1], 
                                        activation='linear', 
                                        name="output_layer")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.0001))
        self.model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mse', 'mae']
        )
        
        print("✅ Standard ANN model built successfully.")




    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the model with shape => x_train (N, input_dim), y_train (N, time_horizon).
        """
        # Extract array if x_train or x_val is a tuple.
        if isinstance(x_train, tuple):
            x_train = x_train[0]
        if x_val is not None and isinstance(x_val, tuple):
            x_val = x_val[0]

        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")
        
        callbacks = []
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.params['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)
        
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_split=0.2
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        preds_training_mode = self.model(x_train, training=True)
        mae_training_mode = np.mean(np.abs(preds_training_mode - y_train))
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        preds_eval_mode = self.model(x_train, training=False)
        mae_eval_mode = np.mean(np.abs(preds_eval_mode - y_train))
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



    def predict(self, data):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # Extract array if data is a tuple.
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
    

