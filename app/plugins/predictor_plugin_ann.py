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
        Builds a Bayesian ANN using TensorFlow Probability.

        Args:
            input_shape (int): Number of input features.
            x_train (np.ndarray): Training dataset to automatically determine train_size.
        """
        # Force conversion to a NumPy array to avoid tuple issues.
        x_train = np.asarray(x_train)

        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for ANN.")

        train_size = x_train.shape[0]
        kl_weight = 1 / max(1, train_size)

        # Define layer sizes.
        layer_sizes = []
        current_size = self.params['initial_layer_size']
        divisor = self.params.get('layer_size_divisor', 2)
        int_layers = self.params.get('intermediate_layers', 3)
        time_horizon = self.params['time_horizon']

        for _ in range(int_layers):
            layer_sizes.append(current_size)
            current_size = max(current_size // divisor, 1)
        layer_sizes.append(time_horizon)

        print("Bayesian ANN Layer sizes:", layer_sizes)
        print(f"Bayesian ANN input_shape: {input_shape}")

        # Custom posterior function with 5 arguments.
        def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
            dtype = tf.as_dtype(dtype) if isinstance(dtype, tf.DType) else tf.float32
            new_shape = tuple(shape) if shape is not None else ()
            if len(new_shape) == 0:
                new_shape = ()
            loc = add_variable_fn(
                name=name + '_loc',
                shape=new_shape,
                initializer=tf.random_normal_initializer(stddev=0.1),
                dtype=dtype,
                trainable=trainable
            )
            rho = add_variable_fn(
                name=name + '_rho',
                shape=new_shape,
                initializer=tf.constant_initializer(-3.0),
                dtype=dtype,
                trainable=trainable
            )
            scale = tf.nn.softplus(rho)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc, scale=scale),
                reinterpreted_batch_ndims=1
            )

        # Custom prior function with 5 arguments.
        def prior_fn(dtype, shape, name, trainable, add_variable_fn):
            dtype = tf.as_dtype(dtype) if isinstance(dtype, tf.DType) else tf.float32
            new_shape = tuple(shape) if shape is not None else ()
            if len(new_shape) == 0:
                new_shape = ()
            loc = tf.zeros(new_shape, dtype=dtype)
            scale = tf.ones(new_shape, dtype=dtype)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc, scale=scale),
                reinterpreted_batch_ndims=1
            )

        # Build the model.
        inputs = tf.keras.Input(shape=(input_shape,), name="model_input", dtype=tf.float32)
        x = inputs

        for idx, size in enumerate(layer_sizes[:-1]):
            x = tfp.layers.DenseVariational(
                units=size,
                make_posterior_fn=posterior_fn,
                make_prior_fn=prior_fn,
                kl_weight=kl_weight,
                activation=self.params.get('activation', 'tanh'),
                name=f"dense_layer_{idx+1}"
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)

        outputs = tfp.layers.DenseVariational(
            units=layer_sizes[-1],
            make_posterior_fn=posterior_fn,
            make_prior_fn=prior_fn,
            kl_weight=kl_weight,
            activation='linear',
            name="output_layer"
        )(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.0001))
        self.model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mse', 'mae']
        )

        print("✅ Bayesian ANN model built successfully.")



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
    

