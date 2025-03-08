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

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
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


    def build_model(self, input_shape, x_train):
        """
        Builds a Bayesian ANN using TensorFlow Probability.

        Args:
            input_shape (int): Number of input features.
            x_train (np.ndarray): Training dataset to automatically determine train_size.
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for ANN.")

        train_size = x_train.shape[0]  # Automatically get the number of training samples
        kl_weight = 1 / max(1, train_size)  # Normalize KL divergence, avoid division by zero

        # Define layer sizes
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

        # Define posterior and prior functions with the expected 5-argument signature.
        def posterior_fn(dtype, shape, name, trainable, add_variable_fn):
            dtype = tf.as_dtype(dtype)
            loc = add_variable_fn(
                name=name,
                shape=shape,
                initial_value=tf.random.normal(shape, stddev=0.1, dtype=dtype),
                dtype=dtype,
                trainable=trainable
            )
            rho = add_variable_fn(
                name=name + '_rho',
                shape=shape,
                initial_value=tf.constant(-3.0, shape=shape, dtype=dtype),
                dtype=dtype,
                trainable=trainable
            )
            scale = tf.nn.softplus(rho)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc, scale=scale),
                reinterpreted_batch_ndims=1
            )

        def prior_fn(dtype, shape, name, trainable, add_variable_fn):
            dtype = tf.as_dtype(dtype)
            loc = tf.zeros(shape, dtype=dtype)
            scale = tf.ones(shape, dtype=dtype)
            return tfp.distributions.Independent(
                tfp.distributions.Normal(loc=loc, scale=scale),
                reinterpreted_batch_ndims=1
            )

        # Build the Bayesian ANN
        inputs = Input(shape=(input_shape,), name="model_input", dtype=tf.float32)
        x = inputs

        # Intermediate Bayesian layers
        for idx, size in enumerate(layer_sizes[:-1]):
            x = tfp.layers.DenseVariational(
                units=size,
                make_posterior_fn=posterior_fn,
                make_prior_fn=prior_fn,
                kl_weight=kl_weight,
                activation=self.params.get('activation', 'tanh'),
                name=f"dense_layer_{idx+1}"
            )(x)
            x = BatchNormalization()(x)

        # Final Bayesian output layer
        outputs = tfp.layers.DenseVariational(
            units=layer_sizes[-1],
            make_posterior_fn=posterior_fn,
            make_prior_fn=prior_fn,
            kl_weight=kl_weight,
            activation='linear',
            name="output_layer"
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.0001))
        self.model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mse', 'mae']
        )

        print("✅ Bayesian ANN model built successfully.")



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
            shuffle=True,  # Enable shuffling
            callbacks=callbacks,
            #validation_data=(x_val, y_val)
            validation_split = 0.2
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

        # Force the model to run in "training mode"
        preds_training_mode = self.model(x_train, training=True)
        mae_training_mode = np.mean(np.abs(preds_training_mode - y_train))
        print(f"MAE in Training Mode (manual): {mae_training_mode:.6f}")

        # Compare with evaluation mode
        preds_eval_mode = self.model(x_train, training=False)
        mae_eval_mode = np.mean(np.abs(preds_eval_mode - y_train))
        print(f"MAE in Evaluation Mode (manual): {mae_eval_mode:.6f}")

        # Evaluate on the full training dataset for consistency
        train_eval_results = self.model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        train_loss, train_mse, train_mae = train_eval_results
        print(f"Restored Weights - Loss: {train_loss}, MSE: {train_mse}, MAE: {train_mae}")
        
        val_eval_results = self.model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        val_loss, val_mse, val_mae = val_eval_results
        
        # Predict validation data for evaluation
        train_predictions = self.predict(x_train)  # Predict train data
        val_predictions = self.predict(x_val)      # Predict validation data

        # Calculate R² scores
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        return history, train_mae, train_r2, val_mae, val_r2, train_predictions, val_predictions



    def predict(self, data, num_samples=50):
        """
        Predict with multiple stochastic forward passes for uncertainty estimation.
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        predictions_samples = np.array([
            self.model(data, training=True).numpy()
            for _ in range(num_samples)
        ])

        predictions_mean = np.mean(predictions_samples, axis=0)
        predictions_std = np.std(predictions_samples, axis=0)

        return predictions_mean, predictions_std


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
    

