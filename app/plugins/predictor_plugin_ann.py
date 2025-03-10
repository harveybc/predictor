import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianNoise
from keras import backend as K
from sklearn.metrics import r2_score 
import logging
import os

class Plugin:
    """
    ANN Predictor Plugin using Keras for multi-step forecasting.
    
    This plugin builds, trains, and evaluates an ANN that outputs (N, time_horizon).
    """

    # Default parameters (note new 'kl_weight' added to tune KL divergence in DenseFlipout)
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
        Builds a Bayesian ANN using Keras Dense layers with TensorFlow Probability's DenseFlipout for
        uncertainty estimation, employing KL annealing and custom posterior/prior functions.
        The final output layer is replaced by a DenseFlipout layer wrapped in a Lambda layer.
        
        Args:
            input_shape (int): Number of input features.
            x_train (np.ndarray): Training dataset to automatically determine train_size.
        """
        import tensorflow as tf
        import tensorflow_probability as tfp
        import numpy as np
        from tensorflow.keras.losses import Huber

        # ---------------------------
        # Print versions of key packages
        # ---------------------------
        print("DEBUG: tensorflow version:", tf.__version__)
        print("DEBUG: tensorflow_probability version:", tfp.__version__)
        print("DEBUG: numpy version:", np.__version__)

        # ---------------------------
        # Convert x_train to a numpy array and print details
        # ---------------------------
        x_train = np.array(x_train)
        print("DEBUG: x_train converted to numpy array.")
        print("       Expected type: <class 'numpy.ndarray'>, Actual type:", type(x_train))
        print("       Expected shape: (n_samples, n_features), Actual shape:", x_train.shape)
        
        # ---------------------------
        # Validate input_shape
        # ---------------------------
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}; must be int for ANN.")
        print("DEBUG: input_shape is valid. Expected type int. Actual input_shape:", input_shape)
        
        # ---------------------------
        # Determine training sample count
        # ---------------------------
        train_size = x_train.shape[0]
        print("DEBUG: Number of training samples (expected):", train_size)
        
        # ---------------------------
        # Compute layer sizes based on parameters
        # ---------------------------
        layer_sizes = []
        current_size = self.params['initial_layer_size']
        print("DEBUG: Initial layer size (expected):", current_size)
        divisor = self.params.get('layer_size_divisor', 2)
        print("DEBUG: Layer size divisor (expected):", divisor)
        int_layers = self.params.get('intermediate_layers', 3)
        print("DEBUG: Number of intermediate layers (expected):", int_layers)
        time_horizon = self.params['time_horizon']
        print("DEBUG: Time horizon (expected final layer size):", time_horizon)
        
        for i in range(int_layers):
            layer_sizes.append(current_size)
            print(f"DEBUG: Appended layer size at layer {i+1}: {current_size}")
            current_size = max(current_size // divisor, 1)
            print(f"DEBUG: Updated current_size after division at layer {i+1}: {current_size}")
        layer_sizes.append(time_horizon)
        print("DEBUG: Final layer sizes (expected):", layer_sizes)
        
        print("DEBUG: Standard ANN input_shape (expected):", input_shape)
        
        # ---------------------------
        # Build input layer
        # ---------------------------
        inputs = tf.keras.Input(shape=(input_shape,), name="model_input", dtype=tf.float32)
        print("DEBUG: Created input layer. Expected shape: (None, {})".format(input_shape))
        x = inputs
        print("DEBUG: Initial x tensor from inputs. Shape:", x.shape, "Type:", type(x))
        
        # ---------------------------
        # Build intermediate Dense layers with BatchNormalization
        # ---------------------------
        for idx, size in enumerate(layer_sizes[:-1]):
            print(f"DEBUG: Building Dense layer {idx+1} with size {size}")
            x = tf.keras.layers.Dense(
                units=size, 
                activation=self.params.get('activation', 'tanh'),
                kernel_initializer='glorot_uniform',
                name=f"dense_layer_{idx+1}"
            )(x)
            print(f"DEBUG: After Dense layer {idx+1}, x shape:", x.shape, "Type:", type(x))
            x = tf.keras.layers.BatchNormalization(name=f"batchnorm_{idx+1}")(x)
            print(f"DEBUG: After BatchNormalization at layer {idx+1}, x shape:", x.shape, "Type:", type(x))
        
        # ---------------------------
        # Check if x is already a KerasTensor; skip conversion if so.
        # ---------------------------
        if hasattr(x, '_keras_history'):
            print("DEBUG: x is already a KerasTensor; skipping tf.convert_to_tensor conversion.")
        else:
            x = tf.convert_to_tensor(x)
            print("DEBUG: Converted x to tensor using tf.convert_to_tensor. New type:", type(x))
        
        # ---------------------------
        # KL Annealing: Initialize KL weight variable.
        # ---------------------------
        target_kl = self.params.get('kl_weight', 1e-4)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='kl_weight_var')
        print("DEBUG: Initialized kl_weight_var with 0.0; target kl_weight is", target_kl)
        
        # ---------------------------
        # Define custom posterior and prior functions for stability.
        # These functions now accept extra arguments via *args.
        # ---------------------------
        def posterior_mean_field_custom(*args, **kwargs):
            # If the first argument is a DType, assume the order is: dtype, kernel_shape, bias_size, ...
            if isinstance(args[0], tf.dtypes.DType):
                dtype = args[0]
                kernel_shape = args[1]
                bias_size = args[2] if len(args) > 2 else 0
            else:
                kernel_shape = args[0]
                bias_size = args[1] if len(args) > 1 else 0
                dtype = args[2] if len(args) > 2 else None
            n = int(np.prod(kernel_shape)) + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Independent(
                        tfp.distributions.Normal(loc=t[..., :n],
                                                scale=1e-3 + tf.nn.softplus(c + t[..., n:])),
                        reinterpreted_batch_ndims=1))
            ])

        def prior_fn(*args, **kwargs):
            if isinstance(args[0], tf.dtypes.DType):
                dtype = args[0]
                kernel_shape = args[1]
                bias_size = args[2] if len(args) > 2 else 0
            else:
                kernel_shape = args[0]
                bias_size = args[1] if len(args) > 1 else 0
                dtype = args[2] if len(args) > 2 else None
            n = int(np.prod(kernel_shape)) + bias_size
            return tf.keras.Sequential([
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Independent(
                        tfp.distributions.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
                        reinterpreted_batch_ndims=1))
            ])

        # ---------------------------
        # Build final Bayesian output layer using DenseFlipout, wrapped in a Lambda layer.
        # ---------------------------
        DenseFlipout = tfp.layers.DenseFlipout
        print("DEBUG: Creating DenseFlipout final layer with units (expected):", layer_sizes[-1])
        flipout_layer = DenseFlipout(
            units=layer_sizes[-1],
            activation='linear',
            kernel_posterior_fn=posterior_mean_field_custom,
            kernel_prior_fn=prior_fn,
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight_var,
            name="output_layer"
        )
        outputs = tf.keras.layers.Lambda(
            lambda t: flipout_layer(t),
            output_shape=lambda s: (s[0], layer_sizes[-1]),
            name="bayesian_dense_flipout"
        )(x)
        print("DEBUG: After DenseFlipout final layer (via Lambda), outputs shape:", outputs.shape, "Type:", type(outputs))
        
        # ---------------------------
        # Create and compile the model
        # ---------------------------
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print("DEBUG: Model created.")
        print("       Model input shape (actual):", self.model.input_shape)
        print("       Model output shape (actual):", self.model.output_shape)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.0001))
        print("DEBUG: Adam optimizer created with learning_rate (expected):", self.params.get('learning_rate', 0.0001))
        self.model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mae']
        )
        print("DEBUG: Model compiled with loss=Huber, metrics=['mae']")
        print("Predictor Model Summary:")
        self.model.summary()

        print("✅ Standard ANN model built successfully.")


    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        """
        Train the model with shape => x_train (N, input_dim), y_train (N, time_horizon).
        Implements KL annealing via a custom callback.
        """
        import tensorflow as tf
        # Ensure x_train and x_val are proper NumPy arrays.
        if isinstance(x_train, tuple):
            x_train = x_train[0]
        if x_val is not None and isinstance(x_val, tuple):
            x_val = x_val[0]
        
        print(f"Training with data => X: {x_train.shape}, Y: {y_train.shape}")
        exp_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != exp_horizon:
            raise ValueError(f"y_train shape {y_train.shape}, expected (N,{exp_horizon}).")
        
        # ---------------------------
        # Define KL Annealing Callback
        # ---------------------------
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

        # ---------------------------
        # Get annealing parameters and create callbacks
        # ---------------------------
        anneal_epochs = config.get("kl_anneal_epochs", 10) if config is not None else 10
        target_kl = self.params.get('kl_weight', 1e-4)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        callbacks = [kl_callback]
        
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.params.get('patience', 10),
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
        print("DEBUG: All predictions collected. Expected predictions array shape: (mc_samples, n_samples, time_horizon), Actual shape:", predictions.shape)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        print("DEBUG: Mean predictions computed. Shape:", mean_predictions.shape)
        print("DEBUG: Uncertainty estimates computed (std dev). Shape:", uncertainty_estimates.shape)
        return mean_predictions, uncertainty_estimates





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
