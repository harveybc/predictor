import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import logging
import os

class Plugin:
    """
    ANN Predictor Plugin using Keras for multi-step forecasting.

    This plugin builds, trains, and evaluates an Artificial Neural Network (ANN) model
    that outputs (N, time_horizon) predictions to match the pipeline's multi-step data.
    """

    # Default parameters for the ANN
    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.002,
        'activation': 'relu',
        'patience': 10,
        'l2_reg': 1e-4
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']
    
    def __init__(self):
        """
        Initializes the Plugin with default parameters and no model.
        """
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        """
        Updates the plugin parameters with provided keyword arguments.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Retrieves current debug info from plugin parameters.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds the plugin's debug info to an external dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Builds the ANN model for multi-step forecasting, with final layer dimension
        = self.params['time_horizon'].

        Args:
            input_shape (int): Number of input features (for ANN).
        """
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}. Must be int for ANN.")

        # Store input dimension for debugging
        self.params['input_dim'] = input_shape

        # Retrieve L2 factor and time horizon
        l2_reg = self.params.get('l2_reg', 1e-4)
        time_horizon = self.params['time_horizon']

        # Dynamically configure layer sizes
        layers = []
        current_size = self.params['initial_layer_size']
        divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // divisor, 1)
            int_layers += 1
        # Final layer = time_horizon for multi-step output
        layers.append(time_horizon)

        print(f"ANN Layer sizes: {layers}")
        print(f"ANN input_shape: {input_shape}")

        # Define model input
        model_input = Input(shape=(input_shape,), name="model_input")
        x = model_input

        # Hidden Dense layers
        for size in layers[:-1]:
            x = Dense(
                units=size,
                activation=self.params['activation'],
                kernel_initializer=HeNormal(),
                kernel_regularizer=l2(l2_reg)
            )(x)
            x = BatchNormalization()(x)

        # Output layer: shape (None, time_horizon)
        model_output = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)

        self.model = Model(inputs=model_input, outputs=model_output, name="ANN_Predictor_Model")

        # Adam optimizer
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        # Compile with Huber loss and typical regression metrics
        self.model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae']
        )
        
        print("Predictor Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Trains the ANN model using training data, multi-step shape = (N, time_horizon).
        Uses EarlyStopping (monitor=val_loss).
        """
        print(f"Training with data shape => X: {x_train.shape}, Y: {y_train.shape}")

        expected_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != expected_horizon:
            raise ValueError(
                f"y_train has shape {y_train.shape}, but expected (N,{expected_horizon}). "
                "Check multi-step logic."
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
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
            validation_data=(x_val, y_val) if (x_val is not None and y_val is not None) else None
        )

        print("Training completed.")
        final_loss = history.history['loss'][-1]
        print(f"Final training loss (Huber): {final_loss}")

        if final_loss > threshold_error:
            print(f"Warning: final_loss={final_loss} > threshold_error={threshold_error}.")

    def predict(self, data):
        """
        Generate predictions => shape (N, time_horizon).
        """
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        preds = self.model.predict(data)
        return preds

    def calculate_mse(self, y_true, y_pred):
        """
        Flatten-based MSE => consistent with multi-step shape (N, time_horizon).
        """
        print(f"Calculating MSE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        print(f"Calculated MSE => {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Flatten-based MAE => consistent with multi-step shape (N, time_horizon).
        """
        print(f"Calculating MAE => y_true={y_true.shape}, y_pred={y_pred.shape}")
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Mismatch => y_true={y_true.shape}, y_pred={y_pred.shape}")
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        return mae

    def save(self, file_path):
        """
        Saves the trained model to file.
        """
        save_model(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a model from file.
        """
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}")
