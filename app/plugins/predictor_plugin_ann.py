import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import logging
import os

class Plugin:
    """
    ANN Predictor Plugin using Keras.
    
    This plugin builds, trains, and evaluates an Artificial Neural Network (ANN) model.
    It supports dynamic configuration of layer sizes, Early Stopping with patience, and L2 Regularization.
    """
    
    # Default parameters for the ANN
    plugin_params = {
        'batch_size': 128,                # Number of samples per gradient update
        'intermediate_layers': 3,         # Number of hidden Dense layers
        'initial_layer_size': 128,         # Number of neurons in the first Dense layer
        'layer_size_divisor': 2,          # Factor to reduce neurons in subsequent layers
        'learning_rate': 0.002,          # Learning rate for the Adam optimizer
        'activation': 'relu',             # Activation function for Dense layers
        'patience': 10,                     # Patience parameter for Early Stopping
        'l2_reg': 1e-4                    # L2 regularization factor
    }

    # Variables to include in debug information
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']
    
    def __init__(self):
        """
        Initializes the Plugin with default parameters and no model.
        """
        self.params = self.plugin_params.copy()  # Initialize parameters
        self.model = None                          # Placeholder for the Keras model

    def set_params(self, **kwargs):
        """
        Updates the plugin parameters with provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to update plugin parameters.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Retrieves the current values of debug variables.
        
        Returns:
            dict: Dictionary containing debug information.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds the plugin's debug information to an external debug_info dictionary.
        
        Args:
            debug_info (dict): External dictionary to update with debug information.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_model(self, input_shape):
        """
        Builds the ANN model based on the specified input shape and plugin parameters.
        Incorporates L2 regularization and uses the specified activation function.
        
        Args:
            input_shape (int): Number of input features.
        
        Raises:
            ValueError: If input_shape is not an integer.
        """
        # Validate input_shape
        if not isinstance(input_shape, int):
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}. Must be int for ANN.")
        
        self.params['input_dim'] = input_shape  # Store input dimension for debugging
        
        # Retrieve L2 regularization factor
        l2_reg = self.params.get('l2_reg', 1e-4)
        
        # Configure layer sizes dynamically
        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)  # Prevent layer size from becoming 0
            int_layers += 1
        layers.append(self.params['time_horizon'])  # Output layer size based on time_horizon
        
        print(f"ANN Layer sizes: {layers}")
        print(f"ANN input_shape: {input_shape}")
        
        # Define model input
        model_input = Input(shape=(input_shape,), name="model_input")
        x = model_input
        
        # Add hidden Dense layers with activation and L2 regularization
        for size in layers[:-1]:
            if size > 1:
                x = Dense(
                    units=size,
                    activation=self.params['activation'],
                    kernel_initializer=HeNormal(),
                    kernel_regularizer=l2(l2_reg),
                    name=f"Dense_{size}"
                )(x)
                #add a batch normalization layer
                x = BatchNormalization()(x)
        
        # Add output Dense layer with linear activation for regression
        model_output = Dense(
            units=layers[-1],
            activation='linear',  # 'linear' activation for regression tasks
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)
        # Add a batch normalization layer to the output
        #model_output = BatchNormalization()(model_output)

        # Create the Keras Model
        self.model = Model(inputs=model_input, outputs=model_output, name="ANN_Predictor_Model")
        
        # Define the Adam optimizer with specified learning rate
        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        
        # Compile the model with Huber loss and evaluation metrics
        self.model.compile(
            optimizer=adam_optimizer,
            loss=Huber(),
            metrics=['mse', 'mae']
        )
        
        print("Predictor Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Trains the ANN model using the provided training data.
        Implements Early Stopping with specified patience to prevent overfitting.
        
        Args:
            x_train (numpy.ndarray): Training input data of shape (N, features).
            y_train (numpy.ndarray): Training target data of shape (N, time_horizon).
            epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per gradient update.
            threshold_error (float): Threshold for Mean Squared Error to trigger warnings.
            x_val (numpy.ndarray, optional): Validation input data.
            y_val (numpy.ndarray, optional): Validation target data.
        
        Raises:
            ValueError: If y_train does not match the expected time_horizon.
        """
        # Debug message indicating the start of training
        print(f"Training predictor model with data shape: X: {x_train.shape}, Y: {y_train.shape}")
        
        # Check if y_train matches the expected time_horizon
        expected_horizon = self.params['time_horizon']
        if y_train.ndim != 2 or y_train.shape[1] != expected_horizon:
            raise ValueError(
                f"y_train has shape {y_train.shape}, but expected time_horizon is {expected_horizon}. "
                f"Ensure these match for correct training."
            )
        
        # Initialize callbacks list
        callbacks = []
        
        # Set up EarlyStopping callback with patience=5
        early_stopping_monitor = EarlyStopping(
            monitor= 'val_loss',  # Monitor validation loss if available

            patience=self.params['patience'],                                            # Number of epochs with no improvement
            restore_best_weights=True,                                                   # Restore model weights from the epoch with the best value
            verbose=1                                                                    # Verbosity mode
        )
        callbacks.append(early_stopping_monitor)
        
        # Train the model with or without validation data
        print("Training with only training data...")
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
            validation_data=(x_val, y_val) if x_val is not None else None
        )
        
        print("Training completed.")
        
        # Retrieve the final loss from the training history
        final_loss = history.history['loss'][-1]
        print(f"Final training loss (Huber): {final_loss}")
        
        # Check if the final loss exceeds the threshold_error
        if final_loss > threshold_error:
            print(f"Warning: Model training completed with loss {final_loss} exceeding the threshold error {threshold_error}.")

    def predict(self, data):
        """
        Generate predictions using the trained ANN model.

        Args:
            data (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted values of shape (N, time_horizon).
        """
                # Suppress TensorFlow logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        #print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        #print(f"Predicted data shape: {predictions.shape}")
        return predictions


    def calculate_mse(self, y_true, y_pred):
        """
        Calculates the Mean Squared Error (MSE) between true and predicted values.
        
        Args:
            y_true (numpy.ndarray): True target values of shape (N, time_horizon).
            y_pred (numpy.ndarray): Predicted target values of shape (N, time_horizon).
        
        Returns:
            float: Calculated MSE.
        
        Raises:
            ValueError: If the shapes of y_true and y_pred do not match.
        """
        print(f"Calculating MSE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_mse: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        
        # Flatten the arrays to 1D for MSE calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        print(f"Shapes after flattening: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")
        
        # Calculate Mean Squared Error
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        print(f"Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Calculates the Mean Absolute Error (MAE) between true and predicted values.
        
        Args:
            y_true (numpy.ndarray): True target values of shape (N, time_horizon).
            y_pred (numpy.ndarray): Predicted target values of shape (N, time_horizon).
        
        Returns:
            float: Calculated MAE.
        
        Raises:
            ValueError: If the shapes of y_true and y_pred do not match.
        """
        print(f"Calculating MAE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        
        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_mae: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )
        
        # Flatten the arrays to 1D for MAE calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        print(f"Shapes after flattening: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")
        
        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        #print(f"Calculated MAE: {mae}")
        return mae

    def save(self, file_path):
        """
        Saves the trained model to the specified file path.
        
        Args:
            file_path (str): Path to save the model.
        """
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        """
        Loads a trained model from the specified file path.
        
        Args:
            file_path (str): Path to load the model from.
        """
        self.model = load_model(file_path)
        print(f"Predictor model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.build_model(input_shape=8)  # Adjusted to 8 as per your data
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
