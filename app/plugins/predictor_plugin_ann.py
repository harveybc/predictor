import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2

class Plugin:
    """
    A predictor plugin using a simple Artificial Neural Network (ANN) based on Keras, with dynamically configurable size.
    This plugin supports Early Stopping and L2 Regularization to prevent overfitting.
    """

    # Default parameters for the ANN plugin
    plugin_params = {
        'epochs': 200,                    # Number of training epochs
        'batch_size': 128,                # Batch size for training
        'intermediate_layers': 3,         # Number of intermediate Dense layers
        'initial_layer_size': 64,         # Number of neurons in the first Dense layer
        'layer_size_divisor': 2,          # Factor to reduce the number of neurons in subsequent layers
        'learning_rate': 0.0001,          # Learning rate for the Adam optimizer
        'activation': 'tanh',             # Activation function for Dense layers
        'patience': 5,                     # Patience for Early Stopping
        'l2_reg': 1e-4,                    # L2 regularization factor
        'time_horizon': 6                  # Number of future steps to predict (should be set externally)
    }

    # Variables to include in debug information
    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

    def __init__(self):
        """
        Initializes the Plugin instance with default parameters and no model.
        """
        self.params = self.plugin_params.copy()  # Copy default parameters to instance
        self.model = None                          # Placeholder for the Keras model

    def set_params(self, **kwargs):
        """
        Updates the plugin parameters with provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments to update plugin parameters.
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                print(f"Warning: Parameter '{key}' is not recognized and will be ignored.")

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
            input_shape (int or tuple): Shape of the input data.
                - For ANN, it should be an integer representing the number of features.
                - For other plugins (if extended), it might be a tuple.
        
        Raises:
            ValueError: If input_shape is not an integer for ANN.
        """
        # Validate and set input_dim based on input_shape
        if isinstance(input_shape, tuple):
            if len(input_shape) == 1 and isinstance(input_shape[0], int):
                input_dim = input_shape[0]
            else:
                raise ValueError(f"For ANN, input_shape should be an integer or a single-element tuple, got {input_shape}.")
        elif isinstance(input_shape, int):
            input_dim = input_shape
        else:
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}. Must be int or single-element tuple.")

        self.params['input_dim'] = input_dim  # Store input_dim for debugging

        # Retrieve L2 regularization factor
        l2_reg = self.params.get('l2_reg', 1e-4)

        # Configure the layer sizes
        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0

        # Dynamically create intermediate layer sizes
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)  # Ensure layer size doesn't go below 1
            int_layers += 1

        # Append the output layer size based on time_horizon
        layers.append(self.params['time_horizon'])

        print(f"ANN Layer sizes: {layers}")
        print(f"ANN input_shape: {input_dim}")

        # Define the model input
        model_input = Input(shape=(input_dim,), name="model_input")
        x = model_input

        # Add intermediate Dense layers with activation and L2 regularization
        for size in layers[:-1]:
            if size > 1:
                x = Dense(
                    units=size,
                    activation=self.params['activation'],
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name=f"Dense_{size}"
                )(x)
                # Uncomment the next line to add Batch Normalization after each Dense layer
                # x = BatchNormalization()(x)
                # Optionally, add Dropout for regularization
                # x = Dropout(0.5)(x)

        # Add the final output Dense layer
        model_output = Dense(
            units=layers[-1],
            activation='linear',  # Typically 'linear' for regression tasks
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)

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

    def train(self, x_train, y_train, epochs, batch_size, threshold_error):
        """
        Trains the ANN model using the provided training data.
        Implements Early Stopping with specified patience to prevent overfitting.

        Args:
            x_train (numpy.ndarray): Training input data of shape (N, features).
            y_train (numpy.ndarray): Training target data of shape (N, time_horizon).
            epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per gradient update.
            threshold_error (float): Threshold for Mean Squared Error to trigger warnings.
        
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
            monitor='loss',                   # Monitor training loss
            patience=self.params['patience'],# Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True,        # Restore model weights from the epoch with the best value of the monitored quantity
            verbose=1                         # Verbosity mode
        )
        callbacks.append(early_stopping_monitor)

        # Train the model
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
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
        Generates predictions using the trained ANN model.

        Args:
            data (numpy.ndarray): Input data for prediction of shape (N, features).
        
        Returns:
            numpy.ndarray: Predicted values of shape (N, time_horizon).
        """
        print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        print(f"Predicted data shape: {predictions.shape}")
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
        print(f"Calculated MAE: {mae}")
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
        # Example usage for debugging purposes
        plugin = Plugin()
        plugin.build_model(input_shape=8)  # Adjusted to 8 as per your data
        debug_info = plugin.get_debug_info()
        print(f"Debug Info: {debug_info}")
   