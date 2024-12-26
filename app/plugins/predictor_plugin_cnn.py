import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2

class Plugin:
    """
    A predictor plugin using a convolutional neural network (CNN) based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001
        

    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_shape', 'intermediate_layers', 'initial_layer_size']

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

    def build_model(self, input_shape):
        """
        Build a CNN-based model with sliding window input.

        Parameters:
            input_shape (tuple): Shape of the input data (window_size, features).
        """
        self.params['input_shape'] = input_shape
        print(f"CNN input_shape: {input_shape}")

        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        # Output layer size is time_horizon
        layers.append(self.params['time_horizon'])  # Output layer size

        # Debugging message
        print(f"CNN Layer sizes: {layers}")

        # Correct Input layer: Use the input_shape directly without extra tuple
        inputs = Input(shape=input_shape, name="model_input")
        x = inputs

        for idx, size in enumerate(layers[:-1]):
            if size > 1:
                x = Conv1D(
                    filters=size, 
                    kernel_size=3, 
                    activation='relu', 
                    kernel_initializer=HeNormal(), 
                    padding='same',
                    kernel_regularizer=l2(self.params.get('l2_reg', 1e-4)),
                    name=f"conv1d_{idx+1}"
                )(x)
                x = BatchNormalization(name=f"batch_norm_{idx+1}")(x)
                # Dropout lines are removed as per your request
                x = MaxPooling1D(pool_size=2, name=f"max_pool_{idx+1}")(x)

        x = Flatten(name="flatten")(x)
        # Dropout after Flatten is removed as per your request
        model_output = Dense(
            layers[-1], 
            activation='tanh', 
            kernel_initializer=GlorotUniform(), 
            kernel_regularizer=l2(self.params.get('l2_reg', 1e-4)),
            name="model_output"
        )(x)

        self.model = Model(inputs=inputs, outputs=model_output, name="cnn_model")

        adam_optimizer = Adam(
            learning_rate=self.params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        self.model.compile(
            optimizer=adam_optimizer, 
            loss=Huber(), 
            metrics=['mse','mae'], 
            run_eagerly=False  # Set to False for better performance unless debugging
        )

        # Debugging messages to trace the model configuration
        print("CNN Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train the CNN model with Early Stopping to prevent overfitting.

        Parameters:
            x_train (numpy.ndarray): Training input data.
            y_train (numpy.ndarray): Training target data.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            threshold_error (float): Threshold for loss to trigger warnings.
            x_val (numpy.ndarray, optional): Validation input data.
            y_val (numpy.ndarray, optional): Validation target data.
        """
        callbacks = []

        # Early Stopping based on validation loss if available
        patience = self.params.get('patience', 5)  # default patience is 5 epochs
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss' if (x_val is not None and y_val is not None) else 'loss',
            patience=patience, 
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)

        print(f"Training CNN model with data shape: {x_train.shape}")
        if x_val is not None and y_val is not None:
            history = self.model.fit(
                x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=1, 
                validation_data=(x_val, y_val),
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=1, 
                callbacks=callbacks
            )
        print("Training completed.")
        mse = history.history['val_loss'][-1] if (x_val is not None and y_val is not None) else history.history['loss'][-1]
        if mse > threshold_error:
            print(f"Warning: Model training completed with MSE {mse} exceeding the threshold error {threshold_error}.")

    def predict(self, data):
        """
        Generate predictions using the trained CNN model.

        Parameters:
            data (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted outputs.
        """
        # CNN expects data to be (samples, window_size, features)

        print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        print(f"Predicted data shape: {predictions.shape}")
        return predictions


    def calculate_mse(self, y_true, y_pred):
        """
        Calculate MSE without losing the 2D alignment (N, time_horizon).
        We do flatten them both consistently into 1D, but preserve the step count.
        """
        print(f"Calculating MSE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_mse: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )

        # Flatten them consistently
        y_true_flat = y_true.reshape(-1)  # (N * time_horizon,)
        y_pred_flat = y_pred.reshape(-1)

        print(f"Shapes after flattening: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")

        # Calculate absolute diffs and then MSE
        abs_difference = np.abs(y_true_flat - y_pred_flat)
        print(f"Absolute differences: {abs_difference}")
        squared_abs_difference = abs_difference ** 2
        print(f"Squared absolute differences: {squared_abs_difference}")

        mse = np.mean(squared_abs_difference)
        print(f"Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        """
        Calculate MAE without losing the 2D alignment (N, time_horizon).
        """
        print(f"Calculating MAE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")

        # Ensure both y_true and y_pred have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in calculate_mae: y_true={y_true.shape}, y_pred={y_pred.shape}"
            )

        # Flatten them consistently
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        print(f"Shapes after flattening: y_true={y_true_flat.shape}, y_pred={y_pred_flat.shape}")

        abs_difference = np.abs(y_true_flat - y_pred_flat)
        print(f"Absolute differences: {abs_difference}")

        mae = np.mean(abs_difference)
        print(f"Calculated MAE: {mae}")
        return mae


    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Predictor model loaded from {file_path}")

# Debugging usage example
if __name__ == "__main__":
    plugin = Plugin()
    plugin.build_model(input_shape=8)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
