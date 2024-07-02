import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam

class Plugin:
    """
    A predictor plugin using a simple LSTM network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'initial_layer_size': 64,
        'layer_size_divisor': 2
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size']

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
        self.params['input_dim'] = input_shape

        # Layer configuration
        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        int_layers = 0
        while int_layers < self.params['intermediate_layers']:
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
            int_layers += 1
        layers.append(1)  # Output layer size

        # Debugging message
        print(f"LSTM Layer sizes: {layers}")

        # Model
        model_input = Input(shape=(input_shape, 1), name="model_input")
        print(f"LSTM input_shape: {input_shape}")

        x = model_input
        for size in layers[:-1]:
            x = LSTM(size, activation='relu', return_sequences=True)(x)
        x = LSTM(layers[-2], activation='relu')(x)
        model_output = Dense(layers[-1], activation='linear', name="model_output")(x)
        
        self.model = Model(inputs=model_input, outputs=model_output, name="predictor_model")
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Predictor Model Summary:")
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size, threshold_error):
        print(f"Training predictor model with data shape: {x_train.shape}")
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")
        mse = history.history['loss'][-1]
        if mse > threshold_error:
            print(f"Warning: Model training completed with MSE {mse} exceeding the threshold error {threshold_error}.")

    def predict(self, data):
        print(f"Predicting data with shape: {data.shape}")
        predictions = self.model.predict(data)
        print(f"Predicted data shape: {predictions.shape}")
        return predictions

    def calculate_mse(self, y_true, y_pred):
        print(f"Calculating MSE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        y_pred = y_pred.flatten()  # Ensure y_pred is a 1D array
        abs_difference = np.abs(np.array(y_true) - np.array(y_pred))
        squared_abs_difference = abs_difference ** 2
        mse = np.mean(squared_abs_difference)
        print(f"Calculated MSE: {mse}")
        return mse

    def calculate_mae(self, y_true, y_pred):
        print(f"Calculating MAE for shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
        y_pred = y_pred.flatten()  # Ensure y_pred is a 1D array
        abs_difference = np.abs(np.array(y_true) - np.array(y_pred))
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
    plugin.build_model(input_shape=128)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
