import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense, Dropout, Add, LayerNormalization, GlobalAveragePooling1D, Flatten
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention

class Plugin:
    """
    A predictor plugin using a simple Transformer network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 10,
        'batch_size': 256,
        'intermediate_layers': 1,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'num_heads': 2,
        'dropout_rate': 0.1
    }

    plugin_debug_vars = ['epochs', 'batch_size', 'input_dim', 'intermediate_layers', 'initial_layer_size', 'num_heads', 'dropout_rate']

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
        print(f"Transformer Layer sizes: {layers}")

        # Model
        inputs = Input(shape=(input_shape, 1), name="model_input")
        print(f"Transformer input_shape: {input_shape}")

        x = inputs
        for i, size in enumerate(layers[:-1]):
            print(f"Adding MultiHeadAttention with size {size} and num_heads {self.params['num_heads']}")
            x = self.multi_head_attention_block(x, size, self.params['num_heads'], self.params['dropout_rate'])
            print(f"Shape after MultiHeadAttention block {i + 1}: {x.shape}")

        # Final layer
        print(f"Adding final Dense layer with size {layers[-1]}")
        x = Dense(layers[-1], activation='linear', name="model_output")(x)
        print(f"Shape after final Dense layer: {x.shape}")

        self.model = Model(inputs=inputs, outputs=x, name="predictor_model")
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Debugging messages to trace the model configuration
        print("Predictor Model Summary:")
        self.model.summary()

    def multi_head_attention_block(self, x, size, num_heads, dropout_rate):
        print(f"Adding MultiHeadAttention with size {size} and num_heads {num_heads}")
        attn_output = MultiHeadAttention(head_num=num_heads)(x)
        print(f"Shape after MultiHeadAttention: {attn_output.shape}")

        attn_output = Dropout(dropout_rate)(attn_output)
        print(f"Shape after Dropout (MultiHeadAttention): {attn_output.shape}")

        out1 = Add()([x, attn_output])
        print(f"Shape after Add (residual connection - MultiHeadAttention): {out1.shape}")

        out1 = LayerNormalization(epsilon=1e-6)(out1)
        print(f"Shape after LayerNormalization (MultiHeadAttention): {out1.shape}")

        print(f"Adding feed-forward network with size {size}")
        ffn_output = Dense(size, activation='relu')(out1)
        print(f"Shape after Dense (feed-forward network): {ffn_output.shape}")

        ffn_output = Dropout(dropout_rate)(ffn_output)
        print(f"Shape after Dropout (feed-forward network): {ffn_output.shape}")

        ffn_output = Dense(size)(ffn_output)
        print(f"Shape after Dense (matching dimensions for residual connection): {ffn_output.shape}")

        out2 = Add()([out1, ffn_output])
        print(f"Shape after Add (residual connection - feed-forward network): {out2.shape}")

        out2 = LayerNormalization(epsilon=1e-6)(out2)
        print(f"Shape after LayerNormalization (feed-forward network): {out2.shape}")

        return out2

    def train(self, x_train, y_train, epochs, batch_size, threshold_error):
        # Ensure x_train is 3D
        if x_train.ndim == 2:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        print(f"Training predictor model with data shape: {x_train.shape}")
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Training completed.")
        mse = history.history['loss'][-1]
        if mse > threshold_error:
            print(f"Warning: Model training completed with MSE {mse} exceeding the threshold error {threshold_error}.")

    def predict(self, data):
        # Ensure data is 3D
        if data.ndim == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
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
    plugin.build_model(input_shape=8)
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
