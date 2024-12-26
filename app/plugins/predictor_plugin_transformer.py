import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, Add, LayerNormalization, GlobalAveragePooling1D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import tensorflow as tf


class Plugin:
    """
    A predictor plugin using a Transformer network based on Keras, with dynamically configurable size.
    """

    plugin_params = {
        'epochs': 200,  # Increased number of epochs
        'batch_size': 128,
        'intermediate_layers': 3,  
        'initial_layer_size': 64,  
        'layer_size_divisor': 2,
        'num_heads': 2,  # Keeping the number of heads dependent on size as before
        'learning_rate': 0.0001,
        'dropout_rate': 0.1
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
        """
        Build a Transformer-based model using `keras_multi_head.MultiHeadAttention`, 
        even if seq_len=1. This code disables mask usage in MHA calls to avoid 
        rank-4 mask errors for single-time-step cases. All other code remains 
        unchanged so you can keep using TensorFlow 2.10 and the 'keras-multi-head' library.
        """

        # For clarity, define or fetch hyperparameters from self.params
        seq_len, num_features = input_shape  # e.g. (1, 50) if you've reshaped x_train
        time_horizon = self.params['time_horizon']  # required
        d_model = self.params.get('d_model', 64)
        num_heads = self.params.get('num_heads', 2)
        ff_dim = self.params.get('ff_dim', d_model * 4)
        num_blocks = self.params.get('num_blocks', 2)
        dropout_rate = self.params.get('dropout_rate', 0.1)

        # Extra dense layer sizes if you like
        layers = self._resolve_layers()

        print("\n--- Transformer Build Info ---")
        print(f"Detected seq_len={seq_len}, num_features={num_features}")
        print(f"time_horizon={time_horizon}, d_model={d_model}, num_heads={num_heads}, ff_dim={ff_dim}, num_blocks={num_blocks}")
        print(f"Extra layer sizes for final Dense: {layers}")
        print("---------------------------------\n")

        inputs = Input(shape=(seq_len, num_features), name="model_input")

        # 1) Project from num_features -> d_model
        x = Dense(d_model, name="feature_projection")(inputs)

        # 2) Sinusoidal positional encoding (always used)
        pos_encoding = self._positional_encoding(seq_len, d_model)
        x = Add(name="add_pos_encoding")([x, pos_encoding])

        # 3) Stack multiple encoder blocks
        for i in range(num_blocks):
            block_name = f"transformer_block_{i+1}"
            x = self._transformer_encoder_block(
                x,
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                block_name=block_name
            )

        # 4) Pool across time dimension -> (batch_size, d_model)
        x = GlobalAveragePooling1D(name="gap")(x)

        # 5) Optional extra Dense layers from layers[:-1]
        for size in layers[:-1]:
            if size > 1:
                x = Dense(size, activation='relu')(x)

        # 6) Final output => (batch_size, time_horizon)
        final_output_dim = layers[-1]
        model_output = Dense(
            final_output_dim,
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            name="model_output"
        )(x)

        # 7) Build & compile
        self.model = Model(inputs=inputs, outputs=model_output, name="predictor_model")

        adam_optimizer = Adam(
            learning_rate=self.params.get('learning_rate', 1e-3),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )

        # Using 'mse' to match prior usage, or Huber() if you prefer
        self.model.compile(
            optimizer=adam_optimizer,
            loss='mean_squared_error',
            metrics=['mse','mae']
        )

        print("Predictor Model Summary:")
        self.model.summary()


    def _transformer_encoder_block(self, x, d_model, num_heads, ff_dim, block_name):
        """
        A single Transformer encoder block using 'keras_multi_head.MultiHeadAttention'
        but forcing mask=None to avoid rank-4 mask errors when seq_len=1.
        """

        # 1) Multi-Head Self-Attention, disabling mask
        #    If we pass 'mask=None', the library won't try to create a 4D mask.
        attn_output = MultiHeadAttention(
            head_num=num_heads,
            name=f"{block_name}_mha"
        )(x, x, mask=None)  # <--- forcibly passing mask=None

        x = Add(name=f"{block_name}_add_attn")([x, attn_output])
        x = LayerNormalization(name=f"{block_name}_ln_attn")(x)

        # 2) 2-layer feed-forward sub-block
        ff = Dense(ff_dim, activation='relu', name=f"{block_name}_ff1")(x)
        ff = Dense(d_model, name=f"{block_name}_ff2")(ff)

        x = Add(name=f"{block_name}_add_ff")([x, ff])
        x = LayerNormalization(name=f"{block_name}_ln_ff")(x)
        return x


    def _positional_encoding(self, seq_len, d_model):
        """
        Sinusoidal positional encoding => (1, seq_len, d_model).
        Always used, even if seq_len=1.
        """
        import numpy as np

        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pos_encoding[pos, i] = np.sin(angle)
                if i + 1 < d_model:
                    pos_encoding[pos, i+1] = np.cos(angle)

        pos_encoding = np.expand_dims(pos_encoding, axis=0)   # => (1, seq_len, d_model)
        return tf.constant(pos_encoding, dtype=tf.float32)


    def _resolve_layers(self):
        """
        Helper method to build the list of Dense layer sizes for final config.
        E.g. might read from self.params['initial_layer_size'], 
        self.params['intermediate_layers'], etc., then append time_horizon as last.
        """
        layers = []
        current_size = self.params.get('initial_layer_size', 64)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        int_layers = self.params.get('intermediate_layers', 2)
        for _ in range(int_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        # Final layer => time_horizon
        layers.append(self.params['time_horizon'])
        return layers



    def train(self, x_train, y_train, epochs, batch_size, threshold_error):
        """
        Train method expects y_train to already be shaped for multi-step output
        (i.e., y_train.shape == (num_samples, time_horizon)).
        The data pipeline (process_data) must ensure a sliding window with stride = time_horizon.
        """
        # Ensure x_train is 3D
        if x_train.ndim == 2:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)


        callbacks = []
    
        patience = self.params.get('patience', 5)  # default patience is 10 epochs
        early_stopping_monitor = EarlyStopping(
            monitor='loss', 
            patience=patience, 
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)

        print(f"Training predictor model with data shape: {x_train.shape}")
        history = self.model.fit(
            x_train, 
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        print("Training completed.")

        # Check final training loss
        mse = history.history['loss'][-1]
        if mse > threshold_error:
            print(f"Warning: Model training completed with MSE {mse} exceeding the threshold error {threshold_error}.")

    def predict(self, data):
        """
        If the model outputs multiple steps (time_horizon), the returned predictions
        will have shape (num_samples, time_horizon).
        """
        # Ensure data is 3D
        if data.ndim == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)

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
    plugin.build_model(input_shape=8)  # Adjusted to 8 as per your data
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")
