import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, Add, LayerNormalization, GlobalAveragePooling1D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber


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
        Build a Transformer-based model with mandatory positional encoding.

        Automatically determines:
        - seq_len = input_shape[0]
        - num_features = input_shape[1]
        Also uses self.params for hyperparameters (with defaults if missing):
        - time_horizon
        - d_model (int) => dimension to embed num_features
        - num_heads (int)
        - ff_dim (int) => feed-forward expansion dimension
        - num_blocks (int) => how many Transformer blocks
        - dropout_rate (float)
        - learning_rate (float)
        - layer_size_divisor, intermediate_layers, etc. for final Dense config (optional usage)
        """

        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Input,
            Dense,
            GlobalAveragePooling1D,
            LayerNormalization,
            Add
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.initializers import GlorotUniform
        from multi_head_attention import MultiHeadAttention  # or your local MHA
        # If you're using a different MHA import, adjust accordingly.

        # 1) Extract shape & hyperparams
        seq_len, num_features = input_shape
        time_horizon = self.params['time_horizon']  # Required param
        d_model = self.params.get('d_model', 64)    # Default: 64
        num_heads = self.params.get('num_heads', 4)
        ff_dim = self.params.get('ff_dim', d_model * 4)
        num_blocks = self.params.get('num_blocks', 2)
        dropout_rate = self.params.get('dropout_rate', 0.1)

        # 2) (Optional) handle your 'layers' logic if you want extra layers
        #    Or you can skip this if you're just doing a pure Transformer final output.
        layers = []
        current_size = self.params.get('initial_layer_size', 64)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        int_layers = self.params.get('intermediate_layers', 0)
        for _ in range(int_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        # The final layer is always time_horizon (for multi-step output).
        layers.append(time_horizon)

        print(f"\n--- Transformer Build Info ---")
        print(f"Detected seq_len={seq_len}, num_features={num_features}")
        print(f"time_horizon={time_horizon}, d_model={d_model}, num_heads={num_heads}, ff_dim={ff_dim}, num_blocks={num_blocks}")
        print(f"Extra layer sizes for final Dense: {layers}")
        print("---------------------------------\n")

        # 3) Define input: (batch_size, seq_len, num_features)
        inputs = Input(shape=(seq_len, num_features), name="model_input")

        # 4) Project features from 'num_features' -> 'd_model'
        #    This is effectively the embedding step for the Transformer.
        x = Dense(d_model, name="feature_projection")(inputs)

        # 5) Mandatory sinusoidal positional encoding for time dimension
        pos_encoding = self._positional_encoding(seq_len, d_model)
        # Broadcast-add => (batch_size, seq_len, d_model)
        x = Add(name="add_pos_encoding")([x, pos_encoding])

        # 6) Stack num_blocks Transformer encoder blocks
        for i in range(num_blocks):
            block_name = f"transformer_block_{i+1}"
            x = self._transformer_encoder_block(
                x,
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout_rate,
                block_name=block_name
            )

        # 7) Pool across the time dimension => (batch_size, d_model)
        x = GlobalAveragePooling1D(name="gap")(x)

        # 8) Optionally apply extra Dense layers from 'layers[:-1]'
        #    If you want intermediate feed-forward layers beyond the Transformer blocks.
        for size in layers[:-1]:
            if size > 1:
                x = Dense(size, activation='relu')(x)

        # 9) Final output Dense => shape (time_horizon,)
        final_output_dim = layers[-1]
        model_output = Dense(
            final_output_dim,
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            name="model_output"
        )(x)

        # 10) Build & compile
        self.model = Model(inputs=inputs, outputs=model_output, name="predictor_model")
        adam_optimizer = Adam(
            learning_rate=self.params.get('learning_rate', 1e-3),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False
        )
        self.model.compile(
            optimizer=adam_optimizer,
            loss='mean_squared_error',
            metrics=['mse','mae']
        )

        print("Predictor Model Summary:")
        self.model.summary()

    def _transformer_encoder_block(self, x, d_model, num_heads, ff_dim, dropout, block_name):
        """
        A single Transformer encoder block with:
        - Multi-head self-attention
        - Residual + LayerNorm
        - 2-layer feed-forward sub-block
        - Residual + LayerNorm
        """
        # 1) Multi-Head Self-Attention
        attn_output = MultiHeadAttention(head_num=num_heads, name=f"{block_name}_mha")(x, x)
        x = Add(name=f"{block_name}_add_attn")([x, attn_output])
        x = LayerNormalization(name=f"{block_name}_ln_attn")(x)

        # 2) Feed-forward sub-block
        ffn = Dense(ff_dim, activation='relu', name=f"{block_name}_ff1")(x)
        ffn = Dense(d_model, name=f"{block_name}_ff2")(ffn)

        # Residual + LayerNorm
        x = Add(name=f"{block_name}_add_ff")([x, ffn])
        x = LayerNormalization(name=f"{block_name}_ln_ff")(x)
        return x

    def _positional_encoding(self, seq_len, d_model):
        """
        Always-used sinusoidal positional encoding.
        Returns a constant tensor of shape (1, seq_len, d_model),
        which will be broadcast-added to the projected inputs.
        """
        import numpy as np
        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pos_encoding[pos, i]   = np.sin(angle)
                if i + 1 < d_model:
                    pos_encoding[pos, i+1] = np.cos(angle)

        # shape => (1, seq_len, d_model)
        pos_encoding = np.expand_dims(pos_encoding, axis=0)
        return tf.constant(pos_encoding, dtype=tf.float32)


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
