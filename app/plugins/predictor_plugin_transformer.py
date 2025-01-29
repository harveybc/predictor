import numpy as np
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Input, Dropout, Add, LayerNormalization, GlobalAveragePooling1D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import tensorflow as tf
from tensorflow.keras.regularizers import l2


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
        Build the Transformer-based model with Dropout and L2 Regularization.
        """
        # Extract shapes and hyperparameters
        seq_len, num_features = input_shape  # e.g., (50, 1)
        time_horizon = self.params['time_horizon']
        d_model = self.params.get('d_model', 64)
        num_heads = self.params.get('num_heads', 2)
        ff_dim = self.params.get('ff_dim', d_model * 4)
        num_blocks = self.params.get('num_blocks', 2)
        dropout_rate = self.params.get('dropout_rate', 0.1)
        l2_reg = self.params.get('l2_reg', 1e-4)

        # Resolve additional Dense layer sizes
        layers = self._resolve_layers()

        print("\n--- Transformer Build Info ---")
        print(f"Detected seq_len={seq_len}, num_features={num_features}")
        print(f"time_horizon={time_horizon}, d_model={d_model}, num_heads={num_heads}, ff_dim={ff_dim}, num_blocks={num_blocks}")
        print(f"Extra layer sizes for final Dense: {layers}")
        print("---------------------------------\n")

        # 1. Define Input
        inputs = Input(shape=(seq_len, num_features), name="model_input")

        # 2. Feature Projection with Dropout and L2 Regularization
        x = Dense(
            d_model,
            activation='tanh',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="feature_projection"
        )(inputs)
        #x = Dropout(dropout_rate, name="projection_dropout")(x)  # Dropout after projectio

        # 4. Transformer Encoder Blocks with Dropout and L2 Regularization
        for i in range(num_blocks):
            block_name = f"transformer_block_{i+1}"
            x = self._transformer_encoder_block(
                x,
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                block_name=block_name
            )

        # 5. Global Average Pooling
        x = GlobalAveragePooling1D(name="global_avg_pool")(x)

        # 6. Intermediate Dense Layers with Dropout and L2 Regularization
        for idx, size in enumerate(layers[:-1]):
            if size > 1:
                x = Dense(
                    size,
                    activation='relu',
                    kernel_initializer=GlorotUniform(),
                    kernel_regularizer=l2(l2_reg),
                    name=f"intermediate_dense_{idx+1}"
                )(x)
                #x = Dropout(dropout_rate, name=f"intermediate_dropout_{idx+1}")(x)  # Dropout after intermediate Dense

        # 7. Final Output Layer with L2 Regularization
        # Output layer => shape (N, time_horizon)
        model_output = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name="model_output"
        )(x)

        # 8. Define and Compile Model
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
            metrics=['mse', 'mae']
        )

        print("Predictor Model Summary:")
        self.model.summary()

    def _transformer_encoder_block(self, x, d_model, num_heads, ff_dim, dropout_rate, l2_reg, block_name):
        """
        Define a single Transformer encoder block with Dropout and L2 Regularization.
        """
        # 1. Multi-Head Self-Attention without mask
        attn_layer = MultiHeadAttention(
            head_num=num_heads,
            name=f"{block_name}_mha"
        )
        attn_output = attn_layer([x, x, x])  # q, k, v

        # 2. Add & LayerNorm
        x = Add(name=f"{block_name}_add_attn")([x, attn_output])
        x = LayerNormalization(name=f"{block_name}_layer_norm_attn")(x)

        # 3. Feed-Forward Network with Dropout and L2 Regularization
        ff = Dense(
            ff_dim,
            activation='relu',
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name=f"{block_name}_ffn_dense_1"
        )(x)
        #ff = Dropout(dropout_rate, name=f"{block_name}_ffn_dropout_1")(ff)  # Dropout after first FF layer
        ff = Dense(
            d_model,
            activation=None,
            kernel_initializer=GlorotUniform(),
            kernel_regularizer=l2(l2_reg),
            name=f"{block_name}_ffn_dense_2"
        )(ff)
        #ff = Dropout(dropout_rate, name=f"{block_name}_ffn_dropout_2")(ff)  # Dropout after second FF layer

        # 4. Add & LayerNorm
        x = Add(name=f"{block_name}_add_ffn")([x, ff])
        x = LayerNormalization(name=f"{block_name}_layer_norm_ffn")(x)

        return x

    def _positional_encoding(self, seq_len, d_model):
        """
        Create sinusoidal positional encoding.
        """
        pos_encoding = np.zeros((seq_len, d_model), dtype=np.float32)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pos_encoding[pos, i] = np.sin(angle)
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(angle)
        pos_encoding = np.expand_dims(pos_encoding, axis=0)  # Shape: (1, seq_len, d_model)
        return tf.constant(pos_encoding, dtype=tf.float32)

    def _resolve_layers(self):
        """
        Determine the sizes of intermediate Dense layers based on parameters.
        """
        layers = []
        current_size = self.params.get('initial_layer_size', 64)
        layer_size_divisor = self.params.get('layer_size_divisor', 2)
        intermediate_layers = self.params.get('intermediate_layers', 3)
        for _ in range(intermediate_layers):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        layers.append(self.params['time_horizon'])  # Final layer size
        return layers


    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None):
        """
        Train method expects y_train to already be shaped for multi-step output
        (i.e., y_train.shape == (num_samples, time_horizon)).
        The data pipeline (process_data) must ensure a sliding window with stride = time_horizon.
        """
        # Ensure x_train is 3D
        callbacks = []
        patience = self.params.get('patience', 10)

        # Early stopping based on validation loss
        early_stopping_monitor = EarlyStopping(
            monitor='loss',  # Monitor validation loss
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_monitor)

        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None

        # Fit the model
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            #validation_data=validation_data,  # Provide validation data
            verbose=1,
            callbacks=callbacks,
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
