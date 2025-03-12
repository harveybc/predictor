import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, MultiHeadAttention, Add, LayerNormalization, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K
import gc


class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: ReduceLROnPlateau patience counter: {self.patience_counter}")


class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wait > 0:
            self.patience_counter = self.wait
        else:
            self.patience_counter = 0
        print(f"DEBUG: EarlyStopping patience counter: {self.patience_counter}")


class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()


class Plugin:
    """
    Transformer Predictor Plugin using Keras for multi-step forecasting.
    """

    plugin_params = {
        'batch_size': 128,
        'intermediate_layers': 3,
        'initial_layer_size': 128,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-2,
        'kl_weight': 1e-3,
        'num_heads': 4
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
        Builds a Transformer model with a Bayesian output layer.
        """
        self.params['input_dim'] = input_shape
        l2_reg = self.params.get('l2_reg', 1e-4)

        layers = []
        current_size = self.params['initial_layer_size']
        layer_size_divisor = self.params['layer_size_divisor']
        for _ in range(self.params['intermediate_layers']):
            layers.append(current_size)
            current_size = max(current_size // layer_size_divisor, 1)
        layers.append(self.params['time_horizon'])

        print(f"Transformer Layer sizes: {layers}")
        print(f"Transformer input shape: {input_shape}")

        model_input = Input(shape=(input_shape,), name="model_input")
        x = model_input

        for idx, size in enumerate(layers[:-1]):
            x_dense = Dense(
                units=size,
                activation=self.params['activation'],
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name=f"dense_layer_{idx+1}"
            )(x)

            sequence_length = 8
            feature_dim = max(1, size // sequence_length)

            x_reshaped = Reshape((sequence_length, feature_dim), name=f"reshape_{idx+1}")(x_dense)

            num_heads = self.params['num_heads']
            key_dim = max(1, feature_dim // num_heads)

            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                name=f"mha_layer_{idx+1}"
            )(query=x_reshaped, value=x_reshaped)

            attention_proj = Dense(
                units=feature_dim,
                activation='tanh',
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_reg),
                name=f"mha_projection_{idx+1}"
            )(attention_output)

            x_att_proj = Reshape((sequence_length * feature_dim,), name=f"reshape_back_{idx+1}")(attention_proj)

            x = Add(name=f"residual_add_{idx+1}")([x_dense, x_att_proj])

        x = BatchNormalization(name="batch_norm_final")(x)

        # Bayesian Output Layer (DenseFlipout + Deterministic Bias)
        DenseFlipout = tfp.layers.DenseFlipout
        bayesian_output = DenseFlipout(
            units=layers[-1],
            activation='linear',
            name="output_layer"
        )(x)

        bias_layer = Dense(
            units=layers[-1],
            activation='linear',
            kernel_initializer=GlorotUniform(),
            name="deterministic_bias"
        )(x)

        outputs = bayesian_output + bias_layer
        self.model = Model(inputs=model_input, outputs=outputs, name="predictor_model")

        self.model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss=self.custom_loss,
            metrics=['mae']
        )

        print("Predictor Model Summary:")
        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        return Huber()(y_true, y_pred)

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val=None, y_val=None, config=None):
        patience_value = self.params.get('early_patience', 10)
        min_delta = 1e-4

        early_stopping_monitor = EarlyStoppingWithPatienceCounter(
            monitor='val_loss',
            patience=patience_value,
            restore_best_weights=True,
            verbose=1,
            min_delta=min_delta
        )

        reduce_lr_patience = max(1, patience_value // 3)
        reduce_lr_monitor = ReduceLROnPlateauWithCounter(
            monitor='val_loss',
            factor=0.1,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )

        callbacks = [early_stopping_monitor, reduce_lr_monitor, ClearMemoryCallback()]

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
            validation_data=(x_val, y_val)
        )

        #train_predictions = self.predict(x_train)
        mc_samples = config.get("mc_samples", 100)
        train_predictions, uncertainty_estimates = self.predict_with_uncertainty(x_train, mc_samples=mc_samples)
        #val_predictions = self.predict(x_val)
        val_predictions, uncertainty_estimates =  self.predict_with_uncertainty(x_val, mc_samples=mc_samples)
        return history, train_predictions, val_predictions


    def predict(self, data):
        return self.model.predict(data)

    def predict_with_uncertainty(self, data, mc_samples=100):
        predictions = np.array([self.model(data, training=True).numpy() for _ in range(mc_samples)])
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty_estimates = np.std(predictions, axis=0)
        return mean_predictions, uncertainty_estimates

    def calculate_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))

    def save(self, file_path):
        save_model(self.model, file_path)
        print(f"Predictor model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path)
        print(f"Predictor model loaded from {file_path}")
