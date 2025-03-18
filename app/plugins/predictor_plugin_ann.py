import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error
import tensorflow.keras.backend as K
import gc

class Plugin:
    """
    ANN Predictor Plugin adapted for direct multi-output.
    """

    plugin_params = {
        'batch_size': 32,
        'intermediate_layers': 3,
        'initial_layer_size': 64,
        'layer_size_divisor': 2,
        'learning_rate': 0.0001,
        'activation': 'tanh',
        'l2_reg': 1e-5,
        'kl_weight': 1e-3,
        'time_horizon': 6,
        'mmd_lambda': 0.01
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def build_model(self, input_shape):
        """
        Builds a Bayesian ANN for multi-output forecasting.
        """
        KL_WEIGHT = self.params['kl_weight']
        mmd_lambda = self.params['mmd_lambda']

        inputs = Input(shape=(input_shape,), name="model_input")

        x = inputs
        current_size = self.params['initial_layer_size']

        # Build intermediate layers
        for idx in range(self.params['intermediate_layers']):
            x = Dense(
                units=current_size,
                activation=self.params['activation'],
                kernel_initializer='he_normal',
                name=f"dense_layer_{idx+1}"
            )(x)
            current_size = max(current_size // self.params['layer_size_divisor'], 4)

        # Bayesian DenseFlipout layer
        DenseFlipout = tfp.layers.DenseFlipout
        bayesian_output = DenseFlipout(
            units=self.params['time_horizon'],
            activation='linear',
            kernel_divergence_fn=(lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT),
            name="bayesian_output_layer"
        )(x)

        # Deterministic bias layer for stability
        bias_output = Dense(
            units=self.params['time_horizon'],
            activation='linear',
            kernel_initializer='zeros',
            name="bias_output_layer"
        )(x)

        final_outputs = bayesian_output + bias_output

        # Split outputs explicitly for direct multi-output
        outputs_list = tf.split(final_outputs, num_or_size_splits=self.params['time_horizon'], axis=1)
        outputs_list = [tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1), name=f"output_{i+1}")(o) 
                        for i, o in enumerate(outputs_list)]

        self.model = Model(inputs=inputs, outputs=outputs_list)

        # Compile with individual loss for each output
        self.model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss=[self.custom_loss for _ in range(self.params['time_horizon'])],
            metrics=['mae']
        )
        self.mmd_lambda = mmd_lambda

        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        """
        Custom loss combining Huber loss and Maximum Mean Discrepancy (MMD).
        """
        huber_loss = Huber()(y_true, y_pred)
        mmd_loss = self.compute_mmd(y_pred, y_true)
        return huber_loss + self.mmd_lambda * mmd_loss

    def compute_mmd(self, x, y, sigma=1.0, sample_size=256):
        """
        Compute Maximum Mean Discrepancy (MMD) using Gaussian Kernel.
        """
        idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:sample_size]
        x_sample = tf.gather(x, idx)
        y_sample = tf.gather(y, idx)

        def gaussian_kernel(a, b):
            a = tf.expand_dims(a, 1)
            b = tf.expand_dims(b, 0)
            dist = tf.reduce_sum(tf.square(a - b), axis=-1)
            return tf.exp(-dist / (2 * sigma**2))

        K_xx = gaussian_kernel(x_sample, x_sample)
        K_yy = gaussian_kernel(y_sample, y_sample)
        K_xy = gaussian_kernel(x_sample, y_sample)
        return tf.reduce_mean(K_xx) + tf.reduce_mean(K_yy) - 2 * tf.reduce_mean(K_xy)

    def train(self, x_train, y_train, epochs, batch_size, x_val=None, y_val=None):
        """
        Trains the ANN model.
        """
        if isinstance(y_train, np.ndarray):
            y_train = [y_train[:, i] for i in range(self.params['time_horizon'])]
        if y_val is not None and isinstance(y_val, np.ndarray):
            y_val = [y_val[:, i] for i in range(self.params['time_horizon'])]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: (gc.collect(), K.clear_session()))
        ]

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val) if x_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, x):
        """
        Predict using the ANN model.
        """
        predictions = self.model.predict(x)
        return np.stack(predictions, axis=1)

    def predict_with_uncertainty(self, x, mc_samples=100):
        """
        Predicts mean and uncertainty estimates via Monte Carlo sampling.
        """
        predictions = [self.model(x, training=True) for _ in range(mc_samples)]
        predictions = np.array([np.stack(p, axis=1) for p in predictions])
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        return mean_pred, uncertainty

    def calculate_mae(self, y_true, y_pred):
        return mean_absolute_error(y_true.flatten(), y_pred.flatten())

    def save(self, file_path):
        save_model(self.model, file_path)

    def load(self, file_path):
        self.model = load_model(file_path)
