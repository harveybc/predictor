"""Serializable Keras layers used by the temporal model components."""

import tensorflow as tf

keras = tf.keras
@keras.utils.register_keras_serializable(package="modular_temporal")
class FeatureSelect(keras.layers.Layer):
    """Fixed column routing: gather declared input channels for one branch.

    This is NOT learned feature selection. It is a deterministic
    ``tf.gather(inputs, indices, axis=-1)`` over the channel axis with indices
    fixed by the configuration (each branch's ``features`` resolved against the
    ordered ``feature_names``); it has no weights and never changes the time
    axis. Diagrams label it "column routing". Any learned selector or gate is a
    separately declared ablation, never a silent replacement. The registered
    Keras name (``modular_temporal>FeatureSelect``) is kept for serialization
    compatibility.

    Parameters
    ----------
    indices : sequence[int]
        Channel positions to retain, in branch input order.
    **kwargs
        Standard Keras layer options such as ``name`` and ``trainable``.
    """

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = tuple(indices)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], len(self.indices))

    def get_config(self):
        return {**super().get_config(), "indices": list(self.indices)}


@keras.utils.register_keras_serializable(package="modular_temporal")
class PositionalEncoding(keras.layers.Layer):
    """Add stateless sinusoidal positions to a rank-three sequence tensor.

    Input and output have shape ``(batch, steps, channels)``. Odd channel widths
    are supported and serialization uses the registered Keras layer name.
    """

    def call(self, inputs):
        length, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        position = tf.cast(tf.range(length)[:, None], tf.float32)
        channel = tf.range(width)[None, :]
        rates = tf.pow(10000.0, -2.0 * tf.cast(channel // 2, tf.float32)
                       / tf.cast(width, tf.float32))
        angle = position * rates
        encoding = tf.where(channel % 2 == 0, tf.sin(angle), tf.cos(angle))
        return inputs + tf.cast(encoding[None, :, :], inputs.dtype)



@keras.utils.register_keras_serializable(package="modular_temporal")
class SeasonalNaiveBaseline(keras.layers.Layer):
    """Seasonal naive read from INSIDE the input window: for horizon h with period P, the target
    channel at window position ``window - 1 - (P - h)`` (time t + h - P). Fixed gathers, no weights.
    Output ``(batch, len(horizons), len(target channels))`` -- added to the forecast head's output, so
    the head learns y - seasonal_naive (config ``target_residual``)."""

    def __init__(self, positions, channels, **kwargs):
        super().__init__(**kwargs)
        self.positions, self.channels = tuple(positions), tuple(channels)

    def call(self, inputs):
        return tf.gather(tf.gather(inputs, self.positions, axis=1), self.channels, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.positions), len(self.channels))

    def get_config(self):
        return {**super().get_config(), "positions": list(self.positions), "channels": list(self.channels)}


@keras.utils.register_keras_serializable(package="modular_temporal")
class WindowMean(keras.layers.Layer):
    """Per-channel mean of the last ``length`` steps of the window: (B, T, F) -> (B, 1, F). Every step it
    reads is observed by the decision time t (no value after t), but each in-window step of a centered
    input then depends on later in-window steps: the normalized path is causal w.r.t. t, not per step."""

    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)

    def call(self, inputs):
        return tf.reduce_mean(inputs[:, -self.length:, :], axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[2])

    def get_config(self):
        return {**super().get_config(), "length": self.length}


@keras.utils.register_keras_serializable(package="modular_temporal")
class TargetMeanBroadcast(keras.layers.Layer):
    """(B, 1, F) window mean -> (B, H, T): the target channels' mean repeated over the H horizons."""

    def __init__(self, channels, horizons, **kwargs):
        super().__init__(**kwargs)
        self.channels, self.horizons = tuple(channels), int(horizons)

    def call(self, mean):
        return tf.repeat(tf.gather(mean, self.channels, axis=-1), self.horizons, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.horizons, len(self.channels))

    def get_config(self):
        return {**super().get_config(), "channels": list(self.channels), "horizons": self.horizons}
