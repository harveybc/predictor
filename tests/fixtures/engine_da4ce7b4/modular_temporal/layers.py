"""Serializable Keras layers used by the temporal model components."""

import tensorflow as tf

keras = tf.keras
@keras.utils.register_keras_serializable(package="modular_temporal")
class FeatureSelect(keras.layers.Layer):
    """Select ordered feature channels without changing the time axis.

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

