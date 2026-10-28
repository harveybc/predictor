"""Serializable Keras layers used by the temporal model components."""

import tensorflow as tf

keras = tf.keras


@keras.utils.register_keras_serializable(package="modular_temporal")
class CausalFrames(keras.layers.Layer):
    """Expose a fixed trailing context at every input time step.

    The output has shape ``(batch, steps, context, channels)``. Missing history
    before the first sample is left-padded with zeros; output step ``t`` can
    therefore depend only on input steps ``<= t``. The layer has no weights.

    Parameters
    ----------
    context : int
        Number of current-and-past samples in each frame.
    **kwargs
        Standard Keras layer options.
    """

    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        if type(context) is not int or context <= 0:
            raise ValueError("context must be a positive integer")
        self.context = context

    def call(self, inputs):
        padded = tf.pad(inputs, [[0, 0], [self.context - 1, 0], [0, 0]])
        return tf.signal.frame(
            padded, frame_length=self.context, frame_step=1, axis=1
        )

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.context, input_shape[-1])

    def get_config(self):
        return {**super().get_config(), "context": self.context}


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
