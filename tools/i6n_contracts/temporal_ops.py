"""Candidate N0 TensorFlow primitives, not a deployed training protocol.

Temporal reduction is anchored at the last observed tick. Supervised losses
mask labels before subtraction and normalize each head by its valid count.
The caller must derive masks with C1 and fit target scales on eligible TRAIN
labels. No market calendar or label availability is inferred here.
"""

import tensorflow as tf


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def retained_indices(length, factor):
    """Return causal reduction positions, always including ``length - 1``.

    Apply after a stride-one causal convolution, not after an already
    left-anchored strided convolution. This preserves support at the origin;
    it does not claim lossless compression or zero filtering delay.
    """
    length = _positive_int("length", length)
    factor = _positive_int("factor", factor)
    return tuple(range((length - 1) % factor, length, factor))


@tf.keras.utils.register_keras_serializable(package="predictor_i6n")
class RightAnchoredSubsample(tf.keras.layers.Layer):
    """Select sequence positions from the right; supports features and masks.

    Parameters
    ----------
    factor : int
        Positive temporal reduction factor. Use the same layer configuration
        for a sequence and its time-axis availability mask.
    """

    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = _positive_int("factor", factor)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        tf.debugging.assert_positive(length, message="empty temporal sequence")
        start = tf.math.floormod(length - 1, self.factor)
        return inputs[:, start::self.factor, ...]

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if shape[1] is not None:
            shape[1] = (shape[1] + self.factor - 1) // self.factor
        return tuple(shape)

    def get_config(self):
        return dict(super().get_config(), factor=self.factor)


def masked_huber(y_true, y_pred, mask, head_weights, delta=1.0):
    """Return weighted mean of per-head masked Huber losses.

    All tensors have shape ``(examples, heads)`` except the positive weight
    vector. Invalid positions are selected away *before* residuals, so NaNs
    there neither poison the loss nor acquire gradients. An empty head has
    no weight; an empty batch returns zero. The trainer must skip optimizer
    updates for an entirely empty batch (including weight decay).

    This is a batch training primitive. Epoch/annual metrics must accumulate
    sums and valid counts across batches, not average these batch means.
    """
    prediction = tf.convert_to_tensor(y_pred)
    if not prediction.dtype.is_floating:
        prediction = tf.cast(prediction, tf.float32)
    target = tf.cast(tf.convert_to_tensor(y_true), prediction.dtype)
    validity = tf.convert_to_tensor(mask)
    if validity.dtype != tf.bool:
        raise ValueError("mask must be boolean")
    weights = tf.cast(tf.convert_to_tensor(head_weights), prediction.dtype)
    threshold = tf.cast(tf.convert_to_tensor(delta), prediction.dtype)
    tf.debugging.assert_rank(prediction, 2)
    tf.debugging.assert_rank(target, 2)
    tf.debugging.assert_rank(validity, 2)
    tf.debugging.assert_rank(weights, 1)
    tf.debugging.assert_rank(threshold, 0)
    tf.debugging.assert_equal(tf.shape(target), tf.shape(prediction))
    tf.debugging.assert_equal(tf.shape(validity), tf.shape(prediction))
    tf.debugging.assert_equal(tf.size(weights), tf.shape(prediction)[1])
    tf.debugging.assert_all_finite(weights, "nonfinite head weights")
    tf.debugging.assert_positive(weights, message="positive head weights required")
    tf.debugging.assert_all_finite(threshold, "nonfinite Huber delta")
    tf.debugging.assert_positive(threshold)
    safe_target = tf.where(validity, target, tf.zeros_like(target))
    safe_prediction = tf.where(validity, prediction, tf.zeros_like(prediction))
    tf.debugging.assert_all_finite(safe_target, "invalid eligible target")
    tf.debugging.assert_all_finite(safe_prediction, "invalid eligible prediction")
    error = tf.abs(safe_prediction - safe_target)
    quadratic = tf.minimum(error, threshold)
    loss = .5 * tf.square(quadratic) + threshold * (error - quadratic)
    counts = tf.reduce_sum(tf.cast(validity, prediction.dtype), axis=0)
    per_head = tf.math.divide_no_nan(tf.reduce_sum(loss, axis=0), counts)
    active_weights = tf.where(counts > 0, weights, tf.zeros_like(weights))
    return tf.math.divide_no_nan(tf.reduce_sum(per_head * active_weights),
                                 tf.reduce_sum(active_weights))
