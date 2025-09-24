"""Common loss and metric functions shared across predictor plugins.

All functions here are pure / stateless and TensorFlow-friendly so they can be
safely imported inside model building contexts. Keep signatures stable to avoid
serialization issues.
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.losses import Huber

# --- Metrics ---

def mae_magnitude(y_true, y_pred):
    """Mean Absolute Error on first column (magnitude).
    Expands y_true to two columns if it is shape (N,) or (N,1) to preserve
    backward compatibility with existing plugin logic.
    """
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))

def r2_metric(y_true, y_pred):
    """R^2 on first column (magnitude)."""
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
        y_true = tf.concat([y_true, tf.zeros_like(y_true)], axis=1)
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    ss_res = tf.reduce_sum(tf.square(mag_true - mag_pred))
    ss_tot = tf.reduce_sum(tf.square(mag_true - tf.reduce_mean(mag_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# --- Auxiliary kernels ---

def _gaussian_kernel(x, y, sigma):
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    dist = tf.reduce_sum(tf.square(x - y), axis=-1)
    return tf.exp(-dist / (2.0 * sigma ** 2))

# --- MMD ---

def compute_mmd(x, y, sigma=1.0, sample_size=256):
    """Compute Maximum Mean Discrepancy with optional subsampling."""
    idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))[:sample_size]
    x_sample = tf.gather(x, idx)
    y_sample = tf.gather(y, idx)
    k_xx = _gaussian_kernel(x_sample, x_sample, sigma)
    k_yy = _gaussian_kernel(y_sample, y_sample, sigma)
    k_xy = _gaussian_kernel(x_sample, y_sample, sigma)
    return tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)

# --- Composite Loss Variants ---

def composite_loss_basic(y_true, y_pred, mmd_lambda=0.0, sigma=1.0):
    """Huber + optional MMD on magnitude column."""
    if y_true.shape.ndims == 1 or (y_true.shape.ndims == 2 and y_true.shape[1] == 1):
        y_true = tf.reshape(y_true, [-1, 1])
    mag_true = y_true[:, 0:1]
    mag_pred = y_pred[:, 0:1]
    huber_loss_val = Huber()(mag_true, mag_pred)
    if mmd_lambda != 0.0:
        mmd_loss_val = compute_mmd(mag_pred, mag_true, sigma=sigma)
    else:
        mmd_loss_val = 0.0
    return huber_loss_val + mmd_lambda * mmd_loss_val

# Legacy signature adapter for multi-head plugins

def composite_loss_multihead(y_true, y_pred, head_index, mmd_lambda, sigma,
                             p, i, d,
                             list_last_signed_error,
                             list_last_stddev,
                             list_last_mmd,
                             list_local_feedback):
    """Adapter wrapping composite_loss_basic keeping legacy callable shape.
    Currently ignores control feedback lists (placeholders) but keeps them
    for interface compatibility.
    """
    return composite_loss_basic(y_true, y_pred, mmd_lambda=mmd_lambda, sigma=sigma)

def random_normal_initializer_44(shape, dtype=None):
    return tf.random.normal(shape, mean=0.0, stddev=0.05, dtype=dtype, seed=44)

__all__ = [
    'mae_magnitude', 'r2_metric', 'compute_mmd',
    'composite_loss_basic', 'composite_loss_multihead',
    'random_normal_initializer_44'
]
