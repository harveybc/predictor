"""Bayesian utility functions shared by predictor plugins.

Centralizes posterior/prior factories, KL annealing callback builder and
Monte-Carlo Welford uncertainty estimation to reduce duplication across
ANN/CNN/LSTM/Transformer style plugins.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import Callback

# ---------------------------------------------------------------------------
# Posterior / Prior (mean-field) factories
# ---------------------------------------------------------------------------

def posterior_mean_field(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str):
        name = None
    bias_size = 0  # ignore bias for simplification
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.0))
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=123), dtype=dtype, trainable=trainable,
                      name=(f"{name}_loc" if name else "posterior_loc"))
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=124), dtype=dtype, trainable=trainable,
                        name=(f"{name}_scale" if name else "posterior_scale"))
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)
    loc_r = tf.reshape(loc, kernel_shape)
    scale_r = tf.reshape(scale, kernel_shape)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_r, scale=scale_r),
        reinterpreted_batch_ndims=len(kernel_shape),
    )

def prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str):
        name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    loc_r = tf.reshape(loc, kernel_shape)
    scale_r = tf.reshape(scale, kernel_shape)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc_r, scale=scale_r),
        reinterpreted_batch_ndims=len(kernel_shape),
    )

# ---------------------------------------------------------------------------
# KL Annealing Callback
# ---------------------------------------------------------------------------
class _KLAnnealCallback(Callback):
    def __init__(self, plugin, target_kl: float, anneal_epochs: int):
        super().__init__()
        self.plugin = plugin
        self.target_kl = target_kl
        self.anneal_epochs = max(1, anneal_epochs)
    def on_epoch_begin(self, epoch, logs=None):
        frac = min(1.0, (epoch + 1) / self.anneal_epochs)
        self.plugin.kl_weight_var.assign(self.target_kl * frac)

def build_kl_anneal_callback(plugin, target_kl: float, anneal_epochs: int):
    return _KLAnnealCallback(plugin, target_kl, anneal_epochs)

# ---------------------------------------------------------------------------
# Monte Carlo predictive mean & std (Welford incremental)
# ---------------------------------------------------------------------------

def predict_mc_welford(model, x_test, mc_samples: int = 50):
    if model is None:
        raise ValueError("Model not built.")
    sample = model(x_test[:1], training=True)
    sample = [sample] if not isinstance(sample, list) else sample
    heads = len(sample)
    n = x_test.shape[0]
    d = sample[0].shape[-1]
    means = [np.zeros((n, d), dtype=np.float32) for _ in range(heads)]
    m2 = [np.zeros_like(means[0]) for _ in range(heads)]
    counts = [0] * heads
    for _ in range(mc_samples):
        preds = model(x_test, training=False)
        preds = [preds] if not isinstance(preds, list) else preds
        for h in range(heads):
            arr = preds[h].numpy()
            if arr.ndim == 1:
                arr = np.expand_dims(arr, -1)
            counts[h] += 1
            delta = arr - means[h]
            means[h] += delta / counts[h]
            delta2 = arr - means[h]
            m2[h] += delta * delta2
    stds = []
    for h in range(heads):
        var = np.full_like(means[h], np.nan) if counts[h] < 2 else m2[h] / (counts[h] - 1)
        stds.append(np.sqrt(np.maximum(var, 0)))
    return means, stds

__all__ = [
    'posterior_mean_field', 'prior_fn', 'build_kl_anneal_callback', 'predict_mc_welford'
]
