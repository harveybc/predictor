"""Sparse-clock adapter around the established causal per-feature branch factory.

The final input channel is observed-market-bar presence, separate from each
feature's source-availability channel. Missing slots never become quotes.
Branches and core preserve every clock position; only the predictive head
reads the final position. All resolution arms have identical parameter shapes.
"""

import numpy as np
import tensorflow as tf

from predictor_plugins.modular_temporal.components import causal_conv1d

K = tf.keras


@K.utils.register_keras_serializable(package="predictor_i6e")
class ClockView(K.layers.Layer):
    """Serializable feature slicing and physical-time encoding without Lambdas."""

    def __init__(self, mode, index=0, spacing_hours=1., **kwargs):
        super().__init__(**kwargs)
        self.mode, self.index, self.spacing_hours = mode, index, spacing_hours

    def call(self, inputs):
        if self.mode == "feature":
            return inputs[:, :, 2 * self.index:2 * self.index + 2]
        if self.mode == "mask":
            return inputs[:, :, -1:] > .5
        if self.mode == "clean":
            return tf.where(inputs[:, :, -1:] > .5, inputs[:, :, :-1], 0.)
        if self.mode == "gate":
            values, mask = inputs
            return tf.where(mask, values, tf.zeros_like(values))
        if self.mode == "attention":
            mask = inputs[:, :, 0]
            length = tf.shape(mask)[1]
            causal = tf.linalg.band_part(tf.ones((length, length), tf.int32), -1, 0) > 0
            return mask[:, :, None] & mask[:, None, :] & causal[None, :, :]
        if self.mode == "position":
            length, width = tf.shape(inputs)[1], inputs.shape[-1]
            # Unit is physical hours, not ordinal token number.
            positions = tf.cast(tf.range(length), inputs.dtype) * self.spacing_hours
            bands = tf.cast(tf.range(width), inputs.dtype)
            angle = positions[:, None] / tf.pow(10000., 2 * tf.floor(bands / 2) / width)
            code = tf.where(tf.cast(tf.range(width) % 2, tf.bool), tf.cos(angle), tf.sin(angle))
            return inputs + code[None, :, :]
        if self.mode == "last":
            return inputs[:, -1, :]
        raise ValueError(f"unknown clock view: {self.mode}")

    def get_config(self):
        return dict(super().get_config(), mode=self.mode, index=self.index,
                    spacing_hours=self.spacing_hours)


def build_model(samples, n_features, spacing_hours):
    """Build equal-capacity compact models with explicit missing-slot handling.

    Input has shape ``(samples, 2*n_features+1)``: alternating scaled feature /
    source mask, followed by market mask. Transformer defaults are width 32,
    two blocks, two heads, FFN 64. Core Conv1D channels descend 24->16->8;
    all strides are one because 3/5/9 tokens must not be compressed further.
    """
    if samples not in (3, 5, 9) or type(n_features) is not int or n_features < 1:
        raise ValueError("compact 3/5/9-token model requires positive feature count")
    if not np.isfinite(spacing_hours) or spacing_hours <= 0:
        raise ValueError("positive physical sample spacing required")
    inp = K.Input((samples, 2 * n_features + 1), name="clock_input")
    mask = ClockView("mask", name="market_presence")(inp)
    clean = ClockView("clean", name="ignore_placeholders")(inp)
    parts = []
    grid = tuple((i - samples + 1) * spacing_hours for i in range(samples))
    for index in range(n_features):
        feature = ClockView("feature", index=index, name=f"feature_{index}")(clean)
        branch = causal_conv1d(input_shape=(samples, 2), time_grid=grid,
                               output_steps=samples, name=f"branch_{index}",
                               params={"channels": 16, "kernel_size": 3})
        parts.append(ClockView("gate", name=f"branch_mask_{index}")([branch.model(feature), mask]))
    fused = parts[0] if len(parts) == 1 else K.layers.Concatenate(name="fusion")(parts)
    x = ClockView("position", spacing_hours=float(spacing_hours), name="physical_position")(fused)
    x = K.layers.Conv1D(32, 1, name="core_projection")(x)
    x = ClockView("gate", name="core_entry_mask")([x, mask])
    attention_mask = ClockView("attention", name="causal_market_attention")(mask)
    for block in range(2):
        attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.,
                                                name=f"attention_{block}")(
            x, x, attention_mask=attention_mask)
        x = K.layers.LayerNormalization(name=f"attention_norm_{block}")(
            K.layers.Add(name=f"attention_add_{block}")([x, attention]))
        ff = K.layers.Dense(64, activation="gelu", name=f"ff_expand_{block}")(x)
        ff = K.layers.Dense(32, name=f"ff_project_{block}")(ff)
        x = K.layers.LayerNormalization(name=f"ff_norm_{block}")(
            K.layers.Add(name=f"ff_add_{block}")([x, ff]))
        x = ClockView("gate", name=f"transformer_mask_{block}")([x, mask])
    for stage, channels in enumerate((24, 16, 8)):
        residual = K.layers.Conv1D(channels, 1, name=f"core_skip_{stage}")(x)
        update = K.layers.Conv1D(channels, 3, padding="causal", activation="gelu",
                                 name=f"core_conv_{stage}")(x)
        x = K.layers.LayerNormalization(name=f"core_norm_{stage}")(
            K.layers.Add(name=f"core_add_{stage}")([residual, update]))
        x = ClockView("gate", name=f"core_mask_{stage}")([x, mask])
    output = K.layers.Dense(1, name="forecast_head")(ClockView("last", name="head_input")(x))
    return K.Model(inp, output, name="I6E_masked_compact")
