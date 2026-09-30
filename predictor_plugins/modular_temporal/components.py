"""Built-in Keras branch, fusion, core, and forecasting-head components."""

from dataclasses import dataclass
import math

import tensorflow as tf

from .common import _keys, _partition, _positive_int
from .layers import PositionalEncoding

keras = tf.keras

@dataclass(frozen=True)
class TemporalComponent:
    """Keras component paired with the exact timestamps represented by its output.

    Parameters
    ----------
    model : keras.Model
        A rank-three Keras model returning a temporal sequence.
    time_grid : tuple
        Ordered right-edge timestamps, in the task's declared time unit.
    """

    model: keras.Model
    time_grid: tuple


def causal_conv1d(*, input_shape, time_grid, output_steps, name, params):
    """Build a causal Conv1D branch that preserves every input time step.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Window length and feature count for this branch.
    time_grid : tuple
        Input timestamps represented by the window.
    output_steps : int
        Must equal the input-grid length. Branches do not downsample time.
    name : str
        Stable Keras model name.
    params : dict
        ``channels`` (default 16) and ``kernel_size`` (default 3).

    Returns
    -------
    TemporalComponent
        Branch model and its output time grid.
    """
    channels = _positive_int(params.get("channels", 16), "channels")
    kernel = _positive_int(params.get("kernel_size", 3), "kernel_size")
    _keys(params, {"channels", "kernel_size"}, "branch params")
    if output_steps != len(time_grid):
        raise ValueError("Branch extractors must preserve the full temporal grid")
    inputs = keras.Input(input_shape)
    x = keras.layers.Conv1D(channels, kernel, padding="causal", activation="gelu")(inputs)
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


def sequence_concat(*, input_shapes, time_grid, name, params):
    """Concatenate branch channels while preserving their common time axis.

    Parameters
    ----------
    input_shapes : list[tuple[int, int]]
        Each branch's ``(time_steps, channels)`` output shape.
    time_grid : tuple
        Shared branch output timestamps.
    name : str
        Stable Keras model name.
    params : dict
        Must be empty; this built-in fusion has no trainable weights.

    Returns
    -------
    TemporalComponent
        Raw channel concatenation and the unchanged time grid.
    """
    _keys(params, set(), "fusion params")
    inputs = [keras.Input(shape) for shape in input_shapes]
    x = keras.layers.Concatenate(axis=-1)(inputs) if len(inputs) > 1 else keras.layers.Activation("linear")(inputs[0])
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


def transformer_conv(*, input_shape, time_grid, output_steps, output_channels, name, params):
    """Build the positional Transformer core and staged temporal bottleneck.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Fused ``(time_steps, channels)`` shape.
    time_grid : tuple
        Fused sequence timestamps.
    output_steps : int
        Bottleneck time length.
    output_channels : int
        Bottleneck channel width.
    name : str
        Stable Keras model name.
    params : dict
        Transformer width, head/block counts, FFN width, dropout, and reduction
        stage settings. Defaults are documented in the package API page.

    Returns
    -------
    TemporalComponent
        Core model and its output time grid.
    """
    _keys(params, {"d_model", "heads", "blocks", "ff_dim", "dropout",
                  "stage_channels", "time_factors", "kernel_size"}, "core params")
    width = _positive_int(params.get("d_model", 64), "d_model")
    heads = _positive_int(params.get("heads", 4), "heads")
    blocks = _positive_int(params.get("blocks", 2), "blocks")
    ff_dim = _positive_int(params.get("ff_dim", 128), "ff_dim")
    kernel = _positive_int(params.get("kernel_size", 3), "kernel_size")
    dropout = params.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1 or width % heads:
        raise ValueError("Invalid dropout or d_model/head divisibility")
    grid = _partition(time_grid, output_steps)
    channels = params.get("stage_channels", [32, 16, output_channels])
    factors = params.get("time_factors", _default_time_factors(
        len(time_grid) // output_steps, len(channels)
    ))
    if len(channels) not in (3, 4) or len(factors) != len(channels):
        raise ValueError("Expected three or four matching temporal-reduction stages")
    for n in [*channels, *factors]:
        _positive_int(n, "temporal-reduction stage")
    if (channels[-1] != output_channels or math.prod(factors) != len(time_grid) // output_steps
            or any(a <= b for a, b in zip([width, *channels[:-1]], channels))):
        raise ValueError("Stages must progressively reduce channels to output_channels and time to output_steps")
    inputs = keras.Input(input_shape)
    x = PositionalEncoding(name="positional_encoding")(inputs)
    x = keras.layers.Dense(width, name="model_projection")(x)
    for i in range(blocks):
        attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=width // heads,
                                                    dropout=dropout, name=f"attention_{i}")
        a = attention(x, x, use_causal_mask=True)
        a = keras.layers.Dropout(dropout)(a)
        x = keras.layers.LayerNormalization(name=f"attention_norm_{i}")(keras.layers.Add()([x, a]))
        f = keras.layers.Dense(ff_dim, activation="gelu", name=f"ffn_expand_{i}")(x)
        f = keras.layers.Dropout(dropout)(f)
        f = keras.layers.Dense(width, name=f"ffn_project_{i}")(f)
        x = keras.layers.LayerNormalization(name=f"ffn_norm_{i}")(keras.layers.Add()([x, f]))
    for i, (channel, factor) in enumerate(zip(channels, factors)):
        if int(x.shape[1]) % factor:
            raise ValueError("Temporal stage requires exact divisibility")
        x = _residual_temporal_stage(x, channel, factor, kernel, f"stage_{i}")
    return TemporalComponent(keras.Model(inputs, x, name=name), grid)


def _default_time_factors(total_factor, stage_count):
    """Distribute prime factors across stages while keeping all samples aligned."""
    factors = [1] * stage_count
    remainder = total_factor
    prime = 2
    while prime * prime <= remainder:
        while remainder % prime == 0:
            target = min(range(stage_count), key=factors.__getitem__)
            factors[target] *= prime
            remainder //= prime
        prime += 1
    if remainder > 1:
        target = min(range(stage_count), key=factors.__getitem__)
        factors[target] *= remainder
    return factors


def _residual_temporal_stage(inputs, channels, factor, kernel, name):
    """Downsample complete adjacent blocks and refine them with a residual TCN."""
    shortcut = keras.layers.Conv1D(
        channels, kernel_size=factor, strides=factor, padding="valid",
        name=name + "_skip_downsample",
    )(inputs)
    values = keras.layers.Conv1D(
        channels, kernel_size=factor, strides=factor, padding="valid",
        name=name + "_block_projection",
    )(inputs)
    values = keras.layers.LayerNormalization(name=name + "_downsample_norm")(
        keras.layers.Add(name=name + "_downsample_residual")([shortcut, values])
    )
    values = keras.layers.Activation("gelu", name=name + "_downsample_activation")(values)
    residual = keras.layers.Conv1D(
        channels, kernel_size=kernel, padding="causal", activation="gelu",
        name=name + "_temporal_conv",
    )(values)
    residual = keras.layers.Conv1D(
        channels, kernel_size=1, padding="causal", name=name + "_temporal_projection",
    )(residual)
    values = keras.layers.LayerNormalization(name=name + "_temporal_norm")(
        keras.layers.Add(name=name + "_temporal_residual")([values, residual])
    )
    return keras.layers.Activation("gelu", name=name + "_output")(values)


def forecast(*, input_shape, time_grid, horizons, target_count, name, params):
    """Build the direct multi-horizon regression output head.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Core latent ``(time_steps, channels)`` shape.
    time_grid : tuple
        Declared future horizon timestamps.
    horizons : list[int]
        Strictly increasing positive forecast offsets.
    target_count : int
        Number of predicted target series.
    name : str
        Stable Keras model name.
    params : dict
        Must be empty for this implementation.

    Returns
    -------
    TemporalComponent
        Forecast model with shape ``(batch, horizons, targets)``.
    """
    _keys(params, set(), "head params")
    inputs = keras.Input(input_shape)
    x = keras.layers.Flatten(name="task_tokens")(inputs)
    x = keras.layers.Dense(len(horizons) * target_count, name="forecast_projection")(x)
    x = keras.layers.Reshape((len(horizons), target_count), name="forecast_horizons")(x)
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))
