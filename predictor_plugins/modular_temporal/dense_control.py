"""Non-temporal Dense control for the I6-D causal-window comparison.

This module intentionally lives beside, rather than inside, the temporal
assembly path. A Dense branch may consume an ordered causal window, but its
latent units do not inherit timestamps from that window.
"""

from dataclasses import dataclass
import json
import math

import tensorflow as tf

from .common import _keys, _positive_int
from .components import TemporalComponent, component

keras = tf.keras

CONFIG_SCHEMA = "predictor.modular.dense_control.v1"
UNORDERED_LATENT_VECTOR = "unordered_latent_vector"


@dataclass(frozen=True)
class DenseLatentComponent:
    """Keras component whose rank-two output has no temporal semantics.

    ``support_grid`` identifies the ordered observations consumed to produce
    the vector. It describes causal support, not the meaning of latent units.
    """

    model: keras.Model
    support_grid: tuple
    semantics: str = UNORDERED_LATENT_VECTOR

    @property
    def available_at(self):
        """Right edge of the causal input support."""
        return self.support_grid[-1]


def _hidden_units(value):
    if not isinstance(value, list):
        raise ValueError("hidden_units must be a list of positive integers")
    return [_positive_int(item, "hidden_units") for item in value]


def _support_grid(value):
    if not isinstance(value, (tuple, list)) or len(value) != 24:
        raise ValueError("Dense control requires one exact 24-hour support grid")
    grid = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(item) for item in grid):
        raise ValueError("support_grid must contain finite hour offsets")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("support_grid must be strictly increasing")
    if any(not math.isclose(right - left, 1.0) for left, right in zip(grid, grid[1:])):
        raise ValueError("Dense control requires 24 consecutive hourly observations")
    return grid


def _effective_params(params):
    _keys(params, {"hidden_units", "latent_units", "activation", "use_bias"},
          "dense control params")
    activation = params.get("activation", "gelu")
    if not isinstance(activation, str) or not activation:
        raise ValueError("activation must be a nonempty Keras activation name")
    use_bias = params.get("use_bias", True)
    if type(use_bias) is not bool:
        raise ValueError("use_bias must be boolean")
    return {
        "hidden_units": _hidden_units(params.get("hidden_units", [16, 8])),
        "latent_units": _positive_int(params.get("latent_units", 8), "latent_units"),
        "activation": activation,
        "use_bias": use_bias,
    }


@component(
    "control_branch",
    "1.1.0",
    {"hidden_units", "latent_units", "activation", "use_bias"},
    "(batch, 24 hourly observations, channels of one semantic feature) -> "
    "(batch, latent units); "
    "unordered latent vector with causal support through the window right edge; "
    "no temporal-preservation claim",
    defaults={"hidden_units": [16, 8], "latent_units": 8,
              "activation": "gelu", "use_bias": True},
)
def causal_window_dense(*, input_shape, support_grid, name, params):
    """Build one per-feature Dense control over an exact 24-hour window.

    Flattening is valid here precisely because this is the non-temporal I6-D
    control. The output is never passed to sequence fusion or the temporal core.
    """
    if (not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2
            or input_shape[0] != 24 or isinstance(input_shape[1], bool)
            or not isinstance(input_shape[1], int) or input_shape[1] <= 0):
        raise ValueError("Dense control requires one exact 24-hour per-feature window")
    grid = _support_grid(support_grid)
    effective = _effective_params(params)
    inputs = keras.Input(shape=tuple(input_shape), name=f"{name}_window")
    values = keras.layers.Flatten(name=f"{name}_ordered_window")(inputs)
    for index, units in enumerate(effective["hidden_units"]):
        values = keras.layers.Dense(
            units,
            activation=effective["activation"],
            use_bias=effective["use_bias"],
            name=f"{name}_dense_{index + 1}",
        )(values)
    latent = keras.layers.Dense(
        effective["latent_units"],
        activation=effective["activation"],
        use_bias=effective["use_bias"],
        name=f"{name}_latent",
    )(values)
    return DenseLatentComponent(keras.Model(inputs, latent, name=name), grid)


def dense_control_config(*, feature_name, hidden_units=None, latent_units=8,
                         activation="gelu", use_bias=True,
                         preserves_temporal_axis=False):
    """Return the validated, JSON-ready identity of one Dense control branch."""
    if preserves_temporal_axis is not False:
        raise ValueError("A Dense control temporal-preservation claim is invalid")
    if not isinstance(feature_name, str) or not feature_name:
        raise ValueError("feature_name must be a nonempty string")
    params = _effective_params({
        "hidden_units": [16, 8] if hidden_units is None else hidden_units,
        "latent_units": latent_units,
        "activation": activation,
        "use_bias": use_bias,
    })
    return {
        "schema": CONFIG_SCHEMA,
        "plugin": "causal_window_dense",
        "semantic_type": UNORDERED_LATENT_VECTOR,
        "window_hours": 24,
        "feature_name": feature_name,
        "params": params,
    }


def canonical_dense_control_json(config):
    """Serialize a validated Dense-control config deterministically."""
    normalized = load_dense_control_config(config)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_dense_control_config(value):
    """Parse and validate a Dense-control mapping or JSON document."""
    config = json.loads(value) if isinstance(value, str) else dict(value)
    _keys(config, {"schema", "plugin", "semantic_type", "window_hours",
                   "feature_name", "params"}, "dense control config")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Unsupported Dense control config schema")
    if config.get("plugin") != "causal_window_dense":
        raise ValueError("Dense control config names the wrong plugin")
    if config.get("semantic_type") != UNORDERED_LATENT_VECTOR:
        raise ValueError("Dense control cannot claim temporal semantics")
    if config.get("window_hours") != 24:
        raise ValueError("Dense control requires a 24-hour window")
    params = config.get("params")
    if not isinstance(params, dict):
        raise ValueError("Dense control params must be an object")
    return dense_control_config(feature_name=config.get("feature_name"), **params)


class DenseControlAdapter:
    """Explicitly combine unordered Dense latents without inventing time.

    The adapter is the only bridge exposed by this control. Its result remains
    rank two and can feed a non-temporal comparison head. It cannot enter
    ``sequence_concat`` or ``transformer_conv``.
    """

    def __init__(self, *, latent_units=None):
        self.latent_units = (None if latent_units is None
                             else _positive_int(latent_units, "latent_units"))

    def combine(self, components, *, name):
        if not isinstance(components, (list, tuple)) or not components:
            raise ValueError("DenseControlAdapter needs at least one latent component")
        if any(isinstance(item, TemporalComponent) for item in components):
            raise TypeError("DenseControlAdapter refuses TemporalComponent inputs")
        if any(not isinstance(item, DenseLatentComponent) for item in components):
            raise TypeError("DenseControlAdapter accepts only DenseLatentComponent inputs")
        support = components[0].support_grid
        if any(item.support_grid != support for item in components[1:]):
            raise ValueError("Dense controls must share identical causal support")
        inputs = [keras.Input(shape=item.model.output_shape[1:], name=f"latent_{index}")
                  for index, item in enumerate(components)]
        merged = (keras.layers.Concatenate(name=f"{name}_concat")(inputs)
                  if len(inputs) > 1 else keras.layers.Activation("linear")(inputs[0]))
        if self.latent_units is not None:
            merged = keras.layers.Dense(self.latent_units, name=f"{name}_projection")(merged)
        return DenseLatentComponent(keras.Model(inputs, merged, name=name), support)

    @staticmethod
    def as_temporal(component):
        del component
        raise TypeError("Unordered latent units cannot be reshaped into fake timestamps")
