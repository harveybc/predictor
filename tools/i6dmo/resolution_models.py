"""Bounded DS2/DS4 resolution models and exact elapsed-horizon heads, R0 only.

``conv`` reuses the existing branch4/core8,4/head8 components and physical-time
encoder without changing the legacy model schema. ``linear`` is a multivariate
flattened affine readout plus an independent affine offset/closure readout.

``dlinear`` is explicitly an ADAPTED multivariate DLinear, inspired by
https://github.com/cure-lab/LTSF-Linear (models/DLinear.py), not an exact
literature reproduction: kernel3 rather than kernel25, flattened multivariate
maps to exact scalar price heads rather than per-channel sequence forecasts,
and an additional affine physical-time context. Endpoint replication and the
trend/residual split follow that decomposition. The centered moving average
uses only observations inside the supplied window ending at t; its right edge
repeats t, never a future observation. Targets are never smoothed or supplied
to the model. All maps start from seeded kernels and zero biases; no pretrained
autoencoder, donor, fitted decomposition or extra hidden nonlinearities.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.i6dmo import model as M

keras, tf = M.keras, M.tf
SCHEMA = "i6dmo.resolution_model.v1"
SLOTS = (6, 7, 9, 13, 18, 24, 25, 49)
SPACINGS = (2, 4)
HORIZONS = ((2, 3, 4, 5, 6, 12), (18, 24, 36, 48, 60, 72), (72,))
KINDS = ("conv", "linear", "dlinear")
_KEYS = {"schema", "features", "slots", "spacing_open_market_hours", "horizons_hours", "kind", "seed"}


def build_config(features, slots, spacing, horizons, kind="conv", seed=6102026):
    """Return the independent, JSON-serializable fixed resolution-model config."""
    if not isinstance(features, (list, tuple)) or not features or \
            any(not isinstance(f, str) or not f for f in features) or \
            len(set(features)) != len(features):
        raise ValueError("features must be unique nonempty names")
    if type(slots) is not int or slots not in SLOTS:
        raise ValueError(f"slots must be one of {SLOTS}")
    if type(spacing) is not int or spacing not in SPACINGS:
        raise ValueError(f"spacing must be one of {SPACINGS} open-market hours")
    if not isinstance(horizons, (list, tuple)) or any(type(h) is not int for h in horizons) or \
            tuple(horizons) not in HORIZONS:
        raise ValueError(f"horizons must be exactly one of {HORIZONS}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return {"schema": SCHEMA, "features": list(features), "slots": slots,
            "spacing_open_market_hours": spacing, "horizons_hours": list(horizons),
            "kind": kind, "seed": seed}


def _validated(config):
    if not isinstance(config, dict) or set(config) != _KEYS or config.get("schema") != SCHEMA:
        raise ValueError(f"schema/fields must match {SCHEMA}")
    return build_config(config["features"], config["slots"], config["spacing_open_market_hours"],
                        config["horizons_hours"], kind=config["kind"], seed=config["seed"])


@keras.utils.register_keras_serializable(package="i6dmo")
class TimeInputGuard(keras.layers.Layer):
    """Validate signed physical offsets and binary closure flags without encoding them."""

    def __init__(self, spacing, **kwargs):
        super().__init__(**kwargs)
        if type(spacing) is not int or spacing not in SPACINGS:
            raise ValueError(f"spacing must be one of {SPACINGS}")
        self.spacing = spacing

    def call(self, inputs):
        offsets, closure = inputs
        zero = tf.cast(0, offsets.dtype)
        checks = [
            tf.debugging.assert_all_finite(offsets, "INVALID_TIME_INPUT: nonfinite offsets"),
            tf.debugging.assert_all_finite(closure, "INVALID_TIME_INPUT: nonfinite closure"),
            tf.debugging.assert_equal(offsets[:, -1], zero, message="INVALID_TIME_INPUT: origin must be included"),
            tf.debugging.assert_less_equal(offsets, zero, message="INVALID_TIME_INPUT: future offsets"),
            tf.debugging.assert_greater_equal(
                offsets[:, 1:] - offsets[:, :-1], tf.cast(self.spacing, offsets.dtype),
                message="INVALID_TIME_INPUT: compressed or unordered offsets"),
            tf.debugging.assert_equal(closure[:, 0], tf.cast(0, closure.dtype),
                                      message="INVALID_TIME_INPUT: first closure must be zero"),
            tf.debugging.assert_equal(tf.reduce_all((closure == 0) | (closure == 1)), True,
                                      message="INVALID_TIME_INPUT: closure must be binary"),
        ]
        with tf.control_dependencies(checks):
            return [tf.identity(offsets), tf.identity(closure)]

    def get_config(self):
        return {**super().get_config(), "spacing": self.spacing}


@keras.utils.register_keras_serializable(package="i6dmo")
class WindowDecomposition(keras.layers.Layer):
    """ADAPTED DLinear kernel3: (residual, trend), endpoint-replicated within [past, t]."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        if type(kernel_size) is not int or kernel_size != 3:
            raise ValueError("the adapted DLinear moving-average kernel is fixed to 3")
        self.kernel_size = kernel_size

    def call(self, values):
        padded = tf.concat([values[:, :1, :], values, values[:, -1:, :]], axis=1)
        trend = tf.nn.avg_pool1d(padded, ksize=3, strides=1, padding="VALID")
        return [values - trend, trend]

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size}


@dataclass
class ResolutionBundle:
    """The config/forecast_model protocol consumed by train.fit_dmo and its fresh loader."""

    config: dict
    forecast_model: keras.Model


@keras.utils.register_keras_serializable(package="i6dmo")
class HeadListModel(keras.Model):
    """Keep the trainer's list-of-heads contract even when Keras unwraps a singleton."""

    def call(self, inputs, training=None, mask=None):
        outputs = super().call(inputs, training=training, mask=mask)
        return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def _conv_outputs(c, observations, offsets, closure):
    slots = c["slots"]
    grid = tuple(range(slots))
    sequences = []
    for i, _ in enumerate(c["features"]):
        name = f"branch_{i:02d}"
        comp = M.dmo_branch(input_shape=(slots, 1), time_grid=grid, output_steps=slots, name=name,
                            params=M._branch_params(M._FIXED))
        branch = M._validate_component(comp, [(slots, 1)], grid, 4)
        M._probe_causal(branch, grid, grid, "branch " + name)
        sequences.append(branch(M.FeatureSelect([i], name="select_" + name)(observations)))
    factory, _ = M._resolve("fusion", {"plugin": "sequence_concat"}, {"fusion": "modular.fusion"})
    shapes = [tuple(s.shape[1:]) for s in sequences]
    fusion = M._validate_component(factory(input_shapes=shapes, time_grid=grid,
                                           name="sequence_fusion", params={}),
                                   shapes, grid, 4 * len(sequences))
    encoded = M.DynamicPhysicalTimeEncoder(min_adjacent_hours=float(c["spacing_open_market_hours"]),
                                           name="physical_time_encoder")([fusion(sequences), offsets, closure])
    shape = tuple(encoded.shape[1:])
    core = M._validate_component(M.dmo_core(input_shape=shape, time_grid=grid, name="temporal_core",
                                            params=M._core_params(M._FIXED)), [shape], grid, 4)
    M._probe_causal(core, grid, grid, "core")
    latent = core(encoded)
    return [M.dmo_head(input_shape=tuple(latent.shape[1:]), horizon_hours=h, name=M.head_name(h),
                        params=M._FIXED["heads"])(latent) for h in c["horizons_hours"]]


def _affine_outputs(c, observations, offsets, closure):
    context = keras.layers.Concatenate(name="time_context")([
        keras.layers.Rescaling(1.0 / 24.0, name="offset_hours_over_24")(offsets), closure])
    if c["kind"] == "dlinear":
        residual, trend = WindowDecomposition(name="window_decomposition")(observations)
        components = [("trend", keras.layers.Flatten(name="trend_flatten")(trend)),
                      ("residual", keras.layers.Flatten(name="residual_flatten")(residual))]
    else:
        components = [("values", keras.layers.Flatten(name="values_flatten")(observations))]
    outputs = []
    for h in c["horizons_hours"]:
        name = M.head_name(h)
        parts = [keras.layers.Dense(1, activation="linear", bias_initializer="zeros",
                                     name=f"{name}_{label}")(values) for label, values in components]
        parts.append(keras.layers.Dense(1, activation="linear", bias_initializer="zeros",
                                         name=name + "_context")(context))
        outputs.append(keras.layers.Add(name=name)(parts))
    return outputs


def build(config):
    """Revalidate and seed-build one resolution model; outputs are always a list of heads."""
    c = _validated(config)
    M.enable_determinism()
    keras.utils.set_random_seed(c["seed"])
    observations = keras.Input((c["slots"], len(c["features"])), name="observations")
    offsets = keras.Input((c["slots"],), name="offsets_hours")
    closure = keras.Input((c["slots"],), name="closure_crossed")
    checked_offsets, checked_closure = TimeInputGuard(
        c["spacing_open_market_hours"], name="time_input_guard")([offsets, closure])
    output_fn = _conv_outputs if c["kind"] == "conv" else _affine_outputs
    outputs = output_fn(c, observations, checked_offsets, checked_closure)
    forecast = HeadListModel([observations, offsets, closure], outputs, name="i6dmo_resolution_forecast")
    return ResolutionBundle(c, forecast)
