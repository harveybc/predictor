"""Explicit R0 continuation architectures, separate from all retained model identities.

Conv branches remain Conv1D4(k3), followed by channel concatenation and ONLY
adjacent actual elapsed hours/24 plus closure flags. No additive positional
code, sinusoid, Transformer, Bayesian component, donor or pretrained encoder.
Core choices and independent Dense/causal-Conv/LSTM heads are sealed in config.

Linear and ADAPTED multivariate DLinear reuse resolution_models' affine math
and raw offset/closure context unchanged. The latter's kernel3, endpoint-
replicated observed-window decomposition is inspired by
https://github.com/cure-lab/LTSF-Linear, not an exact literature reproduction;
it neither smooths targets nor accesses beyond t. Unused head/core settings
canonicalize to ``none``/[] so identical controls have one config identity.

The caller supplies full TRAIN and full reused development VALIDATION to
train.fit_dmo (whose second argument retains the historical name ``inner``).
This module creates no split or holdout and reads no dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.i6dmo import model as M, resolution_models as R

keras, tf = M.keras, M.tf
SCHEMA = "i6dmo.continuation_model.v1"
HEADS = ("dense", "conv", "lstm")
CORES = ((2,), (4,), (4, 2), (8, 4))
HORIZONS = (*R.HORIZONS, (4,))
_KEYS = {"schema", "features", "slots", "spacing_open_market_hours", "horizons_hours",
         "kind", "seed", "head", "core"}


def build_config(features, slots, spacing, horizons, kind="conv", seed=6102026,
                 head="conv", core=(8, 4)):
    """Return a canonical independent config; unused affine architecture choices are erased."""
    if not isinstance(horizons, (list, tuple)) or any(type(h) is not int for h in horizons) or \
            tuple(horizons) not in HORIZONS:
        raise ValueError(f"horizons must be exactly one of {HORIZONS}")
    # Share the established feature/grid/kind/seed checks without broadening its allowed heads.
    c = R.build_config(features, slots, spacing, R.HORIZONS[0], kind=kind, seed=seed)
    if kind == "conv":
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        if not isinstance(core, (tuple, list)) or any(type(f) is not int for f in core) or \
                tuple(core) not in CORES:
            raise ValueError(f"core must be one of {CORES}")
        core = list(core)
    else:
        head, core = "none", []
    c.update(schema=SCHEMA, horizons_hours=list(horizons), head=head, core=core)
    return c


def _validated(config):
    if not isinstance(config, dict) or set(config) != _KEYS or config.get("schema") != SCHEMA:
        raise ValueError(f"schema/fields must match {SCHEMA}")
    return build_config(config["features"], config["slots"], config["spacing_open_market_hours"],
                        config["horizons_hours"], kind=config["kind"], seed=config["seed"],
                        head=config["head"], core=config["core"])


@keras.utils.register_keras_serializable(package="i6dmo")
class ElapsedContext(keras.layers.Layer):
    """Preserve fused values exactly and append real adjacent hours/24 plus closure flags."""

    def call(self, inputs):
        values, offsets, closure = inputs
        offsets = tf.cast(offsets, values.dtype)
        adjacent = tf.concat([tf.zeros_like(offsets[:, :1]), offsets[:, 1:] - offsets[:, :-1]], axis=1)
        context = tf.stack([adjacent / 24.0, tf.cast(closure, values.dtype)], axis=-1)
        return tf.concat([values, context], axis=-1)


@dataclass
class ContinuationBundle:
    """The config/forecast_model protocol for fit_dmo and canonical fresh reconstruction."""

    config: dict
    forecast_model: keras.Model


def _private_head(c, shape, horizon):
    name = M.head_name(horizon)
    if c["head"] == "dense":
        return M.dmo_head(input_shape=shape, horizon_hours=horizon, name=name, params=M._FIXED["heads"])
    inputs = keras.Input(shape)
    if c["head"] == "conv":
        x = keras.layers.Conv1D(4, 3, strides=1, padding="causal", activation="relu",
                                bias_initializer="zeros", name=name + "_conv")(inputs)
        x = keras.layers.Flatten(name=name + "_flatten")(x)
    else:
        x = keras.layers.LSTM(4, return_sequences=False, go_backwards=False,
                              unit_forget_bias=False, bias_initializer="zeros", name=name + "_lstm")(inputs)
    x = keras.layers.Dense(8, activation="relu", bias_initializer="zeros", name=name + "_hidden")(x)
    outputs = keras.layers.Dense(1, activation="linear", bias_initializer="zeros", name=name + "_out")(x)
    return keras.Model(inputs, outputs, name=name)


def _conv_outputs(c, observations, offsets, closure):
    slots, grid = c["slots"], tuple(range(c["slots"]))
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
                                           name="sequence_fusion", params={}), shapes, grid, 4 * len(sequences))
    if fusion.weights:
        raise ValueError("fusion must be weightless concatenation")
    encoded = ElapsedContext(name="physical_context")([fusion(sequences), offsets, closure])
    shape = tuple(encoded.shape[1:])
    params = {"filters": c["core"], "kernel_sizes": [3] * len(c["core"]),
              "strides": [1] * len(c["core"]), "activation": "relu"}
    core = M._validate_component(M.dmo_core(input_shape=shape, time_grid=grid, name="temporal_core", params=params),
                                 [shape], grid, c["core"][-1])
    # Narrow ReLU cores can be inactive at initialization; probe the graph, not that random gate.
    seeded_weights = core.get_weights()
    try:
        core.set_weights([np.full_like(w, 0.03) if w.ndim > 1 else np.zeros_like(w) for w in seeded_weights])
        M._probe_causal(core, grid, grid, "core")
    finally:
        core.set_weights(seeded_weights)
    latent = core(encoded)
    return [_private_head(c, tuple(latent.shape[1:]), h)(latent) for h in c["horizons_hours"]]


def build(config):
    """Validate and seed-build a new identity, with list outputs even for a single head."""
    c = _validated(config)
    M.enable_determinism()
    keras.utils.set_random_seed(c["seed"])
    observations = keras.Input((c["slots"], len(c["features"])), name="observations")
    offsets = keras.Input((c["slots"],), name="offsets_hours")
    closure = keras.Input((c["slots"],), name="closure_crossed")
    checked_offsets, checked_closure = R.TimeInputGuard(
        c["spacing_open_market_hours"], name="time_input_guard")([offsets, closure])
    output_fn = _conv_outputs if c["kind"] == "conv" else R._affine_outputs
    outputs = output_fn(c, observations, checked_offsets, checked_closure)
    forecast = R.HeadListModel([observations, offsets, closure], outputs, name="i6dmo_continuation_forecast")
    return ContinuationBundle(c, forecast)
