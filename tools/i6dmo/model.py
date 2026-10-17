"""I6-DMO minimal modular model, bridged to the hierarchical modular_temporal plugin.

Two families share the identical architecture: short (offsets -24..0 h every
2 h, heads 2/3/4/5/6/12 h) and long (offsets -144..0 h every 12 h, heads
18/24/36/48/60/72 h; PLAN.md section 5). Only the grid and head names differ.

Fixed identity (owner order section C): one branch per feature ``(batch, 13, 1)``
with causal Conv1D 4 filters / kernel 3 / stride 1 / ReLU; raw concatenation on
the filter axis (the plugin's built-in weightless ``sequence_concat`` fusion);
explicit sinusoidal positional encoding immediately after fusion whose
positions are the PHYSICAL offsets in hours (-24..0, origin kept at 0); core
causal Conv1D 8 then 4 filters, kernel 3, stride 1; six independent heads
(2/3/4/5/6/12 h), each Flatten -> Dense(8, relu) -> Dense(1, linear). Only the
heads collapse time. No LSTM, Transformer, Bayesian layer, MC sampling, decoder
or denoising.

Bridge, not fork: components are declared with the plugin's ``component()``
protocol (role, semantic version, parameters, contract), column routing reuses
``FeatureSelect``, fusion is resolved through the plugin registry, every branch
and the core pass the plugin's ``probe_alignment`` causality probe, and weight
identity uses the plugin's ``weights_hash``. ``build_modular`` itself is not
called because (a) its built-in branch/core/head defaults (GELU, Transformer
core, single Dense head) are historical identities that must not be modified,
(b) non-built-in components resolve only through installed entry points, and
(c) it labels horizons as multiples of ``sample_hours``, which would turn the
exact 3 h / 5 h endpoints into 6 h / 10 h under two-hour spacing.

R0 only: weights come from the fixed seed. Certified R1/R2 donors are a later
contrast and are refused here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import os as _os
import sys as _sys

# Static GPU algorithm selection, set BEFORE TensorFlow initializes in every entrypoint that
# imports this module: cuDNN/cuBLAS AUTOTUNE state is per-process, so a process that trained
# selects different (individually deterministic) kernels than one that only loads; op
# determinism makes each choice repeatable, not process-invariant (lane 4 witness verdict).
#: Init-ordering witness: if TensorFlow was already in sys.modules at THIS instant, the env
#: below may not govern its kernel selection, so records from this process are not scoreable
#: (lane 4 found the flags round bit-identical to the pre-flags round under the pytest
#: harness, which imports TF at collection; production spawns flags-first interpreters).
INIT_ORDER = {
    "tf_imported_before_flags": "tensorflow" in _sys.modules,
    "flags_at_import": {k: _os.environ.get(k) for k in
                        ("TF_CUDNN_USE_AUTOTUNE", "TF_CUDNN_DETERMINISTIC",
                         "TF_DETERMINISTIC_OPS", "TF_XLA_FLAGS")},
}
_os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
_os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
_os.environ["TF_DETERMINISTIC_OPS"] = "1"
if "--tf_xla_auto_jit=0" not in _os.environ.get("TF_XLA_FLAGS", ""):
    _os.environ["TF_XLA_FLAGS"] = (_os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_auto_jit=0").strip()

import tensorflow as tf

from predictor_plugins.modular_temporal import (FeatureSelect, PositionalEncoding, TemporalComponent,
                                                component, probe_alignment, weights_hash)
from predictor_plugins.modular_temporal.assembly import _validate_component
from predictor_plugins.modular_temporal.common import _file_hash, _major_minor, keras_version
from predictor_plugins.modular_temporal.registry import _resolve

keras = tf.keras
MODEL_SCHEMA = "i6dmo.model.v1"

#: Deterministic kernels for training AND inference so DMO10 save/load prediction identity is
#: bitwise on the SAME device (lane 4 GPU replay found nondeterministic kernel selection between
#: sessions). Enabled by every build/load path below; recorded in MODEL.json and the train receipt.
DETERMINISM = {"op_determinism": False, "api": "tf.config.experimental.enable_op_determinism",
               "autotune": "static", "jit_autoclustering": False, "env": {},
               "init_order": INIT_ORDER}
_DETERMINISM_ENV = ("TF_DETERMINISTIC_OPS", "TF_CUDNN_DETERMINISTIC", "TF_CUDNN_USE_AUTOTUNE",
                    "TF_XLA_FLAGS")


def enable_determinism():
    """Idempotently force deterministic, STATICALLY selected kernels; returns the flags.

    Op determinism plus autotune OFF (cuDNN) and XLA autoclustering OFF, so kernel
    selection is process-invariant, not merely repeatable within one process. The
    4.8k-parameter model makes the performance cost irrelevant.
    """
    _os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    _os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    _os.environ["TF_DETERMINISTIC_OPS"] = "1"
    if "--tf_xla_auto_jit=0" not in _os.environ.get("TF_XLA_FLAGS", ""):
        _os.environ["TF_XLA_FLAGS"] = (_os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_auto_jit=0").strip()
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    # Grappler remapping/fusion rewrites a traced forward differently in a process that has
    # trained; disable remapping and the sibling fusion passes so every process executes the
    # same op-for-op graph (XLA autoclustering is already off above).
    tf.config.optimizer.set_experimental_options({"remapping": False, "layout_optimizer": False,
                                                  "auto_mixed_precision": False})
    DETERMINISM["grappler"] = {"remapping": False, "layout_optimizer": False,
                               "auto_mixed_precision": False}
    # TF32 matmul/conv (Ampere+) rounds float32 mantissas per execution mode, not per weight;
    # disable explicitly so trained and loading processes share one FP32 forward semantics.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    DETERMINISM["tensor_float_32"] = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    DETERMINISM["op_determinism"] = True
    DETERMINISM["env"] = {k: _os.environ.get(k) for k in _DETERMINISM_ENV}
    return json.loads(_canonical(DETERMINISM))


def determinism_flags():
    """The canonical flag dict recorded in monitor records and enforced at replay."""
    enable_determinism()
    return {"op_determinism": True, "jit_autoclustering": False,
            "grappler_remapping": False, "grappler_layout_optimizer": False,
            "grappler_auto_mixed_precision": False,
            "tensor_float_32": DETERMINISM.get("tensor_float_32", False),
            "tf_imported_before_flags": INIT_ORDER["tf_imported_before_flags"],
            "TF_DETERMINISTIC_OPS": DETERMINISM["env"]["TF_DETERMINISTIC_OPS"],
            "TF_CUDNN_DETERMINISTIC": DETERMINISM["env"]["TF_CUDNN_DETERMINISTIC"],
            "TF_CUDNN_USE_AUTOTUNE": DETERMINISM["env"]["TF_CUDNN_USE_AUTOTUNE"]}
ARTIFACT_SCHEMA = "i6dmo.model_artifact.v1"
_KEYS = {"schema", "features", "offsets_hours", "horizons_hours", "branch", "core", "heads",
         "positional_encoding", "regime", "seed"}
_FIXED = {
    "branch": {"filters": 4, "kernel_size": 3, "strides": 1, "padding": "causal", "activation": "relu"},
    "core": {"filters": [8, 4], "kernel_sizes": [3, 3], "strides": [1, 1], "padding": "causal",
             "activation": "relu"},
    "heads": {"hidden_units": 8, "hidden_activation": "relu", "output_activation": "linear"},
    "positional_encoding": "physical_offset_sinusoid_after_fusion",
}
#: Exact (input offsets, head horizons) pairs per family; nothing else is accepted.
FAMILIES = {
    "short": {"offsets_hours": list(range(-24, 1, 2)), "horizons_hours": [2, 3, 4, 5, 6, 12]},
    "long": {"offsets_hours": list(range(-144, 1, 12)), "horizons_hours": [18, 24, 36, 48, 60, 72]},
    # Paired hourly-input control: same 24 elapsed hours and heads at hourly spacing (25 obs).
    # The only forced recipe change: each head flattens a 25x4 instead of a 13x4 core sequence.
    "short_hourly": {"offsets_hours": list(range(-24, 1, 1)), "horizons_hours": [2, 3, 4, 5, 6, 12]},
}
#: Market-time families (MARKET_TIME_PAIR_PLAN section 2/3): offsets are NOT a fixed grid, so the
#: config declares the slot count and the market_time block instead of offsets_hours, and the model
#: takes per-example time inputs. Elapsed spans vary across closures and DST by construction.
MARKET_FAMILIES = {
    "short_plus18": {"slots": 13, "horizons_hours": [2, 3, 4, 5, 6, 12, 18], "rule": "MG2_MARKET_EXACT",
                     "spacing_open_market_hours": 2, "span_open_market_hours": 24},
    "single18": {"slots": 13, "horizons_hours": [18], "rule": "MG2_MARKET_EXACT",
                 "spacing_open_market_hours": 2, "span_open_market_hours": 24},
    "short_window7": {"slots": 7, "horizons_hours": [2, 3, 4, 5, 6, 12], "rule": "MG2_MARKET_EXACT",
                      "spacing_open_market_hours": 2, "span_open_market_hours": 12},
    "short_window25": {"slots": 25, "horizons_hours": [2, 3, 4, 5, 6, 12], "rule": "MG2_MARKET_EXACT",
                       "spacing_open_market_hours": 2, "span_open_market_hours": 48},
    "short_mg2": {"slots": 13, "horizons_hours": [2, 3, 4, 5, 6, 12], "rule": "MG2_MARKET_EXACT",
                  "spacing_open_market_hours": 2, "span_open_market_hours": 24},
    "long_mg12": {"slots": 13, "horizons_hours": [18, 24, 36, 48, 60, 72], "rule": "MG12_MARKET_EXACT",
                  "spacing_open_market_hours": 12, "span_open_market_hours": 144},
    # NEXT_CONTRASTS_PLAN section 2, MG1 matched control: 25 real observations including t, one
    # expected open-market hour apart, the same 24 open-market-hour span and the same six heads.
    # Recipe unchanged; the only consequence is the wider head Flatten (25x4 instead of 13x4).
    "short_mg1": {"slots": 25, "horizons_hours": [2, 3, 4, 5, 6, 12], "rule": "MG1_MARKET_EXACT",
                  "spacing_open_market_hours": 1, "span_open_market_hours": 24},
}
#: The sealed effective config's model block is the authority: it carries ``offsets_hours: None``
#: (variable per example, never a fixed grid) and the sealed ``time_encoder`` identity, so both
#: are accepted here and normalized away. Review B-1 (RB04).
MARKET_KEYS = {"schema", "features", "slots", "market_time", "horizons_hours", "branch", "core",
               "heads", "positional_encoding", "regime", "seed"}
MARKET_OPTIONAL_KEYS = {"offsets_hours", "time_encoder"}
TIME_ENCODER_VERSION = "1.0.0"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def model_config(features, seed=6102026, family="short"):
    """The fixed R0 model config for an ordered feature list and family."""
    family = MATCHED_MODEL_FAMILIES.get(family, family)
    if family in MARKET_FAMILIES:
        spec = MARKET_FAMILIES[family]
        return normalize_model_config({
            "schema": MODEL_SCHEMA, "features": list(features), "regime": "R0", "seed": seed,
            "branch": json.loads(_canonical(_FIXED["branch"])),
            "core": json.loads(_canonical(_FIXED["core"])),
            "heads": json.loads(_canonical(_FIXED["heads"])),
            "positional_encoding": "dynamic_physical_time_after_fusion",
            "slots": spec["slots"], "horizons_hours": list(spec["horizons_hours"]),
            "market_time": {"rule": spec["rule"],
                            "spacing_open_market_hours": spec["spacing_open_market_hours"],
                            "span_open_market_hours": spec["span_open_market_hours"],
                            "observations": spec["slots"], "include_origin": True,
                            "elapsed_span_fixed": False, "require_exact_market_slots": True}})
    return normalize_model_config({"schema": MODEL_SCHEMA, "features": list(features), "regime": "R0",
                                   "seed": seed, **json.loads(_canonical(_FIXED)),
                                   **json.loads(_canonical(FAMILIES[family]))})


def is_market_time(config):
    return "market_time" in config


def market_family(config):
    """The MARKET_FAMILIES key whose rule the config declares (None for elapsed configs)."""
    if not is_market_time(config):
        return None
    rule = (config.get("market_time") or {}).get("rule")
    return next((name for name, spec in MARKET_FAMILIES.items() if spec["rule"] == rule), None)


def model_family(config):
    for name, grid in FAMILIES.items():
        if config["offsets_hours"] == grid["offsets_hours"] and config["horizons_hours"] == grid["horizons_hours"]:
            return name
    raise ValueError("FIXED_VALUE: offsets/horizons must be exactly one declared family pair "
                     f"{json.dumps(FAMILIES)}")


def _normalize_market_config(config):
    c = json.loads(_canonical(config))
    extra = set(c) - MARKET_KEYS - MARKET_OPTIONAL_KEYS
    missing = MARKET_KEYS - set(c)
    if extra or missing:
        raise ValueError(f"UNKNOWN_FIELD/schema mismatch in market-time model config: "
                         f"{sorted(extra | missing)}")
    if c.pop("offsets_hours", None) is not None:
        raise ValueError("FIXED_VALUE: a market-time model has no fixed offsets_hours grid; "
                         "offsets are per-example inputs")
    encoder = c.pop("time_encoder", None)
    if encoder is not None:
        if encoder.get("layer") != "i6dmo.DynamicPhysicalTimeEncoder" or \
                encoder.get("version") != TIME_ENCODER_VERSION or \
                encoder.get("unit_conversion_hours") != 24 or encoder.get("fitted") is not False or \
                encoder.get("context_channels") != ["adjacent_elapsed_hours_over_24",
                                                    "closure_crossed"]:
            raise ValueError("FIXED_VALUE: the sealed time_encoder identity does not match "
                             f"{TIME_ENCODER_VERSION} DynamicPhysicalTimeEncoder")
    if c["schema"] != MODEL_SCHEMA:
        raise ValueError(f"model schema must be {MODEL_SCHEMA}")
    if c["regime"] != "R0":
        raise ValueError("only R0 is built here; no donor is compatible by shape alone")
    if c["positional_encoding"] != "dynamic_physical_time_after_fusion":
        raise ValueError("FIXED_VALUE: market-time models use the dynamic physical-time encoder")
    for key in ("branch", "core", "heads"):
        if c[key] != _FIXED[key]:
            raise ValueError(f"FIXED_VALUE: model {key} is fixed to {_FIXED[key]!r}")
    mt = c["market_time"]
    match = next((spec for spec in MARKET_FAMILIES.values()
                  if spec["rule"] == mt.get("rule") and spec["slots"] == c["slots"]
                  and spec["horizons_hours"] == c["horizons_hours"]), None)
    if match is None:
        raise ValueError(f"FIXED_VALUE: unknown market-time rule {mt.get('rule')!r}")
    if (c["slots"], c["horizons_hours"]) != (match["slots"], match["horizons_hours"]) or \
            mt.get("spacing_open_market_hours") != match["spacing_open_market_hours"] or \
            mt.get("span_open_market_hours") != match["span_open_market_hours"] or \
            mt.get("elapsed_span_fixed") is not False or \
            mt.get("require_exact_market_slots") is not True or \
            mt.get("include_origin") is not True:
        raise ValueError("FIXED_VALUE: market-time scope contradicts the frozen plan")
    names = c["features"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError("features must be unique nonempty names")
    if type(c["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return c


def normalize_model_config(config):
    if isinstance(config, dict) and "market_time" in config:
        return _normalize_market_config(config)
    if not isinstance(config, dict) or set(config) != _KEYS:
        extra = sorted(set(config) - _KEYS) if isinstance(config, dict) else []
        raise ValueError(f"UNKNOWN_FIELD/schema mismatch in model config: {extra or 'missing keys'}")
    c = json.loads(_canonical(config))
    if c["schema"] != MODEL_SCHEMA:
        raise ValueError(f"model schema must be {MODEL_SCHEMA}")
    if c["regime"] != "R0":
        raise ValueError("only R0 (explicit random initialization) is built here; "
                         "R1/R2 need certified compatible donors (later contrast)")
    for key, value in _FIXED.items():
        if c[key] != value:
            raise ValueError(f"FIXED_VALUE: model {key} is fixed to {value!r}")
    model_family(c)
    names = c["features"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError("features must be unique nonempty names")
    if type(c["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return c


def head_name(h):
    return f"h{h}"


@keras.utils.register_keras_serializable(package="i6dmo")
class PhysicalOffsetEncoding(PositionalEncoding):
    """Sinusoidal encoding whose positions are physical hour offsets (origin = 0).

    Same frequency ladder as the plugin's ``PositionalEncoding``; only the
    position coordinate changes from the slot index to the declared physical
    offset, so the encoder sees the real two-hour spacing and the origin slot.
    """

    def __init__(self, offsets_hours, **kwargs):
        super().__init__(**kwargs)
        self.offsets_hours = tuple(int(v) for v in offsets_hours)

    def call(self, inputs):
        width = tf.shape(inputs)[2]
        position = tf.constant(self.offsets_hours, tf.float32)[:, None]
        channel = tf.range(width)[None, :]
        rates = tf.pow(10000.0, -2.0 * tf.cast(channel // 2, tf.float32) / tf.cast(width, tf.float32))
        angle = position * rates
        encoding = tf.where(channel % 2 == 0, tf.sin(angle), tf.cos(angle))
        return inputs + tf.cast(encoding[None, :, :], inputs.dtype)

    def get_config(self):
        return {**super().get_config(), "offsets_hours": list(self.offsets_hours)}


@keras.utils.register_keras_serializable(package="i6dmo")
class DynamicPhysicalTimeEncoder(keras.layers.Layer):
    """Dynamic physical-time encoder: the ONLY model extension for the market-time pair.

    Call signature ``[values, offsets_hours, closure_crossed]`` where ``values`` is the fused
    branch output ``(batch, slots, C)``, ``offsets_hours`` is the per-example signed UTC offset
    from the origin in hours ``(batch, slots)`` (the last entry exactly 0) and
    ``closure_crossed`` is the per-example adjacent-closure flag ``(batch, slots)`` (slot 0 is 0).

    It adds the same deterministic sinusoidal code as the plugin's ``PositionalEncoding`` but
    evaluated on the ACTUAL offsets instead of ordinal slot indices, then APPENDS exactly two
    context channels before the core: the adjacent elapsed hours divided by
    ``unit_conversion_hours`` (a fixed unit conversion, never fitted) and the closure flag.
    Output is ``(batch, slots, C + 2)``; with 19 branches that is 76 + 2 = 78. The layer has no
    weights and no fixed per-batch grid can replace its per-example inputs.
    """

    def __init__(self, unit_conversion_hours=24.0, min_adjacent_hours=2.0,
                 version=TIME_ENCODER_VERSION, **kwargs):
        super().__init__(**kwargs)
        if float(unit_conversion_hours) <= 0:
            raise ValueError("unit_conversion_hours must be positive")
        if float(min_adjacent_hours) <= 0:
            raise ValueError("min_adjacent_hours must be positive")
        if version != TIME_ENCODER_VERSION:
            raise ValueError(f"unsupported time encoder version {version!r}")
        self.unit_conversion_hours = float(unit_conversion_hours)
        # The declared market spacing in open-market hours: adjacent ELAPSED hours can only be
        # larger (closures stretch them), so a tighter ladder - an ordinal index sequence above
        # all - is physically impossible for this design and is refused. Default 2.0 is the
        # tightest declared market grid (MG2). Review B-4.
        self.min_adjacent_hours = float(min_adjacent_hours)
        self.version = version

    def build(self, input_shape):
        values, offsets, closure = input_shape
        if len(values) != 3 or len(offsets) != 2 or len(closure) != 2:
            raise ValueError("expected [values(batch, slots, C), offsets(batch, slots), "
                             "closure(batch, slots)]; a fixed per-batch grid is not accepted")
        if offsets[1] != values[1] or closure[1] != values[1]:
            raise ValueError("per-example time inputs must have one entry per slot")
        super().build(input_shape)

    def call(self, inputs):
        values, offsets, closure = inputs
        offsets = tf.cast(offsets, values.dtype)
        closure = tf.cast(closure, values.dtype)
        # Review B-4: an ordinal-index encoding or a physically impossible future offset must be
        # CAUGHT, not merely numerically different. The origin is the last slot (offset exactly 0),
        # every offset is in the past, and slot 0 has no adjacent delta and no closure flag.
        zero = tf.cast(0.0, values.dtype)
        checks = [
            tf.debugging.Assert(tf.reduce_all(tf.equal(offsets[:, -1], zero)),
                                ["ORDINAL_OR_INVALID_OFFSETS: the last offset must be exactly 0",
                                 offsets[:, -1]]),
            tf.debugging.Assert(tf.reduce_all(offsets <= zero),
                                ["ORDINAL_OR_INVALID_OFFSETS: offsets must be <= 0 (no future "
                                 "offsets, no ordinal slot indices)", tf.reduce_max(offsets)]),
            tf.debugging.Assert(tf.reduce_all(tf.equal(closure[:, 0], zero)),
                                ["ORDINAL_OR_INVALID_OFFSETS: the first closure flag must be 0",
                                 closure[:, 0]]),
        ]
        with tf.control_dependencies(checks):
            offsets = tf.identity(offsets)
        width = tf.shape(values)[2]
        channel = tf.range(width)[None, None, :]
        rates = tf.pow(tf.cast(10000.0, values.dtype),
                       -2.0 * tf.cast(channel // 2, values.dtype) / tf.cast(width, values.dtype))
        angle = offsets[:, :, None] * rates
        code = tf.where(channel % 2 == 0, tf.sin(angle), tf.cos(angle))
        encoded = values + code
        # Adjacent elapsed hours: slot 0 has delta 0 by declaration.
        deltas = offsets[:, 1:] - offsets[:, :-1]
        minimum = tf.cast(self.min_adjacent_hours, values.dtype)
        with tf.control_dependencies([tf.debugging.Assert(
                tf.reduce_all(deltas >= minimum),
                ["ORDINAL_OR_INVALID_OFFSETS: adjacent elapsed hours below the declared market "
                 "spacing (ordinal slot indices or a compressed ladder)",
                 tf.reduce_min(deltas), minimum])]):
            deltas = tf.identity(deltas)
        adjacent = tf.concat([tf.zeros_like(offsets[:, :1]), deltas], axis=1)
        context = tf.stack([adjacent / self.unit_conversion_hours, closure], axis=-1)
        return tf.concat([encoded, context], axis=-1)

    def compute_output_shape(self, input_shape):
        values = input_shape[0]
        return (*values[:-1], None if values[-1] is None else values[-1] + 2)

    def get_config(self):
        return {**super().get_config(), "unit_conversion_hours": self.unit_conversion_hours,
                "min_adjacent_hours": self.min_adjacent_hours, "version": self.version}


@component("branch", "1.0.0", {"filters", "kernel_size", "strides", "activation"},
           "(batch, 13, 1) -> (batch, 13, filters); causal Conv1D stride 1, no time reduction",
           defaults={"filters": 4, "kernel_size": 3, "strides": 1, "activation": "relu"})
def dmo_branch(*, input_shape, time_grid, output_steps, name, params):
    if output_steps != len(time_grid):
        raise ValueError("branches preserve the temporal grid")
    inputs = keras.Input(input_shape)
    x = keras.layers.Conv1D(params["filters"], params["kernel_size"], strides=params["strides"],
                            padding="causal", activation=params["activation"], name=name + "_conv")(inputs)
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


@component("core", "1.0.0", {"filters", "kernel_sizes", "strides", "activation"},
           "(batch, 13, C) -> (batch, 13, filters[-1]); causal Conv1D stack, stride 1",
           defaults={"filters": [8, 4], "kernel_sizes": [3, 3], "strides": [1, 1], "activation": "relu"})
def dmo_core(*, input_shape, time_grid, name, params):
    inputs = keras.Input(input_shape)
    x = inputs
    for i, (f, k, s) in enumerate(zip(params["filters"], params["kernel_sizes"], params["strides"])):
        x = keras.layers.Conv1D(f, k, strides=s, padding="causal", activation=params["activation"],
                                name=f"core_conv_{i}")(x)
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


@component("head", "1.0.0", {"hidden_units", "hidden_activation", "output_activation"},
           "(batch, 13, 4) -> (batch, 1); Flatten -> Dense(hidden) -> Dense(1); one horizon, own weights",
           defaults={"hidden_units": 8, "hidden_activation": "relu", "output_activation": "linear"})
def dmo_head(*, input_shape, horizon_hours, name, params):
    inputs = keras.Input(input_shape)
    x = keras.layers.Flatten(name=name + "_flatten")(inputs)
    x = keras.layers.Dense(params["hidden_units"], activation=params["hidden_activation"],
                           name=name + "_hidden")(x)
    x = keras.layers.Dense(1, activation=params["output_activation"], name=name + "_out")(x)
    return keras.Model(inputs, x, name=name)


def _identity(factory):
    return {"name": factory.__name__, "implementation": f"tools.i6dmo.model:{factory.__name__}",
            "role": factory.component_role, "version": factory.component_version,
            "contract": factory.component_contract}


@dataclass
class DmoBundle:
    config: dict
    forecast_model: keras.Model
    branch_models: dict
    fusion_model: keras.Model
    core_model: keras.Model
    head_models: dict
    fusion_identity: dict

    def component_manifests(self):
        c = self.config
        return {
            "schema": 1,
            "model_config_sha256": hashlib.sha256(_canonical(c).encode()).hexdigest(),
            "branch": {**_identity(dmo_branch), "params": _branch_params(c), "count": len(self.branch_models),
                       "input_shape": [c["slots"] if is_market_time(c) else len(c["offsets_hours"]), 1]},
            "fusion": self.fusion_identity,
            "positional_encoding": ({"layer": "tools.i6dmo.model:DynamicPhysicalTimeEncoder",
                                     "offsets": "per_example_actual_utc_hours"}
                                    if is_market_time(c) else
                                    {"layer": "tools.i6dmo.model:PhysicalOffsetEncoding",
                                     "base": "predictor_plugins.modular_temporal:PositionalEncoding",
                                     "offsets_hours": c["offsets_hours"]}),
            "core": {**_identity(dmo_core), "params": _core_params(c),
                     "output_shape": list(self.core_model.output_shape[1:])},
            "time_encoder": (dict(DynamicPhysicalTimeEncoder(
                                      min_adjacent_hours=float(c["market_time"]["spacing_open_market_hours"])
                                  ).get_config(),
                                  layer="i6dmo.DynamicPhysicalTimeEncoder",
                                  position="immediately_after_branch_fusion",
                                  context_channels=["adjacent_elapsed_hours_over_24",
                                                    "closure_crossed"])
                             if is_market_time(c) else None),
            "heads": {**_identity(dmo_head), "params": c["heads"], "names": list(self.head_models),
                      "horizons_hours": c["horizons_hours"]},
            "parameters": {"branches": int(sum(m.count_params() for m in self.branch_models.values())),
                           "core": int(self.core_model.count_params()),
                           "heads": {n: int(m.count_params()) for n, m in self.head_models.items()},
                           "total": int(self.forecast_model.count_params())},
        }


def _branch_params(c):
    b = c["branch"]
    return {"filters": b["filters"], "kernel_size": b["kernel_size"], "strides": b["strides"],
            "activation": b["activation"]}


def _core_params(c):
    k = c["core"]
    return {"filters": k["filters"], "kernel_sizes": k["kernel_sizes"], "strides": k["strides"],
            "activation": k["activation"]}


def _probe_causal(model, in_grid, out_grid, label):
    """Plugin causality probe with ReLU-aware retries.

    A look-ahead violation fails on the FIRST seed (hard). The participation half
    can false-negative when one random perturbation lands every filter in a dead
    ReLU region, so it is retried over a few fixed seeds and must pass on one.
    """
    last = None
    for seed in (0, 1, 2, 3):
        try:
            return probe_alignment(model, in_grid, out_grid, seed=seed, label=label)
        except ValueError as exc:
            if "depends on the later" in str(exc):
                raise
            last = exc
    raise last


def build_dmo(config):
    """Build the fixed R0 model. Seeds the global RNG first (explicit random init)."""
    c = normalize_model_config(config)
    enable_determinism()
    keras.utils.set_random_seed(c["seed"])
    market = is_market_time(c)
    if market:
        steps, n = c["slots"], len(c["features"])
        grid = tuple(range(steps))          # SLOT ORDER ONLY (not wall-clock time) for the probes
    else:
        grid = tuple(c["offsets_hours"])
        steps, n = len(grid), len(c["features"])
    inputs = keras.Input((steps, n), name="observations")
    time_inputs = ([keras.Input((steps,), name="offsets_hours"),
                    keras.Input((steps,), name="closure_crossed")] if market else [])
    branches, sequences = {}, []
    for i, feature in enumerate(c["features"]):
        name = f"branch_{i:02d}"
        comp = dmo_branch(input_shape=(steps, 1), time_grid=grid, output_steps=steps, name=name,
                          params=_branch_params(c))
        model = _validate_component(comp, [(steps, 1)], grid, c["branch"]["filters"])
        _probe_causal(model, grid, grid, "branch " + name)
        branches[feature] = model
        sequences.append(model(FeatureSelect([i], name="select_" + name)(inputs)))
    factory, fusion_identity = _resolve("fusion", {"plugin": "sequence_concat"}, {"fusion": "modular.fusion"})
    shapes = [tuple(s.shape[1:]) for s in sequences]
    fusion_comp = factory(input_shapes=shapes, time_grid=grid, name="sequence_fusion", params={})
    fusion = _validate_component(fusion_comp, shapes, grid, sum(s[1] for s in shapes))
    if fusion.weights:
        raise ValueError("fusion must be weightless concatenation")
    fused = fusion(sequences)
    if market:
        encoder = DynamicPhysicalTimeEncoder(
            min_adjacent_hours=float(c["market_time"]["spacing_open_market_hours"]),
            name="physical_time_encoder")
        encoded = encoder([fused, *time_inputs])
    else:
        encoded = PhysicalOffsetEncoding(grid, name="positional_encoding")(fused)
    core_shape = tuple(encoded.shape[1:])
    core_comp = dmo_core(input_shape=core_shape, time_grid=grid, name="temporal_core",
                         params=_core_params(c))
    core = _validate_component(core_comp, [core_shape], grid, c["core"]["filters"][-1])
    _probe_causal(core, grid, grid, "core")
    latent = core(encoded)
    heads, outputs = {}, []
    for h in c["horizons_hours"]:
        name = head_name(h)
        head = dmo_head(input_shape=tuple(latent.shape[1:]), horizon_hours=h, name=name, params=c["heads"])
        heads[name] = head
        outputs.append(head(latent))
    forecast = keras.Model([inputs, *time_inputs], outputs, name="i6dmo_forecast")
    fusion_model = keras.Model(inputs, fused, name="fusion_model")
    return DmoBundle(c, forecast, branches, fusion_model, core, heads, fusion_identity)


# ----------------------------------------------------------------------------- capacity (SC07)

CAPACITY_SCHEMA = "i6dmo.capacity.v1"
#: NEXT_CONTRASTS_PLAN "Capacity disclosure": the ONLY declared capacity difference between the
#: matched arms is the wider head Flatten, (25 - 13) slots * 4 core channels * 8 hidden units *
#: 6 heads. Any other difference (branch, fusion, time encoder, core, head width/depth) is an
#: unreported capacity change and is refused.
#: Data-lane arm ids (design.MATCHED_SCOPE) -> model family. The sealed matched design families
#: short_mg1_matched / short_mg2_matched build exactly these recipes (MATCHED_MODEL_FAMILIES).
MATCHED_ARMS = {"mg1": "short_mg1", "mg2m": "short_mg2"}
MATCHED_MODEL_FAMILIES = {"short_mg1_matched": "short_mg1", "short_mg2_matched": "short_mg2"}
MATCHED_HEAD_DELTA = (25 - 13) * 4 * 8 * 6
#: Declared per-component trainable parameter counts of the fixed R0 recipe with the 19 RAW
#: features, written out literally so a code change that alters capacity cannot pass silently:
#: the build must reproduce BOTH the analytic recipe count and this literal disclosure.
DECLARED_CAPACITY = {
    "short_mg1": {"slots": 25, "features": 19, "core_input_channels": 78,
                  "branches": 304, "per_branch": 16, "fusion": 0, "time_encoder": 0,
                  "core": 1980, "core_layers": [1880, 100],
                  "per_head": 817, "per_head_hidden": 808, "per_head_output": 9,
                  "heads": 4902, "total": 7186},
    "short_mg2": {"slots": 13, "features": 19, "core_input_channels": 78,
                  "branches": 304, "per_branch": 16, "fusion": 0, "time_encoder": 0,
                  "core": 1980, "core_layers": [1880, 100],
                  "per_head": 433, "per_head_hidden": 424, "per_head_output": 9,
                  "heads": 2598, "total": 4882},
}


class CapacityError(ValueError):
    """Typed CAPACITY_UNREPORTED_CHANGE: the built network differs from the declared recipe."""

    code = "CAPACITY_UNREPORTED_CHANGE"

    def __init__(self, detail):
        super().__init__(f"{self.code}: {detail}")


def declared_parameter_counts(config):
    """Analytic trainable parameter counts implied by the declared fixed recipe."""
    c = normalize_model_config(config)
    market = is_market_time(c)
    slots = c["slots"] if market else len(c["offsets_hours"])
    n = len(c["features"])
    b, k, hd = c["branch"], c["core"], c["heads"]
    per_branch = b["kernel_size"] * 1 * b["filters"] + b["filters"]
    channels = n * b["filters"] + (2 if market else 0)
    core_input = channels
    core_layers = []
    for filters, kernel in zip(k["filters"], k["kernel_sizes"]):
        core_layers.append(kernel * channels * filters + filters)
        channels = filters
    flat = slots * channels
    hidden = flat * hd["hidden_units"] + hd["hidden_units"]
    output = hd["hidden_units"] * 1 + 1
    heads = len(c["horizons_hours"])
    total = per_branch * n + sum(core_layers) + heads * (hidden + output)
    return {"slots": slots, "features": n, "core_input_channels": core_input,
            "branches": per_branch * n, "per_branch": per_branch, "fusion": 0, "time_encoder": 0,
            "core": sum(core_layers), "core_layers": core_layers,
            "per_head": hidden + output, "per_head_hidden": hidden, "per_head_output": output,
            "heads": heads * (hidden + output), "total": total}


def measured_parameter_counts(bundle):
    """Per-component trainable parameter counts MEASURED on the built Keras graph."""
    model = bundle.forecast_model
    branches = {f: int(m.count_params()) for f, m in bundle.branch_models.items()}
    core_layers = [int(layer.count_params()) for layer in bundle.core_model.layers
                   if layer.weights]
    heads = {}
    for name, head in bundle.head_models.items():
        heads[name] = {"hidden": int(head.get_layer(name + "_hidden").count_params()),
                       "output": int(head.get_layer(name + "_out").count_params()),
                       "total": int(head.count_params())}
    encoder = sum(int(layer.count_params()) for layer in model.layers
                  if isinstance(layer, (DynamicPhysicalTimeEncoder, PhysicalOffsetEncoding)))
    fusion = int(bundle.fusion_model.count_params()) - sum(branches.values())
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    per_head = sorted({v["total"] for v in heads.values()})
    out = {"slots": int(bundle.core_model.input_shape[1]), "features": len(branches),
           "core_input_channels": int(bundle.core_model.input_shape[-1]),
           "branches": sum(branches.values()),
           "per_branch": (sorted(set(branches.values())) if len(set(branches.values())) != 1
                          else next(iter(branches.values()))),
           "fusion": fusion, "time_encoder": encoder,
           "core": int(bundle.core_model.count_params()), "core_layers": core_layers,
           "per_head": per_head[0] if len(per_head) == 1 else per_head,
           "per_head_hidden": (heads[next(iter(heads))]["hidden"]
                               if len({v["hidden"] for v in heads.values()}) == 1
                               else sorted({v["hidden"] for v in heads.values()})),
           "per_head_output": (heads[next(iter(heads))]["output"]
                               if len({v["output"] for v in heads.values()}) == 1
                               else sorted({v["output"] for v in heads.values()})),
           "heads": sum(v["total"] for v in heads.values()), "total": total}
    out["_detail"] = {"branches": branches, "heads": heads, "trainable_total": trainable,
                      "non_trainable_total": total - trainable}
    return out


def capacity_record(bundle):
    """Machine-readable capacity disclosure; raises CapacityError on any unreported change.

    The build must reproduce (a) the analytic count implied by the declared recipe, (b) the
    literal DECLARED_CAPACITY entry for a matched arm, and (c) the accounting identity
    total = branches + fusion + time_encoder + core + heads with no non-trainable weights.
    """
    declared = declared_parameter_counts(bundle.config)
    measured = measured_parameter_counts(bundle)
    detail = measured.pop("_detail")
    problems = [f"{k}: measured {measured[k]!r} != recipe {declared[k]!r}"
                for k in declared if measured.get(k) != declared[k]]
    family = market_family(bundle.config)
    literal = DECLARED_CAPACITY.get(family)
    if literal is not None:
        problems += [f"{k}: measured {measured[k]!r} != declared disclosure {literal[k]!r}"
                     for k in literal if measured.get(k) != literal[k]]
    parts = sum(measured[k] for k in ("branches", "fusion", "time_encoder", "core", "heads"))
    if parts != measured["total"] or detail["non_trainable_total"] != 0:
        problems.append(f"unaccounted parameters: components {parts}, total {measured['total']}, "
                        f"non-trainable {detail['non_trainable_total']}")
    if problems:
        raise CapacityError("; ".join(problems))
    record = {"schema": CAPACITY_SCHEMA, "family": family,
              "arm": next((a for a, f in MATCHED_ARMS.items() if f == family), None),
              "model_config_sha256": hashlib.sha256(_canonical(bundle.config).encode()).hexdigest(),
              "counts": measured, "per_branch_detail": detail["branches"],
              "per_head_detail": detail["heads"], "recipe_match": True,
              "declared_disclosure_match": literal is not None}
    record["capacity_sha256"] = hashlib.sha256(_canonical(record).encode()).hexdigest()
    return record


def matched_capacity_delta(record_mg1, record_mg2):
    """The disclosed MG1-minus-MG2 difference; refuses anything but the declared head delta."""
    for record, arm in ((record_mg1, "mg1"), (record_mg2, "mg2m")):
        if not isinstance(record, dict) or record.get("schema") != CAPACITY_SCHEMA or \
                record.get("arm") != arm:
            raise CapacityError(f"{arm} capacity record missing or not the {arm} arm")
        literal = DECLARED_CAPACITY[MATCHED_ARMS[arm]]
        if any(record["counts"].get(k) != v for k, v in literal.items()):
            raise CapacityError(f"{arm} record disagrees with the declared disclosure")
    a, b = record_mg1["counts"], record_mg2["counts"]
    delta = {k: a[k] - b[k] for k in ("branches", "fusion", "time_encoder", "core", "heads",
                                       "total")}
    expected = {"branches": 0, "fusion": 0, "time_encoder": 0, "core": 0,
                "heads": MATCHED_HEAD_DELTA, "total": MATCHED_HEAD_DELTA}
    if delta != expected:
        raise CapacityError(f"mg1-mg2m delta {delta} != declared {expected}")
    return {"schema": "i6dmo.capacity_delta.v1", "delta": delta,
            "formula": "(25-13)*4*8*6", "declared_head_delta": MATCHED_HEAD_DELTA,
            "interpretation": "practical input/readout recipe comparison; not proof that "
                              "additional signal alone caused any gain"}


def predict_matrix(model, x, batch_size=4096):
    """(n, heads) float64 predictions in horizon order, in the training coordinate.

    ``x`` is one array, or a list of arrays for a market-time model
    ``[observations, offsets_hours, closure_crossed]`` (all sliced consistently).
    """
    if isinstance(x, (list, tuple)):
        arrays = [np.asarray(part, dtype=np.float32) for part in x]
        if len({len(part) for part in arrays}) != 1:
            raise ValueError("every model input must carry the same number of examples")
    else:
        arrays = [np.asarray(x, dtype=np.float32)]
    x = arrays[0]
    parts = []
    for start in range(0, len(x), batch_size):
        batch = [part[start:start + batch_size] for part in arrays]
        out = model(batch if len(batch) > 1 else batch[0], training=False)
        parts.append(np.concatenate([np.asarray(o, dtype=np.float64) for o in out], axis=1))
    width = len(model.outputs)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, width))


def weight_tensor_digests(model):
    """Per-tensor identity: shape, dtype and sha256 of each weight, in model order."""
    out = []
    for value in model.get_weights():
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(_canonical([list(value.shape), str(value.dtype)]).encode("ascii"))
        digest.update(value.tobytes())
        out.append({"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest.hexdigest()})
    return out


def save_dmo(bundle, directory, extra=None):
    """Write the weights file (canonical), full archive (compatibility) and MODEL.json."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "forecast_model.keras"
    bundle.forecast_model.save(archive)
    weights_file = out / "model.weights.h5"
    bundle.forecast_model.save_weights(weights_file)
    market = is_market_time(bundle.config)
    document = {"schema": ARTIFACT_SCHEMA, "config": bundle.config, "keras_version": keras_version(),
                "determinism": json.loads(_canonical(DETERMINISM)),
                "input_names": [t.name.split(":")[0].split("/")[0] for t in bundle.forecast_model.inputs],
                "market_time": (dict(bundle.config["market_time"],
                                     slot_order_is_not_wall_clock=True,
                                     time_inputs=["offsets_hours", "closure_crossed"],
                                     time_encoder_version=TIME_ENCODER_VERSION)
                                if market else None),
                "weight_tensors": weight_tensor_digests(bundle.forecast_model),
                "weights_file_sha256": _file_hash(weights_file),
                "tensorflow_version": tf.__version__, "output_names": list(bundle.forecast_model.output_names),
                "components": bundle.component_manifests(),
                "weights_sha256": weights_hash(bundle.forecast_model), "archive_sha256": _file_hash(archive),
                "extra": extra or {}}
    (out / "MODEL.json").write_text(json.dumps(document, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return document


def load_dmo(directory, compatibility_check=True):
    """Canonical load: REBUILD the architecture from the saved config via ``build_dmo`` and
    ``load_weights`` from the weights file, verifying every tensor digest recorded at save
    time (typed WEIGHTS_DIGEST_MISMATCH). Keras full-model deserialization is only a
    secondary compatibility check on the WEIGHTS (never the scoring graph): on GPU a
    deserialized graph can fuse/trace differently from the built one (lane 4, DMO10).
    """
    src = Path(directory)
    document = json.loads((src / "MODEL.json").read_text(encoding="utf-8"))
    if document.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported model artifact schema")
    if _major_minor(document["keras_version"]) != _major_minor(keras_version()):
        raise ValueError(f"saved under Keras {document['keras_version']}, running {keras_version()}")
    enable_determinism()
    archive = src / "forecast_model.keras"
    if _file_hash(archive) != document["archive_sha256"]:
        raise ValueError("model archive hash mismatch")
    weights_file = src / "model.weights.h5"
    if _file_hash(weights_file) != document["weights_file_sha256"]:
        raise ValueError("weights file hash mismatch")
    if (document.get("market_time") is not None) != is_market_time(document["config"]):
        raise ValueError("MARKET_TIME_INCOMPATIBLE: artifact metadata and config disagree; no "
                         "donor is compatible by shape alone")
    rebuilt = build_dmo(document["config"])
    rebuilt.forecast_model.load_weights(weights_file)
    tensors = weight_tensor_digests(rebuilt.forecast_model)
    if tensors != document["weight_tensors"]:
        bad = [i for i, (a, b) in enumerate(zip(tensors, document["weight_tensors"])) if a != b]
        raise ValueError(f"WEIGHTS_DIGEST_MISMATCH: loaded weight tensors {bad} differ from the "
                         "digests recorded at save time")
    if weights_hash(rebuilt.forecast_model) != document["weights_sha256"]:
        raise ValueError("rebuilt architecture does not hold the saved weights")
    if list(rebuilt.forecast_model.output_names) != document["output_names"]:
        raise ValueError("output names drifted")
    document = dict(document, load_path="rebuild+load_weights")
    if compatibility_check:
        stored = keras.models.load_model(archive, compile=False, safe_mode=True)
        if weights_hash(stored) != document["weights_sha256"]:
            raise ValueError("archived full model carries different weights (compatibility check)")
    return rebuilt, document
