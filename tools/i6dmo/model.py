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
import math
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
CONTEXT_BRIDGE_MODEL_FAMILIES = ("long_mg12_bridge_control", "long_mg2_bridge")
MARKET_FAMILIES = {
    "short_mg2": {"slots": 13, "horizons_hours": [2, 3, 4, 5, 6, 12], "rule": "MG2_MARKET_EXACT",
                  "spacing_open_market_hours": 2, "span_open_market_hours": 24},
    "long_mg12": {"slots": 13, "horizons_hours": [18, 24, 36, 48, 60, 72], "rule": "MG12_MARKET_EXACT",
                  "spacing_open_market_hours": 12, "span_open_market_hours": 144},
    # NEXT_CONTRASTS_PLAN section 2, MG1 matched control: 25 real observations including t, one
    # expected open-market hour apart, the same 24 open-market-hour span and the same six heads.
    # Recipe unchanged; the only consequence is the wider head Flatten (25x4 instead of 13x4).
    "short_mg1": {"slots": 25, "horizons_hours": [2, 3, 4, 5, 6, 12], "rule": "MG1_MARKET_EXACT",
                  "spacing_open_market_hours": 1, "span_open_market_hours": 24},
    "long_mg1": {"slots": 145, "horizons_hours": [18, 24, 36, 48, 60, 72], "rule": "MG1_MARKET_EXACT",
                 "spacing_open_market_hours": 1, "span_open_market_hours": 144},
    "long_mg12_bridge_control": {
        "slots": 13, "horizons_hours": [18, 24, 36, 48, 60, 72], "rule": "MG12_MARKET_EXACT",
        "spacing_open_market_hours": 12, "span_open_market_hours": 144},
    "long_mg2_bridge": {
        "slots": 13, "horizons_hours": [18, 24, 36, 48, 60, 72], "rule": "MG2_MARKET_EXACT",
        "spacing_open_market_hours": 2, "span_open_market_hours": 24},
}
#: The sealed effective config's model block is the authority: it carries ``offsets_hours: None``
#: (variable per example, never a fixed grid) and the sealed ``time_encoder`` identity, so both
#: are accepted here and normalized away. Review B-1 (RB04).
MARKET_KEYS = {"schema", "features", "slots", "market_time", "horizons_hours", "branch", "core",
               "heads", "positional_encoding", "regime", "seed"}
MARKET_OPTIONAL_KEYS = {"offsets_hours", "time_encoder", "family"}
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
            **({"family": family} if family in CONTEXT_BRIDGE_MODEL_FAMILIES else {}),
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
    """Match rule/slots/heads; bridge tags distinguish its control from legacy MG12."""
    if not is_market_time(config):
        return None
    rule = (config.get("market_time") or {}).get("rule")
    tagged = config.get("family")
    if "family" in config:
        if tagged not in CONTEXT_BRIDGE_MODEL_FAMILIES:
            return None
        candidates = ((tagged, MARKET_FAMILIES[tagged]),)
    else:
        candidates = MARKET_FAMILIES.items()
    return next((name for name, spec in candidates
                 if spec["rule"] == rule and spec["slots"] == config.get("slots")
                 and spec["horizons_hours"] == config.get("horizons_hours")), None)


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
    match = MARKET_FAMILIES.get(market_family(c))
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
    if len(in_grid) > 25:
        return _probe_long_causal(model, in_grid, out_grid, label)
    last = None
    for seed in (0, 1, 2, 3):
        try:
            return probe_alignment(model, in_grid, out_grid, seed=seed, label=label)
        except ValueError as exc:
            if "depends on the later" in str(exc):
                raise
            last = exc
    raise last


def _probe_long_causal(model, in_grid, out_grid, label):
    """Every long-window slot must participate, across fixed ReLU activation baselines.

    No-lookahead is checked on every perturbation. Participation is accumulated per slot,
    since requiring all 145 ReLUs to activate on one random baseline is not reliable.
    No weights, input ordering or tolerances are changed.
    """
    length, channels = int(model.input_shape[1]), int(model.input_shape[2])
    if length != len(in_grid) or int(model.output_shape[1]) != len(out_grid):
        raise ValueError(f"{label}: probe grid does not match model shape")
    future = np.asarray(out_grid)[None, :] < np.asarray(in_grid)[:, None]
    first = np.asarray([next((k for k, edge in enumerate(out_grid) if edge >= t), -1)
                        for t in in_grid])
    active = np.zeros(length, dtype=bool)
    for seed in range(8):
        base = np.random.default_rng(seed).normal(size=(1, length, channels)).astype("float32")
        batch = np.repeat(base, 2 * length + 1, axis=0)
        for i in range(length):
            batch[i + 1, i, :] += 3.0
            batch[length + i + 1, i, :] -= 3.0
        out = np.asarray(model(batch, training=False), dtype="float64")
        moved = np.max(np.abs(out[1:] - out[:1]), axis=2)
        moved = np.maximum(moved[:length], moved[length:])
        scale = max(1.0, float(np.max(np.abs(out[0]))))
        if np.any(moved[future] > 1e-5 * scale):
            raise ValueError(f"{label}: time alignment violated; output depends on the later input")
        active |= (first >= 0) & (moved[np.arange(length), np.maximum(first, 0)] > 1e-7 * scale)
        if active.all():
            return {"checked_inputs": length, "checked_outputs": len(out_grid)}
    raise ValueError(f"{label}: time alignment violated; inputs at "
                     f"{np.asarray(in_grid)[~active].tolist()} do not reach their right-edge outputs")


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
MATCHED_ARMS = {"mg1": "short_mg1", "mg2m": "short_mg2",
                "mg1l": "long_mg1", "mg12m": "long_mg12",
                "mg12c": "long_mg12_bridge_control", "mg2l": "long_mg2_bridge"}
MATCHED_MODEL_FAMILIES = {"short_mg1_matched": "short_mg1", "short_mg2_matched": "short_mg2",
                          "long_mg1_matched": "long_mg1", "long_mg12_matched": "long_mg12",
                          "long_mg12_bridge_control": "long_mg12_bridge_control",
                          "long_mg2_bridge": "long_mg2_bridge"}
MATCHED_HEAD_DELTA = (25 - 13) * 4 * 8 * 6
LONG_MATCHED_HEAD_DELTA = (145 - 13) * 4 * 8 * 6
CONTEXT_BRIDGE_HEAD_DELTA = (13 - 13) * 4 * 8 * 6
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
    "long_mg1": {"slots": 145, "features": 19, "core_input_channels": 78,
                 "branches": 304, "per_branch": 16, "fusion": 0, "time_encoder": 0,
                 "core": 1980, "core_layers": [1880, 100],
                 "per_head": 4657, "per_head_hidden": 4648, "per_head_output": 9,
                 "heads": 27942, "total": 30226},
    "long_mg12": {"slots": 13, "features": 19, "core_input_channels": 78,
                  "branches": 304, "per_branch": 16, "fusion": 0, "time_encoder": 0,
                  "core": 1980, "core_layers": [1880, 100],
                  "per_head": 433, "per_head_hidden": 424, "per_head_output": 9,
                  "heads": 2598, "total": 4882},
    "long_mg12_bridge_control": {
        "slots": 13, "features": 19, "core_input_channels": 78,
        "branches": 304, "per_branch": 16, "fusion": 0, "time_encoder": 0,
        "core": 1980, "core_layers": [1880, 100],
        "per_head": 433, "per_head_hidden": 424, "per_head_output": 9,
        "heads": 2598, "total": 4882},
    "long_mg2_bridge": {
        "slots": 13, "features": 19, "core_input_channels": 78,
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
    """Disclosed ordered-arm capacity delta for exactly one matched contrast."""
    if not isinstance(record_mg1, dict) or not isinstance(record_mg2, dict):
        raise CapacityError("two matched capacity records required")
    arms = (record_mg1.get("arm"), record_mg2.get("arm"))
    if arms == ("mg1", "mg2m"):
        head_delta, formula = MATCHED_HEAD_DELTA, "(25-13)*4*8*6"
    elif arms == ("mg1l", "mg12m"):
        head_delta, formula = LONG_MATCHED_HEAD_DELTA, "(145-13)*4*8*6"
    elif arms == ("mg12c", "mg2l"):
        head_delta, formula = CONTEXT_BRIDGE_HEAD_DELTA, "(13-13)*4*8*6"
    else:
        raise CapacityError(f"not an ordered matched arm pair: {arms}")
    for record, arm in zip((record_mg1, record_mg2), arms):
        if not isinstance(record, dict) or record.get("schema") != CAPACITY_SCHEMA or \
                record.get("arm") != arm:
            raise CapacityError(f"{arm} capacity record missing or not the {arm} arm")
        if "family" in record and record["family"] != MATCHED_ARMS[arm]:
            raise CapacityError(f"{arm} record declares another model family")
        literal = DECLARED_CAPACITY[MATCHED_ARMS[arm]]
        if not isinstance(record.get("counts"), dict) or \
                any(record["counts"].get(k) != v for k, v in literal.items()):
            raise CapacityError(f"{arm} record disagrees with the declared disclosure")
    a, b = record_mg1["counts"], record_mg2["counts"]
    delta = {k: a[k] - b[k] for k in ("branches", "fusion", "time_encoder", "core", "heads",
                                       "total")}
    expected = {"branches": 0, "fusion": 0, "time_encoder": 0, "core": 0,
                "heads": head_delta, "total": head_delta}
    if delta != expected:
        raise CapacityError(f"{arms} delta {delta} != declared {expected}")
    return {"schema": "i6dmo.capacity_delta.v1", "delta": delta,
            "formula": formula, "declared_head_delta": head_delta,
            "interpretation": "practical input/readout recipe comparison; not proof that "
                              "additional signal alone caused any gain"}


# ----------------------------------------------------------------------------- R2 branch donors
#
# NEXT_CONTRASTS_PLAN section 3 / R2_DESIGN_PROPOSAL.md: R2 = the SAME network as R0 whose 19
# per-feature branch encoders start from compatible pretrained weights, imported and kept
# TRAINABLE. Conv1D kernel shapes (3, 1, 4) do not depend on the slot count, so an MG1 (25-slot),
# DS2/hourly/lagged, closed full-population or FS4/I6-B branch has weights of IDENTICAL shape:
# shape is never evidence of compatibility. Every donor must carry and match the full identity
# below; anything else is a typed refusal. Design layer only: no pretraining runner here.

BRANCH_DONOR_SCHEMA = "i6dmo.branch_donor.v1"
R2_OBJECTIVE_UNDECIDED = "UNDECIDED_REQUIRES_REVIEW"
BRANCH_DONOR_KEYS = ("schema", "arm", "rule", "slots", "spacing_open_market_hours", "feature_name",
                     "feature_index", "feature_names_sha256", "branch_model_name", "conv_layer_name",
                     "branch_params", "input_shape", "output_shape", "scaler_sha256",
                     "population_sha256", "objective", "weights")


class DonorImportError(ValueError):
    """Typed refusal of an R2 branch-donor import (the message starts with the code)."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def scaler_identity_sha256(feature_names, mean, std, fit_row_digest, n_rows):
    """Identity of the shared FIT-only scaler (exact float64 values, feature order, row digest)."""
    mean = [float(v).hex() for v in np.asarray(mean, dtype=np.float64).reshape(-1)]
    std = [float(v).hex() for v in np.asarray(std, dtype=np.float64).reshape(-1)]
    return hashlib.sha256(_canonical({"feature_names": list(feature_names), "mean": mean, "std": std,
                                      "fit_row_digest": str(fit_row_digest),
                                      "n_rows": int(n_rows)}).encode()).hexdigest()


def _array_digests(arrays):
    out = []
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(_canonical([list(value.shape), str(value.dtype)]).encode("ascii"))
        digest.update(value.tobytes())
        out.append({"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest.hexdigest()})
    return out


R2_SEED_BASE = 6102026
R2_SEED_RULE = ("keras.utils.set_random_seed(6102026 + feature_index) immediately before building "
                "each autoencoder; applied seed recorded in each donor terminal")
R2_PRETRAIN_TERMINAL_SCHEMA = "i6dmo.r2_pretraining_terminal.v1"
R2_PURGE_RULE = "INNER selection windows whose earliest slot instant precedes the INNER start are purged"
R2_PROVENANCE_KEYS = ("campaign_id", "ruling_seal_sha256", "code_pin", "units", "corpus")
#: Hard budget stops of the registered loop. Reaching max_epochs is an ordinary end (best epoch
#: restored); a max_updates / max_seconds stop is a typed failure and never yields a donor.
R2_BUDGET_STOPS = ("max_updates", "max_seconds")


def r2_seed(feature_index):
    return R2_SEED_BASE + int(feature_index)


def branch_donor_context(bundle, *, population_sha256, scaler_sha256, objective, provenance=None):
    """The identity every donor for ``bundle`` must match (derived from the BUILT network).

    ``provenance`` (campaign_id, ruling_seal_sha256, code_pin, units{feature: unit_id},
    corpus{feature: learned-corpus sha256}) is what binds a donor to THIS campaign's own
    authenticated pretraining terminals; without it nothing is importable (D3).
    """
    c = bundle.config
    family = market_family(c)
    arm = next((a for a, f in MATCHED_ARMS.items() if f == family), None)
    if arm is None or not is_market_time(c):
        raise DonorImportError("R2_DONOR_ARM_MISMATCH", "R2 imports only into a matched market-time arm")
    features = list(c["features"])
    branches = {}
    for index, feature in enumerate(features):
        model = bundle.branch_models[feature]
        branches[feature] = {"feature_index": index, "branch_model_name": model.name,
                             "conv_layer_name": model.name + "_conv",
                             "input_shape": list(model.input_shape[1:]),
                             "output_shape": list(model.output_shape[1:]),
                             "weights": [{"shape": list(w.shape), "dtype": str(np.asarray(w).dtype)}
                                         for w in model.get_weights()]}
    return {"arm": arm, "rule": c["market_time"]["rule"], "slots": int(c["slots"]),
            "spacing_open_market_hours": c["market_time"]["spacing_open_market_hours"],
            "features": features,
            "feature_names_sha256": hashlib.sha256(_canonical(features).encode()).hexdigest(),
            "branch_params": _branch_params(c), "scaler_sha256": scaler_sha256,
            "population_sha256": population_sha256, "objective": objective, "branches": branches,
            "provenance": provenance}


def branch_donor_manifest(bundle, feature, context):
    """Manifest of ONE branch of ``bundle`` (e.g., after pretraining) under ``context``."""
    model = bundle.branch_models[feature]
    entry = context["branches"][feature]
    return {"schema": BRANCH_DONOR_SCHEMA, "arm": context["arm"], "rule": context["rule"],
            "slots": context["slots"], "spacing_open_market_hours": context["spacing_open_market_hours"],
            "feature_name": feature, "feature_index": entry["feature_index"],
            "feature_names_sha256": context["feature_names_sha256"],
            "branch_model_name": model.name, "conv_layer_name": model.name + "_conv",
            "branch_params": dict(context["branch_params"]),
            "input_shape": list(model.input_shape[1:]), "output_shape": list(model.output_shape[1:]),
            "scaler_sha256": context["scaler_sha256"], "population_sha256": context["population_sha256"],
            "objective": json.loads(_canonical(context["objective"])),
            "weights": _array_digests(model.get_weights())}


def check_branch_donor(manifest, weights, expected, feature):
    """Typed IDENTITY check of one donor (never by shape alone). Identity is necessary but NOT
    sufficient: a self-declared manifest is only importable through an authenticated terminal."""
    if not isinstance(manifest, dict) or manifest.get("schema") != BRANCH_DONOR_SCHEMA:
        raise DonorImportError("R2_DONOR_SCHEMA", f"{feature}: donor schema must be {BRANCH_DONOR_SCHEMA}")
    missing = [k for k in BRANCH_DONOR_KEYS if k not in manifest]
    if missing:
        raise DonorImportError("R2_DONOR_IDENTITY_INCOMPLETE",
                               f"{feature}: donor lacks {missing}; a shape match alone is never accepted")
    objective = expected["objective"]
    if objective == R2_OBJECTIVE_UNDECIDED or not isinstance(objective, dict):
        raise DonorImportError("R2_OBJECTIVE_UNDECIDED",
                               "the pretraining objective has not been ruled; no donor is importable")
    entry = expected["branches"].get(feature)
    if entry is None:
        raise DonorImportError("R2_DONOR_SET", f"{feature} is not a branch of the target network")
    checks = (
        ("R2_DONOR_ARM_MISMATCH", ("arm",)),
        ("R2_DONOR_SAMPLING_MISMATCH", ("rule", "slots", "spacing_open_market_hours")),
        ("R2_DONOR_FEATURE_ORDER", ("feature_names_sha256",)),
        ("R2_DONOR_ARCHITECTURE_MISMATCH", ("branch_params",)),
        ("R2_DONOR_SCALER_MISMATCH", ("scaler_sha256",)),
        ("R2_DONOR_POPULATION_MISMATCH", ("population_sha256",)),
    )
    for code, keys in checks:
        for key in keys:
            if manifest[key] != expected[key]:
                raise DonorImportError(code, f"{feature}: donor {key}={manifest[key]!r} != "
                                             f"expected {expected[key]!r}")
    if manifest["feature_name"] != feature or manifest["feature_index"] != entry["feature_index"]:
        raise DonorImportError("R2_DONOR_FEATURE_ORDER",
                               f"donor {manifest['feature_name']!r}@{manifest['feature_index']} offered "
                               f"for {feature!r}@{entry['feature_index']}")
    if (manifest["branch_model_name"], manifest["conv_layer_name"]) != \
            (entry["branch_model_name"], entry["conv_layer_name"]):
        raise DonorImportError("R2_DONOR_LAYER_MISMATCH",
                               f"{feature}: donor layers {manifest['branch_model_name']}/"
                               f"{manifest['conv_layer_name']} != {entry['branch_model_name']}/"
                               f"{entry['conv_layer_name']}")
    if (manifest["input_shape"], manifest["output_shape"]) != (entry["input_shape"], entry["output_shape"]):
        raise DonorImportError("R2_DONOR_ARCHITECTURE_MISMATCH", f"{feature}: donor branch shapes differ")
    if _canonical(manifest["objective"]) != _canonical(objective):
        raise DonorImportError("R2_DONOR_OBJECTIVE_MISMATCH",
                               f"{feature}: donor objective differs from the ruled objective")
    actual = _array_digests(weights)
    if [{k: d[k] for k in ("shape", "dtype")} for d in actual] != entry["weights"]:
        raise DonorImportError("R2_DONOR_ARCHITECTURE_MISMATCH", f"{feature}: weight tensors differ in shape")
    if actual != manifest["weights"]:
        raise DonorImportError("R2_DONOR_WEIGHTS_DIGEST",
                               f"{feature}: supplied weights do not hash to the donor manifest")
    return actual


# --- learned corpus, seeded pretraining unit body, authenticated donor terminals (D2/D3) ---------

def branch_pretraining_corpus(fit_x, fit_row_ids, inner_x, inner_row_ids, inner_slot_instants, *,
                              feature_index, scaler_mean, scaler_std, inner_start_epoch):
    """FIT windows and purged INNER selection windows of ONE feature, z-scored by the shared scaler.

    ``fit_x``/``inner_x`` are the arm tensors' RAW ``x`` (n, slots, n_features). Returns the
    (n, slots, 1) float32 arrays the objective consumes and the learned-corpus digest that binds a
    donor to exactly these windows (population rows, feature index, purge rule, scaler values).
    """
    j = int(feature_index)
    mean, std = float(np.asarray(scaler_mean)[j]), float(np.asarray(scaler_std)[j])
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        raise DonorImportError("R2_CORPUS_SCALER", f"feature {j}: invalid shared scaler entry")
    keep = np.asarray(inner_slot_instants, np.int64)[:, 0] >= int(inner_start_epoch)
    fit = ((np.asarray(fit_x, np.float64)[:, :, j:j + 1] - mean) / std).astype(np.float32)
    sel = ((np.asarray(inner_x, np.float64)[keep][:, :, j:j + 1] - mean) / std).astype(np.float32)
    fit_ids = np.asarray(fit_row_ids, np.int64)
    sel_ids = np.asarray(inner_row_ids, np.int64)[keep]
    if len(fit) < 1 or len(sel) < 1:
        raise DonorImportError("R2_CORPUS_EMPTY", f"feature {j}: FIT {len(fit)} / selection {len(sel)}")
    body = {"feature_index": j, "purge_rule": R2_PURGE_RULE,
            "scaler": [float(mean).hex(), float(std).hex()],
            "fit_row_ids": _array_digests([fit_ids])[0]["sha256"],
            "fit_windows": _array_digests([fit])[0]["sha256"],
            "selection_row_ids": _array_digests([sel_ids])[0]["sha256"],
            "selection_windows": _array_digests([sel])[0]["sha256"],
            "counts": {"fit": int(len(fit)), "selection": int(len(sel)), "purged": int((~keep).sum())}}
    return fit, sel, dict(body, sha256=hashlib.sha256(_canonical(body).encode()).hexdigest())


def _objective_fit_settings(objective):
    fs = objective["fit_settings"]
    return {k: fs[k] for k in ("max_epochs", "patience", "batch_size", "learning_rate", "weight_decay",
                               "min_delta", "max_updates", "max_seconds")}


def fit_branch_mean_monitor(model, fit_windows, selection_windows, settings, *, loss="mse"):
    """Versioned O1 orchestration: evaluate FIT/INNER at identical end-of-epoch weights."""
    import time
    from predictor_plugins.modular_temporal import objectives as O
    from tools.i6dmo.r2_design import MEAN_MONITOR_VERSION
    s = O._settings(settings)
    x, vx = np.asarray(fit_windows), np.asarray(selection_windows)
    if x.ndim != 3 or vx.ndim != 3 or x.shape[1:] != vx.shape[1:] or not len(x) or not len(vx) or \
            not np.isfinite(x).all() or not np.isfinite(vx).all() or loss not in ("mse", "mae"):
        raise DonorImportError("R2_PRETRAINING_INPUT", "finite nonempty matching FIT/INNER windows required")
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=s["learning_rate"],
                  weight_decay=s["weight_decay"]), loss=loss)
    started = time.monotonic()
    updates, stale, best_epoch, selected_updates = 0, 0, 0, 0
    best, weights, history, selected_inner = math.inf, None, [], None
    initial_updates = int(model.optimizer.iterations.numpy())
    rng = np.random.default_rng(s["seed"])

    def budget():
        if updates >= s["max_updates"] or time.monotonic() - started >= s["max_seconds"]:
            raise DonorImportError("R2_PRETRAINING_BUDGET_STOP", "hard update/time budget; no donor selected")

    def evaluate(values):
        sums, absolute, count = [], [], 0
        for start in range(0, len(values), s["batch_size"]):
            budget()
            batch = values[start:start + s["batch_size"]]
            pred = np.asarray(model(batch, training=False), dtype=np.float64)
            if pred.shape != batch.shape or not np.isfinite(pred).all():
                raise DonorImportError("R2_PRETRAINING_NONFINITE", "invalid reconstruction")
            error = pred - batch
            sums.append(float(np.square(error).sum(dtype=np.float64)))
            absolute.append(float(np.abs(error).sum(dtype=np.float64)))
            count += error.size
        return dict(mse_z=math.fsum(sums) / count, mae_z=math.fsum(absolute) / count)

    stop = "max_epochs"
    for epoch in range(1, s["max_epochs"] + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), s["batch_size"]):
            budget()
            batch = x[order[start:start + s["batch_size"]]]
            model.reset_metrics()
            before = int(model.optimizer.iterations.numpy())
            value = model.train_on_batch(batch, batch)
            if not np.isfinite(np.asarray(value)).all():
                raise DonorImportError("R2_PRETRAINING_NONFINITE", "invalid update loss")
            delta = int(model.optimizer.iterations.numpy()) - before
            if delta != 1:
                raise DonorImportError("R2_PRETRAINING_UPDATES", "expected one actual optimizer update")
            updates += delta
        digest = _array_digests(model.get_weights())
        fit, inner = evaluate(x), evaluate(vx)
        if digest != _array_digests(model.get_weights()):
            raise DonorImportError("R2_MONITOR_WEIGHT_CHANGE", "inference monitor changed weights")
        key = "mse_z" if loss == "mse" else "mae_z"
        monitored = .5 * (fit[key] + inner[key])
        if not math.isfinite(monitored):
            raise DonorImportError("R2_PRETRAINING_NONFINITE", "nonfinite end-of-epoch mean")
        history.append(dict(epoch=epoch, train_loss=fit[key], validation_loss=inner[key],
                            monitored_loss=monitored, updates=updates, weights_sha256=_sha_arrays(model.get_weights())))
        if monitored < best - s["min_delta"]:
            best, best_epoch, stale, selected_updates = monitored, epoch, 0, updates
            selected_inner = inner[key]
            weights = [np.array(w, copy=True) for w in model.get_weights()]
        else:
            stale += 1
        budget()
        if stale >= s["patience"]:
            stop = "patience"
            break
    if weights is None:
        raise DonorImportError("R2_PRETRAINING_NO_CHECKPOINT", "no complete monitored epoch")
    model.set_weights(weights)
    if _array_digests(model.get_weights()) != _array_digests(weights):
        raise DonorImportError("R2_RESTORE_MISMATCH", "best weights did not restore")
    if int(model.optimizer.iterations.numpy()) - initial_updates != updates:
        raise DonorImportError("R2_PRETRAINING_UPDATES", "optimizer updates changed outside training")
    reconstruction = dict(state="MEASURED", fit=evaluate(x), inner=evaluate(vx))
    return dict(selected_epoch=best_epoch, selected_updates=selected_updates, observed_updates=updates,
                best_validation_loss=selected_inner, best_monitored_loss=best, stop_reason=stop,
                stop_class="no_improvement" if stop == "patience" else "max_epochs",
                monitor_version=MEAN_MONITOR_VERSION, restored_best_weights=True,
                elapsed_seconds=time.monotonic() - started, history=history, reconstruction=reconstruction)


def _sha_arrays(values):
    return hashlib.sha256(_canonical(_array_digests(values)).encode()).hexdigest()


def pretrain_branch(bundle, feature, fit_windows, selection_windows, objective):
    """ONE pretraining unit body: the ruled registered objective trains this branch IN PLACE.

    The seed is bound immediately before the autoencoder is built (D2):
    ``keras.utils.set_random_seed(6102026 + feature_index)``. A budget stop (max_updates /
    max_seconds) is a typed failure (R2_PRETRAINING_BUDGET_STOP): no donor is ever selected from it.
    """
    from predictor_plugins.modular_temporal import objectives as O
    if objective == R2_OBJECTIVE_UNDECIDED or not isinstance(objective, dict):
        raise DonorImportError("R2_OBJECTIVE_UNDECIDED", "no pretraining before the objective is ruled")
    index = list(bundle.config["features"]).index(feature)
    cls, identity = O.resolve_objective({"plugin": objective["name"], "params": objective["params"]})
    if identity["version"] != objective["version"] or identity["params"] != objective["params"]:
        raise DonorImportError("R2_OBJECTIVE_UNKNOWN", "registered objective differs from the ruling")
    settings = dict(_objective_fit_settings(objective), seed=r2_seed(index))
    branch = bundle.branch_models[feature]
    branch.trainable = True
    seed = r2_seed(index)
    keras.utils.set_random_seed(seed)                       # immediately before the AE is built
    fs = objective["fit_settings"]
    if market_family(bundle.config) == "long_mg12" and "monitor_version" not in fs:
        raise DonorImportError("R2_OBJECTIVE_UNKNOWN", "new long pretraining requires the versioned mean monitor")
    if "monitor_version" in fs:
        from tools.i6dmo.r2_design import MEAN_MONITOR, MEAN_MONITOR_VERSION
        if fs["monitor_version"] != MEAN_MONITOR_VERSION or fs.get("monitor") != MEAN_MONITOR or \
                market_family(bundle.config) != "long_mg12" or objective["name"] != "autoencoder_reconstruction":
            raise DonorImportError("R2_OBJECTIVE_UNKNOWN", "unsupported versioned branch monitor")
        from predictor_plugins.modular_temporal.pretraining import build_autoencoder
        ae = build_autoencoder(branch, channels=identity["params"]["decoder_channels"])
        result = fit_branch_mean_monitor(ae, fit_windows, selection_windows, settings, loss=identity["params"]["loss"])
    else:
        result = cls(identity["params"]).fit(branch, fit_windows, selection_windows, settings)
    if result.get("stop_reason") in R2_BUDGET_STOPS:
        raise DonorImportError("R2_PRETRAINING_BUDGET_STOP",
                               f"{feature}: stopped by {result.get('stop_reason')}; no donor selected")
    fit = {k: result.get(k) for k in ("selected_epoch", "selected_updates", "observed_updates",
                                    "best_validation_loss", "stop_reason", "stop_class", "elapsed_seconds", "history")}
    if "monitor_version" in result:
        fit.update({k: result[k] for k in ("monitor_version", "best_monitored_loss", "restored_best_weights")})
    return {"applied_seed": seed, "objective_identity": identity,
            "weights": [np.array(w) for w in branch.get_weights()],
            "fit": fit,
            "reconstruction": result.get("reconstruction")}


def write_branch_donor_terminal(out_dir, *, bundle, feature, context, unit_id, corpus, pretrained):
    """The pretraining unit's atomic terminal (donor weights + full identity + provenance)."""
    prov = context.get("provenance") or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = context["branches"][feature]["feature_index"]
    weights_path = out / "donor.npz"
    np.savez(weights_path, kernel=pretrained["weights"][0], bias=pretrained["weights"][1])
    manifest = branch_donor_manifest(bundle, feature, context)
    if manifest["weights"] != _array_digests(pretrained["weights"]):
        raise DonorImportError("R2_DONOR_WEIGHTS_DIGEST", f"{feature}: branch is not the pretrained one")
    terminal = {"schema": R2_PRETRAIN_TERMINAL_SCHEMA, "status": "COMPLETE",
                "campaign_id": prov.get("campaign_id"), "unit_id": unit_id,
                "feature": feature, "feature_index": index,
                "applied_seed": pretrained["applied_seed"], "seed_rule": R2_SEED_RULE,
                "code_pin": prov.get("code_pin"), "ruling_seal_sha256": prov.get("ruling_seal_sha256"),
                "objective": context["objective"], "learned_corpus": corpus,
                "fit": json.loads(json.dumps(pretrained["fit"], default=float)),
                "reconstruction": json.loads(json.dumps(pretrained["reconstruction"], default=float)),
                "artifacts": {"donor.npz": _file_hash(weights_path)}, "manifest": manifest}
    path = out / "TERMINAL.json"
    tmp = out / ".TERMINAL.json.tmp"
    tmp.write_text(json.dumps(terminal, sort_keys=True, indent=1, allow_nan=False))
    tmp.replace(path)
    return {"path": str(path), "sha256": _file_hash(path)}


def load_branch_donor(reference, expected, feature):
    """Load ONE donor through its authenticated stored terminal (D3); typed refusal otherwise.

    ``reference`` = {"path", "sha256"} as authenticated by the OPS store (the digest it retained
    when the pretraining unit completed). The trust boundary is that retained reference: unkeyed
    hashes cannot authenticate a fully rewritten store on their own.
    """
    prov = expected.get("provenance")
    if not isinstance(prov, dict) or any(k not in prov for k in R2_PROVENANCE_KEYS):
        raise DonorImportError("R2_DONOR_PROVENANCE_MISSING", "no campaign provenance to bind donors to")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise DonorImportError("R2_DONOR_PROVENANCE_MISSING",
                               f"{feature}: donors are importable only through an authenticated terminal")
    path = Path(reference["path"])
    if not path.is_file() or _file_hash(path) != reference["sha256"]:
        raise DonorImportError("R2_DONOR_TERMINAL_DIGEST", f"{feature}: terminal bytes differ from the reference")
    terminal = json.loads(path.read_text())
    index = expected["branches"][feature]["feature_index"] if feature in expected["branches"] else None
    wanted = {"schema": R2_PRETRAIN_TERMINAL_SCHEMA, "status": "COMPLETE",
              "campaign_id": prov["campaign_id"], "unit_id": prov["units"].get(feature),
              "feature": feature, "feature_index": index, "applied_seed": None if index is None
              else r2_seed(index), "seed_rule": R2_SEED_RULE, "code_pin": prov["code_pin"],
              "ruling_seal_sha256": prov["ruling_seal_sha256"], "objective": expected["objective"]}
    for key, value in wanted.items():
        if _canonical(terminal.get(key)) != _canonical(value):
            raise DonorImportError("R2_DONOR_PROVENANCE_MISMATCH",
                                   f"{feature}: terminal {key}={terminal.get(key)!r} != {value!r}")
    corpus = (terminal.get("learned_corpus") or {}).get("sha256")
    if corpus is None or corpus != prov["corpus"].get(feature):
        raise DonorImportError("R2_DONOR_CORPUS_MISMATCH",
                               f"{feature}: donor learned from other windows than this population's")
    if (terminal.get("fit") or {}).get("stop_reason") in R2_BUDGET_STOPS:
        raise DonorImportError("R2_PRETRAINING_BUDGET_STOP", f"{feature}: a budget stop is not a donor")
    weights_path = path.parent / "donor.npz"
    if _file_hash(weights_path) != (terminal.get("artifacts") or {}).get("donor.npz"):
        raise DonorImportError("R2_DONOR_WEIGHTS_DIGEST", f"{feature}: donor artifact bytes differ")
    with np.load(weights_path, allow_pickle=False) as z:
        weights = [z["kernel"], z["bias"]]
    check_branch_donor(terminal.get("manifest"), weights, expected, feature)
    return weights, reference["sha256"]


def _non_branch_digest(bundle):
    weights = list(bundle.core_model.get_weights())
    for head in bundle.head_models.values():
        weights.extend(head.get_weights())
    return hashlib.sha256(_canonical(_array_digests(weights)).encode()).hexdigest()


def import_branch_donors(bundle, donors, expected):
    """R2 initialization: import ALL 19 branch donors into a fresh R0-seeded ``bundle``.

    ``donors`` maps every feature to ``{"terminal": {"path", "sha256"}}``: the authenticated
    stored terminal of THIS campaign's pretraining unit for that feature. A bare
    ``{"manifest", "weights"}`` entry is identity-checked (so a wrong recipe keeps its specific
    code) and then refused R2_DONOR_PROVENANCE_MISSING: a self-declared manifest never passes.
    Every donor is verified before any weight is set; a failure while setting weights rolls every
    branch back (R2_DONOR_IMPORT_ROLLED_BACK). Core and head weights stay bitwise identical to the
    R0 seed build and every branch stays trainable (R2, never R1-frozen).
    """
    features = expected["features"]
    if list(bundle.config["features"]) != features:
        raise DonorImportError("R2_DONOR_FEATURE_ORDER", "target network feature order differs")
    if set(donors) != set(features):
        raise DonorImportError("R2_DONOR_SET", f"missing {sorted(set(features) - set(donors))}, "
                                               f"unexpected {sorted(set(donors) - set(features))}")
    loaded, terminals = {}, {}
    for feature in features:
        entry = donors[feature]
        if isinstance(entry, dict) and "terminal" not in entry and "manifest" in entry:
            check_branch_donor(entry["manifest"], entry.get("weights") or [], expected, feature)
            raise DonorImportError("R2_DONOR_PROVENANCE_MISSING",
                                   f"{feature}: a self-declared manifest is never importable; donors come "
                                   "only from this campaign's authenticated pretraining terminals")
        weights, digest = load_branch_donor((entry or {}).get("terminal"), expected, feature)
        loaded[feature], terminals[feature] = weights, digest
    before = _non_branch_digest(bundle)
    snapshot = {f: [np.array(w) for w in bundle.branch_models[f].get_weights()] for f in features}
    branches, touched = {}, []
    try:
        for feature in features:
            model = bundle.branch_models[feature]
            touched.append(feature)
            model.set_weights([np.asarray(w) for w in loaded[feature]])
            model.trainable = True
            branches[feature] = {"r0_init": _array_digests(snapshot[feature]),
                                 "imported": _array_digests(model.get_weights()),
                                 "terminal_sha256": terminals[feature]}
    except Exception as exc:
        failed = []
        for feature in touched:
            model = bundle.branch_models[feature]
            try:
                model.set_weights(snapshot[feature])
            except Exception:  # noqa: BLE001 - fall back to per-variable assignment
                for variable, value in zip(model.weights, snapshot[feature]):
                    variable.assign(value)
            if _array_digests(model.get_weights()) != _array_digests(snapshot[feature]):
                failed.append(feature)
        if failed:
            raise DonorImportError("R2_DONOR_IMPORT_ROLLBACK_FAILED",
                                   f"{failed}: discard this network and rebuild from the seed") from None
        raise DonorImportError("R2_DONOR_IMPORT_ROLLED_BACK", f"{type(exc).__name__}: {exc}") from None
    if _non_branch_digest(bundle) != before:
        raise DonorImportError("R2_NON_BRANCH_CHANGED", "core/head weights changed during import")
    if not all(bundle.branch_models[f].trainable for f in features) or not bundle.forecast_model.trainable:
        raise DonorImportError("R2_FROZEN_BRANCH", "R2 branches must stay trainable")
    record = {"schema": "i6dmo.r2_import.v1", "arm": expected["arm"],
              "population_sha256": expected["population_sha256"],
              "scaler_sha256": expected["scaler_sha256"], "objective": expected["objective"],
              "provenance": expected["provenance"], "non_branch_weights_sha256": before,
              "donor_terminals": dict(terminals), "branches": branches, "trainable": True,
              "trainable_parameters": int(sum(int(np.prod(w.shape))
                                              for w in bundle.forecast_model.trainable_weights))}
    record["import_sha256"] = hashlib.sha256(_canonical(
        {k: v for k, v in record.items() if k != "branches"}).encode()).hexdigest()
    return record


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
