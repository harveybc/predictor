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
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def model_config(features, seed=6102026, family="short"):
    """The fixed R0 model config for an ordered feature list and family (short or long)."""
    return normalize_model_config({"schema": MODEL_SCHEMA, "features": list(features), "regime": "R0",
                                   "seed": seed, **json.loads(_canonical(_FIXED)),
                                   **json.loads(_canonical(FAMILIES[family]))})


def model_family(config):
    for name, grid in FAMILIES.items():
        if config["offsets_hours"] == grid["offsets_hours"] and config["horizons_hours"] == grid["horizons_hours"]:
            return name
    raise ValueError("FIXED_VALUE: offsets/horizons must be exactly one declared family pair "
                     f"{json.dumps(FAMILIES)}")


def normalize_model_config(config):
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
                       "input_shape": [len(c["offsets_hours"]), 1]},
            "fusion": self.fusion_identity,
            "positional_encoding": {"layer": "tools.i6dmo.model:PhysicalOffsetEncoding",
                                    "base": "predictor_plugins.modular_temporal:PositionalEncoding",
                                    "offsets_hours": c["offsets_hours"]},
            "core": {**_identity(dmo_core), "params": _core_params(c),
                     "output_shape": list(self.core_model.output_shape[1:])},
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


def build_dmo(config):
    """Build the fixed R0 model. Seeds the global RNG first (explicit random init)."""
    c = normalize_model_config(config)
    enable_determinism()
    keras.utils.set_random_seed(c["seed"])
    grid = tuple(c["offsets_hours"])
    steps, n = len(grid), len(c["features"])
    inputs = keras.Input((steps, n), name="observations")
    branches, sequences = {}, []
    for i, feature in enumerate(c["features"]):
        name = f"branch_{i:02d}"
        comp = dmo_branch(input_shape=(steps, 1), time_grid=grid, output_steps=steps, name=name,
                          params=_branch_params(c))
        model = _validate_component(comp, [(steps, 1)], grid, c["branch"]["filters"])
        probe_alignment(model, grid, grid, label="branch " + name)
        branches[feature] = model
        sequences.append(model(FeatureSelect([i], name="select_" + name)(inputs)))
    factory, fusion_identity = _resolve("fusion", {"plugin": "sequence_concat"}, {"fusion": "modular.fusion"})
    shapes = [tuple(s.shape[1:]) for s in sequences]
    fusion_comp = factory(input_shapes=shapes, time_grid=grid, name="sequence_fusion", params={})
    fusion = _validate_component(fusion_comp, shapes, grid, sum(s[1] for s in shapes))
    if fusion.weights:
        raise ValueError("fusion must be weightless concatenation")
    fused = fusion(sequences)
    encoded = PhysicalOffsetEncoding(grid, name="positional_encoding")(fused)
    core_comp = dmo_core(input_shape=tuple(fused.shape[1:]), time_grid=grid, name="temporal_core",
                         params=_core_params(c))
    core = _validate_component(core_comp, [tuple(fused.shape[1:])], grid, c["core"]["filters"][-1])
    probe_alignment(core, grid, grid, label="core")
    latent = core(encoded)
    heads, outputs = {}, []
    for h in c["horizons_hours"]:
        name = head_name(h)
        head = dmo_head(input_shape=tuple(latent.shape[1:]), horizon_hours=h, name=name, params=c["heads"])
        heads[name] = head
        outputs.append(head(latent))
    forecast = keras.Model(inputs, outputs, name="i6dmo_forecast")
    fusion_model = keras.Model(inputs, fused, name="fusion_model")
    return DmoBundle(c, forecast, branches, fusion_model, core, heads, fusion_identity)


def predict_matrix(model, x, batch_size=4096):
    """(n, heads) float64 predictions in horizon order, in the training coordinate."""
    x = np.asarray(x, dtype=np.float32)
    parts = []
    for start in range(0, len(x), batch_size):
        out = model(x[start:start + batch_size], training=False)
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
    document = {"schema": ARTIFACT_SCHEMA, "config": bundle.config, "keras_version": keras_version(),
                "determinism": json.loads(_canonical(DETERMINISM)),
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
