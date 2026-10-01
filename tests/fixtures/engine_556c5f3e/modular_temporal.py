"""Reusable modular temporal Keras engine (no fitting, CLI or device mutation).

Harness API
-----------
``b = build_modular(config)`` returns a ModularBundle with normalized ``config``,
``branch_models`` (local ordered feature inputs), ``fusion_model`` (full input ->
raw concatenated sequence), ``core_model`` (fused input -> bottleneck),
``encoder_model`` (full input -> bottleneck), and ``forecast_model`` (full input
-> forecast). All models share live weights. Inputs are float arrays shaped
(batch, window, len(feature_names)); branch inputs follow branch feature order.

Required config: feature_names, branches, sample_hours (explicit, positive).
Defaults: window=24, branch_steps=12, output_steps=6, output_channels=8.
Forecast shape is independently (batch, len(horizons), target_count), with
top-level horizons=[1] (positive sample offsets) and target_count=1 by default.
The forecast head consumes ALL bottleneck tokens; flattening is confined to it.
Each branch: name, features, plugin='causal_conv1d', params={}, regime='R0',
donor=None. Core: plugin='transformer_conv', params={}, regime='R0', donor=None.
Fusion/head: plugin='sequence_concat'/'forecast', params={}. Names of entry point
groups default to modular.branch/core/fusion/head; override via entry_point_groups
mapping branch/core/fusion/head -> group. Built-ins require no setup.py changes.
Unknown or ambiguous plugins fail. Factories are keyword-callables returning
TemporalComponent(model, time_grid): branch/core take input_shape, time_grid,
output_steps, name, params; core additionally output_channels. Fusion takes
input_shapes, time_grid, name, params; head takes input_shape, time_grid,
horizons, target_count, name, params. Factories must preserve rank-three
sequences and report exact right-edge grids. Branch/core output grids must be
the complete equal-block partition of the input grid, not merely equal shapes.
Fusion is raw channel concatenation: external factories must implement this
contract without trainable transformations. Head grid labels future horizons.
Custom Keras layers must be registered before safe-mode donor loading.

Core params: d_model=64, heads=4, blocks=2, ff_dim=128, dropout=0,
stage_channels=[32,16,output_channels], time_factors=[branch_steps/output_steps,1,1],
kernel_size=3. Three or four strictly decreasing channel stages are supported.
Branch params: channels=16, kernel_size=3. Every compression block consumes all
its samples, assigns its right edge, and learns a projection; compression is
lossy. Input timestamps are interval right edges sample_hours, 2*sample_hours,
...; caller must validate actual input timestamps/gaps and causal feature
availability. Padding in causal convolutions is history padding, never invented
observations to repair an incompatible compression length.

Pretraining (parent owns train/validation splits, optimizers and early stopping):
``ae = build_autoencoder(b.branch_models[name])`` reconstructs local branch input;
``ae = build_autoencoder(b.core_model)`` reconstructs materialized fused sequences.
Both update the supplied encoder in place. ``build_decoder(latent_shape,
output_steps, output_channels, channels=32)`` also works independently. Explicit
``branch_autoencoder(b, branch_name)`` and ``core_autoencoder(b)`` are aliases.
``default_config(feature_names)`` returns a normalized hourly R0 configuration
with one branch per feature, suitable for copying and overriding before building.
After best-weight restore call ``save_donor(model, 'encoder.keras',
b.donor_manifest('branch', name))`` or ``b.donor_manifest('core')``. The latter
computes CURRENT branch weights and fusion identity, so freeze branches before
materialization/core fitting. Load with ``load_donor(path, expected_manifest)``.
Use model.save/load_model for whole-model exports; import this module before load.

R0 forbids donors; R1 loads then recursively freezes; R2 loads then unfreezes.
The manifest excludes regime and donor paths so R1/R2 share identical initial
weights and configuration identity. Core identity binds ordered features, branch
component identities and actual weights, plus fusion identity. Donor archives
have .manifest.json sidecars: exact schema, canonical manifest SHA256 and archive
SHA256. Hashes detect corruption/mismatch, not a malicious author who can rewrite
both files; accept donors only from trusted producers. No fallback is permitted.
The bundle's manifests are fresh deep copies, not model-quality certifications.

Run CPU validation with CUDA_VISIBLE_DEVICES='' and TF_NUM_INTRAOP_THREADS=1,
TF_NUM_INTEROP_THREADS=1, OMP_NUM_THREADS=1 before importing TensorFlow. This module
does not modify the host's devices, global thread pools or running jobs.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import entry_points
import json
import math
from pathlib import Path
import re

import numpy as np
import tensorflow as tf

keras = tf.keras


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _digest(value):
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weights_hash(model):
    """Ordered shape/dtype/byte identity, independent of trainability and names."""
    h = sha256()
    for weight in model.get_weights():
        h.update(_json([list(weight.shape), str(weight.dtype)]).encode("ascii"))
        h.update(np.ascontiguousarray(weight).tobytes())
    return h.hexdigest()


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _partition(grid, steps):
    _positive_int(steps, "output_steps")
    if steps > len(grid) or len(grid) % steps:
        raise ValueError("Temporal compression requires exact divisibility")
    return tuple(grid[len(grid) // steps - 1::len(grid) // steps])


@keras.utils.register_keras_serializable(package="modular_temporal")
class FeatureSelect(keras.layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = tuple(indices)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], len(self.indices))

    def get_config(self):
        return {**super().get_config(), "indices": list(self.indices)}


@keras.utils.register_keras_serializable(package="modular_temporal")
class PositionalEncoding(keras.layers.Layer):
    """Stateless sinusoidal encoding, serializable including odd channel widths."""

    def call(self, inputs):
        length, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        position = tf.cast(tf.range(length)[:, None], tf.float32)
        channel = tf.range(width)[None, :]
        rates = tf.pow(10000.0, -2.0 * tf.cast(channel // 2, tf.float32)
                       / tf.cast(width, tf.float32))
        angle = position * rates
        encoding = tf.where(channel % 2 == 0, tf.sin(angle), tf.cos(angle))
        return inputs + tf.cast(encoding[None, :, :], inputs.dtype)


@dataclass(frozen=True)
class TemporalComponent:
    model: keras.Model
    time_grid: tuple


def _compress(x, steps, channels, name):
    current, width = int(x.shape[1]), int(x.shape[2])
    _partition(tuple(range(current)), steps)
    # Adjacent blocks retain every position, including the final observation.
    x = keras.layers.Reshape((steps, (current // steps) * width), name=name + "_blocks")(x)
    return keras.layers.Dense(channels, name=name + "_projection")(x)


def causal_conv1d(*, input_shape, time_grid, output_steps, name, params):
    channels = _positive_int(params.get("channels", 16), "channels")
    kernel = _positive_int(params.get("kernel_size", 3), "kernel_size")
    _keys(params, {"channels", "kernel_size"}, "branch params")
    inputs = keras.Input(input_shape)
    x = keras.layers.Conv1D(channels, kernel, padding="causal", activation="gelu")(inputs)
    x = _compress(x, output_steps, channels, "compression")
    return TemporalComponent(keras.Model(inputs, x, name=name), _partition(time_grid, output_steps))


def sequence_concat(*, input_shapes, time_grid, name, params):
    _keys(params, set(), "fusion params")
    inputs = [keras.Input(shape) for shape in input_shapes]
    x = keras.layers.Concatenate(axis=-1)(inputs) if len(inputs) > 1 else keras.layers.Activation("linear")(inputs[0])
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


def transformer_conv(*, input_shape, time_grid, output_steps, output_channels, name, params):
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
    factors = params.get("time_factors", [len(time_grid) // output_steps] + [1] * (len(channels) - 1))
    if len(channels) not in (3, 4) or len(factors) != len(channels):
        raise ValueError("Expected three or four matching compression stages")
    for n in [*channels, *factors]:
        _positive_int(n, "compression stage")
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
        x = _compress(x, int(x.shape[1]) // factor, channel, f"stage_{i}")
        x = keras.layers.Conv1D(channel, kernel, padding="causal", activation="gelu",
                                name=f"stage_conv_{i}")(x)
    return TemporalComponent(keras.Model(inputs, x, name=name), grid)


def forecast(*, input_shape, time_grid, horizons, target_count, name, params):
    _keys(params, set(), "head params")
    inputs = keras.Input(input_shape)
    x = keras.layers.Flatten(name="task_tokens")(inputs)
    x = keras.layers.Dense(len(horizons) * target_count, name="forecast_projection")(x)
    x = keras.layers.Reshape((len(horizons), target_count), name="forecast_horizons")(x)
    return TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


DEFAULTS = {"modular.branch": "causal_conv1d", "modular.core": "transformer_conv",
            "modular.fusion": "sequence_concat", "modular.head": "forecast"}
BUILTINS = {"modular.branch": {"causal_conv1d": causal_conv1d},
            "modular.core": {"transformer_conv": transformer_conv},
            "modular.fusion": {"sequence_concat": sequence_concat},
            "modular.head": {"forecast": forecast}}


def _keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Invalid {label}: expected keys in {sorted(allowed)}")


def _resolve(role, spec, groups):
    default_group = "modular." + role
    group = groups[role]
    name = spec["plugin"]
    if group == default_group and name in BUILTINS[default_group]:
        return BUILTINS[default_group][name], {"group": group, "name": name,
                                               "implementation": "modular_temporal.v1:" + name}
    matches = list(entry_points(group=group, name=name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one plugin {group}:{name}; got {len(matches)}")
    ep = matches[0]
    identity = {"group": group, "name": name, "implementation": ep.value,
                "distribution": ep.dist.name if ep.dist else None,
                "version": ep.dist.version if ep.dist else None}
    return ep.load(), identity


def _normalize(config):
    c = _copy(config)
    _keys(c, {"window", "sample_hours", "feature_names", "branches", "branch_steps",
              "core", "fusion", "head", "output_steps", "output_channels", "entry_point_groups",
              "horizons", "target_count"}, "config")
    for key, default in (("window", 24), ("branch_steps", 12), ("output_steps", 6), ("output_channels", 8)):
        c[key] = _positive_int(c.get(key, default), key)
    c["target_count"] = _positive_int(c.get("target_count", 1), "target_count")
    c.setdefault("horizons", [1])
    if not isinstance(c["horizons"], list) or not c["horizons"]:
        raise ValueError("horizons must be a nonempty ordered list")
    for horizon in c["horizons"]:
        _positive_int(horizon, "horizon")
    if c["horizons"] != sorted(set(c["horizons"])):
        raise ValueError("horizons must be strictly increasing")
    period = c.get("sample_hours")
    if (isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period)
            or period <= 0 or period * c["window"] < 24):
        raise ValueError("Explicit sample_hours and at least 24h physical window are required")
    names = c.get("feature_names")
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("feature_names must be unique ordered strings")
    branches = c.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ValueError("At least one branch is required")
    groups = c.setdefault("entry_point_groups", {})
    _keys(groups, {"branch", "core", "fusion", "head"}, "entry_point_groups")
    for role in ("branch", "core", "fusion", "head"):
        groups.setdefault(role, "modular." + role)
        if not isinstance(groups[role], str) or not groups[role]:
            raise ValueError("Entry point group must be a nonempty string")
    seen = set()
    for spec in branches:
        _keys(spec, {"name", "features", "plugin", "params", "regime", "donor"}, "branch")
        name, features = spec.get("name"), spec.get("features")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in seen:
            raise ValueError("Branch names must be unique valid identifiers")
        seen.add(name)
        if (not isinstance(features, list) or not features or any(f not in names for f in features)
                or len(set(features)) != len(features)):
            raise ValueError("Branch features must be unique members of feature_names")
    for role, specs in (("branch", branches), ("core", [c.setdefault("core", {})]),
                        ("fusion", [c.setdefault("fusion", {})]), ("head", [c.setdefault("head", {})])):
        for spec in specs:
            if role != "branch":
                _keys(spec, {"plugin", "params", "regime", "donor"} if role == "core" else {"plugin", "params"}, role)
            spec.setdefault("plugin", DEFAULTS["modular." + role])
            spec.setdefault("params", {})
            if not isinstance(spec["plugin"], str) or not isinstance(spec["params"], dict):
                raise ValueError("Plugin and params must be a string and dict")
            if role in ("branch", "core"):
                spec.setdefault("regime", "R0")
                spec.setdefault("donor", None)
                if spec["regime"] not in ("R0", "R1", "R2"):
                    raise ValueError("Regime must be R0, R1 or R2")
                if (spec["regime"] == "R0" and spec["donor"] is not None
                        or spec["regime"] != "R0" and not spec["donor"]):
                    raise ValueError("R0 forbids donors; R1/R2 require explicit donors")
    _partition(tuple(range(c["window"])), c["branch_steps"])
    _partition(tuple(range(c["branch_steps"])), c["output_steps"])
    return c


def default_config(feature_names):
    """Normalized hourly defaults with one independently addressable branch/feature."""
    return _normalize({"feature_names": list(feature_names), "sample_hours": 1,
                       "branches": [{"name": f"branch_{i}", "features": [feature]}
                                    for i, feature in enumerate(feature_names)]})


def _validate_component(component, input_shapes, grid, output_channels=None):
    if not isinstance(component, TemporalComponent) or not isinstance(component.model, keras.Model):
        raise ValueError("Plugin must return TemporalComponent(Keras model, time_grid)")
    model = component.model
    if len(model.inputs) != len(input_shapes) or any(tuple(t.shape[1:]) != tuple(s) for t, s in zip(model.inputs, input_shapes)):
        raise ValueError("Plugin input shape does not match contract")
    if (len(model.outputs) != 1 or len(model.output_shape) != 3
            or model.output_shape[1] != len(grid) or model.output_shape[2] is None):
        raise ValueError("Plugin must output a fixed rank-three sequence")
    if tuple(component.time_grid) != tuple(grid):
        raise ValueError("Plugin time grid does not match right-edge grid contract")
    if output_channels is not None and model.output_shape[2] != output_channels:
        raise ValueError("Plugin channel contract mismatch")
    return model


def _manifest(role, config, spec, plugin, model, input_grid, output_grid):
    return {"schema": 1, "role": role, "plugin": plugin, "params": _copy(spec["params"]),
            "features": _copy(spec.get("features", config["feature_names"])),
            "feature_names": _copy(config["feature_names"]),
            "name": spec.get("name", role), "sample_hours": config["sample_hours"],
            "input_shape": list(model.input_shape[1:]), "output_shape": list(model.output_shape[1:]),
            "input_grid": list(input_grid), "output_grid": list(output_grid)}


def _apply_regime(model, spec, manifest):
    if spec["regime"] != "R0":
        loaded = load_donor(spec["donor"], manifest)
        try:
            model.set_weights(loaded.get_weights())
        except ValueError as exc:
            raise ValueError("Donor weights incompatible with component manifest") from exc
    model.trainable = spec["regime"] != "R1"


@dataclass
class ModularBundle:
    config: dict
    branch_models: dict
    fusion_model: keras.Model
    core_model: keras.Model
    encoder_model: keras.Model
    forecast_model: keras.Model
    branch_time_grid: tuple
    core_time_grid: tuple
    _branch_manifests: dict = field(repr=False)
    _core_manifest: dict = field(repr=False)
    _fusion_component: keras.Model = field(repr=False)
    _fusion_identity: dict = field(repr=False)

    def donor_manifest(self, role, name=None):
        """Snapshot identity at export time, after restoring selected weights."""
        if role == "branch":
            if name not in self._branch_manifests:
                raise ValueError("Unknown branch name")
            return _copy(self._branch_manifests[name])
        if role != "core" or name is not None:
            raise ValueError("Expected branch/name or core without name")
        manifest = _copy(self._core_manifest)
        manifest["upstream"] = _upstream(self.branch_models, self._branch_manifests,
                                          self._fusion_component, self._fusion_identity)
        return manifest


def _upstream(branches, manifests, fusion, identity):
    return {"branches": [{"manifest": manifests[name], "weights_sha256": weights_hash(model)}
                         for name, model in branches.items()],
            "fusion": {"identity": identity, "weights_sha256": weights_hash(fusion)}}


def build_modular(config: dict) -> ModularBundle:
    c = _normalize(config)
    input_grid = tuple((i + 1) * c["sample_hours"] for i in range(c["window"]))
    branch_grid = _partition(input_grid, c["branch_steps"])
    core_grid = _partition(branch_grid, c["output_steps"])
    inputs = keras.Input((c["window"], len(c["feature_names"])), name="observations")
    branches, manifests, sequences = {}, {}, []
    groups = c["entry_point_groups"]
    for spec in c["branches"]:
        name = spec["name"]
        shape = (c["window"], len(spec["features"]))
        factory, identity = _resolve("branch", spec, groups)
        component = factory(input_shape=shape, time_grid=input_grid, output_steps=c["branch_steps"],
                            name=name, params=_copy(spec["params"]))
        model = _validate_component(component, [shape], branch_grid)
        manifest = _manifest("branch", c, spec, identity, model, input_grid, branch_grid)
        _apply_regime(model, spec, manifest)
        branches[name], manifests[name] = model, manifest
        local = FeatureSelect([c["feature_names"].index(f) for f in spec["features"]],
                              name="select_" + name)(inputs)
        sequences.append(model(local))
    factory, fusion_identity = _resolve("fusion", c["fusion"], groups)
    fusion_identity["params"] = _copy(c["fusion"]["params"])
    shapes = [tuple(x.shape[1:]) for x in sequences]
    component = factory(input_shapes=shapes, time_grid=branch_grid, name="sequence_fusion", params=_copy(c["fusion"]["params"]))
    fusion = _validate_component(component, shapes, branch_grid, sum(s[1] for s in shapes))
    if fusion.weights:
        raise ValueError("Raw sequence fusion cannot contain weights")
    fused = fusion(sequences)
    fusion_model = keras.Model(inputs, fused, name="fusion_model")
    factory, identity = _resolve("core", c["core"], groups)
    component = factory(input_shape=tuple(fused.shape[1:]), time_grid=branch_grid,
                        output_steps=c["output_steps"], output_channels=c["output_channels"],
                        name="temporal_core", params=_copy(c["core"]["params"]))
    core = _validate_component(component, [tuple(fused.shape[1:])], core_grid, c["output_channels"])
    core_manifest = _manifest("core", c, c["core"], identity, core, branch_grid, core_grid)
    core_manifest["upstream"] = _upstream(branches, manifests, fusion, fusion_identity)
    _apply_regime(core, c["core"], core_manifest)
    latent = core(fused)
    encoder = keras.Model(inputs, latent, name="encoder_model")
    factory, _ = _resolve("head", c["head"], groups)
    forecast_grid = tuple(input_grid[-1] + h * c["sample_hours"] for h in c["horizons"])
    component = factory(input_shape=tuple(latent.shape[1:]), time_grid=forecast_grid,
                        horizons=c["horizons"], target_count=c["target_count"],
                        name="forecast_head", params=_copy(c["head"]["params"]))
    head = _validate_component(component, [tuple(latent.shape[1:])], forecast_grid, c["target_count"])
    model = keras.Model(inputs, head(latent), name="forecast_model")
    return ModularBundle(c, branches, fusion_model, core, encoder, model, branch_grid,
                         core_grid, manifests, core_manifest, fusion, fusion_identity)


def build_decoder(latent_shape, output_steps, output_channels, channels=32):
    """Learned reconstruction decoder; not a point-in-time feature producer."""
    steps = _positive_int(output_steps, "output_steps")
    width = _positive_int(output_channels, "output_channels")
    _positive_int(channels, "channels")
    if len(latent_shape) != 2:
        raise ValueError("latent_shape must be (steps, channels)")
    for n in latent_shape:
        _positive_int(n, "latent_shape")
    if steps % latent_shape[0]:
        raise ValueError("Decoder expansion requires exact divisibility")
    inputs = keras.Input(tuple(latent_shape))
    x = keras.layers.UpSampling1D(steps // latent_shape[0])(inputs)
    x = keras.layers.Conv1D(channels, 3, padding="same", activation="gelu")(x)
    x = keras.layers.Conv1D(width, 1, padding="same")(x)
    return keras.Model(inputs, x, name="reconstruction_decoder")


def build_autoencoder(encoder, *, channels=32):
    """Share encoder weights and reconstruct its input; caller compiles/fits."""
    if len(encoder.inputs) != 1 or len(encoder.outputs) != 1 or len(encoder.input_shape) != 3 or len(encoder.output_shape) != 3:
        raise ValueError("Autoencoder requires a single rank-three encoder")
    decoder = build_decoder(tuple(encoder.output_shape[1:]), *encoder.input_shape[1:], channels=channels)
    inputs = keras.Input(tuple(encoder.input_shape[1:]))
    return keras.Model(inputs, decoder(encoder(inputs)), name=encoder.name + "_autoencoder")


def branch_autoencoder(bundle, branch_name, *, channels=32):
    return build_autoencoder(bundle.branch_models[branch_name], channels=channels)


def core_autoencoder(bundle, *, channels=32):
    return build_autoencoder(bundle.core_model, channels=channels)


def _donor_path(path):
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError("Donor path must end in .keras")
    return path, path.with_suffix(".manifest.json")


def save_donor(model, path, manifest):
    """Save selected encoder and strict integrity sidecar; return sidecar dict."""
    path, sidecar = _donor_path(path)
    manifest = _copy(manifest)
    _check_manifest_model(manifest, model)
    model.save(path)
    document = {"schema": 1, "manifest": manifest, "manifest_sha256": _digest(manifest),
                "model_sha256": _file_hash(path), "weights_sha256": weights_hash(model)}
    sidecar.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return document


def _check_manifest_model(manifest, model):
    required = {"schema", "role", "plugin", "params", "features", "feature_names", "name",
                "sample_hours", "input_shape", "output_shape", "input_grid", "output_grid"}
    if manifest.get("role") == "core":
        required.add("upstream")
    if set(manifest) != required or manifest.get("schema") != 1 or manifest.get("role") not in ("branch", "core"):
        raise ValueError("Invalid donor manifest schema")
    if (list(model.input_shape[1:]) != manifest["input_shape"]
            or list(model.output_shape[1:]) != manifest["output_shape"]):
        raise ValueError("Donor manifest/model shapes differ")


def load_donor(path, expected_manifest):
    """Verify bytes and exact expected identity before safe-mode deserialization."""
    path, sidecar = _donor_path(path)
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid donor manifest JSON") from exc
    required = {"schema", "manifest", "manifest_sha256", "model_sha256", "weights_sha256"}
    if not isinstance(document, dict) or set(document) != required or document["schema"] != 1:
        raise ValueError("Invalid donor manifest schema")
    if _digest(document["manifest"]) != document["manifest_sha256"]:
        raise ValueError("Donor manifest hash mismatch")
    if _json(document["manifest"]) != _json(expected_manifest):
        raise ValueError("Donor manifest mismatch (features/config/grid/upstream)")
    if _file_hash(path) != document["model_sha256"]:
        raise ValueError("Donor archive hash mismatch")
    model = keras.models.load_model(path, compile=False, safe_mode=True)
    _check_manifest_model(document["manifest"], model)
    if weights_hash(model) != document["weights_sha256"]:
        raise ValueError("Donor weights hash mismatch")
    return model
