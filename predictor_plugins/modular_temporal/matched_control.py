"""Matched Dense-versus-Conv control models for I6-D.

The two arms share observations, targets, horizons, seed, fitting contract and
the exact predictive-head implementation. Their representation paths remain
deliberately different: Dense emits an unordered vector, while Conv preserves
time through fusion and the temporal core. The Conv representation is flattened
only at the predictive-head boundary; the Dense representation is never
reshaped or described as a sequence.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
import tensorflow as tf

from .assembly import build_modular
from .dense_control import DenseControlAdapter, UNORDERED_LATENT_VECTOR, causal_window_dense
from .layers import FeatureSelect

keras = tf.keras

CONFIG_SCHEMA = "predictor.i6d.matched_control.v1"
TEMPORAL_SEQUENCE = "ordered_temporal_sequence"


def canonical_sha256(value):
    """Return a deterministic SHA-256 for finite JSON-compatible data."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer_list(value, name, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a list of positive integers")
    return [_positive_int(item, name) for item in value]


def normalize_matched_config(config):
    """Validate and canonicalize the complete matched-control contract."""
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    value = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    expected = {"schema", "experiment_id", "feature_names", "feature_groups", "window", "sample_hours",
                "horizons", "target_names", "seed", "branch",
                "core", "head", "fit"}
    if set(value) != expected:
        raise ValueError(f"config fields differ: {sorted(set(value) ^ expected)}")
    if value["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"schema must be {CONFIG_SCHEMA}")
    if not isinstance(value["experiment_id"], str) or not value["experiment_id"].strip():
        raise ValueError("experiment_id must be nonempty")
    names = value["feature_names"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("feature_names must be unique nonempty strings")
    groups = value["feature_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("feature_groups must be a nonempty list")
    group_names, routed = [], []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"name", "channels"}:
            raise ValueError("each feature group needs name and channels")
        if not isinstance(group["name"], str) or not group["name"]:
            raise ValueError("feature group name must be nonempty")
        channels = group["channels"]
        if (not isinstance(channels, list) or not channels or len(set(channels)) != len(channels)
                or any(channel not in names for channel in channels)):
            raise ValueError("feature group channels must be declared and unique")
        group_names.append(group["name"])
        routed.extend(channels)
    if len(set(group_names)) != len(group_names):
        raise ValueError("feature group names must be unique")
    if len(routed) != len(set(routed)) or set(routed) != set(names):
        raise ValueError("feature_groups must route every input channel exactly once")
    if value["window"] != 24 or value["sample_hours"] != 1:
        raise ValueError("I6-D requires exact causal 24-hour windows sampled hourly")
    horizons = _integer_list(value["horizons"], "horizons")
    if horizons != sorted(set(horizons)):
        raise ValueError("horizons must be strictly increasing")
    targets = value["target_names"]
    if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
            or any(not isinstance(name, str) or not name for name in targets)):
        raise ValueError("target_names must be unique nonempty strings")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int) or value["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")

    branch = value["branch"]
    if not isinstance(branch, dict) or set(branch) != {"dense", "conv"}:
        raise ValueError("branch must declare dense and conv")
    dense = branch["dense"]
    if not isinstance(dense, dict) or set(dense) != {
            "hidden_units", "latent_units", "activation", "use_bias"}:
        raise ValueError("dense branch fields differ")
    dense["hidden_units"] = _integer_list(dense["hidden_units"], "hidden_units", allow_empty=True)
    dense["latent_units"] = _positive_int(dense["latent_units"], "latent_units")
    if not isinstance(dense["activation"], str) or not dense["activation"]:
        raise ValueError("dense activation must be nonempty")
    if type(dense["use_bias"]) is not bool:
        raise ValueError("dense use_bias must be boolean")
    conv = branch["conv"]
    if not isinstance(conv, dict) or set(conv) != {"channels", "kernel_size"}:
        raise ValueError("conv branch fields differ")
    for key in conv:
        conv[key] = _positive_int(conv[key], key)

    core = value["core"]
    if not isinstance(core, dict) or set(core) != {
            "vector_width", "dense_hidden_units", "conv", "conv_output_steps",
            "conv_output_channels"}:
        raise ValueError("core fields differ")
    core["vector_width"] = _positive_int(core["vector_width"], "vector_width")
    core["dense_hidden_units"] = _integer_list(
        core["dense_hidden_units"], "dense_hidden_units", allow_empty=True)
    core["conv_output_steps"] = _positive_int(core["conv_output_steps"], "conv_output_steps")
    core["conv_output_channels"] = _positive_int(
        core["conv_output_channels"], "conv_output_channels")
    if core["vector_width"] != core["conv_output_steps"] * core["conv_output_channels"]:
        raise ValueError("vector_width must equal flattened Conv temporal-core width")
    if not isinstance(core["conv"], dict):
        raise ValueError("core conv params must be an object")

    head = value["head"]
    if not isinstance(head, dict) or set(head) != {"initializer_seed"}:
        raise ValueError("head must declare only initializer_seed")
    head["initializer_seed"] = _positive_int(head["initializer_seed"], "initializer_seed")

    fit = value["fit"]
    expected_fit = {"max_epochs", "patience", "batch_size", "learning_rate", "weight_decay",
                    "loss", "huber_delta", "min_delta", "max_updates", "max_seconds",
                    "monitor", "monitor_every"}
    if not isinstance(fit, dict) or set(fit) != expected_fit:
        raise ValueError("fit fields differ")
    for key in ("max_epochs", "patience", "batch_size", "max_updates", "monitor_every"):
        fit[key] = _positive_int(fit[key], key)
    for key in ("learning_rate", "huber_delta", "max_seconds"):
        fit[key] = _number(fit[key], key, positive=True)
    for key in ("weight_decay", "min_delta"):
        fit[key] = _number(fit[key], key)
        if fit[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    if fit["loss"] not in ("huber", "mae", "mse") or fit["monitor"] != "validation_loss":
        raise ValueError("fit loss/monitor contract is invalid")
    return value


def _weight_sha256(model):
    digest = hashlib.sha256()
    for weight in model.get_weights():
        value = np.ascontiguousarray(weight)
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _shared_head(width, horizons, targets, seed, name):
    inputs = keras.Input((width,), name=f"{name}_input")
    projected = keras.layers.Dense(
        len(horizons) * len(targets),
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        bias_initializer="zeros",
        name="matched_forecast_projection",
    )(inputs)
    outputs = keras.layers.Reshape(
        (len(horizons), len(targets)), name="matched_forecast_horizons"
    )(projected)
    return keras.Model(inputs, outputs, name=name)


@dataclass(frozen=True)
class MatchedArm:
    """One executable arm and its pre-head representation."""

    model: keras.Model
    representation: keras.Model
    head: keras.Model
    semantics: str
    time_grid: tuple | None
    parameter_groups: dict


@dataclass(frozen=True)
class MatchedControl:
    """Both I6-D arms and their immutable architecture report."""

    config: dict
    dense: MatchedArm
    conv: MatchedArm
    report: dict


def _dense_arm(config):
    names = config["feature_names"]
    support = tuple(range(-23, 1))
    inputs = keras.Input((24, len(names)), name="observations")
    components, latents = [], []
    for index, group in enumerate(config["feature_groups"]):
        columns = [names.index(channel) for channel in group["channels"]]
        component = causal_window_dense(
            input_shape=(24, len(columns)), support_grid=support,
            name=f"dense_branch_{index:03d}",
            params=config["branch"]["dense"],
        )
        components.append(component)
        selected = FeatureSelect(columns, name=f"dense_select_{index:03d}")(inputs)
        latents.append(component.model(selected))
    adapter = DenseControlAdapter().combine(components, name="dense_unordered_fusion")
    fused = adapter.model(latents)
    values = fused
    for index, units in enumerate(config["core"]["dense_hidden_units"]):
        values = keras.layers.Dense(units, activation="gelu",
                                    name=f"dense_core_hidden_{index + 1}")(values)
    values = keras.layers.Dense(config["core"]["vector_width"], activation="gelu",
                                name="dense_core_vector")(values)
    representation = keras.Model(inputs, values, name="dense_non_temporal_representation")
    head = _shared_head(config["core"]["vector_width"], config["horizons"],
                        config["target_names"], config["head"]["initializer_seed"],
                        "matched_predictive_head")
    model = keras.Model(inputs, head(representation(inputs)), name="i6d_dense_control")
    branch_params = sum(component.model.count_params() for component in components)
    core_params = representation.count_params() - branch_params
    return MatchedArm(model, representation, head, UNORDERED_LATENT_VECTOR, None,
                      {"branches": branch_params, "core_and_fusion": core_params,
                       "head": head.count_params(), "total": model.count_params()})


def _conv_arm(config):
    names = config["feature_names"]
    model_config = {
        "window": 24,
        "sample_hours": 1,
        "feature_names": names,
        "branches": [
            {"name": f"conv_branch_{index:03d}", "features": group["channels"],
             "plugin": "causal_conv1d", "params": config["branch"]["conv"],
             "regime": "R0", "donor": None}
            for index, group in enumerate(config["feature_groups"])
        ],
        "branch_steps": 24,
        "fusion": {"plugin": "sequence_concat", "params": {}},
        "core": {"plugin": "transformer_conv", "params": config["core"]["conv"],
                 "regime": "R0", "donor": None},
        "head": {"plugin": "forecast", "params": {}},
        "output_steps": config["core"]["conv_output_steps"],
        "output_channels": config["core"]["conv_output_channels"],
        "horizons": config["horizons"],
        "target_count": len(config["target_names"]),
        "alignment_probe": False,
    }
    bundle = build_modular(model_config)
    inputs = keras.Input((24, len(names)), name="observations")
    sequence = bundle.encoder_model(inputs)
    vector = keras.layers.Flatten(name="predictive_head_boundary_flatten")(sequence)
    representation = keras.Model(inputs, sequence, name="conv_temporal_representation")
    head = _shared_head(config["core"]["vector_width"], config["horizons"],
                        config["target_names"], config["head"]["initializer_seed"],
                        "matched_predictive_head")
    model = keras.Model(inputs, head(vector), name="i6d_conv_control")
    branch_params = sum(model.count_params() for model in bundle.branch_models.values())
    core_params = bundle.core_model.count_params()
    return MatchedArm(model, representation, head, TEMPORAL_SEQUENCE,
                      tuple(bundle.core_time_grid),
                      {"branches": branch_params, "core_and_fusion": core_params,
                       "head": head.count_params(), "total": model.count_params()})


def build_matched_control(config):
    """Build both arms and prove their shared versus intentionally different parts."""
    normalized = normalize_matched_config(config)
    tf.keras.utils.set_random_seed(normalized["seed"])
    dense = _dense_arm(normalized)
    tf.keras.utils.set_random_seed(normalized["seed"])
    conv = _conv_arm(normalized)
    dense_head, conv_head = _weight_sha256(dense.head), _weight_sha256(conv.head)
    if dense_head != conv_head or dense.head.count_params() != conv.head.count_params():
        raise ValueError("matched predictive heads differ at initialization")
    fit_sha = canonical_sha256(normalized["fit"])
    shared = {
        "feature_names": normalized["feature_names"],
        "selected_feature_union": [group["name"] for group in normalized["feature_groups"]],
        "window_hours": 24,
        "horizons": normalized["horizons"],
        "target_names": normalized["target_names"],
        "seed": normalized["seed"],
        "fit_sha256": fit_sha,
        "predictive_head_sha256": dense_head,
    }
    report = {
        "schema": "predictor.i6d.architecture_report.v1",
        "config_sha256": canonical_sha256(normalized),
        "matched_contract": shared,
        "arms": {
            "DENSE": {
                "branch": "causal_window_dense per selected feature",
                "fusion": "concatenation of unordered latent vectors",
                "core": "non-temporal Dense control core",
                "representation_shape": list(dense.representation.output_shape[1:]),
                "representation_semantics": dense.semantics,
                "parameters": dense.parameter_groups,
                "branches": len(normalized["feature_groups"]),
                "head_parameters": dense.head.count_params(),
                "head_sha256": dense_head,
            },
            "CONV": {
                "branch": "causal time-preserving Conv1D per selected feature",
                "fusion": "sequence channel concatenation",
                "core": "positional causal Transformer plus residual Conv1D temporal reduction",
                "representation_shape": list(conv.representation.output_shape[1:]),
                "representation_semantics": conv.semantics,
                "time_grid": list(conv.time_grid),
                "parameters": conv.parameter_groups,
                "branches": len(normalized["feature_groups"]),
                "head_parameters": conv.head.count_params(),
                "head_sha256": conv_head,
            },
        },
        "parameter_equivalence_forced": False,
    }
    return MatchedControl(normalized, dense, conv, report)


def array_sha256(array):
    """Hash array dtype, shape and bytes without ambiguous object encoding."""
    value = np.asarray(array)
    if value.dtype.kind == "O":
        raise ValueError("object arrays are not hashable evidence")
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([contiguous.dtype.str, contiguous.shape]).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def train_only_plumbing(config, windows, targets, row_ids):
    """Exercise both real graphs on TRAIN-shaped arrays without fitting or other splits."""
    harness = build_matched_control(config)
    x, y, rows = np.asarray(windows), np.asarray(targets), np.asarray(row_ids)
    expected_y = (len(x), len(harness.config["horizons"]), len(harness.config["target_names"]))
    if (x.shape != (len(x), 24, len(harness.config["feature_names"]))
            or y.shape != expected_y or rows.shape != (len(x),) or len(set(rows.tolist())) != len(rows)
            or x.dtype.kind not in "fi" or y.dtype.kind not in "fi"
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError("TRAIN plumbing arrays violate matched row/shape/finite contract")
    arms = {}
    for name, arm in (("DENSE", harness.dense), ("CONV", harness.conv)):
        output = np.asarray(arm.model(x, training=False))
        if output.shape != expected_y or not np.all(np.isfinite(output)):
            raise ValueError(f"{name} plumbing output violates target contract")
        arms[name] = {"output_shape": list(output.shape), "output_sha256": array_sha256(output)}
    return {"status": "TRAIN_ONLY_PLUMBING_OK", "rows": len(x),
            "row_identity_sha256": array_sha256(rows), "arms": arms,
            "fit_invocations": 0, "validation_read": False, "test_read": False,
            "architecture": harness.report}
