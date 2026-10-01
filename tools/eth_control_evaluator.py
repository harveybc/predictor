#!/usr/bin/env python3
"""Base-architecture control for the lane F2 financial contrast: flatten + MLP.

The owner's financial contrast compares the differentiated branch+core model with a
non-branching BASE architecture on the same target, rows, scaler, budget and selection
rule. ``evaluate_control(config, train_npz, validation_npz, output_dir)`` reuses the
modular evaluator's own loading contract, settings validation, early-stopping fit loop,
metrics, persistence baseline and digests (``tools.modular_candidate_evaluator``), and only
replaces the model: ``Input(W,F) -> Flatten -> Dense(h1, relu) -> Dense(h2, relu) ->
Dense(H*T) -> Reshape(H,T)``. The receipt has the SAME schema
(``modular.candidate.evaluation.v1``) plus ``architecture`` and ``control`` fields, so the
independent checkpoint scorer verifies it unchanged.

config = {"control": {"kind": "flatten_mlp", "hidden": [h1, h2], "activation": "relu"},
          "model": {window, sample_hours, feature_names, horizons, target_count},
          "evaluator": {...}, "target_feature_indices": [...], "objective": {...}}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools import modular_candidate_evaluator as ev

ARCHITECTURE = "flatten_mlp_control"


def control_parameters(window, features, hidden, horizons, targets):
    """Exact trainable parameter count of the control for a given hidden list."""
    widths = [window * features, *hidden, horizons * targets]
    return sum(a * b + b for a, b in zip(widths, widths[1:]))


def build_control(model_config, control):
    import tensorflow as tf

    if control.get("kind") != "flatten_mlp":
        raise ValueError("control kind must be flatten_mlp")
    hidden = control.get("hidden")
    if (not isinstance(hidden, list) or not hidden or any(type(h) is not int or h < 1 for h in hidden)):
        raise ValueError("control hidden must be a nonempty list of positive integers")
    activation = control.get("activation", "relu")
    w, f = model_config["window"], len(model_config["feature_names"])
    h, t = len(model_config["horizons"]), model_config["target_count"]
    inputs = tf.keras.Input(shape=(w, f), name="window")
    x = tf.keras.layers.Flatten(name="flatten")(inputs)
    for i, units in enumerate(hidden):
        x = tf.keras.layers.Dense(units, activation=activation, name=f"hidden_{i}")(x)
    x = tf.keras.layers.Dense(h * t, name="forecast_flat")(x)
    outputs = tf.keras.layers.Reshape((h, t), name="forecast")(x)
    return tf.keras.Model(inputs, outputs, name=ARCHITECTURE)


def evaluate_control(config, train_path, validation_path, output_dir, progress=None):
    """Mirror of ``evaluate_candidate`` with the control model; same receipt contract."""
    if not isinstance(config, dict) or not isinstance(config.get("control"), dict):
        raise ValueError("config['control'] is required")
    if "branches" in config.get("model", {}):
        raise ValueError("a control config carries no modular branches")
    settings = ev._settings(config)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config = json.loads(canonical)
    model_config = config["model"]
    train = ev._load(train_path, "train", config)
    validation = ev._load(validation_path, "validation", config)
    objective = ev._objective(config, train["metric_space"])
    if train["dataset_id"] != validation["dataset_id"]:
        raise ValueError("dataset identities disagree")
    if np.intersect1d(train["row_ids"], validation["row_ids"]).size:
        raise ValueError("train/validation row identities overlap")
    for key in ("metric_space", "scaler_identity", "scaler_scale"):
        if not np.array_equal(train[key], validation[key]):
            raise ValueError(f"train/validation {key} disagree")
    if train["target_timestamps"].max() >= validation["input_start"].min():
        raise ValueError("train labels must chronologically precede validation input support")
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("output_dir already exists")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(settings["seed"])
    model = build_control(model_config, config["control"])
    batch = settings["batch_size"]
    x, y = train["windows"], train["targets"]
    vx, vy = validation["windows"], validation["targets"]
    ev._predict(model, x[:1], y[:1], batch)
    initial_digest = ev._weight_digest(model.get_weights())
    training = ev.fit_with_early_stopping(model, x, y, vx, vy,
                                          settings if progress is None else {**settings, "progress": progress})
    best_weights = model.get_weights()
    prediction = ev._predict(model, vx, vy, batch)
    baseline = np.repeat(vx[:, -1:, config["target_feature_indices"]], len(model_config["horizons"]), axis=1)
    metrics = ev._metrics(vy, prediction, baseline)
    per_horizon = {str(h): ev._metrics(vy[:, i:i + 1], prediction[:, i:i + 1], baseline[:, i:i + 1])
                   for i, h in enumerate(model_config["horizons"])}
    objective["value"] = metrics[objective["metric"]]
    if objective["value"] is None:
        raise ValueError("objective undefined: persistence denominator is zero")
    destination.mkdir(parents=True, exist_ok=False)
    artifact = destination / "best.keras"
    model.save(artifact)
    restored = tf.keras.models.load_model(artifact, compile=False)
    reloaded_prediction = ev._predict(restored, vx, vy, batch)
    if not np.allclose(prediction, reloaded_prediction, rtol=1e-5, atol=1e-6):
        raise ValueError("saved model reload parity failed")
    weights_digest = ev._weight_digest(best_weights)
    if ev._weight_digest(restored.get_weights()) != weights_digest:
        raise ValueError("saved model weights digest differs after reload")
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    result = dict(
        schema_version="modular.candidate.evaluation.v1", status="completed", architecture=ARCHITECTURE,
        control={**config["control"], "trainable_parameters": trainable, "total_parameters": int(model.count_params())},
        objective=objective, metrics=metrics, per_horizon=per_horizon,
        data=dict(dataset_id=train["dataset_id"], train_rows=len(x), validation_rows=len(vx),
                  horizons=model_config["horizons"], target_names=validation["target_names"].tolist(),
                  metric_space=train["metric_space"], scaler_identity=train["scaler_identity"],
                  scaler_scale=train["scaler_scale"].tolist(), timestamp_unit="seconds",
                  train_target_end=int(train["target_timestamps"].max()),
                  validation_input_start=int(validation["input_start"].min()),
                  validation_origin_start=int(validation["timestamps"].min()), test_used=False),
        training=training,
        digests=dict(config_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                     train_sha256=train["sha256"], validation_sha256=validation["sha256"],
                     initial_weights_sha256=initial_digest, weights_sha256=weights_digest,
                     model_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
        artifacts=dict(best_model=str(artifact.resolve())),
        reload_parity=dict(passed=True, rtol=1e-5, atol=1e-6,
                           max_abs_error=float(np.max(abs(prediction - reloaded_prediction)))),
    )
    (destination / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result, prediction
