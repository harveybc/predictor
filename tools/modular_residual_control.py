#!/usr/bin/env python3
"""Base-architecture control on the seasonal-residual target: flatten + MLP (+ engine seasonal naive).

Answers the architecture question on the ECL testbed: does the differentiated per-feature model beat a
non-branching base of the same parameter count trained on the SAME residual target, rows, scaler, seeds,
budget and selection rule? Pattern of lane F2's ``tools/eth_control_evaluator.py`` (M07), adapted to the
ECL NPZ and with the residual: ``Input(W,F) -> Flatten -> Dense(h_i, relu)... -> Dense(H*T) -> Reshape(H,T)``
PLUS the engine's own ``SeasonalNaiveBaseline`` (window position ``W-1-(P-h)``, the target channels), so the
MLP learns y - seasonal_naive exactly as the modular head does. Loading, settings, early stopping, metrics,
persistence baseline and digests are the modular evaluator's own (``tools.modular_candidate_evaluator``);
the receipt keeps schema ``modular.candidate.evaluation.v1`` plus ``architecture``/``control``.

config = {"control": {"kind": "flatten_mlp", "hidden": [h1, h2], "activation": "relu",
                      "target_residual": {"kind": "seasonal_naive", "period": P} | absent},
          "model": {window, sample_hours, feature_names, horizons, target_count},
          "evaluator": {...}, "target_feature_indices": [...], "objective": {...}}
usage: modular_residual_control.py CONFIG TRAIN_NPZ VALIDATION_NPZ OUTPUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import modular_candidate_evaluator as ev  # noqa: E402

ARCHITECTURE = "flatten_mlp_control"


def control_parameters(window, features, hidden, horizons, targets):
    widths = [window * features, *hidden, horizons * targets]
    return sum(a * b + b for a, b in zip(widths, widths[1:]))


def residual_positions(window, horizons, period):
    positions = [window - 1 - (period - h) for h in horizons]
    if any(p < 0 or p > window - 1 for p in positions):
        raise ValueError("seasonal reference outside the window")
    return positions


def build_control(model_config, control, target_indices):
    import tensorflow as tf
    from predictor_plugins.modular_temporal.layers import SeasonalNaiveBaseline

    if control.get("kind") != "flatten_mlp":
        raise ValueError("control kind must be flatten_mlp")
    hidden = control.get("hidden")
    if not isinstance(hidden, list) or not hidden or any(type(h) is not int or h < 1 for h in hidden):
        raise ValueError("control hidden must be a nonempty list of positive integers")
    w, f = model_config["window"], len(model_config["feature_names"])
    horizons, t = model_config["horizons"], model_config["target_count"]
    if len(target_indices) != t:
        raise ValueError("target_feature_indices must match target_count")
    inputs = tf.keras.Input(shape=(w, f), name="window")
    x = tf.keras.layers.Flatten(name="flatten")(inputs)
    for i, units in enumerate(hidden):
        x = tf.keras.layers.Dense(units, activation=control.get("activation", "relu"), name=f"hidden_{i}")(x)
    x = tf.keras.layers.Dense(len(horizons) * t, name="forecast_flat")(x)
    outputs = tf.keras.layers.Reshape((len(horizons), t), name="forecast")(x)
    residual = control.get("target_residual")
    if residual:
        if residual.get("kind") != "seasonal_naive":
            raise ValueError("control target_residual kind must be seasonal_naive")
        positions = residual_positions(w, horizons, int(residual["period"]))
        baseline = SeasonalNaiveBaseline(positions, list(target_indices), name="seasonal_naive_baseline")(inputs)
        outputs = tf.keras.layers.Add(name="forecast_plus_seasonal")([outputs, baseline])
    return tf.keras.Model(inputs, outputs, name=ARCHITECTURE)


def evaluate_control(config, train_path, validation_path, output_dir, progress=None):
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
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("output_dir already exists")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(settings["seed"])
    model = build_control(model_config, config["control"], config["target_feature_indices"])
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
    destination.mkdir(parents=True, exist_ok=False)
    artifact = destination / "best.keras"
    model.save(artifact)
    restored = tf.keras.models.load_model(artifact, compile=False)
    reloaded = ev._predict(restored, vx, vy, batch)
    if not np.allclose(prediction, reloaded, rtol=1e-5, atol=1e-6):
        raise ValueError("saved model reload parity failed")
    weights_digest = ev._weight_digest(best_weights)
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    result = dict(
        schema_version="modular.candidate.evaluation.v1", status="completed", architecture=ARCHITECTURE,
        control={**config["control"], "trainable_parameters": trainable, "total_parameters": int(model.count_params())},
        objective=objective, metrics=metrics, per_horizon=per_horizon,
        data=dict(dataset_id=train["dataset_id"], train_rows=len(x), validation_rows=len(vx),
                  horizons=model_config["horizons"], metric_space=train["metric_space"],
                  scaler_identity=train["scaler_identity"], test_used=False),
        training=training,
        digests=dict(config_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                     train_sha256=train["sha256"], validation_sha256=validation["sha256"],
                     initial_weights_sha256=initial_digest, weights_sha256=weights_digest,
                     model_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
        artifacts=dict(best_model=str(artifact.resolve())),
        reload_parity=dict(passed=True, max_abs_error=float(np.max(abs(prediction - reloaded)))))
    (destination / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.save(destination / "validation_prediction.npy", prediction.astype(np.float32))
    return result, prediction


if __name__ == "__main__":
    cfg, tr, va, out = sys.argv[1:5]
    res, _ = evaluate_control(json.loads(Path(cfg).read_text()), tr, va, out)
    print(json.dumps({"MAE": repr(res["metrics"]["MAE"]), "trainable": res["control"]["trainable_parameters"],
                      "selected_epoch": res["training"].get("selected_epoch"),
                      "stop_reason": res["training"].get("stop_reason"),
                      "weights_sha256": res["digests"]["weights_sha256"]}))
