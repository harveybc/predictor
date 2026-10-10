"""Offline, validation-only evaluator for DOIN modular forecasting candidates.

``evaluate_candidate(config, train_path, validation_path, output_dir)`` accepts
a JSON-compatible dict. config['model'] passes unchanged to lazy ``build_modular``
and carries window, sample_hours, feature_names, horizons (positive ordered
offsets), target_count and architecture settings. target_feature_indices is a
top-level evaluator setting. Training overrides live in evaluator:
max_epochs (1..100), patience (1..50), batch_size (1..65536), learning_rate,
weight_decay, loss (huber/mae/mse), huber_delta, min_delta, seed, max_updates,
max_seconds. horizons is always a nonempty rank-one integer array/list, even
for a single horizon; scalar horizons and rank-two scalar targets are refused.

Each NPZ must contain exactly: windows [N,W,F], targets [N,H,T], row_ids [N],
timestamps [N] (forecast origins), target_timestamps [N,H], dataset_id (scalar),
split (train/validation scalar), feature_names [F], target_names [T], horizons
[H], timestamp_unit (scalar 'seconds'), metric_space (scalar, e.g. 'z_train'),
scaler_identity (scalar), scaler_scale [T] (positive source-to-target scales).
Times are integer epoch seconds. Input windows are contiguous samples ending at
timestamps; target_timestamps must equal origin + horizon*sample_hours*3600.
Train target support must precede validation INPUT support, not just its origin.
Target values and their input features must share units/transforms. NPZ metric
space/scaler metadata must match across splits. No scaler is fitted here: MAE
and MSE are direct errors in the declared target space (including z_train).
Persistence repeats each row's last observed target feature across horizons.
Skills are 1 - error / persistence_error (null for a zero denominator).

Optional objective (top-level or modular_experiment.objective) follows DOIN's
metric/split/higher_is_better/unit contract, plus returned value. Supported
metrics: MAE, MSE, skill_MAE, skill_MSE. Default: MAE in the declared space.
No test path, plugin registry, live node, broker, or network calls are used.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, name, low, inclusive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    _finite(value, name)
    if value < low or (not inclusive and value == low):
        raise ValueError(f"{name} must be {'>=' if inclusive else '>'} {low}")
    return value


EARLY_STOP_IMPLEMENTATION = "modular.early_stop.v2"
EARLY_STOP_KEYS = ("monitor", "monitor_every", "patience", "min_delta", "max_epochs",
                   "max_updates", "max_seconds")


def early_stop_identity(settings):
    """Experimental identity of the stopping rule; a changed rule is a new variant.

    Binds the implementation version, the monitored quantity, cadence, patience,
    min_delta and every hard limit, plus best-checkpoint restore. Batch size,
    optimizer and loss are candidate parameters, not part of the stopping rule.
    """
    rule = {"implementation": EARLY_STOP_IMPLEMENTATION, "restore": "best_monitored_checkpoint",
            **{key: settings[key] for key in EARLY_STOP_KEYS}}
    canonical = json.dumps(rule, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"rule": rule, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def _settings(config):
    settings = dict(max_epochs=20, patience=5, batch_size=32, learning_rate=1e-3,
                    weight_decay=1e-4, loss="huber", huber_delta=1.0,
                    min_delta=0.0, seed=42, max_updates=100000, max_seconds=3600.0,
                    monitor="validation_loss", monitor_every=1)
    supplied = config.get("evaluator", {})
    if not isinstance(supplied, dict) or set(supplied) - set(settings):
        raise ValueError("unexpected evaluator settings")
    settings.update(supplied)
    for name, maximum in (("max_epochs", 100), ("patience", 50), ("monitor_every", 100),
                          ("batch_size", 65536), ("seed", 2**31 - 1), ("max_updates", 1000000)):
        _integer(settings[name], name, 0 if name == "seed" else 1, maximum)
    for name in ("learning_rate", "huber_delta", "max_seconds"):
        _number(settings[name], name, 0, inclusive=False)
    if settings["max_seconds"] > 86400:
        raise ValueError("max_seconds must be <= 86400")
    for name in ("weight_decay", "min_delta"):
        _number(settings[name], name, 0)
    if settings["loss"] not in ("huber", "mae", "mse"):
        raise ValueError("loss must be huber, mae or mse")
    if settings["monitor"] not in ("validation_loss", "train_validation_mean"):
        raise ValueError("monitor must be validation_loss or train_validation_mean")
    return settings


def _strings(array, name, shape):
    if array.shape != shape or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be strings with shape {shape}")
    values = array.astype(str)
    if any(not value.strip() for value in values.flat):
        raise ValueError(f"{name} cannot be empty")
    return values


def _times(array, name, shape):
    if array.shape != shape or array.dtype.kind != "i":
        raise ValueError(f"{name} must be integer epoch-second timestamps {shape}")
    if np.any(abs(array.astype(np.float64)) > 10**11):
        raise ValueError(f"{name} outside supported epoch-second range")
    return array


def _load(path, split, config):
    model_config = config["model"]
    # Hash precisely the bytes parsed, even if a producer replaces the file later.
    raw = Path(path).read_bytes()
    required = {"windows", "targets", "row_ids", "timestamps", "target_timestamps",
                "dataset_id", "split", "feature_names", "target_names", "horizons",
                "timestamp_unit", "metric_space", "scaler_identity", "scaler_scale"}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"{split} NPZ missing/unexpected fields: {set(archive.files) ^ required}")
        data = {key: archive[key] for key in required}
    x, y = data["windows"], data["targets"]
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or min(*x.shape, *y.shape) < 1:
        raise ValueError("windows/targets must be nonempty [N,W,F]/[N,H,T]")
    for name in ("windows", "targets"):
        if data[name].dtype.kind not in "fi":
            raise ValueError(f"{name} must be real numeric arrays")
        data[name] = _finite(data[name].astype(np.float32), name)
    n, w, f = x.shape
    _, h, t = y.shape
    if model_config.get("window") != w:
        raise ValueError("window disagrees with NPZ")
    if model_config.get("target_count") != t:
        raise ValueError("target_count disagrees with targets")
    features = _strings(data["feature_names"], "feature_names", (f,)).tolist()
    targets = _strings(data["target_names"], "target_names", (t,)).tolist()
    ids = _strings(data["row_ids"], "row_ids", (n,))
    if len(set(ids)) != n or len(set(features)) != f or len(set(targets)) != t:
        raise ValueError("duplicate row/feature/target identity")
    indices = config.get("target_feature_indices")
    if (not isinstance(indices, list) or len(indices) != t or
            any(type(i) is not int or not 0 <= i < f for i in indices) or
            len(set(indices)) != t):
        raise ValueError("target_feature_indices must identify each target uniquely")
    if features != model_config.get("feature_names") or targets != [features[i] for i in indices]:
        raise ValueError("feature/target identities disagree with config")
    horizons = model_config.get("horizons")
    if (not isinstance(horizons, list) or len(horizons) != h or
            any(type(i) is not int or i <= 0 for i in horizons) or
            horizons != sorted(set(horizons)) or data["horizons"].dtype.kind not in "iu" or
            not np.array_equal(data["horizons"], horizons)):
        raise ValueError("horizons must match ordered positive config offsets")
    data["dataset_id"] = _strings(data["dataset_id"], "dataset_id", ()).item()
    for key in ("timestamp_unit", "metric_space", "scaler_identity"):
        data[key] = _strings(data[key], key, ()).item()
    if data["timestamp_unit"] != "seconds":
        raise ValueError("timestamp_unit must be seconds")
    scale = data["scaler_scale"]
    if scale.shape != (t,) or scale.dtype.kind not in "fi" or np.any(scale <= 0):
        raise ValueError("scaler_scale must contain one positive scale per target")
    _finite(scale, "scaler_scale")
    if _strings(data["split"], "split", ()).item() != split:
        raise ValueError(f"expected {split} split")
    data["timestamps"] = _times(data["timestamps"], "timestamps", (n,))
    data["target_timestamps"] = _times(data["target_timestamps"], "target_timestamps", (n, h))
    origins, ends = data["timestamps"], data["target_timestamps"]
    if (np.any(origins[1:] <= origins[:-1]) or np.any(ends <= origins[:, None]) or
            np.any(ends[:, 1:] <= ends[:, :-1])):
        raise ValueError("timestamps must be chronological with strictly future targets")
    seconds = _number(model_config.get("sample_hours"), "sample_hours", 0, inclusive=False) * 3600
    if seconds != int(seconds) or seconds > 86400 * 365:
        raise ValueError("sample_hours must represent whole seconds, at most one year")
    expected = origins[:, None] + np.asarray(horizons, dtype=np.int64) * int(seconds)
    if not np.array_equal(ends, expected):
        raise ValueError("target_timestamps disagree with horizons and sample_hours in seconds")
    data["input_start"] = origins - (w - 1) * int(seconds)
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def _objective(config, metric_space):
    default = dict(metric="MAE", split="validation",
                   higher_is_better=False, unit=metric_space)
    contract = config.get("modular_experiment", {})
    objective = config.get("objective", contract.get("objective", default))
    if "objective" in contract and objective != contract["objective"]:
        raise ValueError("conflicting DOIN objectives")
    directions = {"MAE": False, "MSE": False, "skill_MAE": True, "skill_MSE": True}
    if (not isinstance(objective, dict) or set(objective) != set(default) or
            objective["metric"] not in directions or objective["split"] != "validation" or
            objective["higher_is_better"] is not directions[objective["metric"]] or
            not isinstance(objective["unit"], str) or not objective["unit"].strip()):
        raise ValueError("objective must name a supported validation metric and direction")
    if objective["metric"] not in ("MAE", "MSE") and objective["unit"] != "dimensionless":
        raise ValueError("skills have dimensionless units")
    if metric_space == "z_train" and objective["metric"] in ("MAE", "MSE") and objective["unit"] not in ("z_train", "dimensionless"):
        raise ValueError("z_train errors cannot be labeled as physical units")
    return dict(objective)


def _metrics(y, prediction, baseline):
    for name, array in (("targets", y), ("predictions", prediction), ("baseline", baseline)):
        _finite(array, name)
        if array.shape != y.shape:
            raise ValueError(f"{name} output shape does not match targets")
    error = prediction.astype(np.float64) - y
    naive = baseline.astype(np.float64) - y
    mae, mse = float(np.mean(abs(error))), float(np.mean(error**2))
    bmae, bmse = float(np.mean(abs(naive))), float(np.mean(naive**2))
    result = dict(MAE=mae, MSE=mse, baseline_MAE=bmae, baseline_MSE=bmse,
                  skill_MAE=1 - mae / bmae if bmae else None,
                  skill_MSE=1 - mse / bmse if bmse else None)
    _finite([v for v in result.values() if v is not None], "metrics")
    return result


def _predict(model, x, y, batch_size):
    chunks = []
    for start in range(0, len(x), batch_size):
        prediction = np.asarray(model(_batch(x, start, batch_size, "prediction inputs"), training=False))
        if prediction.shape != y[start:start + batch_size].shape:
            raise ValueError("forecast output shape must be [batch,horizons,targets]")
        chunks.append(_finite(prediction, "predictions"))
    return np.concatenate(chunks)


def _batch(array, start, size, name):
    value = np.asarray(array[start:start + size])
    if value.dtype.kind not in "fi" or value.shape != (min(size, len(array) - start), *array.shape[1:]):
        raise ValueError(f"{name} batch shape/dtype disagrees with array contract")
    return _finite(value, name)


def _weight_digest(weights):
    digest = hashlib.sha256()
    for value in weights:
        value = np.ascontiguousarray(_finite(value, "weights"))
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_with_early_stopping(model, x_train, y_train, x_val, y_val, fit_config):
    """Fit forecast or AE arrays and restore selected weights, without tf.data.

    fit_config accepts the evaluator training settings, plus an optional Keras
    optimizer instance and a loss instance/callable. Default compilation uses
    AdamW and Huber; compile=False instead uses an already-compiled model.
    Returns history, selected_epoch/selected_updates, observed_updates,
    optimizer_iterations, best_validation_loss, stop_reason, elapsed_seconds.
    Optimizer iterations are observed, not inferred; only model weights are
    restored (this helper is for selection, not optimizer-state resumption).
    Time is checked between batches, so an in-flight batch may exceed the limit.
    A budget-interrupted partial epoch is not eligible for checkpoint selection.

    Monitor cadence: validation runs every ``monitor_every`` epochs and always on
    the final permitted epoch; patience counts monitor evaluations, not epochs.
    ``stop_class`` separates a patience stop ("no_improvement") from a hard
    limit ("budget"). The restored weights are re-hashed and must equal the
    selected checkpoint's digest. ``early_stop`` carries the rule identity: a
    different rule is a different experimental variant. An optional
    ``progress`` callable (popped from fit_config) receives a small dict after
    every update and monitor evaluation; it must not raise or block.
    """
    import tensorflow as tf

    raw = dict(fit_config)
    compile_model = raw.pop("compile", True)
    custom_optimizer = raw.pop("optimizer", None)
    progress = raw.pop("progress", None)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    custom_loss = raw.get("loss")
    if custom_loss is not None and not isinstance(custom_loss, str):
        raw["loss"] = "huber"
    else:
        custom_loss = None
    settings = _settings({"evaluator": raw})
    if type(compile_model) is not bool:
        raise ValueError("compile must be boolean")
    if not compile_model and (custom_optimizer is not None or "loss" in fit_config):
        raise ValueError("compile=False cannot override optimizer/loss")
    x, y, vx, vy = x_train, y_train, x_val, y_val
    for name, a in (("x_train", x), ("y_train", y), ("x_val", vx), ("y_val", vy)):
        if len(a.shape) < 2 or min(a.shape) < 1 or np.dtype(a.dtype).kind not in "fi":
            raise ValueError(f"{name} must be nonempty numeric arrays")
    if (len(x) != len(y) or len(vx) != len(vy) or x.shape[1:] != vx.shape[1:] or
            y.shape[1:] != vy.shape[1:]):
        raise ValueError("training/validation array dimensions disagree")
    if compile_model:
        losses = {"huber": lambda: tf.keras.losses.Huber(delta=settings["huber_delta"]),
                  "mae": tf.keras.losses.MeanAbsoluteError,
                  "mse": tf.keras.losses.MeanSquaredError}
        loss = custom_loss if custom_loss is not None else losses[settings["loss"]]()
        optimizer = custom_optimizer if custom_optimizer is not None else tf.keras.optimizers.AdamW(
            learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
        model.compile(optimizer=optimizer, loss=loss)
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise ValueError("compile=False requires a compiled model")
    batch = settings["batch_size"]
    _predict(model, x[:1], y[:1], batch)
    _predict(model, vx[:1], vy[:1], batch)
    _weight_digest(model.get_weights())
    history, best_weights, best_digest = [], None, None
    best_loss, stale, updates, best_epoch, selected_updates = float("inf"), 0, 0, 0, 0
    selected_validation_loss = None
    monitor_every, monitor_count = settings["monitor_every"], 0

    def report(**fields):
        if progress is not None:
            progress(dict(updates=updates, best_epoch=best_epoch,
                          best_validation_loss=selected_validation_loss,
                          best_monitored_loss=None if best_weights is None else best_loss,
                          max_epochs=settings["max_epochs"], max_updates=settings["max_updates"],
                          max_seconds=settings["max_seconds"],
                          elapsed_seconds=time.monotonic() - started, **fields))
    started = time.monotonic()
    initial_iterations = int(optimizer.iterations.numpy())
    stop_reason = "max_epochs"

    def budget_reason():
        if updates >= settings["max_updates"]:
            return "max_updates"
        if time.monotonic() - started >= settings["max_seconds"]:
            return "max_seconds"
        return None

    for epoch in range(1, settings["max_epochs"] + 1):
        train_loss, interrupted = 0.0, False
        for start in range(0, len(x), batch):
            reason = budget_reason()
            if reason:
                stop_reason, interrupted = reason, True
                break
            model.reset_metrics()
            before = int(optimizer.iterations.numpy())
            xb, yb = _batch(x, start, batch, "x_train"), _batch(y, start, batch, "y_train")
            value = model.train_on_batch(xb, yb)
            _finite(value, "training loss")
            delta = int(optimizer.iterations.numpy()) - before
            if delta != 1:
                raise ValueError("expected exactly one observed optimizer update per batch")
            updates += delta
            train_loss += float(np.asarray(value).reshape(-1)[0]) * len(xb)
            report(event="update", epoch=epoch)
        if interrupted:
            break
        epoch_digest = _weight_digest(model.get_weights())
        if epoch % monitor_every and epoch != settings["max_epochs"]:
            history.append(dict(epoch=epoch, train_loss=train_loss / len(x), validation_loss=None,
                                monitored=False, updates=updates, weights_sha256=epoch_digest))
            reason = budget_reason()
            if reason:
                stop_reason = reason
                break
            continue
        val_loss = 0.0
        for start in range(0, len(vx), batch):
            if time.monotonic() - started >= settings["max_seconds"]:
                stop_reason, interrupted = "max_seconds", True
                break
            model.reset_metrics()
            xb, yb = _batch(vx, start, batch, "x_val"), _batch(vy, start, batch, "y_val")
            value = model.test_on_batch(xb, yb)
            _finite(value, "validation loss")
            val_loss += float(np.asarray(value).reshape(-1)[0]) * len(xb)
        if interrupted:
            break
        val_loss /= len(vx)
        monitor_count += 1
        monitored_loss = (val_loss if settings["monitor"] == "validation_loss"
                          else 0.5 * (train_loss / len(x) + val_loss))
        history.append(dict(epoch=epoch, train_loss=train_loss / len(x), validation_loss=val_loss,
                            monitored_loss=monitored_loss,
                            monitored=True, updates=updates, weights_sha256=epoch_digest))
        if monitored_loss < best_loss - settings["min_delta"]:
            best_loss, best_epoch, stale = monitored_loss, epoch, 0
            selected_validation_loss = val_loss
            best_weights = [v.copy() for v in model.get_weights()]
            best_digest, selected_updates = epoch_digest, updates
        else:
            stale += 1
        report(event="monitor", epoch=epoch, validation_loss=val_loss, stale_monitors=stale)
        reason = budget_reason()
        if reason:
            stop_reason = reason
            break
        if stale >= settings["patience"]:
            stop_reason = "patience"
            break
    if best_weights is None or updates == 0:
        raise ValueError("training produced no fully validated checkpoint within budget")
    last_digest = _weight_digest(model.get_weights())
    model.set_weights(best_weights)
    restored_digest = _weight_digest(model.get_weights())
    if restored_digest != best_digest:
        raise ValueError("restored weights differ from the selected checkpoint")
    final_iterations = int(optimizer.iterations.numpy())
    if final_iterations - initial_iterations != updates:
        raise ValueError("optimizer update count changed outside training")
    report(event="restored", epoch=best_epoch)
    return dict(settings=settings, history=history, epochs_completed=len(history),
                monitor_evaluations=monitor_count,
                selected_epoch=best_epoch, best_epoch=best_epoch, selected_updates=selected_updates,
                best_validation_loss=selected_validation_loss,
                best_monitored_loss=best_loss, observed_updates=updates,
                initial_optimizer_iterations=initial_iterations,
                optimizer_iterations=final_iterations, stop_reason=stop_reason,
                stop_class="no_improvement" if stop_reason == "patience" else "budget",
                restored_best_weights=True, restored_weights_sha256=restored_digest,
                last_weights_sha256=last_digest,
                restored_differs_from_last=restored_digest != last_digest,
                early_stop=early_stop_identity(settings),
                elapsed_seconds=time.monotonic() - started)


def evaluate_candidate(config, train_path, validation_path, output_dir, progress=None):
    """Train locally and return a JSON-serializable, measured validation receipt.

    Refuses invalid data/budgets before importing the engine. Output directory
    must not exist, avoiding accidental replacement of another candidate's model.
    The caller controls CPU/GPU visibility before importing TensorFlow.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON-compatible dict")
    if not isinstance(config.get("model"), dict):
        raise ValueError("config['model'] must contain the strict engine configuration")
    settings = _settings(config)
    try:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("config must be finite JSON") from exc
    config = json.loads(canonical)
    model_config = config["model"]
    train = _load(train_path, "train", config)
    validation = _load(validation_path, "validation", config)
    objective = _objective(config, train["metric_space"])
    if train["dataset_id"] != validation["dataset_id"]:
        raise ValueError("dataset identities disagree")
    if train["windows"].shape[1:] != validation["windows"].shape[1:]:
        raise ValueError("train/validation window dimensions disagree")
    if np.intersect1d(train["row_ids"], validation["row_ids"]).size:
        raise ValueError("train/validation row identities overlap")
    for key in ("metric_space", "scaler_identity", "scaler_scale"):
        if not np.array_equal(train[key], validation[key]):
            raise ValueError(f"train/validation {key} disagree")
    if train["target_timestamps"].max() >= validation["input_start"].min():
        raise ValueError("train labels must chronologically precede validation input support")
    identity = config.get("modular_experiment", {}).get("data")
    if identity is not None:
        train_end = np.datetime64(identity["train_end"], "D").astype("datetime64[s]").astype(np.int64)
        validation_start = np.datetime64(identity["validation_start"], "D").astype("datetime64[s]").astype(np.int64)
        day = 86400
        if (identity["dataset_id"] != train["dataset_id"] or train_end >= validation_start or
                train["target_timestamps"].max() >= train_end + day or
                validation["input_start"].min() < validation_start):
            raise ValueError("DOIN data identity/calendar boundaries disagree with NPZ")
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("output_dir already exists")

    import tensorflow as tf
    from predictor_plugins.modular_temporal import build_modular

    tf.keras.utils.set_random_seed(settings["seed"])
    bundle = build_modular(model_config)
    model = bundle.forecast_model
    batch = settings["batch_size"]
    x, y = train["windows"], train["targets"]
    vx, vy = validation["windows"], validation["targets"]
    _predict(model, x[:1], y[:1], batch)
    initial_digest = _weight_digest(model.get_weights())
    training = fit_with_early_stopping(model, x, y, vx, vy,
                                       settings if progress is None else {**settings, "progress": progress})
    best_weights = model.get_weights()
    prediction = _predict(model, vx, vy, batch)
    baseline = np.repeat(vx[:, -1:, config["target_feature_indices"]], len(model_config["horizons"]), axis=1)
    metrics = _metrics(vy, prediction, baseline)
    per_horizon = {str(h): _metrics(vy[:, i:i+1], prediction[:, i:i+1], baseline[:, i:i+1])
                   for i, h in enumerate(model_config["horizons"])}
    objective["value"] = metrics[objective["metric"]]
    if objective["value"] is None:
        raise ValueError("objective undefined: persistence denominator is zero")
    destination.mkdir(parents=True, exist_ok=False)
    artifact = destination / "best.keras"
    model.save(artifact)
    restored = tf.keras.models.load_model(artifact, compile=False)
    reloaded_prediction = _predict(restored, vx, vy, batch)
    if not np.allclose(prediction, reloaded_prediction, rtol=1e-5, atol=1e-6):
        raise ValueError("saved model reload parity failed")
    weights_digest = _weight_digest(best_weights)
    if _weight_digest(restored.get_weights()) != weights_digest:
        raise ValueError("saved model weights digest differs after reload")
    result = dict(
        schema_version="modular.candidate.evaluation.v1", status="completed",
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
    return result
