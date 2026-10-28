"""Finite synthetic weekly integration: calendar -> fit -> C2 price metrics.

No external data is accepted. This is not a historical experiment, a full-year
evaluation or a cost pilot. The sparse synthetic calendar deliberately tests
integration cheaply. ``run`` executes once; ``status`` verifies retained bytes
without framework import. A failed/interrupted run is not automatically retried.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import sys

from tools.i6m_queue import atomic_json
from tools.i6n_contracts import eligibility as c1
from tools.i6n_contracts import paired_metrics as c2
from tools.i6n_contracts import business_metrics as bm
from tools.i6n_weekly_data import build_weekly_partitions

SCOPE = "SYNTHETIC_WEEKLY_INTEGRATION"
BASE = Path(__file__).resolve().parents[1]


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity():
    paths = [Path(__file__), BASE / "tools/i6n_weekly_data.py", BASE / "tools/i6n_training.py",
             BASE / "tools/i6n_model.py", BASE / "tools/i6e_clock_windows.py",
             BASE / "tools/modular_candidate_evaluator.py", BASE / "tools/i6m_queue.py",
             BASE / "tools/i6e_masked_model.py", BASE / "tools/i6e_compact_run.py"]
    paths.extend((BASE / "tools/i6n_contracts").glob("*.py"))
    paths.extend((BASE / "predictor_plugins/modular_temporal").glob("*.py"))
    packages = {}
    for name in ("numpy", "tensorflow", "keras"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return c2.canonical_sha256({"files": {str(p.relative_to(BASE)): _hash(p) for p in paths},
                                "python": sys.version, "packages": packages})


def status(root):
    """Check bound code and artifacts; never interpret internal hashes as signatures."""
    root = Path(root)
    if not (root / "STATUS.json").exists():
        return {"state": "NOT_RUN", "scope": SCOPE}, 3
    try:
        value = json.loads((root / "STATUS.json").read_text())
        if value["code_sha256"] != _identity():
            return {"state": "IDENTITY_MISMATCH", "scope": SCOPE}, 6
        if value["state"] != "COMPLETED":
            if value["state"] == "RUNNING":
                with (root / "RUN.lock").open("r+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        value = dict(value, state="INTERRUPTED", eta_seconds=None)
            return value, 4
        result = json.loads((root / "RESULT.json").read_text())
        if _hash(root / "RESULT.json") != value["result_sha256"]:
            raise ValueError("RESULT digest mismatch")
        if _hash(root / "model.keras") != result["checkpoint_sha256"]:
            raise ValueError("checkpoint digest mismatch")
        for name in ("metric_input", "calendar_input"):
            if _hash(root / f"{name}.json") != result[f"{name}_sha256"]:
                raise ValueError(f"{name} digest mismatch")
        plan = c1.prepare_support_plan(json.loads((root / "calendar_input.json").read_text()))
        document = json.loads((root / "metric_input.json").read_text())
        recomputed = bm.score_business_metrics(
            plan, document, asset="SYNTHETIC", split_id="SYNTHETIC_DEVELOPMENT",
            sealed_at=document["opportunity_population"]["sealed_at"])
        if recomputed != result["metrics"]:
            raise ValueError("metrics differ from input rescore")
        return result, 0
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"state": "DAMAGED", "scope": SCOPE, "error": str(exc)}, 6


def _fixture():
    def dt(text):
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)

    weeks = [c1.OperationalWeek.derive(
        week_id=name, calendar=c1.CalendarRef("SYNTHETIC", "v1"), market_open=dt(start),
        market_close=dt(start) + timedelta(days=5), entry_cutoff=dt(start) + timedelta(days=4),
        entry_delay_after_open_hours=12, force_flat_before_close_hours=12)
        for name, start in (("train", "2023-11-12T22:00"), ("inner", "2023-12-17T22:00"),
                            ("previous", "2023-12-31T22:00"), ("eval", "2024-01-07T22:00"))]
    origins = tuple(w.market_open + timedelta(hours=i) for w in weeks for i in range(120))
    split = c1.SplitBounds("SYNTHETIC_DEVELOPMENT", dt("2024-01-01"), dt("2025-01-01"), None)
    parts = build_weekly_partitions(
        origins=origins, weeks=weeks, observed_targets={t: t for t in origins},
        input_available_at={t: t for t in origins}, horizons=(2, 72),
        evaluation_week_id="eval", evaluation_split=split,
        source_coverage_start=dt("2010-01-01"), coverage_mode="synthetic_sparse")
    return weeks, origins, split, parts


def _execute(root, device, epochs, progress):
    import numpy as np
    import tensorflow as tf
    from tools.i6e_clock_windows import clock_windows
    from tools.i6n_model import DEFAULT_CONFIG, build_model
    from tools.i6n_training import fit_masked

    tf.keras.utils.set_random_seed(6102026)
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        execution = "/CPU:0"
    else:
        if len(tf.config.list_physical_devices("GPU")) != 1:
            raise ValueError("GPU run requires exactly one visible device")
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
        tf.config.set_soft_device_placement(False)
        execution = "/GPU:0"
    with tf.device(execution):
        product = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    if device == "gpu" and "GPU" not in product.device:
        raise ValueError("GPU operation fell back to CPU")
    weeks, origins, split, parts = _fixture()
    first = origins[0] - timedelta(hours=24)
    last = origins[-1]
    ts = np.arange(int(first.timestamp()), int(last.timestamp()) + 1, 3600, dtype=np.int64)
    phase = np.arange(len(ts), dtype=np.float64)
    price = 1.2 + .01 * np.sin(phase / 13) + .004 * np.cos(phase / 31)
    features = np.column_stack((price, np.sin(phase / 17), np.cos(phase / 29)))
    global_index = {datetime.fromtimestamp(int(t), timezone.utc): i for i, t in enumerate(ts)}
    train_fit = [global_index[t] for t, m in zip(parts.train.origins, parts.train.mask) if any(m)]
    center, scale = features[train_fit].mean(axis=0), features[train_fit].std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    standardized = ((features - center) / scale).astype(np.float32)
    doubled = np.empty((len(ts), 6), np.float32)
    doubled[:, 0::2] = standardized
    doubled[:, 1::2] = 1.

    def arrays(part):
        oi = np.array([global_index[t] for t in part.origins], dtype=np.int64)
        x, market = clock_windows(ts, doubled, oi, spacing_hours=1, samples=25)
        assert np.array_equal(x[:, -1], doubled[oi])
        x = np.concatenate((x, market[..., None]), axis=-1).astype(np.float32)
        mask = np.asarray(part.mask, bool)
        labels = np.full(mask.shape, np.nan, np.float32)
        for i, (targets, row) in enumerate(zip(part.target_instants, mask)):
            for j, valid in enumerate(row):
                if valid:
                    labels[i, j] = np.log(price[global_index[targets[j]]] / price[oi[i]])
        return x, labels, mask

    train, inner, evaluation = (arrays(p) for p in (parts.train, parts.inner, parts.evaluation))
    target_scale = np.array([np.std(train[1][train[2][:, j], j]) for j in range(2)], np.float32)
    if not np.isfinite(target_scale).all() or np.any(target_scale <= 0):
        raise ValueError("synthetic TRAIN target scale invalid")
    settings = dict(max_epochs=epochs, patience=2, min_delta=1e-6, batch_size=32,
                    learning_rate=.001, weight_decay=1e-4, huber_delta=1.,
                    seed=6102026, head_weights=[1., 1.])
    with tf.device(execution):
        model = build_model(25, 3, 1, {**DEFAULT_CONFIG, "n_horizons": 2})
        audit = fit_masked(model, train[0], train[1] / target_scale, train[2],
                           inner[0], inner[1] / target_scale, inner[2], settings, progress)
        output_tensor = model(evaluation[0], training=False)
        if device == "gpu":
            variable_devices = [v.value.handle.device for v in model.trainable_variables]
            if "GPU" not in output_tensor.device or any("GPU" not in d for d in variable_devices):
                raise ValueError("training model variables/output are not on GPU")
        prediction = np.asarray(output_tensor) * target_scale
    model.save(root / "model.keras")
    restored = tf.keras.models.load_model(root / "model.keras", compile=False)
    replay = np.asarray(restored(evaluation[0], training=False)) * target_scale
    np.testing.assert_allclose(prediction, replay, rtol=1e-6, atol=1e-7)
    raw_plan = dict(
        schema_version=c1.CONFIG_SCHEMA_VERSION,
        market_calendar={"calendar_id": "SYNTHETIC", "version": "v1",
                         "weeks": [weeks[-1].to_json()]},
        entry_delay_after_open_hours=12, force_flat_before_close_hours=12,
        roles=["MANAGEMENT"], horizons_hours=[2, 72],
        origin_grid={"kind": "EXPLICIT", "timestamps": [c1.iso(t) for t in parts.evaluation.origins]},
        input_unavailable_origins=[], splits=[{k: v for k, v in split.to_json().items() if k != "membership"}],
        label_evidence={"observed_targets": [{"timestamp": c1.iso(t), "available_at": c1.iso(t)}
                                            for t in origins]},
    )
    plan = c1.prepare_support_plan(raw_plan)
    sealed = datetime.now(timezone.utc).isoformat()
    population = bm.build_opportunity_population(plan, asset="SYNTHETIC", split_id=split.split_id, sealed_at=sealed)
    expected_support = {(t, h) for t, row in zip(parts.evaluation.origins, parts.evaluation.mask)
                        for h, valid in zip(parts.horizons, row) if valid}
    scored_support = {(c2.parse_utc("origin", key["origin"]), key["horizon_hours"])
                      for key in population["keys"]}
    if expected_support != scored_support:
        raise ValueError("weekly masks differ from C1/C2 scoring support")
    by_origin = {t: i for i, t in enumerate(parts.evaluation.origins)}
    forecasts = []
    for key in population["keys"]:
        i = by_origin[c2.parse_utc("origin", key["origin"])]
        j = parts.horizons.index(key["horizon_hours"])
        origin, target = parts.evaluation.origins[i], parts.evaluation.target_instants[i][j]
        forecasts.append(dict(key, target_timestamp=c1.iso(target),
                              price_origin=float(price[global_index[origin]]),
                              price_target=float(price[global_index[target]]),
                              prediction=float(price[global_index[origin]] * np.exp(prediction[i, j])),
                              checkpoint=audit["restored_weights_sha256"], unit="SYNTHETIC_PRICE",
                              scale="ORIGINAL_PRICE"))
    document = dict(
        schema_version=c2.INPUT_SCHEMA_VERSION,
        contract=dict(roles=["management_origin"], horizons_hours=[2, 72], calendar_id=bm.calendar_identity(plan),
                      protocol_id="SYNTHETIC_WEEKLY", config_id=_identity(), unit="SYNTHETIC_PRICE", scale="ORIGINAL_PRICE"),
        opportunity_population=population, forecasts=forecasts, forecast_failures=[], usage_receipts=[])
    metrics = bm.score_business_metrics(plan, document, asset="SYNTHETIC", split_id=split.split_id, sealed_at=sealed)
    atomic_json(root / "metric_input.json", document)
    atomic_json(root / "calendar_input.json", raw_plan)
    return dict(training=audit, metrics=metrics, parameters=model.count_params(), device=device,
                probe_device=product.device.split("/device:")[-1],
                training_placement_verified=(device == "gpu"),
                support_sha256=parts.support_sha256, coverage_complete=parts.coverage_complete,
                current_origin_in_window=True, checkpoint_sha256=_hash(root / "model.keras"),
                metric_input_sha256=_hash(root / "metric_input.json"),
                calendar_input_sha256=_hash(root / "calendar_input.json"),
                feature_scaler={"center": center.tolist(), "scale": scale.tolist(), "fit_rows": len(train_fit)},
                target_scale=target_scale.tolist())


def run(root, device="cpu", epochs=3):
    """Run once under caller resource admission; retain progress and paired metrics."""
    if device not in ("cpu", "gpu") or type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("explicit cpu/gpu device and 1..3 synthetic epochs required")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior, code = status(root)
        if code == 0:
            if prior["device"] != device or prior["training"]["settings"]["max_epochs"] != epochs:
                raise ValueError("existing run settings differ")
            return prior
        if code != 3:
            raise ValueError(f"retained run must not be retried: {prior['state']}")
        started = time.monotonic()
        state = dict(state="RUNNING", scope=SCOPE, code_sha256=_identity(), scientific_measurement=False,
                     started_at=datetime.now(timezone.utc).isoformat(), eta_seconds=None)
        atomic_json(root / "STATUS.json", state)

        def progress(event):
            state.update(progress=event, elapsed_seconds=time.monotonic() - started)
            if event["event"] == "epoch":
                state["eta_seconds"] = state["elapsed_seconds"] / event["epoch"] * (epochs - event["epoch"])
            atomic_json(root / "STATUS.json", state)

        try:
            result = _execute(root, device, epochs, progress)
            if _identity() != state["code_sha256"]:
                raise ValueError("code changed during run")
            result.update(state="COMPLETED", scope=SCOPE, scientific_measurement=False,
                          elapsed_seconds=time.monotonic() - started, code_sha256=state["code_sha256"], eta_seconds=0)
            atomic_json(root / "RESULT.json", result)
            state.update(state="COMPLETED", result_sha256=_hash(root / "RESULT.json"), eta_seconds=0)
            atomic_json(root / "STATUS.json", state)
            return result
        except Exception as exc:
            state.update(state="FAILED", error=f"{type(exc).__name__}: {exc}", eta_seconds=None)
            atomic_json(root / "STATUS.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.command == "status":
        value, code = status(args.output)
    else:
        value, code = run(args.output, args.device, args.epochs), 0
    print(json.dumps(value, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
