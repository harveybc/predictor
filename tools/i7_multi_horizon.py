"""Direct multi-output R2 business sweep, with durable weekly terminals.

Reuses I7's authenticated branch donors, modular factories, temporal population
resolver and bounded early stopping. No TEST path or broker is exposed. Each
week fits one 11-output model, not eleven independent scalar models.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HORIZONS = (1, 2, 3, 4, 5, 6, 24, 48, 72, 96, 120)
OPERATING_HORIZONS = HORIZONS[1:]
SCHEMA = "predictor.i7.multi_horizon.v1"


def target_names():
    """Return ordered target column names, retaining 1h as a diagnostic."""
    return tuple(f"Y_s_{h}h" if h <= 6 else f"Y_l_{h}h" for h in HORIZONS)


def exact_targets(timestamps, targets):
    """Mask absent physical future timestamps and stale long-target endpoints."""
    ts = np.asarray(timestamps, dtype="int64")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    missing = set(target_names()) - set(targets)
    missing |= {f"Y_l_{h}h_staleness_h" for h in HORIZONS if h > 6} - set(targets)
    if missing:
        raise ValueError(f"missing target/support columns: {sorted(missing)}")
    y = np.column_stack([targets[name] for name in target_names()]).astype("float64")
    if y.shape != (len(ts), len(HORIZONS)):
        raise ValueError("target shape mismatch")
    for j, h in enumerate(HORIZONS):
        valid = np.isin(ts + h * 3600, ts)
        if h > 6:
            valid &= np.asarray(targets[f"Y_l_{h}h_staleness_h"]) == 0
        y[~valid, j] = np.nan
    return y


def target_scaler(y):
    """Fit robust scales independently per output, using fit rows only."""
    center = np.nanmedian(y, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(y - center), axis=0)
    scale = np.where(scale > np.finfo("float32").eps, scale, np.nanstd(y, axis=0))
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("target scale degenerate")
    return center, scale


def masked_training_targets(y, center, scale):
    """Zero-weight unavailable labels; equalize each horizon's mean loss."""
    valid = np.isfinite(y)
    counts = valid.sum(axis=0)
    if np.any(counts == 0) or np.any(valid.sum(axis=1) == 0):
        raise ValueError("empty training horizon or origin")
    values = np.where(valid, (y - center) / scale, 0).astype("float32")
    weights = (valid * (len(y) / counts)).astype("float32")
    return values[:, :, None], weights


def check_prediction(prediction, n):
    """Reject silent scalar flattening or swapped output dimensions."""
    pred = np.asarray(prediction, dtype="float64")
    if pred.shape != (n, len(HORIZONS), 1):
        raise ValueError(f"prediction shape mismatch: {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("prediction must be finite")
    return pred[:, :, 0]


def error_record(actual, prediction, ids):
    """Sufficient statistics for paired zero-return persistence and model error."""
    from tools.fs4_candidates import digest
    a, p = np.asarray(actual), np.asarray(prediction)
    if a.ndim != 1 or p.shape != a.shape or len(ids) != len(a) or len(a) == 0:
        raise ValueError("nonempty same-row arrays required")
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("metrics require finite values")
    error = p - a
    return dict(n=len(a), rows_sha256=digest(list(ids)),
                absolute_error_sum=float(np.abs(error).sum()),
                squared_error_sum=float(np.square(error).sum()),
                naive_absolute_error_sum=float(np.abs(a).sum()),
                naive_squared_error_sum=float(np.square(a).sum()),
                direction_matches=int((np.sign(p) == np.sign(a)).sum()))


def pool_records(records):
    """Pool by row count, never an unweighted average of weekly errors."""
    for record in records:
        if type(record["n"]) is not int or record["n"] <= 0:
            raise ValueError("invalid scored count")
        for key in ("absolute_error_sum", "squared_error_sum",
                    "naive_absolute_error_sum", "naive_squared_error_sum"):
            if not np.isfinite(record[key]) or record[key] < 0:
                raise ValueError("invalid error sum")
        if type(record["direction_matches"]) is not int or not 0 <= record["direction_matches"] <= record["n"]:
            raise ValueError("invalid direction count")
    n = sum(r["n"] for r in records)
    if n <= 0:
        raise ValueError("empty metric population")
    values = {k: sum(r[k] for r in records) / n for k in (
        "absolute_error_sum", "squared_error_sum", "naive_absolute_error_sum",
        "naive_squared_error_sum", "direction_matches")}
    mae, mse = values["absolute_error_sum"], values["squared_error_sum"]
    nm, ns = values["naive_absolute_error_sum"], values["naive_squared_error_sum"]
    return dict(n=n, mae=mae, mse=mse, naive_mae=nm, naive_mse=ns,
                skill_mae=None if nm == 0 else 1 - mae / nm,
                skill_mse=None if ns == 0 else 1 - mse / ns,
                direction_accuracy=values["direction_matches"],
                beats_naive=mae < nm, unit="raw_log_return", same_row_naive=True)


def _runtime():
    from tools import i7_weekly_branch_regimes as base
    from tools import fs4_weekly_wrapper as weekly
    from tools import fs4_temporal_predictor as predictor
    return base, weekly, predictor


def read_design(root):
    base, _, _ = _runtime()
    d = json.loads((Path(root) / "SWEEP_DESIGN.json").read_text())
    base._verify_seal(d, "design_sha256")
    if (d["schema"] != SCHEMA or d["horizons"] != list(HORIZONS)
            or d["arm"] != "R2_B" or d["test_read"] is not False):
        raise ValueError("sweep design mismatch")
    base.verify_design(d["parent"])
    if d["code_sha256"] != code_digest():
        raise ValueError("sweep code digest mismatch")
    return d


def code_digest():
    """Bind runner, model dependencies and training implementation to terminals."""
    import hashlib
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / "tools/modular_candidate_evaluator.py",
             root / "tools/i7_weekly_branch_regimes.py",
             root / "tools/fs4_temporal_predictor.py",
             root / "tools/fs4_weekly_wrapper.py",
             root / "tools/business_asof_window.py"]
    files += sorted((root / "predictor_plugins/modular_temporal").glob("*.py"))
    value = hashlib.sha256()
    for file in files:
        value.update(str(file.relative_to(root)).encode())
        value.update(file.read_bytes())
    return value.hexdigest()


def initialize(parent_path, root):
    base, _, _ = _runtime()
    parent = base.verify_design(json.loads(Path(parent_path).read_text()))
    document = base._seal(dict(
        schema=SCHEMA, parent=parent, horizons=list(HORIZONS),
        operating_horizons=list(OPERATING_HORIZONS), target_names=list(target_names()),
        arm="R2_B", seed=0, expected_cells=len(parent["weeks"]),
        monitor="train_validation_mean", test_read=False, code_sha256=code_digest(),
        inner_validation_weeks=4,
        feature_policy="fixed selected20 transfer set; not horizon-specific selection",
        label_policy="missing endpoints zero-weighted; loss equalized per horizon",
    ), "design_sha256")
    destination = Path(root) / "SWEEP_DESIGN.json"
    if destination.exists():
        if read_design(root) != document:
            raise ValueError("existing design conflict")
    else:
        base._atomic_json(destination, document)
    return status(root)


def _cell_path(root, week):
    return Path(root) / "cells" / f"week_{week:03d}.json"


def run_cell(root, ordinal, args):
    """Fit one weekly vector model and score each horizon on its valid rows."""
    base, weekly, predictor = _runtime()
    design = read_design(root)
    destination = _cell_path(root, ordinal)
    if destination.exists():
        return validate_cell(design, json.loads(destination.read_text()))
    if ordinal not in range(design["expected_cells"]):
        raise ValueError("unknown week")
    claim = destination.with_suffix(".claim")
    claim.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    started = time.monotonic()
    try:
        parent = design["parent"]
        donors, donor_sha = base._preflight_week_donors(parent, ordinal)
        store = weekly.DataStore.from_paths(
            "EURUSD", args.feature_parquet, args.target_parquet,
            args.validation_feature_parquet, args.validation_target_parquet, bar_hours=1)
        targets = exact_targets(store.ts, store.targets)
        week_dict = parent["weeks"][ordinal]
        task = base.make_task(parent, ordinal, design["seed"])
        protocol = weekly.W.build_protocol(task["validation_year"], task["plan_sha256"])
        week = next(w for w in protocol.weeks()
                    if w.split is weekly.EvaluationSplit.VALIDATION
                    and weekly._iso(w.start) == week_dict["start"])
        purge = dt.timedelta(hours=max(HORIZONS))
        support = weekly.SupportSpec(dt.timedelta(0), purge, dt.timedelta(0),
                                     design["inner_validation_weeks"])
        minimum = int(week.fit_start.timestamp())
        fit_start = minimum + 23 * 3600
        cutoff = int(week.cutoff.timestamp())
        finite = np.isfinite(targets).any(axis=1)
        eligible = np.flatnonzero(finite & (store.ts >= fit_start)
                                  & (store.ts + max(HORIZONS) * 3600 <= cutoff))
        rows = [weekly.AsOfRow(
            record_id=str(store.record_ids[i]),
            event_time=dt.datetime.fromtimestamp(int(store.ts[i]), weekly.UTC),
            available_time=dt.datetime.fromtimestamp(int(store.ts[i]), weekly.UTC),
            target_available_time=dt.datetime.fromtimestamp(
                int(store.ts[i]) + max(HORIZONS) * 3600, weekly.UTC),
            row_digest=base.digest([str(store.record_ids[i]), int(store.ts[i])]))
            for i in eligible]
        resolved = weekly.resolve_asof_window(week, rows, support)
        fi = np.array([store.rid_index[r.record_id] for r in resolved.fit_rows], dtype="int64")
        vi = np.array([store.rid_index[r.record_id] for r in resolved.inner_validation_rows], dtype="int64")
        members = sorted(parent["members"])
        values = store.X[:, [store.col[m] for m in members]]
        scaler = predictor.Standardiser.fit(values[fi])
        spec = base.parent._predictor_spec(parent["parent_design"]["config"])
        xf, kf = predictor._inputs_for(spec, None, values, scaler, fi, store.ts, minimum)
        xv, kv = predictor._inputs_for(spec, None, values, scaler, vi, store.ts, minimum)
        if len(kf) < spec.batch_size or len(kv) == 0:
            raise ValueError("insufficient multi-horizon fit/inner windows")
        center, scale = target_scaler(targets[kf])
        keras = predictor._keras()
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise ValueError("exactly one visible GPU required; CPU fallback prohibited")
        with tf.device("/GPU:0"):
            probe = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        if "GPU:0" not in probe.device or not np.isfinite(probe.numpy()).all():
            raise ValueError("GPU execution probe failed")
        keras.utils.set_random_seed(design["seed"])
        config = base._arm_config(parent, "R2_B", donors)
        config["horizons"] = list(HORIZONS)
        config["target_count"] = 1
        bundle = base.build_modular(config)
        before = base._component_hashes(bundle)
        from tools.modular_candidate_evaluator import fit_with_early_stopping
        heartbeat = Path(root) / "heartbeats" / f"week_{ordinal:03d}.json"
        last_write = [0.]

        def progress(event):
            now = time.monotonic()
            if now - last_write[0] >= 15 or event["event"] != "update":
                base._atomic_json(heartbeat, dict(week=ordinal, utc=weekly._now(), **event))
                last_write[0] = now

        fit = dict(parent["parent_design"]["config"]["fit"],
                   monitor=design["monitor"], seed=design["seed"], progress=progress)
        yf, wf = masked_training_targets(targets[kf], center, scale)
        yv, wv = masked_training_targets(targets[kv], center, scale)
        fit.update(train_sample_weight=wf, validation_sample_weight=wv)
        training = fit_with_early_stopping(
            bundle.forecast_model, xf, yf, xv, yv, fit)
        after = base._component_hashes(bundle)
        changed = sum(before["branches"][name] != after["branches"][name]
                      for name in before["branches"])
        if changed == 0 or training["observed_updates"] <= 0:
            raise ValueError("R2 branch weights did not update")
        score_all = store.range_idx(week_dict["start"], week_dict["end"])
        xs, ks = predictor._inputs_for(spec, None, values, scaler, score_all, store.ts, minimum)
        if len(ks) == 0:
            raise ValueError("no validation input windows")
        chunks = [np.asarray(bundle.forecast_model(xs[i:i+spec.batch_size], training=False))
                  for i in range(0, len(xs), spec.batch_size)]
        prediction = check_prediction(np.concatenate(chunks), len(ks)) * scale + center
        metrics = {}
        for j, h in enumerate(HORIZONS):
            mask = np.isfinite(targets[ks, j])
            ids = store.record_ids[ks[mask]].tolist()
            record = error_record(targets[ks[mask], j], prediction[mask, j], ids)
            metrics[str(h)] = dict(record, **pool_records([record]),
                                  diagnostic_only=h == 1,
                                  excluded_target_rows=int((~mask).sum()))
        artifact = Path(root) / "models" / f"week_{ordinal:03d}.keras"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        bundle.forecast_model.save(artifact)
        body = dict(schema=SCHEMA, status="COMPLETED", design_sha256=design["design_sha256"],
                    week_ordinal=ordinal, week=week_dict, horizons=list(HORIZONS),
                    seed=design["seed"], regime="R2_B", test_read=False,
                    device=probe.device, gpu_details=tf.config.experimental.get_device_details(devices[0]),
                    input_digests=store.digests, row_id_offset=store.row_id_offset,
                    target_center=center.tolist(), target_scale=scale.tolist(),
                    standardiser_sha256=scaler.sha256(),
                    fit_rows=len(kf), inner_rows=len(kv),
                    fit_labels_per_horizon=np.isfinite(targets[kf]).sum(axis=0).tolist(),
                    inner_labels_per_horizon=np.isfinite(targets[kv]).sum(axis=0).tolist(),
                    fit_population_digest=resolved.fit_population_digest,
                    inner_population_digest=resolved.inner_validation_population_digest,
                    training=training, branch_changed_count=changed,
                    branch_weights_before=before["branches"], branch_weights_after=after["branches"],
                    donor_set_sha256=donor_sha, model_path=str(artifact),
                    model_file_sha256=base._file_digest(artifact),
                    model_weights_sha256=predictor.model_weights_sha256(bundle.forecast_model),
                    metrics_by_horizon=metrics, elapsed_seconds=time.monotonic()-started)
        cell = base._seal(body, "cell_sha256")
        validate_cell(design, cell)
        base._atomic_json(destination, cell)
        return cell
    finally:
        claim.unlink(missing_ok=True)


def validate_cell(design, cell):
    base, _, _ = _runtime()
    base._verify_seal(cell, "cell_sha256")
    ordinal = cell["week_ordinal"]
    if type(ordinal) is not int or not 0 <= ordinal < design["expected_cells"]:
        raise ValueError("invalid week identity")
    if (cell["design_sha256"] != design["design_sha256"] or cell["status"] != "COMPLETED"
            or cell["horizons"] != list(HORIZONS) or cell["test_read"] is not False
            or cell["week"] != design["parent"]["weeks"][cell["week_ordinal"]]
            or cell["seed"] != design["seed"] or cell["regime"] != "R2_B"
            or cell["branch_changed_count"] <= 0):
        raise ValueError("cell identity mismatch")
    training = cell["training"]
    changed = sum(cell["branch_weights_before"][name] != cell["branch_weights_after"][name]
                  for name in cell["branch_weights_before"])
    if (training["settings"]["monitor"] != design["monitor"]
            or training["observed_updates"] <= 0 or training["restored_best_weights"] is not True
            or changed != cell["branch_changed_count"]
            or set(cell["branch_weights_before"]) != set(cell["branch_weights_after"])):
        raise ValueError("R2 training evidence mismatch")
    if set(cell["metrics_by_horizon"]) != {str(h) for h in HORIZONS}:
        raise ValueError("horizon population mismatch")
    for record in cell["metrics_by_horizon"].values():
        if not isinstance(record["n"], int) or record["n"] <= 0:
            raise ValueError("invalid scored count")
        expected = pool_records([record])
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("metric reduction mismatch")
    return cell


def status(root, *, close=False):
    """Return progress and measured remaining compute; only full population closes."""
    base, _, _ = _runtime()
    design = read_design(root)
    cells, problems = [], []
    for ordinal in range(design["expected_cells"]):
        path = _cell_path(root, ordinal)
        if path.exists():
            try:
                cells.append(validate_cell(design, json.loads(path.read_text())))
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                problems.append(dict(week=ordinal, reason=str(exc)))
    pending = design["expected_cells"]-len(cells)
    summary = dict(schema=SCHEMA, design_sha256=design["design_sha256"],
                   completed_weeks=len(cells), expected_weeks=design["expected_cells"],
                   pending_weeks=pending, problems=problems, test_read=False,
                   remaining_compute_seconds=None if not cells else
                   pending*sum(c["elapsed_seconds"] for c in cells)/len(cells),
                   state="COMPLETE" if pending == 0 and not problems else "IN_PROGRESS")
    if close and summary["state"] == "COMPLETE":
        summary["annual"] = {
            str(h): dict(pool_records([c["metrics_by_horizon"][str(h)] for c in cells]),
                         diagnostic_only=h == 1,
                         eligible_for_strategy=h != 1 and pool_records([
                             c["metrics_by_horizon"][str(h)] for c in cells])["beats_naive"])
            for h in HORIZONS}
        base._atomic_json(Path(root)/"SWEEP_CLOSURE.json", base._seal(summary, "closure_sha256"))
    base._atomic_json(Path(root)/"SWEEP_STATUS.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run-cell", "worker", "status", "close"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-design")
    parser.add_argument("--week", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--feature-parquet", action="append")
    parser.add_argument("--target-parquet")
    parser.add_argument("--validation-feature-parquet", action="append")
    parser.add_argument("--validation-target-parquet")
    args = parser.parse_args(argv)
    if args.command == "init":
        result = initialize(args.parent_design, args.output)
    elif args.command == "run-cell":
        result = run_cell(args.output, args.week, args)
    elif args.command == "worker":
        design = read_design(args.output)
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard")
        data = sys.argv[1:]
        for week in range(design["expected_cells"]):
            if week % args.shard_count != args.shard_index:
                continue
            if _cell_path(args.output, week).exists():
                validate_cell(design, json.loads(_cell_path(args.output, week).read_text()))
                continue
            subprocess.run([sys.executable, "-m", "tools.i7_multi_horizon", "run-cell",
                            *data[1:], "--week", str(week)], check=True)
            status(args.output)
        result = status(args.output)
    else:
        result = status(args.output, close=args.command == "close")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
