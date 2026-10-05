"""One retained EURUSD PS5 joint reentry comparison, TRAIN folds only.

The four arms refit the same pinned modular temporal architecture. An absent
feature is a zero channel after TRAIN-only scaling, not a changed architecture.
This is development evidence for one pair, never a SelectionRelease.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Never take a GPU from running lanes.

import numpy as np

SOURCE_COMMIT = "26c8ca0a25bedce5f01b5d476750f7c556e17c62"
HOUR = 3600
PURGE_H = 144
DEFAULT_SETTINGS = {"max_epochs": 30, "patience": 5, "batch_size": 64,
                    "learning_rate": 1e-3, "weight_decay": 1e-4, "loss": "mae",
                    "min_delta": 0.0, "max_updates": 5000, "max_seconds": 60.0}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _checked(path, expected):
    path = Path(path)
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f"missing or digest-mismatched retained artifact: {path}")
    return path


def _rows(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_retained(ps1_dir, ps2_dir, *, base, pair, target, horizon, fold_id):
    """Bind one exact PS1/PS2 batch, feature IDs, fold and target; no test path."""
    ps1_dir, ps2_dir = Path(ps1_dir), Path(ps2_dir)
    base, pair = list(base), list(pair)
    if (len(pair) != 2 or pair[0] == pair[1] or not base or
            len(set(base + pair)) != len(base + pair)):
        raise ValueError("one distinct pair and nonempty disjoint base required")
    if target not in ("Y_s", "Y_l") or type(horizon) is not int or horizon <= 0:
        raise ValueError("only a declared regression target/horizon is supported")
    ps1_digest = _json(ps1_dir / "digests.json")["artifacts_sha256"]["admissible_features.json"]
    ps1_path = _checked(ps1_dir / "admissible_features.json", ps1_digest)
    admitted = {f["feature_id"] for f in _json(ps1_path)["features"]
                if f["admissibility"] == "ADMISSIBLE"}
    manifest_path = ps2_dir / "batch_manifest.json"
    manifest = _json(manifest_path)
    if (manifest.get("schema") != "ps2_batch.v1" or manifest.get("asset") != "EURUSD" or
            manifest.get("sampling_period_seconds") != HOUR):
        raise ValueError("not the retained EURUSD hourly PS2 contract")
    names = base + pair
    ps2_digest_path = ps2_dir / "ps2_manifest.json"
    ps2_manifest = _json(ps2_digest_path)
    ready = _json(ps1_dir / "READY")
    provenance = ps2_manifest.get("provenance", {})
    producer = manifest.get("producer", {})
    if (ps2_manifest.get("batch_id") != manifest.get("batch_id") or
            ready.get("batch") != manifest.get("batch_id") or
            provenance.get("lane_a_batch") != ready["batch"] or
            producer.get("lane_a_batch") != ready["batch"] or
            provenance.get("lane_a_ready_digests_sha256") != ready["digests_sha256"] or
            producer.get("lane_a_ready_digests_sha256") != ready["digests_sha256"] or
            provenance.get("lane_a_artifacts_sha256", {}).get("admissible_features.json") != ps1_digest):
        raise ValueError("PS1/PS2 provenance mismatch")
    if not set(names) <= admitted or not set(names) <= set(manifest["features"]):
        raise ValueError("feature IDs are not exactly PS1-admissible and PS2-retained")
    declared = ps2_manifest["output_sha256"]
    evidence = {}
    for name in ("ps2_status.csv", "ps2_synergy.csv"):
        path = ps2_dir / name
        if name in declared:
            _checked(path, declared[name])
        evidence[name] = _sha(path)
    statuses = {(r["feature"], r["target"], int(r["horizon"])): r["status"]
                for r in _rows(ps2_dir / "ps2_status.csv")}
    allowed = {"PROVISIONAL_SURVIVOR", "EXPLORATION", "PROVISIONAL_LOW_PRIORITY"}
    if any(statuses.get((f, target, horizon)) not in allowed for f in names):
        raise ValueError("feature IDs lack a reversible PS2 status for this target")
    synergies = _rows(ps2_dir / "ps2_synergy.csv")
    if not any({r["a"], r["b"]} == set(pair) and r["target"] == target and
               int(r["horizon"]) == horizon and r["fold"] == fold_id for r in synergies):
        raise ValueError("pair is not an exact PS2 synergy row for fold and target")
    folds = [f for f in manifest["folds"] if f["fold_id"] == fold_id and f["split"] == "train"]
    if len(folds) != 1:
        raise ValueError("one TRAIN-only inner fold required")
    column = f"{target}_{horizon}h"
    if column not in manifest["target_columns"].get(target, []):
        raise ValueError("target/horizon absent from retained column contract")
    j = manifest["target_columns"][target].index(column)
    arrays = {}
    for part in ("series", "targets"):
        desc = manifest[part]
        if desc["file"] != f"{part}.npz":
            raise ValueError("retained TRAIN payload filename differs from contract")
        path = _checked(ps2_dir / desc["file"], desc["sha256"])
        with np.load(path, allow_pickle=False) as z:
            arrays[part] = {k: z[k] for k in z.files}
    series, targets = arrays["series"], arrays["targets"]
    ts = np.asarray(series["timestamps"], dtype="int64")
    if (len(ts) < 2 or np.any(np.diff(ts) != HOUR) or
            not np.array_equal(ts, targets["timestamps"]) or
            ts[-1] != int(manifest["train_end_ts"]) or
            ts[-1] >= 1704067200):  # external validation begins 2024-01-01 UTC
        raise ValueError("payload crosses TRAIN boundary or has an invalid hourly grid")
    if (int(ps2_manifest["train_last_time"]) != int(ts[-1]) or
            "Y_s" not in targets or
            int(np.isfinite(targets["Y_s"][:, 0]).sum()) != int(ps2_manifest["train_rows"])):
        raise ValueError("PS2 TRAIN row count or last timestamp differs from retained payload")
    if any(k not in series for k in ("x__" + f for f in names)):
        raise ValueError("exact retained feature IDs lack series arrays")
    cols = {f: np.asarray(series["x__" + f], dtype="float32") for f in names}
    if any(v.shape != ts.shape for v in cols.values()):
        raise ValueError("feature rows differ from retained hourly grid")
    yy = np.asarray(targets[target])
    if yy.ndim != 2 or yy.shape != (len(ts), len(manifest["target_columns"][target])):
        raise ValueError("target rows/columns differ from retained contract")
    fold = folds[0]
    fs, fe = fold["fit"]
    vs, ve = fold["val"]
    if not ts[0] <= fs <= fe < vs <= ve <= ts[-1]:
        raise ValueError("inner fold is not ordered inside TRAIN")
    return {"timestamps": ts, "columns": cols, "y": yy[:, j].astype("float32"),
            "features": names, "base": base, "pair": pair, "target": target,
            "horizon": horizon, "fold": fold, "train_end_ts": int(ts[-1]),
            "identity": {"ps1_features_sha256": ps1_digest,
                         "ps1_ready_digests_sha256": ready["digests_sha256"],
                         "ps2_manifest_sha256": _sha(ps2_digest_path),
                         "ps2_batch_manifest_sha256": _sha(manifest_path),
                         "ps2_status_sha256": evidence["ps2_status.csv"],
                         "ps2_synergy_sha256": evidence["ps2_synergy.csv"],
                         "series_sha256": manifest["series"]["sha256"],
                         "targets_sha256": manifest["targets"]["sha256"]},
            "all_admissible_batch": list(ps2_manifest["control_all_admissible"]),
            "retained_series_batch": list(manifest["features"])}


def _usable_origins(ts, cols, y, names, lo, hi, window, support_h):
    idx = np.flatnonzero((ts >= lo) & (ts <= hi) & (ts + support_h * HOUR <= hi) & np.isfinite(y))
    idx = idx[idx >= window - 1]
    bad = np.zeros(len(ts), dtype="int64")
    for name in names:
        bad |= ~np.isfinite(cols[name])
    cumulative = np.r_[0, np.cumsum(bad)]
    return idx[(cumulative[idx + 1] - cumulative[idx + 1 - window]) == 0]


def _windowed(z, origins, window):
    offsets = np.arange(window, dtype="int64") - window + 1
    return z[origins[:, None] + offsets[None, :]]


def population_control(ps2_dir, active_features):
    """Account for all retained PS2 admissibles; this one-pair run evaluates none as a full control."""
    ps2_dir = Path(ps2_dir)
    batches = {}
    omitted = []
    active = set(active_features)
    for directory in sorted(ps2_dir.parent.glob("batch_*")):
        ps2_path, batch_path = directory / "ps2_manifest.json", directory / "batch_manifest.json"
        if not ps2_path.is_file() or not batch_path.is_file():
            continue
        ps2, batch = _json(ps2_path), _json(batch_path)
        bid = directory.name
        if ps2.get("batch_id") != bid or batch.get("batch_id") != bid:
            raise ValueError(f"population batch identity mismatch: {directory}")
        names = ps2["control_all_admissible"]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate admissible IDs in {bid}")
        retained = set(batch["features"])
        series_path = directory / batch["series"]["file"] if "series" in batch else None
        payload_present = series_path is not None and series_path.is_file()
        if payload_present:
            _checked(series_path, batch["series"]["sha256"])
        for feature in names:
            if directory == ps2_dir and feature in active:
                continue
            state = ("NO_RETAINED_SERIES_PAYLOAD" if feature not in retained else
                     "RETAINED_SERIES_NOT_RUN" if payload_present else "RETAINED_SERIES_PAYLOAD_MISSING")
            omitted.append({"batch": bid, "feature_id": feature,
                            "state": state})
        batches[bid] = {"all_admissible_count": len(names), "retained_series_count": len(retained),
                        "retained_series_payload_present": payload_present,
                        "ps2_manifest_sha256": _sha(ps2_path),
                        "batch_manifest_sha256": _sha(batch_path)}
    if ps2_dir.name not in batches:
        raise ValueError("one-pair batch missing from population inventory")
    return {"all_admissible_count": sum(b["all_admissible_count"] for b in batches.values()),
            "complete_three_batch_population": set(batches) == {"batch_001", "batch_002", "batch_003"},
            "evaluated_all_admissible_control": False,
            "all_admissible_control_state": "PENDING_NOT_EXECUTED",
            "batches": batches, "omitted_set": omitted}


def evaluate_one(data, *, seeds, window, settings, fit_fn, source):
    """Four independent refits for exactly one pair on common hourly origins."""
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or window < 24:
        raise ValueError("distinct seeds and a >=24h temporal window required")
    ts, y, cols, names = data["timestamps"], data["y"], data["columns"], data["features"]
    fs, fe = data["fold"]["fit"]
    vs, ve = data["fold"]["val"]
    fit = _usable_origins(ts, cols, y, names, fs, fe, window, PURGE_H)
    if len(fit) < 3:
        raise ValueError("insufficient common TRAIN origins")
    stop_start = int(ts[fit[int(.85 * len(fit))]])
    train = fit[ts[fit] + PURGE_H * HOUR < stop_start]
    stop = fit[ts[fit] >= stop_start]
    val = _usable_origins(ts, cols, y, names, vs, ve, window, data["horizon"])
    if not len(train) or not len(stop) or not len(val):
        raise ValueError("empty purged TRAIN/stop/validation population")
    lo = int(train[0]) - window + 1
    hi = int(train[-1]) + 1
    zs = []
    for name in names:
        raw = np.asarray(cols[name], dtype="float64")
        finite = raw[lo:hi][np.isfinite(raw[lo:hi])]
        if not len(finite):
            raise ValueError(f"no TRAIN-only scaler observations for {name}")
        mean, std = float(finite.mean()), float(finite.std())
        zs.append(((raw - mean) / (std if std > 1e-12 else 1.0)).astype("float32"))
    z = np.stack(zs, axis=-1)
    windows = [_windowed(z, origins, window) for origins in (train, stop, val)]
    config = {"feature_names": list(names), "window": int(window), "sample_hours": 1,
              "horizons": [int(data["horizon"])], "output_steps": 6,
              "output_channels": 8, "regime": "R0"}
    a, b = data["pair"]
    variants = {"base": data["base"], "base+a": [*data["base"], a],
                "base+b": [*data["base"], b], "base+a+b": names}
    row_sha = hashlib.sha256(np.ascontiguousarray(ts[val]).tobytes()).hexdigest()
    naive_mae = float(np.mean(np.abs(y[val].astype("float64"))))
    arms = {}
    for label, active in variants.items():
        inactive = [i for i, name in enumerate(names) if name not in active]
        x_train, x_stop, x_val = (w.copy() for w in windows)
        for x in (x_train, x_stop, x_val):
            x[:, :, inactive] = 0.0
        per = {}
        for seed in seeds:
            pred, budget = fit_fn(x_train, y[train], x_stop, y[stop], x_val,
                                  seed=seed, settings=dict(settings), config=dict(config), source=source)
            pred = np.asarray(pred).reshape(-1)
            if pred.shape != y[val].shape or not np.all(np.isfinite(pred)):
                raise ValueError("fit returned nonfinite or misaligned validation predictions")
            per[str(seed)] = {"mae": float(np.mean(np.abs(pred.astype("float64") - y[val]))),
                              "budget": budget}
        arms[label] = {"active_features": list(active), "rows_sha256": row_sha,
                       "rows": int(len(val)), "naive_mae": naive_mae,
                       "per_seed": per,
                       "mean_mae": float(np.mean([r["mae"] for r in per.values()]))}
        arms[label]["beats_paired_naive"] = arms[label]["mean_mae"] < naive_mae
        arms[label]["skill_vs_paired_naive"] = (
            1.0 - arms[label]["mean_mae"] / naive_mae if naive_mae > 0 else None)
    joint = arms["base+a+b"]["mean_mae"]
    return {"schema": "ps5.eurusd.one_pair.v1", "evidence_scope": "DEVELOPMENT_INCOMPLETE_POPULATION",
            "selection_release": None, "target": data["target"], "horizon_h": data["horizon"],
            "fold": data["fold"]["fold_id"], "base": data["base"], "pair": data["pair"],
            "seeds": seeds, "settings": dict(settings), "architecture": config,
            "identity": data["identity"], "arms": arms, "refits": 4 * len(seeds),
            "naive": {"kind": "zero_log_return", "mae": naive_mae, "rows_sha256": row_sha},
            "paired_naive_gate": {"status": "PASS" if joint < naive_mae else "FAIL",
                                  "joint_beats_naive": joint < naive_mae,
                                  "strategy_eligible": False,
                                  "reason": "incomplete selection population"},
            "comparison": "RE_ENTERS_THIS_FOLD" if joint < min(arms[k]["mean_mae"]
                for k in ("base", "base+a", "base+b")) else "NO_REENTRY_THIS_FOLD",
            "population": {"train_rows": int(len(train)), "stop_rows": int(len(stop)),
                           "val_rows": int(len(val)), "purge_h": PURGE_H,
                           "max_fit_label_ts": int(ts[train[-1]] + PURGE_H * HOUR),
                           "stop_start_ts": stop_start,
                           "max_stop_label_ts": int(ts[stop[-1]] + PURGE_H * HOUR),
                           "fit_end_ts": int(fe),
                           "max_val_label_ts": int(ts[val[-1]] + data["horizon"] * HOUR),
                           "val_end_ts": int(ve)}}


def verify_source(source):
    """Fail closed unless the modular component comes from an exact clean commit."""
    source = Path(source).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if head != SOURCE_COMMIT or dirty or not (source / "predictor_plugins/modular_temporal/__init__.py").is_file():
        raise ValueError(f"modular source must be clean commit {SOURCE_COMMIT}: {source}")
    return source


def fit_modular(x_train, y_train, x_stop, y_stop, x_val, *, seed, settings, config, source):
    source = verify_source(source)
    sys.path.insert(0, str(source))
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt
    if not Path(mt.__file__).resolve().is_relative_to(source):
        raise ValueError("modular temporal import did not resolve to pinned source")

    module_path = source / "tools/modular_candidate_evaluator.py"
    spec = importlib.util.spec_from_file_location("ps5_pinned_evaluator", module_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    full = mt.default_config(config["feature_names"])
    full.update({k: v for k, v in config.items() if k != "feature_names"})
    bundle = mt.build_modular(full)
    report = evaluator.fit_with_early_stopping(
        bundle.forecast_model, x_train, y_train[:, None, None],
        x_stop, y_stop[:, None, None], dict(settings, seed=int(seed)))
    pred = np.asarray(bundle.forecast_model(x_val, training=False))[:, 0, 0]
    return pred, {"updates": int(report["observed_updates"]),
                  "epochs": int(report["epochs_completed"]), "stop_reason": report["stop_reason"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ps1-dir", type=Path, required=True)
    p.add_argument("--ps2-dir", type=Path, required=True)
    p.add_argument("--modular-source", type=Path, required=True)
    p.add_argument("--base", nargs="+", required=True)
    p.add_argument("--pair", nargs=2, required=True)
    p.add_argument("--target", choices=("Y_s", "Y_l"), required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022])
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--max-epochs", type=int, default=DEFAULT_SETTINGS["max_epochs"])
    p.add_argument("--max-seconds", type=float, default=DEFAULT_SETTINGS["max_seconds"],
                   help="per-refit wall-clock budget; use an outer timeout for the whole comparison")
    args = p.parse_args(argv)
    if args.max_epochs < 1 or not 0 < args.max_seconds <= 86400:
        p.error("max-epochs and max-seconds must be positive")
    source = verify_source(args.modular_source)
    data = load_retained(args.ps1_dir, args.ps2_dir, base=args.base, pair=args.pair,
                         target=args.target, horizon=args.horizon, fold_id=args.fold)
    settings = dict(DEFAULT_SETTINGS, max_epochs=args.max_epochs, max_seconds=args.max_seconds)
    result = evaluate_one(data, seeds=args.seeds, window=args.window, settings=settings,
                          fit_fn=fit_modular, source=source)
    result["modular_source"] = {"commit": SOURCE_COMMIT, "path": str(source)}
    result["control_population"] = population_control(args.ps2_dir, data["features"])
    if args.out.exists():
        raise ValueError("output exists; one-pair executor does not overwrite evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "comparison": result["comparison"],
                      "population": result["population"]}), flush=True)


if __name__ == "__main__":
    main()
