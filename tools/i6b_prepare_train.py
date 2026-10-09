#!/usr/bin/env python3
"""Build the TRAIN-only EURUSD I6-B branch-pretraining population.

The selected population is the sorted union of RAW winners in the sealed FS4
freeze. Each selected feature becomes one branch with two channels: a robustly
standardised value and its observed mask. Windows contain exactly 24 elapsed
hourly timestamps; gaps are not compressed into artificial time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def selected_features(freeze):
    if freeze.get("schema") != "fs4.test_freeze.v1":
        raise ValueError("UNSUPPORTED_FREEZE")
    winners = (freeze.get("winners") or {}).get("EURUSD")
    if not isinstance(winners, dict) or not winners:
        raise ValueError("EURUSD_RAW_WINNERS_REQUIRED")
    members = set()
    for target in winners.values():
        raw = target.get("RAW") if isinstance(target, dict) else None
        if not isinstance(raw, dict) or not raw.get("members"):
            raise ValueError("EURUSD_RAW_WINNER_MEMBERS_REQUIRED")
        members.update(raw["members"])
    if any(not isinstance(name, str) or not name for name in members):
        raise ValueError("INVALID_SELECTED_FEATURE")
    return tuple(sorted(members))


def _merge_sources(paths, features):
    keys = ["t_decision_utc", "row_id"]
    tables = []
    ownership = {}
    for path in paths:
        frame = pd.read_parquet(path)
        if any(key not in frame for key in keys):
            raise ValueError(f"SOURCE_KEYS_MISSING: {path}")
        available = [name for name in features if name in frame]
        for name in available:
            if name in ownership:
                raise ValueError(f"SELECTED_FEATURE_DUPLICATE: {name}")
            ownership[name] = str(path)
        if available:
            tables.append(frame[keys + available].copy())
    missing = sorted(set(features) - set(ownership))
    if missing:
        raise ValueError(f"SELECTED_FEATURE_MISSING: {missing}")
    merged = tables[0]
    for frame in tables[1:]:
        merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one", sort=False)
    if merged.empty or merged[keys].duplicated().any():
        raise ValueError("EMPTY_OR_DUPLICATE_SOURCE_POPULATION")
    return merged, ownership


def _branch_name(feature, index):
    safe = re.sub(r"[^A-Za-z0-9_]", "_", feature)
    if not safe or not safe[0].isalpha():
        safe = "f_" + safe
    return f"b{index:02d}_{safe}"[:120]


def prepare(freeze_path, source_paths, output_path, *, train_end, window=24,
            internal_fraction=0.8):
    freeze_path, output_path = Path(freeze_path), Path(output_path)
    freeze = json.loads(freeze_path.read_text())
    features = selected_features(freeze)
    table, ownership = _merge_sources([Path(path) for path in source_paths], features)
    timestamps = pd.to_datetime(table["t_decision_utc"], utc=True, errors="raise")
    cutoff = pd.Timestamp(train_end)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    table = table.loc[timestamps < cutoff].copy()
    table["t_decision_utc"] = pd.to_datetime(table["t_decision_utc"], utc=True)
    table = table.sort_values("t_decision_utc", kind="stable").reset_index(drop=True)
    if table["t_decision_utc"].duplicated().any():
        raise ValueError("DUPLICATE_TRAIN_TIMESTAMP")
    epoch = (table["t_decision_utc"].astype("int64") // 1_000_000_000).to_numpy(dtype="int64")
    position = {int(ts): i for i, ts in enumerate(epoch)}
    origins, rows = [], []
    hour = 3600
    for origin in epoch:
        wanted = [int(origin - hour * lag) for lag in range(window - 1, -1, -1)]
        if all(ts in position for ts in wanted):
            origins.append(int(origin))
            rows.append([position[ts] for ts in wanted])
    if not rows:
        raise ValueError("NO_COMPLETE_HOURLY_WINDOWS")
    rows = np.asarray(rows, dtype="int64")
    raw = table.loc[:, features].to_numpy(dtype="float64")
    fit_windows = max(1, int(len(rows) * float(internal_fraction)))
    fit_rows = np.unique(rows[:fit_windows].reshape(-1))
    fit = raw[fit_rows]
    median = np.nanmedian(fit, axis=0)
    if not np.all(np.isfinite(median)):
        raise ValueError("FEATURE_HAS_NO_FINITE_TRAIN_PREFIX")
    filled_fit = np.where(np.isfinite(fit), fit, median)
    mean = filled_fit.mean(axis=0)
    std = filled_fit.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    observed = np.isfinite(raw).astype("float32")
    z = ((np.where(np.isfinite(raw), raw, median) - mean) / std).astype("float32")
    channels = np.empty((len(raw), 2 * len(features)), dtype="float32")
    channels[:, 0::2], channels[:, 1::2] = z, observed
    windows = channels[rows]
    channel_names = [item for name in features for item in (name + "__value", name + "__observed")]
    source_digests = {str(Path(path).resolve()): _sha256(path) for path in source_paths}
    population = {
        "schema": "i6b.eurusd.train.v1", "freeze_sha256": freeze.get("freeze_sha256"),
        "train_end_exclusive": cutoff.isoformat(), "window_hours": int(window),
        "selected_features": list(features), "source_sha256": source_digests,
        "origins": len(origins), "normalization_fit_windows": fit_windows,
    }
    dataset_id = "i6b_eurusd_" + _digest(population)[:20]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, windows=windows, timestamps=np.asarray(origins, dtype="int64"),
                            feature_names=np.asarray(channel_names), dataset_id=np.asarray(dataset_id),
                            split=np.asarray("train"), timestamp_unit=np.asarray("seconds"))
    os.replace(temporary, output_path)
    branches = [{"name": _branch_name(name, i),
                 "features": [name + "__value", name + "__observed"]}
                for i, name in enumerate(features)]
    from predictor_plugins.modular_temporal.config import _normalize
    config = _normalize({"feature_names": channel_names, "sample_hours": 1, "branches": branches})
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    report = {**population, "dataset_id": dataset_id, "train_npz": str(output_path),
              "train_npz_sha256": _sha256(output_path), "config": str(config_path),
              "config_sha256": _sha256(config_path), "feature_ownership": ownership,
              "normalization": {name: {"median": float(median[i]), "mean": float(mean[i]),
                                        "std": float(std[i])} for i, name in enumerate(features)},
              "validation_read": False, "test_read": False}
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--feature-parquet", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-end", default="2024-01-01T00:00:00Z")
    args = parser.parse_args(argv)
    report = prepare(args.freeze, args.feature_parquet, args.output, train_end=args.train_end)
    print(json.dumps({"status": "COMPLETE", "features": len(report["selected_features"]),
                      "origins": report["origins"], "sha256": report["train_npz_sha256"]}))


if __name__ == "__main__":
    main()
