#!/usr/bin/env python3
"""Build purged TRAIN-only NPZs for the I6-D fleet cost pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _save(
    path,
    *,
    windows,
    targets,
    row_ids,
    timestamps,
    feature_names,
    dataset_id,
    split,
    horizon,
    target_name,
):
    horizons = np.asarray([horizon], dtype="int64")
    target_values = np.asarray(targets, dtype="float32").reshape(-1, 1, 1)
    np.savez_compressed(
        path,
        windows=np.asarray(windows, dtype="float32"),
        targets=target_values,
        baseline=np.zeros_like(target_values),
        row_ids=np.asarray(row_ids, dtype="U64"),
        timestamps=np.asarray(timestamps, dtype="int64"),
        target_timestamps=np.asarray(timestamps, dtype="int64")[:, None] + 3600 * horizons,
        dataset_id=np.asarray(dataset_id),
        split=np.asarray(split),
        feature_names=np.asarray(feature_names),
        target_names=np.asarray([target_name]),
        horizons=horizons,
        timestamp_unit=np.asarray("seconds"),
        metric_space=np.asarray("native_log_return"),
        scaler_identity=np.asarray("target_unscaled_v1"),
        scaler_scale=np.asarray([1.0], dtype="float64"),
    )


def prepare(windows_npz, targets_parquet, output_dir, train_fraction=0.8, horizon=1):
    with np.load(windows_npz, allow_pickle=False) as source:
        windows = source["windows"]
        timestamps = source["timestamps"]
        feature_names = source["feature_names"]
        dataset_id = source["dataset_id"].item()
    if windows.shape[1:] != (24, 40) or len(windows) != len(timestamps):
        raise ValueError("I6-D pilot expects selected20 windows with shape (rows,24,40)")
    if not 0.5 <= train_fraction < 0.95:
        raise ValueError("train_fraction must be in [0.5,0.95)")

    if horizon not in {1, 2, 3, 4, 5, 6, 24, 48, 72, 96, 120, 144}:
        raise ValueError("unsupported horizon")
    target_name = f"Y_s_{horizon}h" if horizon <= 6 else f"Y_l_{horizon}h"
    targets = pd.read_parquet(
        targets_parquet, columns=["t_decision_utc", "row_id", target_name]
    )
    target_seconds = pd.to_datetime(targets["t_decision_utc"], utc=True).astype("int64") // 10**9
    lookup = targets.assign(_timestamp=target_seconds).set_index("_timestamp")
    selected = lookup.reindex(timestamps)
    valid = selected[target_name].notna().to_numpy()
    windows, timestamps = windows[valid], timestamps[valid]
    selected = selected.iloc[np.flatnonzero(valid)]
    if len(windows) < 100:
        raise ValueError("insufficient aligned TRAIN rows")

    boundary = int(len(windows) * train_fraction)
    validation_start = timestamps[boundary]
    train_mask = timestamps + horizon * 3600 < validation_start - 23 * 3600
    if not train_mask.any():
        raise ValueError("purge removed all TRAIN rows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    train_path, validation_path = output / "train.npz", output / "validation.npz"
    _save(train_path, windows=windows[train_mask], targets=selected[target_name].to_numpy()[train_mask],
          row_ids=selected["row_id"].astype(str).to_numpy()[train_mask], timestamps=timestamps[train_mask],
          feature_names=feature_names, dataset_id=dataset_id, split="train", horizon=horizon,
          target_name=target_name)
    _save(validation_path, windows=windows[boundary:], targets=selected[target_name].to_numpy()[boundary:],
          row_ids=selected["row_id"].astype(str).to_numpy()[boundary:], timestamps=timestamps[boundary:],
          feature_names=feature_names, dataset_id=dataset_id, split="validation", horizon=horizon,
          target_name=target_name)
    report = {
        "schema": "predictor.i6d.cost_pilot_inputs.v1",
        "scope": "TRAIN_ONLY_DIAGNOSTIC",
        "horizon": horizon,
        "target_name": target_name,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(len(windows) - boundary),
        "purged_rows": int(boundary - train_mask.sum()),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "test_read": False,
    }
    (output / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.windows, args.targets, args.output, horizon=args.horizon), sort_keys=True))


if __name__ == "__main__":
    main()
