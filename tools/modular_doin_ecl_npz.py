#!/usr/bin/env python3
"""Build identified train/validation NPZ inputs for the M04 DOIN campaign from governed ECL.

Source: the data-gov ``sota_benchmarks`` ECL CSV, content-addressed by SHA-256
(verified here before parsing). Split: the author loader's chronological 7/1/2
row borders (train rows [0, n_train), validation rows [n_train - window,
n_train + n_vali)). The test rows (>= n_train + n_vali) are dropped immediately
after parsing and before any statistic; no test window is ever constructed.

Scaler: one StandardScaler (population std, ddof=0, the sklearn convention)
fitted on train rows [0, n_train) only. Metric space ``z_train``.

Purge: a train window is kept only if its last target row precedes the first
validation INPUT row. Windows that would span an irregular timestamp gap are
excluded and counted. Both counts are in the manifest.

Output: ``train.npz``, ``validation.npz`` in the evaluator's exact contract and
``MANIFEST.json`` binding source, borders, scaler, purge, digests. Local
arithmetic only; this module grants no governance authority by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ECL_SHA256 = "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ecl(path, expected_sha):
    actual = sha_file(path)
    if actual != expected_sha:
        raise ValueError(f"source digest mismatch: {actual} != {expected_sha}")
    with open(path) as stream:
        header = stream.readline().rstrip("\n").split(",")
    if header[0] != "date":
        raise ValueError("first column must be date")
    dates = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=str).astype("datetime64[s]")
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, len(header)), dtype=np.float64)
    return header[1:], dates.astype(np.int64), values, actual


def author_borders(n_rows, window):
    n_train = int(n_rows * 0.7)
    n_test = int(n_rows * 0.2)
    n_vali = n_rows - n_train - n_test
    return {"n_rows": n_rows, "n_train": n_train, "n_vali": n_vali, "n_test": n_test,
            "train_rows": [0, n_train], "validation_rows": [n_train - window, n_train + n_vali],
            "test_first_row": n_train + n_vali}


def windows_for(rows, times, z, window, horizons, sample_seconds):
    """Origins o (last input row) with complete regular input and target support in rows."""
    start, end = rows
    hmax = max(horizons)
    origins, gaps = [], 0
    for o in range(start + window - 1, end - hmax):
        span = times[o - window + 1:o + hmax + 1]
        if np.any(np.diff(span) != sample_seconds):
            gaps += 1
            continue
        origins.append(o)
    origins = np.asarray(origins, dtype=np.int64)
    offsets = np.arange(-window + 1, 1)
    x = z[origins[:, None] + offsets[None, :]].astype(np.float32)
    y = z[origins[:, None] + np.asarray(horizons)[None, :]].astype(np.float32)
    return origins, x, y, gaps


def build(source, out, *, window, horizons, expected_sha=ECL_SHA256, sample_hours=1):
    names, times, values, source_sha = read_ecl(source, expected_sha)
    borders = author_borders(len(values), window)
    keep = borders["test_first_row"]
    # Drop the test rows before any statistic or window is formed.
    names_arr = np.array(names)
    times, values = times[:keep], values[:keep]
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite source values in train/validation rows")
    train_raw = values[:borders["n_train"]]
    mean = train_raw.mean(axis=0)
    scale = train_raw.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError(f"constant train channels: {names_arr[scale <= 0].tolist()}")
    z = (values - mean) / scale
    seconds = int(sample_hours * 3600)
    scaler = {"kind": "StandardScaler(ddof=0)", "fit_rows": borders["train_rows"],
              "mean_sha256": hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest(),
              "scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest()}
    scaler_identity = "ecl-train-standard:" + hashlib.sha256(json.dumps(scaler, sort_keys=True).encode()).hexdigest()
    dataset_id = f"data-gov:sota_benchmarks:{source_sha}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    val_o, vx, vy, vgaps = windows_for(borders["validation_rows"], times, z, window, horizons, seconds)
    validation_input_start = int(times[val_o.min() - window + 1])
    tr_o, tx, ty, tgaps = windows_for(borders["train_rows"], times, z, window, horizons, seconds)
    target_end = times[tr_o + max(horizons)]
    purge = target_end >= validation_input_start
    tr_o, tx, ty = tr_o[~purge], tx[~purge], ty[~purge]
    manifest = {"schema": "m04.ecl_npz.v1", "dataset_id": dataset_id, "source_sha256": source_sha,
                "channels": len(names), "channel_order_sha256": hashlib.sha256(",".join(names).encode()).hexdigest(),
                "window": window, "horizons": list(horizons), "sample_hours": sample_hours,
                "borders": borders, "scaler": scaler, "scaler_identity": scaler_identity,
                "metric_space": "z_train", "test_rows_read_for_statistics": False,
                "test_windows_constructed": False, "splits": {}}
    for split, origins, x, y, gaps, purged in (("train", tr_o, tx, ty, tgaps, int(purge.sum())),
                                              ("validation", val_o, vx, vy, vgaps, 0)):
        ts = times[origins]
        arrays = dict(windows=x, targets=y, row_ids=np.array([f"ecl:{t}" for t in ts]),
                      timestamps=ts, target_timestamps=ts[:, None] + np.asarray(horizons, np.int64) * seconds,
                      dataset_id=np.array(dataset_id), split=np.array(split),
                      feature_names=names_arr, target_names=names_arr,
                      horizons=np.asarray(horizons, np.int64), timestamp_unit=np.array("seconds"),
                      metric_space=np.array("z_train"), scaler_identity=np.array(scaler_identity),
                      scaler_scale=scale.astype(np.float64))
        path = out / f"{split}.npz"
        np.savez(path, **arrays)
        manifest["splits"][split] = {"path": path.name, "sha256": sha_file(path), "windows": int(len(origins)),
                                     "origin_rows": [int(origins.min()), int(origins.max())],
                                     "origin_start": int(ts.min()), "origin_end": int(ts.max()),
                                     "gap_excluded_windows": gaps, "purged_windows": purged,
                                     "bytes": path.stat().st_size}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizons", default="1-24", help="'a-b' range or comma list")
    parser.add_argument("--expected-sha256", default=ECL_SHA256)
    args = parser.parse_args()
    if "-" in args.horizons:
        a, b = map(int, args.horizons.split("-"))
        horizons = list(range(a, b + 1))
    else:
        horizons = [int(h) for h in args.horizons.split(",")]
    manifest = build(args.source, args.out, window=args.window, horizons=horizons,
                     expected_sha=args.expected_sha256)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.modular_heartbeat import cgroup_memory, process_memory
        manifest["build_resources"] = {"cgroup": cgroup_memory(), "process": process_memory()}
        (Path(args.out) / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass
    print(json.dumps({k: manifest[k] for k in ("dataset_id", "splits", "scaler_identity")}, indent=1))


if __name__ == "__main__":
    main()
