#!/usr/bin/env python
"""Future CLOSE labels on the same rows as the seasonal dataset.

The seasonal masks stay. seasonal_origin is replaced by CLOSE at the
origin so the trainer's naive is CLOSE persistence. No Prediction columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE = Path("/var/tmp/retsu-phase1")
HOUR = 3600 * 10**9
H1_END = np.datetime64("2024-07-01T00:00:00").astype("datetime64[ns]").astype(np.int64)


def close_lookup() -> dict[int, float]:
    series = np.load(PHASE / "stl_series.npz")
    avail = series["avail_ns"].astype(np.int64)
    close = series["close"].astype(np.float64)
    return {int(ts): float(value) for ts, value in zip(avail, close) if np.isfinite(value)}


def future_close(ts: np.ndarray, horizons: list[int], lookup: dict[int, float]) -> np.ndarray:
    out = np.full((len(ts), len(horizons)), np.nan, np.float32)
    for i, origin in enumerate(ts):
        for k, horizon in enumerate(horizons):
            value = lookup.get(int(origin) + int(horizon) * HOUR)
            if value is not None:
                out[i, k] = np.float32(value)
    return out


def persistence(y: np.ndarray, origin: np.ndarray, mask: np.ndarray) -> list[dict]:
    rows = []
    for i in range(y.shape[1]):
        ok = mask & np.isfinite(y[:, i]) & np.isfinite(origin)
        rows.append({
            "horizon_index": i,
            "rows": int(ok.sum()),
            "mae": float(np.mean(np.abs(y[ok, i] - origin[ok]))),
        })
    return rows


def main() -> None:
    lookup = close_lookup()
    meta = json.loads((PHASE / "dataset_meta.json").read_text())
    base = np.load(PHASE / "dataset.npz")
    random_z = np.load(PHASE / "dataset_random_encoder.npz")["z"]
    if random_z.shape != base["z"].shape:
        raise RuntimeError("Z_SHAPE")
    short_h = [int(h) for h in meta["horizons_short"]]
    long_h = [int(h) for h in meta["horizons_long"]]
    ts = base["ts"].astype(np.int64)
    y_short = future_close(ts, short_h, lookup)
    y_long = future_close(ts, long_h, lookup)
    # Same admitted rows as the seasonal labels. Close has to exist there.
    for name, y, old in (
        ("short", y_short, base["y_short"]),
        ("long", y_long, base["y_long"]),
    ):
        if y.shape != old.shape:
            raise RuntimeError(f"SHAPE_{name}")
        admitted = np.isfinite(old)
        missing = int((admitted & ~np.isfinite(y)).sum())
        if missing:
            raise RuntimeError(f"CLOSE_MISSING_{name}_{missing}")
        # Keep non-admitted cells empty so the mask stays the seasonal mask.
        y[~admitted] = np.nan
    origin = np.array([lookup.get(int(stamp), np.nan) for stamp in ts], np.float32)
    if not np.isfinite(origin[base["ok_short"] | base["ok_long"]]).all():
        raise RuntimeError("CLOSE_ORIGIN_MISSING")
    common = {key: base[key] for key in base.files if key not in ("z", "y_short", "y_long", "seasonal_origin")}
    common["y_short"] = y_short
    common["y_long"] = y_long
    common["seasonal_origin"] = origin
    pretrained = PHASE / "dataset_close_pretrained.npz"
    seed7 = PHASE / "dataset_close_seed7.npz"
    np.savez_compressed(pretrained, z=base["z"], **common)
    np.savez_compressed(seed7, z=random_z, **common)
    hold = base["holdout"]
    report = {
        "target": "future_CLOSE",
        "mae_unit": "close_price",
        "masks": "seasonal ok_short and ok_long",
        "naive": "CLOSE at the origin",
        "short_horizons": short_h,
        "long_horizons": long_h,
        "holdout_persistence_short": persistence(y_short, origin, hold & base["ok_short"]),
        "holdout_persistence_long": persistence(y_long, origin, hold & base["ok_long"]),
        "monitor_persistence_short": persistence(y_short, origin, base["monitor"] & base["ok_short"]),
        "monitor_persistence_long": persistence(y_long, origin, base["monitor"] & base["ok_long"]),
        "holdout_h1_rows_short": int(((ts < H1_END) & hold & base["ok_short"]).sum()),
        "holdout_h2_rows_short": int(((ts >= H1_END) & hold & base["ok_short"]).sum()),
    }
    # Attach horizon numbers onto the persistence rows.
    for key, horizons in (("holdout_persistence_short", short_h), ("monitor_persistence_short", short_h),
                          ("holdout_persistence_long", long_h), ("monitor_persistence_long", long_h)):
        for row, horizon in zip(report[key], horizons):
            row["horizon"] = int(horizon)
    (PHASE / "followup" / "close_persistence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["holdout_persistence_short"], indent=2))
    print(json.dumps(report["holdout_persistence_long"], indent=2))
    print("CLOSE_TARGETS_OK")


if __name__ == "__main__":
    main()
