#!/usr/bin/env python
"""CLOSE change on the same rows as the seasonal dataset.

y = future CLOSE - CLOSE at the origin. seasonal_origin is 0 so the
trainer naive is that change, which is CLOSE persistence. The recomputed
2024 persistence must match followup/close_persistence.json before any fit.
No Prediction columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE = Path("/var/tmp/retsu-phase1")
HOUR = 3600 * 10**9
ATOL = 1e-6


def close_lookup() -> dict[int, float]:
    series = np.load(PHASE / "stl_series.npz")
    avail = series["avail_ns"].astype(np.int64)
    close = series["close"].astype(np.float64)
    return {int(ts): float(value) for ts, value in zip(avail, close) if np.isfinite(value)}


def future_close(ts: np.ndarray, horizons: list[int], lookup: dict[int, float]) -> np.ndarray:
    out = np.full((len(ts), len(horizons)), np.nan, np.float64)
    for i, origin in enumerate(ts):
        for k, horizon in enumerate(horizons):
            value = lookup.get(int(origin) + int(horizon) * HOUR)
            if value is not None:
                out[i, k] = value
    return out


def persistence(y: np.ndarray, mask: np.ndarray) -> list[dict]:
    rows = []
    for i in range(y.shape[1]):
        ok = mask & np.isfinite(y[:, i])
        rows.append({
            "horizon_index": i,
            "rows": int(ok.sum()),
            "mae": float(np.mean(np.abs(y[ok, i]))),
        })
    return rows


def assert_matches(fresh: list[dict], previous: list[dict], name: str) -> None:
    old = {int(row["horizon"]): row for row in previous}
    if len(fresh) != len(old):
        raise RuntimeError(f"PERSISTENCE_LEN_{name}")
    for row in fresh:
        horizon = int(row["horizon"])
        ref = old[horizon]
        if int(row["rows"]) != int(ref["rows"]):
            raise RuntimeError(f"PERSISTENCE_ROWS_{name}_{horizon}")
        gap = abs(float(row["mae"]) - float(ref["mae"]))
        if gap > ATOL:
            raise RuntimeError(f"PERSISTENCE_MAE_{name}_{horizon}_{gap}")


def main() -> None:
    lookup = close_lookup()
    meta = json.loads((PHASE / "dataset_meta.json").read_text())
    previous = json.loads((PHASE / "followup" / "close_persistence.json").read_text())
    base = np.load(PHASE / "dataset.npz")
    random_z = np.load(PHASE / "dataset_random_encoder.npz")["z"]
    if random_z.shape != base["z"].shape:
        raise RuntimeError("Z_SHAPE")
    short_h = [int(h) for h in meta["horizons_short"]]
    long_h = [int(h) for h in meta["horizons_long"]]
    ts = base["ts"].astype(np.int64)
    origin = np.array([lookup.get(int(stamp), np.nan) for stamp in ts], np.float64)
    y_short = future_close(ts, short_h, lookup) - origin[:, None]
    y_long = future_close(ts, long_h, lookup) - origin[:, None]
    for name, y, old in (
        ("short", y_short, base["y_short"]),
        ("long", y_long, base["y_long"]),
    ):
        if y.shape != old.shape:
            raise RuntimeError(f"SHAPE_{name}")
        admitted = np.isfinite(old)
        missing = int((admitted & ~np.isfinite(y)).sum())
        if missing:
            raise RuntimeError(f"DCLOSE_MISSING_{name}_{missing}")
        y[~admitted] = np.nan
    if not np.isfinite(origin[base["ok_short"] | base["ok_long"]]).all():
        raise RuntimeError("CLOSE_ORIGIN_MISSING")
    level_std = float(np.std(origin[base["ok_short"]]))
    change_std = float(np.nanstd(y_short[base["ok_short"] & base["pool"], 0]))
    if not (change_std < level_std / 5.0):
        raise RuntimeError(f"CHANGE_STILL_A_LEVEL_{change_std}_{level_std}")
    y_short = y_short.astype(np.float32)
    y_long = y_long.astype(np.float32)
    zeros = np.zeros(len(ts), np.float32)
    common = {key: base[key] for key in base.files if key not in ("z", "y_short", "y_long", "seasonal_origin")}
    common["y_short"] = y_short
    common["y_long"] = y_long
    common["seasonal_origin"] = zeros
    np.savez_compressed(PHASE / "dataset_dclose_pretrained.npz", z=base["z"], **common)
    np.savez_compressed(PHASE / "dataset_dclose_seed7.npz", z=random_z, **common)
    hold = base["holdout"]
    report = {
        "target": "CLOSE_future_minus_origin",
        "mae_unit": "close_change",
        "masks": "seasonal ok_short and ok_long",
        "naive": "zero change, equal to CLOSE persistence",
        "level_std_ok_short": level_std,
        "change_std_h1_pool_short": change_std,
        "short_horizons": short_h,
        "long_horizons": long_h,
        "holdout_persistence_short": persistence(y_short, hold & base["ok_short"]),
        "holdout_persistence_long": persistence(y_long, hold & base["ok_long"]),
        "monitor_persistence_short": persistence(y_short, base["monitor"] & base["ok_short"]),
        "monitor_persistence_long": persistence(y_long, base["monitor"] & base["ok_long"]),
    }
    for key, horizons in (
        ("holdout_persistence_short", short_h),
        ("monitor_persistence_short", short_h),
        ("holdout_persistence_long", long_h),
        ("monitor_persistence_long", long_h),
    ):
        for row, horizon in zip(report[key], horizons):
            row["horizon"] = int(horizon)
    assert_matches(report["holdout_persistence_short"], previous["holdout_persistence_short"], "short")
    assert_matches(report["holdout_persistence_long"], previous["holdout_persistence_long"], "long")
    (PHASE / "followup" / "dclose_persistence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"level_std": level_std, "h1_change_std": change_std}, indent=2))
    print(json.dumps(report["holdout_persistence_short"], indent=2))
    print(json.dumps(report["holdout_persistence_long"], indent=2))
    print("DCLOSE_TARGETS_OK")


if __name__ == "__main__":
    main()
