#!/usr/bin/env python
"""Seasonal increment on the saved causal-STL labels.

y = seasonal(t+h) - seasonal(t). The level forecast is the origin seasonal
plus this increment, so its MAE is the MAE of y. Zero is seasonal persistence.
Same rows, same masks, same timestamps. Does not refit STL.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHASE = Path("/var/tmp/retsu-phase1")


def persistence(y: np.ndarray, mask: np.ndarray, horizons: list[int]) -> list[dict]:
    rows = []
    for i, horizon in enumerate(horizons):
        ok = mask & np.isfinite(y[:, i])
        rows.append({
            "horizon": int(horizon),
            "rows": int(ok.sum()),
            "mae": float(np.mean(np.abs(y[ok, i]))),
        })
    return rows


def main() -> None:
    source = np.load(PHASE / "dataset.npz")
    meta = json.loads((PHASE / "dataset_meta.json").read_text())
    origin = source["seasonal_origin"].astype(np.float64)
    usable = np.isfinite(origin)
    out = {key: source[key] for key in source.files if key not in ("seasonal_origin", "ok_short", "ok_long")}
    out["ok_short"] = source["ok_short"] & usable
    out["ok_long"] = source["ok_long"] & usable
    for key, horizons in (("y_short", meta["horizons_short"]), ("y_long", meta["horizons_long"])):
        y = source[key].astype(np.float64) - origin[:, None]
        admitted = np.isfinite(source[key]) & usable[:, None]
        y[~admitted] = np.nan
        out[key] = y.astype(np.float32)
    out["seasonal_origin"] = np.zeros(len(origin), np.float32)
    np.savez_compressed(PHASE / "dataset_sdelta.npz", **out)
    hold = source["holdout"]
    report = {
        "target": "causal_stl_seasonal_future_minus_origin",
        "level_forecast": "seasonal_origin + predicted_increment",
        "naive": "zero increment, equal to seasonal persistence",
        "dropped_short_missing_origin": int((source["ok_short"] & ~usable).sum()),
        "dropped_long_missing_origin": int((source["ok_long"] & ~usable).sum()),
        "holdout_short": persistence(out["y_short"], hold & out["ok_short"], list(meta["horizons_short"])),
        "holdout_long": persistence(out["y_long"], hold & out["ok_long"], list(meta["horizons_long"])),
    }
    (PHASE / "followup" / "sdelta_persistence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("SDELTA_OK")


if __name__ == "__main__":
    main()
