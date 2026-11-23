#!/usr/bin/env python
"""Count 72h+ labels under the same-segment mask. Does not train and does not refit STL.

The signed rule stays: a horizon is usable only when the cumulative prefix
has at least 1575 train rows and 200 monitor rows. This script does not relax it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

RUNTIME = Path("/var/tmp/sac399-series/runtime")
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, "/var/tmp/retsu-phase1")

import export_dataset as exp  # noqa: E402

PHASE = Path("/var/tmp/retsu-phase1")
LONG = (24, 48, 72, 96, 120, 144)
MIN_POOL = 1575
MIN_MONITOR = 200


def main() -> None:
    bundle = exp.ROOT / "model/61/bundle"
    joined, real = exp.assemble(exp.PARENT, bundle, exp.ROOT / "history397-61-r2")
    risk = json.loads((exp.PARENT / "RISK.json").read_text())
    bounds = {
        **exp.segment_bounds(joined, risk, "2016-01-01", "2024-01-01"),
        **exp.segment_bounds(real, risk, "2024-01-01", "2025-01-01"),
    }
    series = np.load(PHASE / "stl_series.npz")
    avail_ns = series["avail_ns"].astype(np.int64)
    seasonal = series["seasonal"].astype(np.float64)
    by_avail = {int(ts): i for i, ts in enumerate(avail_ns)}
    data = np.load(PHASE / "dataset.npz")
    ts = data["ts"].astype(np.int64)
    pool = data["pool"]
    monitor = data["monitor"]
    holdout = data["holdout"]
    y = np.full((len(ts), len(LONG)), np.nan, np.float64)
    for i, origin in enumerate(ts):
        segment = bounds.get(int(origin))
        if segment is None:
            continue
        op, ff = segment
        for k, horizon in enumerate(LONG):
            label_ts = int(origin) + horizon * exp.HOUR
            if not (op <= label_ts < ff) or bounds.get(label_ts) != segment:
                continue
            j = by_avail.get(label_ts)
            if j is None or not np.isfinite(seasonal[j]):
                continue
            y[i, k] = seasonal[j]
    rows = []
    admitted = 0
    for k, horizon in enumerate(LONG):
        joint = np.isfinite(y[:, : k + 1]).all(axis=1)
        alone = np.isfinite(y[:, k])
        item = {
            "horizon": int(horizon),
            "alone_pool": int((alone & pool).sum()),
            "alone_monitor": int((alone & monitor).sum()),
            "alone_holdout": int((alone & holdout).sum()),
            "prefix_pool": int((joint & pool).sum()),
            "prefix_monitor": int((joint & monitor).sum()),
            "prefix_holdout": int((joint & holdout).sum()),
        }
        item["prefix_admitted"] = item["prefix_pool"] >= MIN_POOL and item["prefix_monitor"] >= MIN_MONITOR
        if item["prefix_admitted"]:
            admitted = k + 1
        rows.append(item)
    report = {
        "rule": "same segment, cumulative prefix, pool>=1575 and monitor>=200",
        "rule_relaxed": False,
        "trained": False,
        "admitted_prefix": [int(h) for h in LONG[:admitted]],
        "not_evaluable": [int(row["horizon"]) for row in rows if not row["prefix_admitted"]],
        "horizons": rows,
    }
    path = PHASE / "followup" / "horizons_72.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("HORIZONS_72_COUNTED")


if __name__ == "__main__":
    main()
