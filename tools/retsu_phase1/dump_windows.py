#!/usr/bin/env python
"""Rebuild the 13x61 windows for the saved dataset rows and check the CPU encoder.

Does not refit STL and does not overwrite dataset.npz. Aborts if the
recomputed embedding disagrees with the saved CPU embedding.
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
OUT = PHASE / "windows.npz"


def main() -> None:
    bundle = exp.ROOT / "model/61/bundle"
    joined, real = exp.assemble(exp.PARENT, bundle, exp.ROOT / "history397-61-r2")
    risk = json.loads((exp.PARENT / "RISK.json").read_text())
    rows_j, avail_j, _ = exp.strict_table(joined, risk, "2016-01-01", "2024-01-01")
    rows_r, avail_r, _ = exp.strict_table(real, risk, "2024-01-01", "2025-01-01")
    windows = np.concatenate([
        np.ascontiguousarray(joined.inputs.windows[rows_j]),
        np.ascontiguousarray(real.inputs.windows[rows_r]),
    ]).astype(np.float32)
    avail = np.concatenate([avail_j, avail_r]).astype(np.int64)
    order = np.argsort(avail, kind="mergesort")
    windows = windows[order]
    avail = avail[order]
    saved = np.load(PHASE / "dataset.npz")
    if windows.shape[0] != saved["ts"].shape[0] or not np.array_equal(avail, saved["ts"]):
        raise SystemExit("WINDOW_ROWS_DO_NOT_MATCH_DATASET")
    if windows.ndim != 3 or windows.shape[1:] != (13, 61):
        raise SystemExit(f"WINDOW_SHAPE {windows.shape}")
    net, _ = exp.load_bundle(bundle, "cpu")
    # Encode in place through the same function the dataset used, on the
    # unsorted row ids, then apply the same order.
    z_j = exp.encode(joined, rows_j, net, "cpu")
    z_r = exp.encode(real, rows_r, net, "cpu")
    z = np.concatenate([z_j, z_r])[order]
    gap = float(np.max(np.abs(z - saved["z"])))
    if gap > 1e-3:
        raise SystemExit(f"WINDOW_ENCODER_GAP {gap}")
    np.savez_compressed(OUT, windows=windows, ts=avail, encoder_max_abs=np.asarray(gap))
    print(json.dumps({"rows": int(len(avail)), "shape": list(windows.shape), "encoder_max_abs": gap}))
    print("WINDOWS_OK")


if __name__ == "__main__":
    main()
