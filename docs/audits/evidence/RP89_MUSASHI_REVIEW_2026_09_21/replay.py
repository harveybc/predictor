"""RP89: independently anchor six new cells and replay each in a fresh process.

Uses the already published RP81 read-only inference probe; no training or services.
Checks accepted preparation data and record as well as forecast/weights/record bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
published = a.repo / "docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_context_daily_lag_v1"
probe = a.repo / "docs/audits/evidence/RP81_MUSASHI_REVIEW_2026_09_21/replay.py"


def check_files(terminal, mapping):
    arts = {art["role"]: art for art in terminal["artifacts"]}
    problems = []
    for role, path in mapping.items():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != arts[role]["sha256"] or len(data) != arts[role]["bytes"]:
            problems.append(f"{role}: byte identity differs from published terminal")
    return problems


prep = json.loads((published / "TERMINALS/prepare.json").read_text())
prep_problems = check_files(prep, {"data": a.root / "BLOCK_DATA.npz", "record": a.root / "BLOCK_DATA.json"})
design = json.loads((published / "DESIGN.json").read_text())
with np.load(a.root / "BLOCK_DATA.npz", allow_pickle=False) as z:
    Y, origins, h = z["Y"], z["common_eval"], int(z["horizon"][0])
    sigma = float(z["scaler_sd"][int(z["target_channel"][0])])
assert sigma == design["benchmark_contract"]["source"]["sd_train"]
rows = []
for cell in design["cells"]:
    unit = cell["cell_id"]
    folder = a.root / "attempts" / unit
    terminal = json.loads((published / "TERMINALS" / f"{unit}.json").read_text())
    problems = check_files(terminal, {"predictions": folder / "arrays.npz", "record": folder / "cell.json", "weights": folder / "weights.weights.h5"})
    with np.load(folder / "arrays.npz", allow_pickle=False) as z:
        for key, expected in (("origins", origins), ("y", Y[origins + h]), ("naive", Y[origins])):
            if not np.array_equal(z[key], expected):
                problems.append(f"{key}: identity mismatch")
        pred, y, naive = (np.asarray(z[k], dtype=np.float64) for k in ("pred", "y", "naive"))
        if not all(np.isfinite(v).all() for v in (pred, y, naive)):
            problems.append("nonfinite")
        mae, baseline = float(np.mean(np.abs(pred - y)) / sigma), float(np.mean(np.abs(naive - y)) / sigma)
    process = subprocess.run([sys.executable, str(probe), "--repo", str(a.repo), "--root", str(a.root), "--unit", unit],
                             capture_output=True, text=True, timeout=120,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
    if process.returncode:
        replay = {"error": process.stderr[-1000:]}
        problems.append("fresh-process replay failed")
    else:
        replay = json.loads(process.stdout.strip().splitlines()[-1])
        if not replay["allclose_1e_6"]:
            problems.append("replay exceeds existing combined tolerance")
    rows.append({**cell, "mae_z": mae, "naive_mae_z": baseline, "skill": 1 - mae / baseline, "replay": replay, "problems": problems})
    print(json.dumps(rows[-1]), flush=True)
out = {"scope": __doc__, "preparation_problems": prep_problems, "cells": rows,
       "all_passed": not prep_problems and all(not r["problems"] for r in rows), "live_warehouse_queried": False}
a.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
