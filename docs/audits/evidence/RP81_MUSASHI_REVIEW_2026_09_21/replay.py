"""Read-only artifact checks plus one fresh inference process per retained RP79 cell.

No training, service changes, new source download or final-holdout access.
Tolerance is the existing runner's allclose(atol=rtol=1e-6), not chosen from replay results.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="1",
                  TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path)
ap.add_argument("--unit")
a = ap.parse_args()


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, a.repo / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


design = json.loads((a.root / "DESIGN.json").read_text())
if a.unit:
    K = load("df_e1_block")
    data = K.load_data(a.root, design)
    rec = json.loads((a.root / "attempts" / a.unit / "cell.json").read_text())
    spec, seed = rec["arm_spec"], int(rec["cell"]["seed"])
    X, assignment = K.arm_inputs(data, spec, assignment=design["source_run"]["graph_assignment"])
    j, h = int(data["target_channel"][0]), int(data["horizon"][0])
    mean, sigma = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    ds = K.Batches(X, data["Y"], data["common_eval"], int(spec["window"]), h, j,
                   int(design["recipe"]["batch"]), mean=mean, sd=sigma, shuffle=False, seed=seed)
    model = K.build_model(spec, assignment, X.shape[1], j, seed)
    model.load_weights(a.root / "attempts" / a.unit / "weights.weights.h5")
    pred = K.predict(model, ds).astype(np.float64) * sigma + mean
    with np.load(a.root / "attempts" / a.unit / "arrays.npz", allow_pickle=False) as z:
        stored, y = z["pred"], z["y"]
    print(json.dumps({"unit": a.unit, "allclose_1e_6": bool(np.allclose(pred, stored, atol=1e-6, rtol=1e-6)),
        "max_abs_prediction_difference": float(np.max(np.abs(pred - stored))),
        "mae_z_replayed": float(np.mean(np.abs(pred - y)) / sigma),
        "mae_z_stored": float(np.mean(np.abs(stored - y)) / sigma)}))
    raise SystemExit(0)

published = a.repo / "docs/audits/evidence/d3_k5_20260917/RP74/blocks/e1_block_arch_x_calendar_v1"
with np.load(a.root / "BLOCK_DATA.npz", allow_pickle=False) as z:
    Y, origins, h = z["Y"], z["common_eval"], int(z["horizon"][0])
    sigma = float(z["scaler_sd"][int(z["target_channel"][0])])
rows = []
for cell in design["cells"]:
    unit = cell["cell_id"]
    folder = a.root / "attempts" / unit
    terminal = json.loads((published / "TERMINALS" / f"{unit}.json").read_text())
    arts = {x["role"]: x for x in terminal["artifacts"]}
    problems = []
    for role, name in (("predictions", "arrays.npz"), ("record", "cell.json"), ("weights", "weights.weights.h5")):
        b = (folder / name).read_bytes()
        if hashlib.sha256(b).hexdigest() != arts[role]["sha256"] or len(b) != arts[role]["bytes"]:
            problems.append(f"{role}: changed from published terminal")
    with np.load(folder / "arrays.npz", allow_pickle=False) as z:
        for key, expected in (("origins", origins), ("y", Y[origins + h]), ("naive", Y[origins])):
            if not np.array_equal(z[key], expected):
                problems.append(f"{key}: identity mismatch")
        pred, y, naive = [np.asarray(z[k], dtype=np.float64) for k in ("pred", "y", "naive")]
        if not all(np.isfinite(v).all() for v in (pred, y, naive)):
            problems.append("nonfinite")
        mae = float(np.mean(np.abs(pred - y)) / sigma)
        baseline = float(np.mean(np.abs(naive - y)) / sigma)
    process = subprocess.run([sys.executable, __file__, "--repo", str(a.repo), "--root", str(a.root), "--unit", unit],
                              capture_output=True, text=True, timeout=120)
    if process.returncode:
        replay = {"error": process.stderr[-1000:], "returncode": process.returncode}
        problems.append("fresh-process replay failed")
    else:
        replay = json.loads(process.stdout.strip().splitlines()[-1])
        if not replay["allclose_1e_6"]:
            problems.append("fresh-process replay exceeds the existing tolerance")
    row = {**cell, "mae_z": mae, "naive_mae_z": baseline, "skill": 1 - mae / baseline,
           "n": int(origins.size), "replay": replay, "problems": problems}
    rows.append(row)
    print(json.dumps(row), flush=True)
result = {"scope": __doc__, "cells": rows, "live_warehouse_queried": False,
          "all_passed": all(not r["problems"] for r in rows), "n": len(rows)}
a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
