#!/usr/bin/env python3
"""Lane H: lane B's Kalman gate on the ETH 4h panel (log close, TRAIN rows only). DEVELOPMENT, CPU."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eth_run = _load("h_kalman_eth_run")
gate = _load("h_kalman_gate_laneb")
eth = eth_run.eth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True)
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, c0 = time.time(), time.process_time()
    fm = json.loads(Path(a.feature_manifest).read_text())
    d = eth_run.load_data(a.view, list(fm["features"]), out / "data", expected_sha=eth.VIEW_SHA256)
    names, times, values, sha = eth.read_view(a.view, eth.VIEW_SHA256)
    n_train = d["train_rows"][1]
    close = values[:n_train, names.index("CLOSE")]
    lc = np.log(close)
    res = gate.lane_b_gate(lc, d["origins"]["train"], d["Y"]["train"], n_train, d["horizons"], d["mu"], d["sigma"], purge=30)
    res.update({"role": a.role, "view_sha256": sha, "train_rows": [0, n_train], "validation_rows_used": False,
                "costs": {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}})
    (out / "GATE_LANE_B_ETH.json").write_text(json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passes": {h: v["passes_all_folds"] for h, v in res["per_horizon"].items()},
                      "mean_model_MAE": {h: v["mean_model_MAE"] for h, v in res["per_horizon"].items()},
                      "mean_zero": {h: v["mean_zero_return_MAE"] for h, v in res["per_horizon"].items()},
                      "mean_intercept": {h: v["mean_intercept_MAE"] for h, v in res["per_horizon"].items()}}))


if __name__ == "__main__":
    main()
