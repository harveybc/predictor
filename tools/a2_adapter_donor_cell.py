#!/usr/bin/env python3
"""Lane A2: ONE single-seed modular cell whose branch is the typed TRAIN-only NPZ adapter donor (R1).

Data: ETH 4h long (eth4h_long_l24_h6to36_v1), all 83 features in one ``npz_adapter_conv`` branch,
horizons 6..36. The adapter output (feature-extractor 6fd7601, already executed on this TRAIN NPZ) is
converted by :func:`predictor_plugins.modular_temporal.adapter_donor.convert_adapter_donor`, which
refuses it unless its row-id digest and input digest are the TRAIN split's. The core is R0 with the
F2 long-campaign core (transformer_conv, residual time factors [2, 2, 1]); evaluator settings are the
F2 base candidate's, seed 2021. Only train.npz and validation.npz are opened; no test rows exist in
either. Writes ``<out>/RESULT.json``: per-horizon validation MAE beside the same-row naives of the
supplied naive table, skill = 1 - MAE_model / MAE_naive, the R1 frozen check (branch bytes in
best.keras == donor bytes), GPU facts and the cgroup/process peak. Label DEVELOPMENT (one seed).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cgroup_peak():
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
        peak = Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.peak"
        return int(peak.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def candidate_config(base, feature_names, donor_path, seed):
    c = copy.deepcopy(base)
    c["evaluator"]["seed"] = seed
    m = c["model"]
    m["feature_names"] = list(feature_names)
    m["branches"] = [{"name": "branch_0", "features": list(feature_names), "plugin": "npz_adapter_conv",
                      "params": {"filters": 32, "kernel_size": 3}, "regime": "R1", "donor": str(donor_path)}]
    m["core"]["regime"], m["core"]["donor"] = "R0", None
    m.pop("regime", None)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--base-candidate", required=True)
    p.add_argument("--naive-table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--gpu-uuid", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import numpy as np
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt  # noqa: F401  (registers engine layers)
    from predictor_plugins.modular_temporal.adapter_donor import convert_adapter_donor
    from tools import modular_candidate_evaluator as evaluator

    gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    if a.gpu_uuid and not gpus:
        raise SystemExit("GPU_REQUEST_FELL_BACK_TO_CPU")
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        placement = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2))).device
    with np.load(a.train, allow_pickle=False) as z:
        features = [str(f) for f in z["feature_names"].tolist()]
    base = json.loads(Path(a.base_candidate).read_text())
    donor = (out / "donor" / "branch_0.keras").resolve()
    cand = candidate_config(base, features, donor, a.seed)
    conversion = convert_adapter_donor(a.adapter_dir, a.train, cand["model"], "branch_0", donor)
    (out / "candidate.json").write_text(json.dumps(cand, indent=1, sort_keys=True) + "\n")
    result = evaluator.evaluate_candidate(cand, a.train, a.validation, out / "candidate")
    best = tf.keras.models.load_model(result["artifacts"]["best_model"], compile=False, safe_mode=True)
    donor_w = [w.tobytes() for w in tf.keras.models.load_model(donor, compile=False).get_weights()]
    frozen = [w.tobytes() for w in best.get_layer("branch_0").get_weights()] == donor_w
    table = json.loads(Path(a.naive_table).read_text())
    horizons = [str(h) for h in cand["model"]["horizons"]]
    if table["rows"] != result["data"]["validation_rows"] or [str(h) for h in table["horizons"]] != horizons:
        raise SystemExit(f"NAIVE_ROWS_MISMATCH: table {table['rows']} rows vs evaluated {result['data']['validation_rows']}")
    rows = {}
    for h in horizons:
        mae = result["per_horizon"][h]["MAE"]
        naives = {k: v[h]["MAE"] for k, v in table["per_naive"].items() if v[h].get("MAE") is not None}
        rows[h] = {"MAE": mae, "naive_MAE": naives,
                   "skill": {k: 1.0 - mae / v for k, v in naives.items()},
                   "beats_every_naive": all(mae < v for v in naives.values())}
    mean_mae = float(np.mean([rows[h]["MAE"] for h in horizons]))
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    document = {
        "schema": "a2.adapter_donor_cell.v1", "label": "DEVELOPMENT", "seed": a.seed, "regime": "R1 branch / R0 core",
        "predictor_commit": commit, "candidate_sha256": _sha(out / "candidate.json"),
        "inputs": {"train_sha256": result["digests"]["train_sha256"],
                   "validation_sha256": result["digests"]["validation_sha256"],
                   "naive_table": str(Path(a.naive_table).resolve()), "naive_table_sha256": _sha(a.naive_table),
                   "test_used": result["data"]["test_used"]},
        "conversion": conversion, "r1_frozen_bit_identical": frozen,
        "validation": {"rows": result["data"]["validation_rows"], "mean_over_horizons_MAE": mean_mae,
                       "objective": result["objective"], "per_horizon": rows},
        "training": {k: result["training"].get(k) for k in ("selected_epoch", "epochs_completed", "stop_reason",
                                                             "observed_updates", "best_validation_loss")},
        "gpu": {"requested_uuid": a.gpu_uuid, "devices": gpus, "matmul_device": placement},
        "cost": {"wall_seconds": time.time() - started, "cgroup_peak_bytes": _cgroup_peak(),
                 "process_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "evaluation": str(Path(result["artifacts"]["best_model"]).parent / "evaluation.json")}
    (out / "RESULT.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"mean_MAE": mean_mae, "frozen": frozen, "per_horizon_MAE": {h: rows[h]["MAE"] for h in horizons},
                      "peak": document["cost"]}))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
