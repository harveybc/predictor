#!/usr/bin/env python3
"""Independent checkpoint scoring of a modular candidate receipt (no training).

Given a completed ``modular.candidate.evaluation.v1`` receipt, this scorer runs
in a fresh process, re-hashes the saved ``best.keras`` bytes and the validation
NPZ, loads the model with ``compile=False, safe_mode=True`` (no optimizer, no
fit code path), predicts every validation row, and recomputes MAE/MSE,
persistence and skills with its own float64 arithmetic. It never imports the
evaluator's fit loop and never reads a train or test array.

Verdict ``VERIFIED`` requires: model/validation digests equal the receipt;
prediction shape equals targets; every metric within ``rtol``/``atol`` of the
receipt; objective recomputed equal to the receipt's objective. Anything else
is ``REFUTED`` with the named mismatch. The output is a separate receipt
(``modular.checkpoint.verification.v1``) with its own digests.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "modular.checkpoint.verification.v1"
VALIDATION_FIELDS = {"windows", "targets", "row_ids", "timestamps", "target_timestamps", "dataset_id",
                     "split", "feature_names", "target_names", "horizons", "timestamp_unit",
                     "metric_space", "scaler_identity", "scaler_scale"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cgroup_resources():
    """Whole-cgroup memory of this verification child (for its declared cap)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.modular_heartbeat import cgroup_memory, process_memory
        return {"cgroup": cgroup_memory(), "process": process_memory()}
    except Exception as exc:  # resource reading must never change a verdict
        return {"error": str(exc)}


def _load_validation(path):
    raw = Path(path).read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != VALIDATION_FIELDS:
            raise ValueError("validation NPZ fields differ from the evaluator contract")
        data = {k: archive[k] for k in VALIDATION_FIELDS}
    if str(data["split"]) != "validation":
        raise ValueError("scorer only accepts the validation split")
    return data, _sha(raw)


def score_metrics(y, prediction, baseline):
    y = np.asarray(y, dtype=np.float64)
    e = np.asarray(prediction, dtype=np.float64) - y
    n = np.asarray(baseline, dtype=np.float64) - y
    mae, mse = float(np.mean(np.abs(e))), float(np.mean(e * e))
    bmae, bmse = float(np.mean(np.abs(n))), float(np.mean(n * n))
    return {"MAE": mae, "MSE": mse, "baseline_MAE": bmae, "baseline_MSE": bmse,
            "skill_MAE": 1 - mae / bmae if bmae else None, "skill_MSE": 1 - mse / bmse if bmse else None}


def _close(a, b, rtol, atol):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * abs(b)


def verify(receipt_path, validation_path, output_path, *, batch_size=512, rtol=1e-5, atol=1e-6,
           target_feature_indices=None):
    started = time.monotonic()
    receipt = json.loads(Path(receipt_path).read_text())
    problems = []
    if receipt.get("schema_version") != "modular.candidate.evaluation.v1" or receipt.get("status") != "completed":
        raise ValueError("a completed modular.candidate.evaluation.v1 receipt is required")
    artifact = Path(receipt["artifacts"]["best_model"])
    model_sha = _sha_file(artifact)
    if model_sha != receipt["digests"]["model_sha256"]:
        problems.append("model_sha256 mismatch")
    data, validation_sha = _load_validation(validation_path)
    if validation_sha != receipt["digests"]["validation_sha256"]:
        problems.append("validation_sha256 mismatch")
    x = data["windows"].astype(np.float32)
    y = data["targets"].astype(np.float64)
    names = data["feature_names"].astype(str).tolist()
    targets = data["target_names"].astype(str).tolist()
    if target_feature_indices is None:
        target_feature_indices = [names.index(t) for t in targets]
    if [names[i] for i in target_feature_indices] != targets:
        raise ValueError("target_feature_indices disagree with target names")

    import tensorflow as tf  # imported only after every cheap check
    from predictor_plugins import modular_temporal  # noqa: F401  registers serializable layers

    model = tf.keras.models.load_model(artifact, compile=False, safe_mode=True)
    if getattr(model, "optimizer", None) is not None:
        problems.append("loaded model carries an optimizer; scoring must be inference-only")
    chunks = []
    for start in range(0, len(x), batch_size):
        chunks.append(np.asarray(model(x[start:start + batch_size], training=False)))
    prediction = np.concatenate(chunks)
    if prediction.shape != y.shape:
        raise ValueError(f"prediction shape {prediction.shape} != targets {y.shape}")
    if not np.all(np.isfinite(prediction)):
        problems.append("nonfinite predictions")
    horizons = data["horizons"].tolist()
    baseline = np.repeat(x[:, -1:, target_feature_indices], len(horizons), axis=1)
    metrics = score_metrics(y, prediction, baseline)
    per_horizon = {str(h): score_metrics(y[:, i:i + 1], prediction[:, i:i + 1], baseline[:, i:i + 1])
                   for i, h in enumerate(horizons)}
    for key, value in metrics.items():
        if not _close(value, receipt["metrics"].get(key), rtol, atol):
            problems.append(f"metric {key}: rescored {value} vs receipt {receipt['metrics'].get(key)}")
    for h, row in per_horizon.items():
        ref = receipt.get("per_horizon", {}).get(h, {})
        if not _close(row["MAE"], ref.get("MAE"), rtol, atol):
            problems.append(f"per-horizon MAE h{h}: rescored {row['MAE']} vs receipt {ref.get('MAE')}")
    objective = dict(receipt["objective"])
    rescored_objective = metrics[objective["metric"]]
    if not _close(rescored_objective, objective["value"], rtol, atol):
        problems.append("objective value mismatch")
    result = {"schema": SCHEMA, "verdict": "REFUTED" if problems else "VERIFIED", "problems": problems,
              "method": "fresh-process checkpoint inference; compile=False, safe_mode=True; no fit, "
                        "no train or test arrays; float64 metric arithmetic independent of the evaluator",
              "receipt_path": str(Path(receipt_path).resolve()),
              "receipt_sha256": _sha_file(receipt_path),
              "digests": {"model_sha256": model_sha, "validation_sha256": validation_sha,
                          "predictions_sha256": _sha(np.ascontiguousarray(prediction.astype("<f4")).tobytes())},
              "objective": {**{k: objective[k] for k in ("metric", "split", "higher_is_better", "unit")},
                            "rescored_value": rescored_objective, "receipt_value": objective["value"]},
              "metrics": metrics, "per_horizon": per_horizon, "validation_rows": int(len(x)),
              "tolerance": {"rtol": rtol, "atol": atol},
              "elapsed_seconds": time.monotonic() - started, "pid": os.getpid(),
              "resources": _cgroup_resources(),
              "environment": {"host_role": os.environ.get("M04_HOST_ROLE"), "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                              "numpy": np.__version__, "executable": sys.executable,
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    Path(output_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    result = verify(args.receipt, args.validation, args.output, batch_size=args.batch_size)
    print(json.dumps({"verdict": result["verdict"], "problems": result["problems"],
                      "objective": result["objective"]}))
    sys.exit(0 if result["verdict"] == "VERIFIED" else 3)


if __name__ == "__main__":
    main()
