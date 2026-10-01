#!/usr/bin/env python3
"""Admitted training child for one lane F2 cell (runs under crispdm-run on the worker).

Trains the nested candidate (a modular ``modular.candidate.v1`` through
``evaluate_candidate``, or a ``control`` config through ``evaluate_control``) on the
declared train/validation NPZ, under an unbuffered heartbeat (<= 30 s) in ``--out``, and
writes ``accepted.json``: the evaluation receipt plus cell identity (cid, label,
config_id), the pinned predictor revision and host role, the environment, the GPU facts
measured inside this process (driver UUID, TensorFlow registration, matmul placement;
``GPU_REQUEST_FELL_BACK_TO_CPU`` is a hard failure when a GPU was requested), the
same-row naive controls per horizon (persistence, zero-return, train-mean, seasonal) and
the whole-cgroup peak. The float32 validation predictions are saved beside the model with
their sha256 so the independent scorer's ``predictions_sha256`` can bind the exported CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.modular_heartbeat import Heartbeat, cgroup_memory, environment, process_memory, progress_adapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--cid", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--manifest", default=None, help="data MANIFEST.json (mu/sigma for the zero-return naive)")
    parser.add_argument("--seasonal-period", type=int, default=6)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    candidate = json.loads(Path(args.candidate).read_text())
    identity = {"campaign_id": args.campaign_id, "cid": args.cid, "label": args.label,
                "host_role": os.environ.get("F2_HOST_ROLE") or os.environ.get("M04_HOST_ROLE"),
                "revision": args.revision, "dataset": Path(args.train).parent.name}
    import subprocess
    try:
        runner_commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                       capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        runner_commit = None
    receipt_extra = {"runner_commit": runner_commit, "candidate": {"cid": args.cid, "label": args.label, "config_id": args.config_id,
                                   "nested_sha256": hashlib.sha256(json.dumps(candidate, sort_keys=True,
                                                                              separators=(",", ":")).encode()).hexdigest()},
                     "bridge": {"predictor_revision": args.revision, "host_role": identity["host_role"],
                                "executor": "f2.local_crispdm", "pid": os.getpid()}}
    with Heartbeat(out / "heartbeat.json", interval=args.heartbeat_interval, identity=identity) as beat:
        beat.update(stage="load")
        from tools.modular_doin_cost_pilot import gpu_facts

        import tensorflow as tf  # noqa: F401

        if args.gpu_uuid:
            facts = gpu_facts(args.gpu_uuid)  # raises GpuFallback: never silently on CPU
        else:
            facts = {"cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        beat.update(stage="build", gpu=facts.get("op_placement"))
        adapt = progress_adapter(beat)
        if "control" in candidate:
            from tools.eth_control_evaluator import evaluate_control
            result, prediction = evaluate_control(candidate, args.train, args.validation, out / "candidate",
                                                  progress=adapt)
        else:
            from tools import modular_candidate_evaluator as evaluator
            result = evaluator.evaluate_candidate(candidate, args.train, args.validation, out / "candidate",
                                                  progress=adapt)
            prediction = _repredict(result, args.validation)
        beat.update(stage="score")
        pred32 = np.ascontiguousarray(np.asarray(prediction, dtype="<f4"))
        np.save(out / "candidate" / "predictions_validation.npy", pred32)
        result["digests"]["predictions_sha256"] = hashlib.sha256(pred32.tobytes()).hexdigest()
        result["artifacts"]["predictions_validation"] = str((out / "candidate" / "predictions_validation.npy").resolve())
        naives = None
        if args.manifest:
            from tools.eth_forecast_naives import _load_validation, _manifest_scaler, naive_table
            data, _ = _load_validation(args.validation)
            mu, sigma = _manifest_scaler(args.manifest, data)
            naives = naive_table(data, mu, sigma, args.seasonal_period)
            naives["model_per_horizon_MAE"] = {h: result["per_horizon"][h]["MAE"] for h in result["per_horizon"]}
            naives["beats_zero_return"] = {h: result["per_horizon"][h]["MAE"] < naives["per_naive"]["zero_return"][h]["MAE"]
                                           for h in result["per_horizon"]}
        receipt = {**result, **receipt_extra, "environment": {**environment(), "gpu_facts": facts},
                   "naives": naives, "label": "DEVELOPMENT",
                   "cost": {"wall_seconds": time.time() - started, "cgroup": cgroup_memory(), "process": process_memory()}}
        (out / "accepted.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
        if "control" not in candidate and any(c.get("donor") for c in [*candidate["model"]["branches"], candidate["model"]["core"]]):
            beat.update(stage="frozen_check")
            from tools.f2_frozen_check import check as frozen_check
            receipt["frozen_check"] = frozen_check(candidate, result["artifacts"]["best_model"])
            (out / "accepted.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
        beat.update(stage="save")
    print(json.dumps({"status": "completed", "objective": receipt["objective"]["value"],
                      "per_horizon_MAE": naives["model_per_horizon_MAE"] if naives else None,
                      "beats_zero_return": naives["beats_zero_return"] if naives else None,
                      "selected_epoch": receipt["training"]["selected_epoch"],
                      "observed_updates": receipt["training"]["observed_updates"],
                      "cgroup_peak_bytes": receipt["cost"]["cgroup"].get("peak_bytes")}))


def _repredict(result, validation_path):
    """The modular evaluator returns no predictions; recompute them from the saved model at the receipt batch."""
    import io

    import tensorflow as tf
    from predictor_plugins import modular_temporal  # noqa: F401
    from tools.modular_candidate_evaluator import _predict

    raw = Path(validation_path).read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        x, y = z["windows"].astype(np.float32), z["targets"].astype(np.float32)
    model = tf.keras.models.load_model(result["artifacts"]["best_model"], compile=False, safe_mode=True)
    return _predict(model, x, y, result["training"]["settings"]["batch_size"])


if __name__ == "__main__":
    main()
