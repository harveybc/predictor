#!/usr/bin/env python3
"""Complete real-model cost pilot for an M04 modular candidate (full data, full model).

Runs the real evaluator (``evaluate_candidate``) once on the full declared
train/validation NPZ populations for a bounded number of epochs, under an
unbuffered heartbeat, and writes ``PILOT.json`` with:

* the three GPU facts measured INSIDE this child (driver UUID visible via
  nvidia-smi, TensorFlow registration, matmul placement) and the armed
  ``GPU_REQUEST_FELL_BACK_TO_CPU`` raise;
* model structure: concatenated width before fusion, total/trainable parameter
  counts, compression path shapes;
* whole-cgroup ``memory.peak`` and TensorFlow device peak (``get_memory_info``),
  cumulative and per stage (the per-stage cgroup reset writes and reads the SAME
  open file descriptor, cgroup v2 per-fd peak semantics);
* per-update cost from the observed optimizer updates and fit wall time.

The validation metrics are a bounded-epoch pilot, labelled PILOT_NOT_A_RESULT.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.modular_heartbeat import Heartbeat, cgroup_memory, process_memory  # noqa: E402


class GpuFallback(RuntimeError):
    pass


class CgroupPeak:
    """One fd on memory.peak: write resets this fd's peak, read returns it."""

    def __init__(self):
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        self.path = Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.peak"
        self.fd = os.open(self.path, os.O_RDWR)
        self.per_fd_reset = True
        try:
            os.write(self.fd, b"reset\n")
        except OSError:
            self.per_fd_reset = False

    def read(self):
        return int(os.pread(self.fd, 64, 0).decode().strip())

    def reset(self):
        if self.per_fd_reset:
            os.write(self.fd, b"reset\n")


def gpu_facts(expected_uuid):
    facts = {"expected_uuid": expected_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "fallback_raise_armed": True}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,temperature.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    facts["driver_visible"] = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    facts["driver_uuid_ok"] = any(expected_uuid in line for line in facts["driver_visible"])
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    facts["tf_version"] = tf.__version__
    facts["tf_registered"] = [{"name": g.name, **{k: str(v) for k, v in
                              tf.config.experimental.get_device_details(g).items()}} for g in gpus]
    if not gpus or not facts["driver_uuid_ok"]:
        raise GpuFallback("GPU_REQUEST_FELL_BACK_TO_CPU: TensorFlow registered no GPU or the driver "
                          f"does not expose {expected_uuid}")
    with tf.device("/GPU:0"):
        a = tf.ones((1024, 1024))  # no random op: valid under TF_DETERMINISTIC_OPS=1
        b = tf.linalg.matmul(a, a)
    facts["op_placement"] = b.device
    if "GPU" not in b.device:
        raise GpuFallback(f"GPU_REQUEST_FELL_BACK_TO_CPU: matmul placed on {b.device}")
    facts["tf_memory_after_probe"] = tf.config.experimental.get_memory_info("GPU:0")
    return facts


def model_structure(bundle):
    fused = bundle.fusion_model.output_shape
    core = []
    for layer in bundle.core_model.layers:
        name = layer.name
        if name.startswith(("stage_", "model_projection", "positional_encoding")):
            core.append({"layer": name, "output_shape": list(layer.output.shape[1:])})
    model = bundle.forecast_model
    trainable = sum(int(w.shape.num_elements()) for w in model.trainable_weights)
    return {"branches": len(bundle.branch_models),
            "branch_output_shapes": sorted({str(m.output_shape[1:]) for m in bundle.branch_models.values()}),
            "concatenated_width_before_core": int(fused[-1]), "fused_shape": list(fused[1:]),
            "compression_path": core, "latent_shape": list(bundle.encoder_model.output_shape[1:]),
            "forecast_shape": list(model.output_shape[1:]),
            "total_parameters": int(model.count_params()), "trainable_parameters": trainable,
            "branch_parameters": int(sum(m.count_params() for m in bundle.branch_models.values())),
            "core_parameters": int(bundle.core_model.count_params())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True, help="nested modular.candidate.v1 JSON")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    peak = CgroupPeak()
    receipt = {"schema": "m04.cost_pilot.v1", "label": "PILOT_NOT_A_RESULT", "started": started,
               "argv": sys.argv, "stages": [], "cgroup_peak_file": str(peak.path),
               "per_fd_peak_reset": peak.per_fd_reset}
    import tensorflow as tf

    if args.gpu_uuid:
        receipt["gpu_facts"] = gpu_facts(args.gpu_uuid)
    else:
        receipt["gpu_facts"] = {"cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    gpu = bool(args.gpu_uuid)

    def device_peak():
        if not gpu:
            return None
        return tf.config.experimental.get_memory_info("GPU:0")

    state = {"stage": "setup", "t": time.monotonic()}

    def close_stage(next_stage):
        now = time.monotonic()
        receipt["stages"].append({"stage": state["stage"], "seconds": now - state["t"],
                                  "cgroup_peak_bytes": peak.read(), "tf_device": device_peak()})
        peak.reset()
        if gpu:
            tf.config.experimental.reset_memory_stats("GPU:0")
        state.update(stage=next_stage, t=now)

    import predictor_plugins.modular_temporal as mt
    from tools import modular_candidate_evaluator as evaluator

    original = mt.build_modular

    def capturing_build(config):
        bundle = original(config)
        receipt["model"] = model_structure(bundle)
        return bundle

    mt.build_modular = capturing_build
    candidate = json.loads(Path(args.candidate).read_text())
    candidate["evaluator"]["max_epochs"] = args.epochs
    candidate["evaluator"]["patience"] = max(candidate["evaluator"]["patience"], args.epochs)
    receipt["candidate"] = candidate
    import keras
    import numpy
    receipt["environment"] = {"tensorflow": tf.__version__, "keras": keras.__version__, "numpy": numpy.__version__,
                              "executable": sys.executable, "host_role": os.environ.get("M04_HOST_ROLE"),
                              "device_pin": "CUDA_VISIBLE_DEVICES environment (no launcher GPU option)"}
    with Heartbeat(out / "heartbeat.json", interval=30.0, identity={"pilot": out.name}) as beat:
        def progress(**fields):
            stage = fields.get("stage")
            coarse = "fit" if stage in ("fit", "validated") else stage
            if coarse != state["stage"]:
                close_stage(coarse)
            beat.update(**fields)

        try:
            result = evaluator.evaluate_candidate(candidate, args.train, args.validation, out / "candidate",
                                                  progress=progress)
        except Exception as exc:
            close_stage("failed")
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (out / "PILOT.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
            raise
    close_stage("end")
    training = result["training"]
    fit = [s for s in receipt["stages"] if s["stage"] == "fit"]
    receipt.update(
        status="completed", wall_seconds=time.time() - started,
        cumulative={"cgroup": cgroup_memory(), "process": process_memory(),
                    "stage_max_cgroup_peak_bytes": max(s["cgroup_peak_bytes"] for s in receipt["stages"]),
                    "stage_max_tf_device_peak_bytes": max(((s["tf_device"] or {}).get("peak", 0)
                                                           for s in receipt["stages"]), default=None)},
        cost={"observed_updates": training["observed_updates"], "epochs_completed": training["epochs_completed"],
              "fit_elapsed_seconds": training["elapsed_seconds"],
              "seconds_per_update_including_validation": training["elapsed_seconds"] / training["observed_updates"],
              "fit_stage_seconds": sum(s["seconds"] for s in fit),
              "updates_per_epoch": training["observed_updates"] // max(1, training["epochs_completed"]),
              "batch_size": training["settings"]["batch_size"]},
        pilot_metrics_PILOT_NOT_A_RESULT={"objective": result["objective"], "metrics": result["metrics"],
                                          "stop_reason": training["stop_reason"],
                                          "selected_epoch": training["selected_epoch"]},
        evaluation_receipt=str(out / "candidate" / "evaluation.json"),
        digests=result["digests"], data=result["data"])
    (out / "PILOT.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({"status": "completed", "cost": receipt["cost"], "cumulative": receipt["cumulative"],
                      "model": receipt["model"]}, default=str))


if __name__ == "__main__":
    main()
