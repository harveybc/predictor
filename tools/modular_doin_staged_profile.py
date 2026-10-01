#!/usr/bin/env python3
"""Staged cost profile of ONE complete modular candidate (no fit to convergence).

Label PROFILE_NOT_A_RESULT. Separates, each with wall time and its own whole-cgroup
peak (per-fd reset of memory.peak) plus the TF device peak when a GPU is pinned:

  import         TensorFlow/Keras import
  construct      build_modular on the candidate's exact nested model (graph construction)
  serialize      full-model save + reload and every donor manifest (branches + core)
  data_load      the evaluator's own NPZ loader on the declared train and validation files
  first_step     compile as the evaluator does (AdamW + candidate loss), first train_on_batch
                 on the first real training batch (trace/compile cost)
  steady         N further real batches (steady per-update cost)
  materialize    branch+fusion forward pass over the full train population in batches
                 (bytes of the fused representation a core pretraining would hold)

The full-population tensors are the candidate's real inputs; nothing is subsampled
except that steady stops after N updates. An unbuffered heartbeat runs throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.modular_doin_cost_pilot import CgroupPeak, gpu_facts, model_structure  # noqa: E402
from tools.modular_heartbeat import Heartbeat, process_memory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steady-steps", type=int, default=20)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(Path(args.candidate).read_text())
    peak = CgroupPeak()
    record = {"schema": "m04.staged_profile.v1", "label": "PROFILE_NOT_A_RESULT", "candidate": args.candidate,
              "per_fd_peak_reset": peak.per_fd_reset, "stages": [], "argv": sys.argv,
              "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TF_DETERMINISTIC_OPS",
                                                              "CUDA_CACHE_MAXSIZE", "M04_HOST_ROLE")}}
    state = {"t": time.monotonic()}
    tf_holder = {}

    def close(stage, **extra):
        now = time.monotonic()
        device = None
        if args.gpu_uuid and "tf" in tf_holder:
            device = tf_holder["tf"].config.experimental.get_memory_info("GPU:0")
            tf_holder["tf"].config.experimental.reset_memory_stats("GPU:0")
        record["stages"].append({"stage": stage, "seconds": now - state["t"], "cgroup_peak_bytes": peak.read(),
                                 "tf_device": device, "process": process_memory(), **extra})
        (out / "PROFILE.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
        peak.reset()
        state["t"] = time.monotonic()

    with Heartbeat(out / "heartbeat.json", interval=30.0, identity={"profile": out.name}) as beat:
        beat.update(stage="import")
        import keras
        import numpy as np
        import tensorflow as tf
        tf_holder["tf"] = tf
        record["environment"].update(tensorflow=tf.__version__, keras=keras.__version__)
        if args.gpu_uuid:
            record["gpu_facts"] = gpu_facts(args.gpu_uuid)
        from predictor_plugins import modular_temporal as mt
        from tools import modular_candidate_evaluator as ev
        close("import")

        beat.update(stage="construct")
        bundle = mt.build_modular(candidate["model"])
        record["model"] = model_structure(bundle)
        close("construct")

        beat.update(stage="serialize")
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            path = Path(tmp) / "full.keras"
            bundle.forecast_model.save(path)
            size = path.stat().st_size
            tf.keras.models.load_model(path, compile=False, safe_mode=True)
            manifests = [bundle.donor_manifest("branch", name) for name in bundle.branch_models]
            manifests.append(bundle.donor_manifest("core"))
        close("serialize", model_bytes=size, manifests=len(manifests))

        beat.update(stage="data_load")
        train = ev._load(args.train, "train", candidate)
        validation = ev._load(args.validation, "validation", candidate)
        x, y = train["windows"], train["targets"]
        close("data_load", train_windows=int(len(x)), validation_windows=int(len(validation["windows"])),
              array_bytes=int(x.nbytes + y.nbytes + validation["windows"].nbytes + validation["targets"].nbytes))

        beat.update(stage="first_step")
        settings = ev._settings(candidate)
        losses = {"huber": lambda: tf.keras.losses.Huber(delta=settings["huber_delta"]),
                  "mae": tf.keras.losses.MeanAbsoluteError, "mse": tf.keras.losses.MeanSquaredError}
        model = bundle.forecast_model
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=settings["learning_rate"],
                                                          weight_decay=settings["weight_decay"]),
                      loss=losses[settings["loss"]]())
        batch = settings["batch_size"]
        model.train_on_batch(x[:batch], y[:batch])
        close("first_step", batch_size=batch)

        beat.update(stage="steady")
        started = time.monotonic()
        steps = 0
        for start in range(batch, len(x), batch):
            if steps >= args.steady_steps:
                break
            model.train_on_batch(x[start:start + batch], y[start:start + batch])
            steps += 1
            beat.update(stage="steady", updates=steps)
        close("steady", updates=steps, seconds_per_update=(time.monotonic() - started) / max(steps, 1),
              updates_per_epoch=-(-len(x) // batch))

        if not args.skip_materialize:
            beat.update(stage="materialize")
            fused = 0
            shape = None
            for start in range(0, len(x), 512):
                part = np.asarray(bundle.fusion_model(x[start:start + 512], training=False))
                fused += part.nbytes
                shape = part.shape[1:]
            close("materialize", fused_float32_bytes=int(fused), fused_shape=list(shape))
    record["cumulative_peak_bytes"] = max(s["cgroup_peak_bytes"] for s in record["stages"])
    record["status"] = "completed"
    (out / "PROFILE.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({s["stage"]: [round(s["seconds"], 2), s["cgroup_peak_bytes"]] for s in record["stages"]}))


if __name__ == "__main__":
    main()
