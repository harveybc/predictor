#!/usr/bin/env python3
"""Build-cost probe for a modular candidate (no data, no fit). Label PROBE_NOT_A_RESULT.

Phase ``build``: build_modular on the candidate's exact nested model; records wall
time, whole-cgroup peak (per-fd reset on memory.peak), process RSS, parameter
count, fused width and compression path. Phase ``first_step`` (optional, a
separate process so its peak is separable): compile exactly as the evaluator does
(AdamW + the candidate loss) and run ONE train_on_batch on random tensors of the
candidate's input/target shapes, recording trace/compile wall time and peak, and
the resolved jit_compile. Random tensors: this measures graph cost, not learning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.modular_doin_cost_pilot import CgroupPeak, model_structure  # noqa: E402
from tools.modular_heartbeat import process_memory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--phase", choices=("build", "first_step"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    peak = CgroupPeak()
    record = {"schema": "m04.build_probe.v1", "label": "PROBE_NOT_A_RESULT", "phase": args.phase,
              "candidate": args.candidate, "per_fd_peak_reset": peak.per_fd_reset,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    t0 = time.monotonic()
    import keras
    import numpy as np
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt

    record["environment"] = {"tensorflow": tf.__version__, "keras": keras.__version__, "executable": sys.executable}
    record["import"] = {"seconds": time.monotonic() - t0, "cgroup_peak_bytes": peak.read()}
    peak.reset()
    t1 = time.monotonic()
    bundle = mt.build_modular(candidate["model"])
    record["build"] = {"seconds": time.monotonic() - t1, "cgroup_peak_bytes": peak.read(),
                       "process": process_memory()}
    record["model"] = model_structure(bundle)
    if args.phase == "first_step":
        peak.reset()
        ev = candidate["evaluator"]
        losses = {"huber": lambda: tf.keras.losses.Huber(delta=ev.get("huber_delta", 1.0)),
                  "mae": tf.keras.losses.MeanAbsoluteError, "mse": tf.keras.losses.MeanSquaredError}
        model = bundle.forecast_model
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=ev["learning_rate"],
                                                          weight_decay=ev["weight_decay"]),
                      loss=losses[ev["loss"]]())
        batch = args.batch_size or ev["batch_size"]
        rng = np.random.default_rng(0)
        x = rng.normal(size=(batch, *model.input_shape[1:])).astype("float32")
        y = rng.normal(size=(batch, *model.output_shape[1:])).astype("float32")
        t2 = time.monotonic()
        model.train_on_batch(x, y)
        first = time.monotonic() - t2
        t3 = time.monotonic()
        for _ in range(3):
            model.train_on_batch(x, y)
        record["first_step"] = {"batch_size": batch, "first_step_seconds": first,
                                "steady_step_seconds": (time.monotonic() - t3) / 3,
                                "resolved_jit_compile": str(getattr(model, "jit_compile", None)),
                                "cgroup_peak_bytes": peak.read(), "process": process_memory(),
                                "devices": [d.name for d in tf.config.list_logical_devices()]}
    Path(args.out).write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({k: record.get(k) for k in ("phase", "build", "first_step")} | {"model": {
        k: record["model"][k] for k in ("branches", "concatenated_width_before_core", "total_parameters")}},
        default=str), flush=True)


if __name__ == "__main__":
    main()
