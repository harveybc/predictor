#!/usr/bin/env python3
"""Prove, in process, which physical GPU TensorFlow sees under CUDA_VISIBLE_DEVICES=<UUID>.

Prints one JSON document. With ``--expect-uuid`` it exits 3 unless exactly one GPU is
visible and the environment pinned that UUID; with ``--expect-none`` it exits 3 unless
TensorFlow sees no GPU (the control that proves a wrong UUID cannot silently fall back).
Run it under crispdm-run; it allocates one small matmul on the device.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-uuid")
    parser.add_argument("--expect-none", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: WPS433 (deliberately late: the environment is the test)

    gpus = tf.config.list_physical_devices("GPU")
    details = []
    for gpu in gpus:
        try:
            info = tf.config.experimental.get_device_details(gpu)
        except Exception as exc:  # pragma: no cover - driver specific
            info = {"error": str(exc)}
        details.append({"name": gpu.name, "details": {k: str(v) for k, v in info.items()}})
    build = tf.sysconfig.get_build_info()
    out = {"schema": "fs4.tf_probe.v1", "tensorflow": tf.__version__,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
           "gpu_count": len(gpus), "gpus": details,
           "build": {k: str(build.get(k)) for k in ("cuda_version", "cudnn_version", "is_cuda_build")}}
    if gpus:
        with tf.device("/GPU:0"):
            x = tf.random.normal((64, 64))
            out["matmul_device"] = tf.matmul(x, x).device
    ok = True
    if args.expect_uuid:
        ok = len(gpus) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == args.expect_uuid
        out["expect_uuid"] = args.expect_uuid
    if args.expect_none:
        ok = ok and not gpus
    out["ok"] = ok
    print(json.dumps(out, sort_keys=True))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
