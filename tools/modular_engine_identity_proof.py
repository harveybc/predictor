#!/usr/bin/env python3
"""Bitwise graph-identity proof of the modular engine between two pinned checkouts (CPU).

Label PROOF_NOT_A_RESULT. ``child`` mode runs inside ONE checkout: for every model
config it builds the graph with build_modular, records config_digest, parameter count
and the ordered weight shapes/dtypes, then overwrites every weight with values drawn
from a fixed seed (per config, per weight index) and runs a forward pass on fixed
inputs; it records the sha256 of the float32 output bytes. ``compare`` runs ``child``
in each checkout in a separate process (two engines never share one interpreter) and
reports IDENTICAL only if every field of every config matches bitwise; otherwise
NOT_IDENTICAL with the first differing config, field and (for weights) layer index.
Run with CUDA_VISIBLE_DEVICES='' and TF_DETERMINISTIC_OPS=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def child(checkout, configs_path, out_path):
    sys.path.insert(0, checkout)
    import numpy as np
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt

    configs = json.loads(Path(configs_path).read_text())
    rows = {}
    for key, config in configs.items():
        try:
            bundle = mt.build_modular(config)
        except ValueError as exc:  # a refusal is part of the engine's behaviour and must also be identical
            rows[key] = {"refused": f"{type(exc).__name__}: {exc}"}
            tf.keras.backend.clear_session()
            continue
        model = bundle.forecast_model
        weights = model.get_weights()
        shapes = [[list(w.shape), str(w.dtype)] for w in weights]
        fixed = []
        for i, w in enumerate(weights):
            rng = np.random.default_rng(int(hashlib.sha256(f"{key}:{i}".encode()).hexdigest()[:8], 16))
            fixed.append((rng.standard_normal(w.shape) * 0.05).astype(w.dtype))
        model.set_weights(fixed)
        rng = np.random.default_rng(12345)
        x = rng.standard_normal((8, *model.input_shape[1:])).astype("float32")
        y = np.asarray(model(x, training=False)).astype("<f4")
        rows[key] = {"config_digest": mt.config_digest(config) if hasattr(mt, "config_digest") else None,
                     "parameters": int(model.count_params()), "weights": shapes,
                     "output_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                     "output_shape": list(y.shape)}
        del bundle, model
        tf.keras.backend.clear_session()
    Path(out_path).write_text(json.dumps({"checkout": checkout, "tensorflow": tf.__version__,
                                          "deterministic": os.environ.get("TF_DETERMINISTIC_OPS"),
                                          "rows": rows}, indent=1) + "\n")


def compare(a, b, configs, out, python):
    results = {}
    for name, checkout in (("a", a), ("b", b)):
        target = Path(out).with_suffix(f".{name}.json")
        subprocess.run([python, "-u", __file__, "child", "--checkout", checkout, "--configs", configs,
                        "--out", str(target)], check=True)
        results[name] = json.loads(target.read_text())
    first = None
    for key in results["a"]["rows"]:
        ra, rb = results["a"]["rows"][key], results["b"]["rows"].get(key)
        if rb is None:
            first = {"config": key, "field": "missing in b"}
            break
        if "refused" in ra or "refused" in rb:
            if ra.get("refused") != rb.get("refused"):
                first = {"config": key, "field": "refusal", "a": ra.get("refused"), "b": rb.get("refused")}
                break
            continue
        for field in ("config_digest", "parameters", "output_shape", "output_sha256"):
            if ra[field] != rb[field]:
                first = {"config": key, "field": field, "a": ra[field], "b": rb[field]}
                break
        if first:
            break
        if ra["weights"] != rb["weights"]:
            idx = next(i for i, (x, y) in enumerate(zip(ra["weights"], rb["weights"])) if x != y) \
                if len(ra["weights"]) == len(rb["weights"]) else min(len(ra["weights"]), len(rb["weights"]))
            first = {"config": key, "field": "weights", "weight_index": idx}
            break
    report = {"schema": "lane_d.engine_identity_proof.v1", "label": "PROOF_NOT_A_RESULT",
              "a": {"checkout": a, "tensorflow": results["a"]["tensorflow"]},
              "b": {"checkout": b, "tensorflow": results["b"]["tensorflow"]},
              "refused_identically": sorted(k for k, v in results["a"]["rows"].items() if "refused" in v),
              "configs": len(results["a"]["rows"]), "verdict": "IDENTICAL" if first is None else "NOT_IDENTICAL",
              "first_difference": first, "deterministic": [results["a"]["deterministic"], results["b"]["deterministic"]]}
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("child", "compare"))
    parser.add_argument("--checkout")
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--configs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "child":
        child(args.checkout, args.configs, args.out)
    else:
        compare(args.a, args.b, args.configs, args.out, sys.executable)


if __name__ == "__main__":
    main()
