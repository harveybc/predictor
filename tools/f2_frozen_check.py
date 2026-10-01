#!/usr/bin/env python3
"""Frozen / updated donor-weight evidence for R1 and R2 cells (lane F2).

Given the nested candidate JSON and the trained ``best.keras`` of a cell, compare every donor-bearing
component's weights in the trained model against the donor file's own weights (sha256 over dtype, shape
and bytes of each array, in order):

* R1 (frozen): the trained component weights must equal the donor weights bit-for-bit
  (``unchanged``); any difference is a defect and the check exits non-zero.
* R2 (fine-tuned): the same donors; the report lists how many components changed and the largest
  absolute weight difference, so "starts from the same donors and shows updates" is a number.
* R0 components carry no donor and are skipped.

The components are located in the trained forecast model by layer name (``branch_i`` for a branch, the
core's declared name otherwise). A component that cannot be located is reported NOT_FOUND (never counted
as unchanged). Output: JSON on stdout and, with --out, a file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def weights_digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(json.dumps([a.dtype.str, list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _find(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
    for layer in model.layers:  # one level of nesting
        for inner in getattr(layer, "layers", []) or []:
            if inner.name == name:
                return inner
    return None


def check(candidate, model_path, core_names=("core", "core_model", "core_encoder", "encoder")):
    import tensorflow as tf
    from predictor_plugins import modular_temporal  # noqa: F401  registers serializable layers

    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    spec = candidate["model"]
    components = [(b["name"], b) for b in spec["branches"]] + [("core", spec["core"])]
    rows, defects = [], 0
    for name, c in components:
        regime, donor = c.get("regime", "R0"), c.get("donor")
        if not donor:
            continue
        donor_model = tf.keras.models.load_model(donor, compile=False, safe_mode=True)
        d_w = donor_model.get_weights()
        layer = _find(model, name) if name != "core" else next(
            (l for l in (_find(model, n) for n in core_names) if l is not None), None)
        if layer is None:
            rows.append({"component": name, "regime": regime, "state": "NOT_FOUND"})
            defects += regime == "R1"
            continue
        m_w = layer.get_weights()
        same_shapes = len(m_w) == len(d_w) and all(a.shape == b.shape for a, b in zip(m_w, d_w))
        if not same_shapes:
            rows.append({"component": name, "regime": regime, "state": "SHAPE_MISMATCH",
                         "model_arrays": len(m_w), "donor_arrays": len(d_w)})
            defects += regime == "R1"
            continue
        equal = all(np.array_equal(a, b) for a, b in zip(m_w, d_w))
        diff = max((float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
                    for a, b in zip(m_w, d_w)), default=0.0)
        row = {"component": name, "regime": regime, "donor": donor, "donor_weights_sha256": weights_digest(d_w),
               "model_weights_sha256": weights_digest(m_w), "state": "unchanged" if equal else "updated",
               "max_abs_diff": diff, "trainable_in_model": bool(getattr(layer, "trainable", True))}
        if regime == "R1" and not equal:
            defects += 1
            row["state"] = "DEFECT_FROZEN_WEIGHTS_CHANGED"
        rows.append(row)
    summary = {"schema": "f2.frozen_check.v1", "model": str(model_path), "components_checked": len(rows),
               "unchanged": sum(r["state"] == "unchanged" for r in rows),
               "updated": sum(r["state"] == "updated" for r in rows),
               "not_found": sum(r["state"] == "NOT_FOUND" for r in rows), "defects": defects, "components": rows}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    summary = check(json.loads(Path(args.candidate).read_text()), args.model)
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "components"}))
    sys.exit(1 if summary["defects"] else 0)


if __name__ == "__main__":
    main()
