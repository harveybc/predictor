"""ONE bounded DMO10 element-level isolation diagnostic (Musashi order, no new rounds).

Freezes the first monitored batch, checkpoint and preprocessing identity from a
RETAINED unit (digests go to Git; arrays stay local), compares predictions
element by element BEFORE any reduction, independently reproduces the float64
``fsum`` partial sums, and runs a controlled flags-on/off comparison in fresh
processes to SUPPORT (or not) the autotune hypothesis. Same-environment and
cross-device comparisons are recorded separately; 5090==4090 is never required.
No production fit, no TEST rows, no seed averaging.

CLI::

    python -m tools.i6dmo.dmo10_isolation freeze  --design D --bundle B --unit U --out DIR
    python -m tools.i6dmo.dmo10_isolation forward --frozen DIR --out PRED.json
    python -m tools.i6dmo.dmo10_isolation diff    --frozen DIR --a A.json --b B.json --out DIFF.json
    python -m tools.i6dmo.dmo10_isolation flags   --frozen DIR --out FLAGS.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.i6dmo import model as M  # noqa: F401  (flags before TF)
from tools.i6dmo import train as T


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, doc):
    Path(path).write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return _sha_bytes(Path(path).read_bytes())


def freeze(design_path, bundle_dir, unit_dir, out_dir, *, split="fit", batch=0):
    """Freeze one monitored batch + checkpoint + preprocessing identity."""
    import tensorflow as tf
    unit, out = Path(unit_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    effective = T.effective_from_design(design_path)
    data = T.load_data_bundle(bundle_dir, effective)
    scales = json.loads((unit / "SCALES.json").read_text())
    training = json.loads((unit / "TRAINING.json").read_text())
    record = training["monitor_evaluation"]
    rows, _ = T.trainable_rows(data[split])
    x, y, mask = T.materialize_monitor_inputs(rows, scales)
    size = record["batch_size"]
    sl = slice(batch * size, batch * size + size)
    xb, yb, mb, ids = x[sl], y[sl], mask[sl], rows["row_id"][sl]
    np.savez(out / "batch.npz", x=xb, y=yb, mask=mb, row_id=ids)
    model_doc = json.loads((unit / "model" / "MODEL.json").read_text())
    doc = {
        "schema": "i6dmo.dmo10_frozen.v1", "split": split, "batch": batch,
        "shape": list(xb.shape), "dtype": str(xb.dtype),
        "x_sha256": _sha_bytes(np.ascontiguousarray(xb).tobytes()),
        "y_sha256": _sha_bytes(np.ascontiguousarray(yb).tobytes()),
        "mask_sha256": _sha_bytes(np.ascontiguousarray(mb).tobytes()),
        "row_ids_sha256": T._ids_sha(ids),
        "scales_sha256": scales.get("scales_sha256"),
        "weights_sha256": model_doc["weights_sha256"],
        "weight_tensors": model_doc["weight_tensors"],
        "model_archive_sha256": model_doc["archive_sha256"],
        "model_dir": str(Path(unit_dir).resolve() / "model"),
        "input_materialization": record.get("input_materialization"),
        "monitor_forward": record.get("monitor_forward"),
        "determinism_flags": record.get("determinism_flags"),
        "recorded_partial_sums": record["witness"][split]["batches"][batch]["partial_sums"],
        "recorded_partial_counts": record["witness"][split]["batches"][batch]["partial_counts"],
        "device": (record.get("recording") or {}).get("device"),
        "runtime": T.floor_protocol_identity()["runtime"],
        "batch_size": size,
    }
    doc["frozen_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write(out / "FROZEN.json", doc)
    return doc


def _load_frozen(frozen_dir):
    frozen = Path(frozen_dir)
    doc = json.loads((frozen / "FROZEN.json").read_text())
    with np.load(frozen / "batch.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if _sha_bytes(np.ascontiguousarray(arrays["x"]).tobytes()) != doc["x_sha256"]:
        raise T.TrainDisposition("FROZEN_DIGEST_MISMATCH", "x bytes changed since freeze")
    return doc, arrays


def forward(frozen_dir, out_path):
    """Fresh forward on the frozen batch via the canonical loader + dedicated monitor forward."""
    import tensorflow as tf
    doc, arrays = _load_frozen(frozen_dir)
    loaded, document = M.load_dmo(doc["model_dir"])
    if document["weights_sha256"] != doc["weights_sha256"]:
        raise T.TrainDisposition("FROZEN_DIGEST_MISMATCH", "checkpoint differs from the frozen one")
    pred = T.monitor_forward(loaded.forecast_model, arrays["x"], doc["batch_size"])
    pred64 = np.ascontiguousarray(pred, dtype=np.float64)
    result = {
        "schema": "i6dmo.dmo10_forward.v1", "frozen_sha256": doc["frozen_sha256"],
        "device": "gpu" if tf.config.list_physical_devices("GPU") else "cpu",
        "process_uuid": T.PROCESS_IDENTITY["uuid"],
        "determinism_flags": T.current_determinism_flags(),
        "tensor_sha256": _sha_bytes(pred64.tobytes()),
        "shape": list(pred64.shape),
        "predictions_hex": pred64.tobytes().hex(),
    }
    _write(out_path, result)
    return result


def diff(frozen_dir, a_path, b_path, out_path):
    """Element-by-element comparison BEFORE any reduction, plus independent fsum reproduction."""
    doc, arrays = _load_frozen(frozen_dir)
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    va = np.frombuffer(bytes.fromhex(a["predictions_hex"]), dtype=np.float64).reshape(a["shape"])
    vb = np.frombuffer(bytes.fromhex(b["predictions_hex"]), dtype=np.float64).reshape(b["shape"])
    delta = np.abs(va - vb)
    differing = np.flatnonzero((va != vb).ravel())
    first = None
    if len(differing):
        i = int(differing[0])
        first = {"flat_index": i, "row": i // va.shape[1], "head": i % va.shape[1],
                 "a": float(va.ravel()[i]), "b": float(vb.ravel()[i]),
                 "a_hex": float(va.ravel()[i]).hex(), "b_hex": float(vb.ravel()[i]).hex()}
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(delta > 0, delta / np.maximum(np.abs(va), np.abs(vb)), 0.0)
    # Independent fsum reproduction of the recorded partial sums from side A.
    y, mask = arrays["y"], arrays["mask"].astype(bool)
    sums = [math.fsum(np.abs(va[mask[:, h], h] - np.asarray(y, np.float64)[mask[:, h], h]).tolist())
            for h in range(va.shape[1])]
    result = {
        "schema": "i6dmo.dmo10_diff.v1", "frozen_sha256": doc["frozen_sha256"],
        "identical": bool(not len(differing)),
        "tensor_sha256_equal": a["tensor_sha256"] == b["tensor_sha256"],
        "n_differing": int(len(differing)), "first_differing_element": first,
        "max_abs_error": float(delta.max()) if delta.size else 0.0,
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
        "max_rel_error": float(rel.max()) if delta.size else 0.0,
        "fsum_reproduction": {"sums": sums,
                              "matches_recorded": sums == doc["recorded_partial_sums"]},
        "comparison_environment": ("same_environment" if a["device"] == b["device"]
                                   else "cross_device"),
        "devices": [a["device"], b["device"]],
        "cross_process": a["process_uuid"] != b["process_uuid"],
    }
    _write(out_path, result)
    return result


def flags_comparison(frozen_dir, out_path):
    """Controlled flags-on/off comparison in FRESH processes (autotune hypothesis test).

    Runs the same frozen forward in two child interpreters: one with the governed
    static-algorithm environment, one with autotune re-enabled. On a CPU-only host
    the verdict is honestly NOT_INFORMATIVE_ON_CPU.
    """
    import tensorflow as tf
    frozen = Path(frozen_dir)
    results = {}
    for name, autotune in (("flags_on", "0"), ("flags_off", "1")):
        env = {**os.environ, "TF_CUDNN_USE_AUTOTUNE": autotune}
        out = frozen / f"pred_{name}.json"
        run = subprocess.run([sys.executable, "-m", "tools.i6dmo.dmo10_isolation", "forward",
                              "--frozen", str(frozen), "--out", str(out)],
                             capture_output=True, text=True, env=env,
                             cwd=Path(__file__).resolve().parents[2], check=False)
        if run.returncode != 0:
            raise T.TrainDisposition("ISOLATION_CHILD_FAILED", run.stderr[-1000:])
        child = json.loads(out.read_text())
        results[name] = {"predictions_sha256": child["tensor_sha256"], "device": child["device"],
                         "flags": child["determinism_flags"], "autotune_env": autotune}
    gpu = results["flags_on"]["device"] == "gpu"
    same = results["flags_on"]["predictions_sha256"] == results["flags_off"]["predictions_sha256"]
    verdict = ("NOT_INFORMATIVE_ON_CPU" if not gpu
               else "SUPPORTS_AUTOTUNE_HYPOTHESIS" if not same
               else "DOES_NOT_SUPPORT_AUTOTUNE_HYPOTHESIS")
    report = {"schema": "i6dmo.dmo10_flags_comparison.v1", **results, "verdict": verdict,
              "fits_run": 0, "test_rows_read": 0,
              "note": "same-environment children on this host; cross-device evidence is recorded "
                      "separately and 5090==4090 arithmetic is never required"}
    _write(out_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.dmo10_isolation")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for flag in ("--design", "--bundle", "--unit", "--out"):
        f.add_argument(flag, required=True)
    f.add_argument("--split", default="fit")
    f.add_argument("--batch", type=int, default=0)
    w = sub.add_parser("forward")
    w.add_argument("--frozen", required=True)
    w.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    for flag in ("--frozen", "--a", "--b", "--out"):
        d.add_argument(flag, required=True)
    g = sub.add_parser("flags")
    g.add_argument("--frozen", required=True)
    g.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            doc = freeze(args.design, args.bundle, args.unit, args.out, split=args.split,
                         batch=args.batch)
        elif args.command == "forward":
            doc = forward(args.frozen, args.out)
            doc = {k: v for k, v in doc.items() if k != "predictions_hex"}
        elif args.command == "diff":
            doc = diff(args.frozen, args.a, args.b, args.out)
        else:
            doc = flags_comparison(args.frozen, args.out)
    except (T.TrainDisposition, T.ContractError) as exc:
        print(json.dumps({"status": "FAILED_TYPED", "code": exc.code, "detail": str(exc)[:500]}))
        return 5
    print(json.dumps(doc, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
