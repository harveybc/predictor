"""I6-DMO single fit: masked normalized loss, same-weight fit/inner stopping, receipt.

Entry point for the automation lane (one subprocess per unit)::

    python -m tools.i6dmo.train --design <design.json> --bundle <frozen_bundle_dir> \
        --out <new_unit_dir> --device {cpu,gpu} [--json]

Exit codes: 0 COMPLETED; 2 CONTRACT_ERROR (design invalid, out dir not empty;
nothing trained, no receipt); 3 FAILED_TYPED (typed disposition, receipt
written); 4 FAILED_UNTYPED (unexpected exception, receipt written). The
receipt is written last, atomically; its presence is the unit terminal.

Training coordinate: ``y_h = (p(t+h) - p(t)) / naive_fit_h`` where
``naive_fit_h`` is the fit-only masked mean ``|p(t+h) - p(t)|``. The per-head
loss is the masked mean absolute error in that coordinate, i.e. exactly
``MAE_h / naive_fit_h``; supported heads are averaged equally. Invalid labels
are removed by boolean selection before any arithmetic, so NaN never enters
the loss or gradients. A zero scale, an unsupported fit head or a constant fit
feature is a typed disposition, never an epsilon. Inputs are z-scored with
fit-only statistics. Early stopping monitors ``0.5 * (fit_loss + inner_loss)``
evaluated at the same end-of-epoch weights; outer VALIDATION is predicted only
after the best weights are restored and never influences any choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

# Imported FIRST so the static-algorithm determinism env is in place before TensorFlow
# initializes in this interpreter (`python -m tools.i6dmo.train` is the production child;
# the init-order witness in tools.i6dmo.model refuses to score records recorded otherwise).
from tools.i6dmo import model as _model_determinism  # noqa: F401
from tools.i6dmo import design as D

RECEIPT_SCHEMA = "i6dmo.train_receipt.v1"
BUNDLE_SCHEMA = "i6dmo.data_bundle.v1"
SPLITS = ("fit", "inner", "validation")
FINAL_SPLITS = ("fit", "inner")
ARRAYS = ("row_id", "x", "origin_price", "target_price", "mask")
OVERRIDABLE = ("max_epochs", "patience")
CODE_FILES = ("tools/i6dmo/design.py", "tools/i6dmo/model.py", "tools/i6dmo/train.py",
              "predictor_plugins/modular_temporal/layers.py",
              "predictor_plugins/modular_temporal/components.py",
              "predictor_plugins/modular_temporal/assembly.py",
              "predictor_plugins/modular_temporal/registry.py",
              "predictor_plugins/modular_temporal/common.py")
REPO = Path(__file__).resolve().parents[2]


class TrainDisposition(Exception):
    """Typed terminal disposition; message starts with the stable code."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


class ContractError(Exception):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _epoch(text):
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


# --------------------------------------------------------------------- design + bundle

def effective_from_design(path):
    try:
        return D.effective_config(D.load_design(path))
    except D.DesignError as exc:
        raise ContractError("DESIGN_INVALID", str(exc)) from None


def required_splits(effective):
    """The model splits a bundle must carry: no outer validation in a FINAL_REFIT."""
    return FINAL_SPLITS if effective.get("mode") == "FINAL_REFIT" else SPLITS


def load_data_bundle(bundle_dir, effective):
    """Authenticate the frozen data bundle and run the model-side boundary checks."""
    expected = required_splits(effective)
    root = Path(bundle_dir)
    try:
        manifest = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainDisposition("BUNDLE_UNREADABLE", str(exc)) from None
    required = {"schema", "features", "feature_manifest_sha256", "horizons_hours", "offsets_hours", "splits"}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise TrainDisposition("BUNDLE_SCHEMA", f"BUNDLE.json requires {sorted(required)}")
    if manifest["schema"] != BUNDLE_SCHEMA:
        raise TrainDisposition("BUNDLE_SCHEMA", f"schema must be {BUNDLE_SCHEMA}")
    splits = manifest["splits"]
    if not isinstance(splits, dict):
        raise TrainDisposition("BUNDLE_SCHEMA", "splits must be an object")
    if any("test" in str(name).lower() for name in splits):
        raise TrainDisposition("TEST_FORBIDDEN", "a TEST split is never read by a fit; the guarded "
                               "final module alone consumes TEST bundles, without training")
    if set(splits) != set(expected):
        raise TrainDisposition("BUNDLE_SCHEMA", f"splits must be exactly {list(expected)}")
    try:
        identity = D.check_feature_identity(manifest["features"])
    except D.DesignError as exc:
        raise TrainDisposition(exc.code, str(exc)) from None
    if manifest["feature_manifest_sha256"] != identity["manifest_sha256"]:
        raise TrainDisposition("FEATURE_IDENTITY_MISMATCH", "declared feature_manifest_sha256 differs")
    if manifest["horizons_hours"] != effective["horizons_hours"]:
        raise TrainDisposition("BUNDLE_HORIZONS", "bundle horizons differ from the design")
    if manifest["offsets_hours"] != effective["offsets_hours"]:
        raise TrainDisposition("BUNDLE_OFFSETS", "bundle input offsets differ from the design")
    test_start = _epoch(effective["splits"]["test"]["start"])
    horizons = np.asarray(effective["horizons_hours"], dtype=np.int64) * 3600
    steps, n_features = len(effective["offsets_hours"]), len(identity["features"])
    out = {"manifest": manifest, "manifest_sha256": file_sha256(root / "BUNDLE.json"),
           "feature_identity": identity}
    for name in expected:
        entry = splits[name]
        path = root / entry["path"]
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise TrainDisposition("BUNDLE_SCHEMA", f"{name} path must be relative inside the bundle")
        if file_sha256(path) != entry["sha256"]:
            raise TrainDisposition("BUNDLE_DIGEST_MISMATCH", name)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAYS):
                raise TrainDisposition("BUNDLE_SCHEMA", f"{name} arrays must be exactly {list(ARRAYS)}")
            arrays = {k: archive[k] for k in ARRAYS}
        rows = len(arrays["row_id"])
        shapes = {"row_id": (rows,), "x": (rows, steps, n_features), "origin_price": (rows,),
                  "target_price": (rows, len(horizons)), "mask": (rows, len(horizons))}
        if rows < 1 or any(arrays[k].shape != s for k, s in shapes.items()):
            raise TrainDisposition("BUNDLE_SHAPE", f"{name}: expected {shapes}")
        if arrays["row_id"].dtype.kind not in "iu" or len(np.unique(arrays["row_id"])) != rows:
            raise TrainDisposition("BUNDLE_ROW_ID", f"{name}: row_id must be unique integers")
        if arrays["mask"].dtype != np.bool_:
            raise TrainDisposition("BUNDLE_SCHEMA", f"{name}: mask must be boolean")
        origin = arrays["row_id"].astype(np.int64)
        if (origin >= test_start).any():
            raise TrainDisposition("TEST_FORBIDDEN", f"{name}: origin at or after the TEST start")
        start, end = (_epoch(effective["splits"][name][k]) for k in ("start", "end"))
        if ((origin < start) | (origin >= end)).any():
            raise TrainDisposition("SPLIT_BOUNDARY_VIOLATION", f"{name}: origin outside [{start}, {end})")
        label_time = origin[:, None] + horizons[None, :]
        crossing = arrays["mask"] & (label_time >= end)
        if crossing.any():
            code = "PURGE_VIOLATION" if name == "fit" else "SPLIT_BOUNDARY_VIOLATION"
            raise TrainDisposition(code, f"{name}: {int(crossing.sum())} valid labels reach {end}")
        if not np.isfinite(arrays["x"]).all():
            raise TrainDisposition("NONFINITE_INPUT", name)
        if not np.isfinite(arrays["origin_price"]).all():
            raise TrainDisposition("NONFINITE_INPUT", f"{name}: origin_price")
        if not np.isfinite(arrays["target_price"][arrays["mask"]]).all():
            raise TrainDisposition("NONFINITE_LABEL", f"{name}: a valid label is not finite")
        out[name] = arrays
    return out


# --------------------------------------------------------------------- scales and coordinates

def trainable_rows(arrays):
    """Rows with at least one valid head; the others are counted, not trained."""
    keep = arrays["mask"].any(axis=1)
    return {k: v[keep] for k, v in arrays.items()}, int((~keep).sum())


def fit_scales(fit):
    """Fit-only input z-score statistics and per-head naive MAE."""
    x = np.asarray(fit["x"], dtype=np.float64).reshape(-1, fit["x"].shape[-1])
    mean, std = x.mean(axis=0), x.std(axis=0)
    if not (np.isfinite(std).all() and (std > 0).all()):
        bad = np.flatnonzero(~(std > 0)).tolist()
        raise TrainDisposition("CONSTANT_FEATURE_IN_FIT", f"feature indices {bad}")
    mask = fit["mask"]
    counts = mask.sum(axis=0)
    if (counts == 0).any():
        raise TrainDisposition("HEAD_UNSUPPORTED_IN_FIT", f"head indices {np.flatnonzero(counts == 0).tolist()}")
    naive = []
    for h in range(mask.shape[1]):
        sel = mask[:, h]
        delta = np.abs(fit["target_price"][sel, h].astype(np.float64) - fit["origin_price"][sel].astype(np.float64))
        naive.append(math.fsum(delta.tolist()) / int(sel.sum()))
    if not all(math.isfinite(v) and v > 0 for v in naive):
        raise TrainDisposition("ZERO_NAIVE_SCALE", f"fit naive MAE per head {naive}")
    scales = {"schema": "i6dmo.scales.v1", "source_split": "fit", "rows": int(len(mask)),
              "input_mean": mean.tolist(), "input_std": std.tolist(), "naive_mae": naive,
              "counts": counts.astype(int).tolist()}
    scales["scales_sha256"] = hashlib.sha256(_canonical(scales).encode()).hexdigest()
    return scales


def transform(arrays, scales):
    """(x float32 z-scored, y in the training coordinate with NaN where invalid, mask)."""
    mean = np.asarray(scales["input_mean"], dtype=np.float64)
    std = np.asarray(scales["input_std"], dtype=np.float64)
    naive = np.asarray(scales["naive_mae"], dtype=np.float64)
    x = ((np.asarray(arrays["x"], dtype=np.float64) - mean) / std).astype(np.float32)
    mask = np.asarray(arrays["mask"], dtype=bool)
    delta = np.where(mask, arrays["target_price"] - arrays["origin_price"][:, None], np.nan)
    with np.errstate(invalid="ignore"):
        y = np.where(mask, delta / naive[None, :], np.nan)
    return x, y, mask


def to_price(pred, arrays, scales):
    """Inverse of the training coordinate: original USD per EUR."""
    naive = np.asarray(scales["naive_mae"], dtype=np.float64)
    return np.asarray(arrays["origin_price"], dtype=np.float64)[:, None] + np.asarray(pred, np.float64) * naive


# --------------------------------------------------------------------- loss and metrics

def masked_normalized_loss(y, pred, mask):
    """Equal-weight mean over supported heads of each head's masked MAE (TF).

    Invalid labels are removed with ``tf.boolean_mask`` before subtraction;
    heads without a valid label in this batch are excluded, not counted as 0.
    Returns ``(loss, per_head_sums, per_head_counts)``.
    """
    import tensorflow as tf

    mask = tf.cast(mask, tf.bool)
    sums, counts = [], []
    for h in range(pred.shape[1]):
        m = mask[:, h]
        p = tf.boolean_mask(pred[:, h], m)
        t = tf.boolean_mask(tf.cast(y[:, h], pred.dtype), m)
        sums.append(tf.reduce_sum(tf.abs(p - t)))
        counts.append(tf.reduce_sum(tf.cast(m, pred.dtype)))
    sums, counts = tf.stack(sums), tf.stack(counts)
    active = counts > 0
    means = tf.where(active, tf.math.divide_no_nan(sums, counts), tf.zeros_like(sums))
    loss = tf.reduce_sum(means) / tf.reduce_sum(tf.cast(active, pred.dtype))
    return loss, sums, counts


def masked_split_metrics(y, pred, mask):
    """Frozen-weight split loss from global per-head sums/counts (float64, fsum)."""
    y, pred, mask = np.asarray(y, np.float64), np.asarray(pred, np.float64), np.asarray(mask, bool)
    if not np.isfinite(y[mask]).all():
        raise TrainDisposition("NONFINITE_LABEL", "a valid label is not finite")
    if not np.isfinite(pred[mask]).all():
        raise TrainDisposition("NONFINITE_PREDICTION", "a prediction on a valid label is not finite")
    sums, counts, per_head = [], [], []
    for h in range(y.shape[1]):
        sel = mask[:, h]
        s = math.fsum(np.abs(pred[sel, h] - y[sel, h]).tolist())
        sums.append(s)
        counts.append(int(sel.sum()))
        per_head.append(s / counts[-1] if counts[-1] else None)
    active = [v for v in per_head if v is not None]
    if not active:
        raise TrainDisposition("NO_SUPPORTED_HEAD", "split has no valid label")
    loss = math.fsum(active) / len(active)
    return {"sums": sums, "counts": counts, "per_head": per_head, "loss": loss,
            "active_heads": len(active)}


MONITOR_EVAL = "i6dmo.monitor_eval.v2"
MONITOR_PARTITION = "sequential_row_id_fixed"
MONITOR_INSTANCE = "canonical_loader_rebuild"
#: THE canonical monitor-input materialization, shared by the fit path and the replay: the
#: bundle arrays are the source of truth, normalization happens in float64 on CPU (numpy),
#: and there is exactly ONE cast to float32, in materialize_monitor_inputs, before any
#: forward. Any other derivation (e.g., a float32 pipeline) yields different bytes and is
#: refused by the input witness.
MONITOR_INPUT = "i6dmo.monitor_input.v1:float64_cpu_normalize_single_float32_cast"


def materialize_monitor_inputs(rows, scales):
    """The ONE shared monitor-input materialization (see MONITOR_INPUT)."""
    x, y, mask = transform(rows, scales)
    return np.ascontiguousarray(x, dtype=np.float32), y, mask
#: Pre-declared monitored-value comparison tolerance, serialization/process-level only.
#: Written justification: process-level GPU algorithm selection, a library property —
#: with autotune forced static on both sides, any residual per-batch partial-sum delta
#: bounded by 1e-5 RELATIVE is accepted for the monitored-value comparison ONLY; reload
#: and prediction identities stay bitwise. It activates ONLY when the record carries the
#: static-algorithm determinism flags and they equal the running process's flags.
MONITOR_TOLERANCE = {"relative_per_partial_sum": 1e-5,
                     "justification": "process-level GPU algorithm selection, library property",
                     "requires_static_flags": True}


def current_determinism_flags():
    """The running process's canonical determinism/static-algorithm flags."""
    from tools.i6dmo.model import determinism_flags
    return determinism_flags()


def _ids_sha(ids):
    import hashlib as _h
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    return _h.sha256(ids.tobytes()).hexdigest()


def evaluate_monitored(model, fit, inner, batch_size, ids=None):
    """THE monitored-loss evaluation procedure (training and replay share it, DMO10).

    One fixed path over the CANONICAL FIXED SEQUENTIAL partition: the split
    arrays in row_id order, cut into consecutive batches of ``batch_size``
    (``predict_matrix``; float32 inputs, float64 outputs per batch, concatenated
    in order), then :func:`masked_split_metrics` global per-head float64
    ``fsum`` sums/counts, equal-weight mean over supported heads, and
    ``0.5 * (fit_loss + inner_loss)``. The training shuffle exists for gradient
    steps ONLY and never reaches monitoring: on GPU a row's float32 forward
    value legitimately varies with batch composition (GEMM tiling), so only
    this fixed partition is bitwise comparable across sessions.

    Returns ``(fit_metrics, inner_metrics, monitored, witness)``. The witness
    carries the sha256 of the exact consumed row-id batch sequence, per-batch
    per-head float64 partial sums/counts, and a code witness (sha256 of this
    function's source plus a nonce derived from the partition parameters,
    computed INSIDE this function) so a stale, cached or fallback evaluation
    path is detectable at replay (``ids=None`` records positional ids).
    """
    import inspect
    from tools.i6dmo.model import predict_matrix
    names = ("fit", "inner")
    splits = {"fit": fit, "inner": inner}
    id_arrays = {}
    for index, name in enumerate(names):
        n = len(splits[name][0])
        given = None if ids is None else np.asarray(ids[index], dtype=np.int64)
        if given is not None and len(given) != n:
            raise TrainDisposition("MONITOR_PATH_MISMATCH", f"{name} ids length {len(given)} != rows {n}")
        id_arrays[name] = np.arange(n, dtype=np.int64) if given is None else given
    nonce = hashlib.sha256(_canonical(
        {"procedure": MONITOR_EVAL, "batch_size": batch_size, "partition": MONITOR_PARTITION,
         "n_fit": len(fit[0]), "n_inner": len(inner[0]),
         "fit_ids_sha256": _ids_sha(id_arrays["fit"]),
         "inner_ids_sha256": _ids_sha(id_arrays["inner"])}).encode()).hexdigest()
    code_witness = hashlib.sha256(
        (inspect.getsource(evaluate_monitored) + nonce).encode()).hexdigest()
    witness = {"ids": "positional" if ids is None else "row_id", "nonce": nonce,
               "code_witness": code_witness}
    results = {}
    for name in names:
        x, y, mask = splits[name]
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))  # the EXACT bytes fed below
        pred = predict_matrix(model, x, batch_size=batch_size)
        results[name] = masked_split_metrics(y, pred, mask)
        y64, mb_all = np.asarray(y, np.float64), np.asarray(mask, bool)
        batches = []
        for b, start in enumerate(range(0, len(x), batch_size)):
            stop = start + batch_size
            pb, yb, mb = pred[start:stop], y64[start:stop], mb_all[start:stop]
            sums = [math.fsum(np.abs(pb[mb[:, h], h] - yb[mb[:, h], h]).tolist())
                    for h in range(yb.shape[1])]
            counts = [int(mb[:, h].sum()) for h in range(yb.shape[1])]
            batches.append({"batch": b, "start": start, "rows": int(len(pb)),
                            "row_ids_sha256": _ids_sha(id_arrays[name][start:stop]),
                            "input_sha256": hashlib.sha256(x[start:stop].tobytes()).hexdigest(),
                            "partial_sums": sums, "partial_counts": counts})
        witness[name] = {"row_ids_sha256": _ids_sha(id_arrays[name]), "batches": batches,
                         # Input witness: post-normalization float32 bytes, digested before any
                         # forward; batch-0 rows individually, the first rows byte-complete.
                         "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                         "batch0_input_row_sha256": [hashlib.sha256(row.tobytes()).hexdigest()
                                                     for row in x[:batch_size]],
                         "batch0_input_rows_hex": [row.tobytes().hex() for row in x[:8]],
                         # Row-level forward witness: batch-0 per-row per-head float64 hex, so a
                         # replay can tell a weight effect (localized rows) from a different
                         # matmul/conv execution mode (uniform small deltas on every row).
                         "batch0_predictions_hex": [[float(v).hex() for v in row]
                                                    for row in pred[:batch_size]]}
    monitored = 0.5 * (results["fit"]["loss"] + results["inner"]["loss"])
    return results["fit"], results["inner"], monitored, witness


def monitor_self_check(model, fit, inner, record, expected, ids=None):
    """Recording-time self-check: a SEPARATE recompute must reproduce ``expected`` bitwise.

    Called in-session before any receipt is written, on FRESHLY materialized
    input copies (never the same tensor object) and through the full
    ``evaluate_monitored`` path, so the comparison is always between two
    independent computations and never the stored value against itself. A
    mismatch is a typed MONITOR_SELF_CHECK_FAILED, never silently accepted.
    """
    fit = tuple(np.array(part, copy=True) for part in fit)
    inner = tuple(np.array(part, copy=True) for part in inner)
    fm, im, monitored, _ = replay_monitored(model, fit, inner, record, ids=ids)
    if monitored != expected:
        raise TrainDisposition("MONITOR_SELF_CHECK_FAILED",
                               f"inference-path recompute {monitored!r} != recorded {expected!r} "
                               "(bitwise, same device, same session)")
    return fm, im, monitored


def replay_monitored(model, fit, inner, record, batch_size=None, ids=None):
    """Recompute the monitored loss through the recorded identical procedure.

    ``record`` is TRAINING.json's ``monitor_evaluation``. A different procedure
    name, batch size, graph, partition or monitor instance is a typed
    MONITOR_PATH_MISMATCH (mixed v1/v2 records are refused): a one-pass or
    permuted recompute of a batched training monitor is a different reduction
    order, not a replay. ``model`` must be the canonical loader-path instance
    (``load_dmo``). When the record carries a witness, the recompute is
    compared batch by batch and the FIRST divergent batch is reported with its
    row-id digest and the pair of partial sums (MONITOR_WITNESS_DIVERGENCE).
    """
    if not isinstance(record, dict) or record.get("procedure") != MONITOR_EVAL:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"recorded procedure {record.get('procedure') if isinstance(record, dict) else record!r}"
                               f" is not {MONITOR_EVAL} (mixed or old records are never comparable)")
    if record.get("monitor_instance", MONITOR_INSTANCE) != MONITOR_INSTANCE:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"monitor instance {record.get('monitor_instance')!r} is not the "
                               f"canonical {MONITOR_INSTANCE}")
    if record.get("input_materialization", MONITOR_INPUT) != MONITOR_INPUT:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"input materialization {record.get('input_materialization')!r} is "
                               f"not the canonical {MONITOR_INPUT}; the bundle arrays with one "
                               "float64 CPU normalization and a single float32 cast are the truth")
    if record.get("graph", "inference") != "inference":
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"a {record.get('graph')}-graph-evaluated monitor is not comparable; "
                               "only the shared inference path is")
    if record.get("partition") != MONITOR_PARTITION:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"monitor partition {record.get('partition')!r} is not the canonical "
                               f"{MONITOR_PARTITION}; a permuted/shuffled monitor evaluation is a "
                               "different batch composition and is never comparable")
    recorded = record.get("batch_size")
    if type(recorded) is not int or recorded < 1:
        raise TrainDisposition("MONITOR_PATH_MISMATCH", "recorded batch_size missing")
    if batch_size is not None and batch_size != recorded:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"replay batch_size {batch_size} != recorded {recorded}; "
                               "the monitored value is only comparable through the identical path")
    flags = record.get("determinism_flags")
    tolerance_active = False
    if flags is not None:
        running = current_determinism_flags()
        if flags.get("tf_imported_before_flags") is True or running["tf_imported_before_flags"] is True:
            side = "recorded" if flags.get("tf_imported_before_flags") is True else "running"
            raise TrainDisposition("DETERMINISM_INIT_ORDER_INVALID",
                                   f"the {side} process imported TensorFlow before the static-"
                                   "algorithm flags were set; its kernel selection is ungoverned "
                                   "and the monitored value is not scoreable")
        if flags != running:
            raise TrainDisposition("DETERMINISM_FLAGS_MISMATCH",
                                   f"recorded flags {flags} != running process {running}; a record "
                                   "scored under other kernel-selection flags is never comparable")
        tolerance_active = flags.get("TF_CUDNN_USE_AUTOTUNE") == "0"
    stored = record.get("witness")
    if stored is not None and stored.get("ids") == "row_id" and ids is None:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               "the recorded witness binds row ids; the replay must supply the same ids")
    fm, im, monitored, witness = evaluate_monitored(model, fit, inner, recorded, ids=ids)
    tolerated = []
    if stored is not None:
        tolerated = _compare_witness(stored, witness, tolerance_active)
        if tolerated and record.get("monitored") is not None:
            bound = MONITOR_TOLERANCE["relative_per_partial_sum"] * max(1.0, abs(record["monitored"]))
            if abs(monitored - record["monitored"]) > bound:
                raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                                       f"monitored {monitored!r} vs recorded {record['monitored']!r} "
                                       f"exceeds the pre-declared bound {bound}")
    return fm, im, monitored, tolerated


def _row_forward_diff(stored_hex, fresh_hex):
    """Batch-0 per-row diff with a delta-pattern verdict (uniform => execution mode)."""
    divergent, deltas = [], []
    for i, (ra, rb) in enumerate(zip(stored_hex, fresh_hex)):
        if ra == rb:
            continue
        divergent.append(i)
        for ha, hb in zip(ra, rb):
            va, vb = float.fromhex(ha), float.fromhex(hb)
            if va != vb:
                deltas.append(abs(va - vb) / max(abs(va), abs(vb), 1e-300))
    if not divergent:
        return None
    uniform = (len(divergent) == min(len(stored_hex), len(fresh_hex)) and deltas
               and max(deltas) <= 10 * min(deltas))
    return {"n_rows": len(stored_hex), "n_divergent_rows": len(divergent),
            "first_divergent_rows": divergent[:5],
            "max_relative_delta": max(deltas) if deltas else 0.0,
            "min_relative_delta": min(deltas) if deltas else 0.0,
            "uniform_delta_pattern": uniform,
            "interpretation": ("UNIFORM small deltas on every row: a different matmul/conv "
                               "execution mode (e.g., TF32 vs FP32), not weights" if uniform else
                               "LOCALIZED row divergence: weight or data effect, not a uniform "
                               "execution-mode rounding")}


def _compare_input_witness(name, a, b):
    """Byte identity of the post-normalization inputs; a mismatch names rows and bytes."""
    if a.get("input_sha256") is None:
        return
    if a["input_sha256"] == b["input_sha256"]:
        return
    rows = [i for i, (ra, rb) in enumerate(zip(a.get("batch0_input_row_sha256") or [],
                                               b["batch0_input_row_sha256"])) if ra != rb]
    detail = (f"{name}: post-normalization input bytes differ (recorded {a['input_sha256']} vs "
              f"recomputed {b['input_sha256']}); the monitor was fed a different tensor than the "
              f"canonical materialization rebuilds")
    bad_batches = [rb["batch"] for ra, rb in zip(a.get("batches") or [], b["batches"])
                   if ra.get("input_sha256") != rb["input_sha256"]]
    detail += f"; divergent batches {bad_batches[:5]} of {len(b['batches'])}"
    if rows:
        detail += f"; batch 0 first divergent rows {[f'row {i}' for i in rows[:5]]} of {len(b['batch0_input_row_sha256'])}"
        stored_hex, fresh_hex = a.get("batch0_input_rows_hex") or [], b["batch0_input_rows_hex"]
        for i, (ha, hb) in enumerate(zip(stored_hex, fresh_hex)):
            if ha != hb:
                sa = np.frombuffer(bytes.fromhex(ha), dtype=np.float32)
                sb = np.frombuffer(bytes.fromhex(hb), dtype=np.float32)
                elems = np.flatnonzero(sa.tobytes() != sb.tobytes() if len(sa) != len(sb)
                                       else sa != sb)[:4]
                pairs = [(int(e), float(sa[e]), float(sb[e])) for e in elems]
                detail += (f"; row {i} byte-level first differing elements "
                           f"(index, recorded, recomputed): {pairs}")
                break
    raise TrainDisposition("INPUT_WITNESS_MISMATCH", detail)


def _compare_witness(stored, fresh, tolerance_active=False):
    """Batch-by-batch comparison; returns the tolerated batches (empty when exact).

    With ``tolerance_active`` (static-algorithm flags confirmed on both sides), a batch
    whose rows/counts/row-ids match exactly and whose per-head partial sums differ by at
    most the pre-declared 1e-5 RELATIVE bound is tolerated and reported, not refused.
    """
    tolerated = []
    if stored.get("code_witness") != fresh["code_witness"]:
        raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                               "code witness differs: the recorded evaluation ran different "
                               "evaluate_monitored source or partition parameters (stale/cached/"
                               f"fallback path): recorded={stored.get('code_witness')} "
                               f"recomputed={fresh['code_witness']}")
    for name in ("fit", "inner"):
        a, b = stored.get(name) or {}, fresh[name]
        _compare_input_witness(name, a, b)
        row_diff = None
        if a.get("batch0_predictions_hex"):
            row_diff = _row_forward_diff(a["batch0_predictions_hex"], b["batch0_predictions_hex"])
        if a.get("row_ids_sha256") != b["row_ids_sha256"]:
            raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                                   f"{name} row-id sequence differs: recorded "
                                   f"{a.get('row_ids_sha256')} recomputed {b['row_ids_sha256']}")
        for ra, rb in zip(a.get("batches") or [], b["batches"]):
            if ra == rb:
                continue
            structural = {k: ra.get(k) for k in ("batch", "start", "rows", "row_ids_sha256",
                                                 "partial_counts")}
            if tolerance_active and structural == {k: rb[k] for k in structural}:
                deltas = [abs(u - v) / max(abs(u), abs(v), 1e-300)
                          for u, v in zip(ra.get("partial_sums") or [], rb["partial_sums"])]
                if deltas and max(deltas) <= MONITOR_TOLERANCE["relative_per_partial_sum"]:
                    tolerated.append({"split": name, "batch": rb["batch"],
                                      "max_relative_delta": max(deltas),
                                      "row_forward_diff": row_diff if rb["batch"] == 0 else None,
                                      "tolerance": MONITOR_TOLERANCE})
                    continue
            detail = (f"{name} batch {rb['batch']} (start {rb['start']}, row_ids "
                      f"{rb['row_ids_sha256']}): recorded partial_sums={ra.get('partial_sums')} "
                      f"counts={ra.get('partial_counts')} vs recomputed "
                      f"partial_sums={rb['partial_sums']} counts={rb['partial_counts']}")
            if rb["batch"] == 0 and row_diff:
                detail += (f"; row-level forward witness: rows "
                           f"{['row ' + str(i) for i in row_diff['first_divergent_rows']]} of "
                           f"{row_diff['n_rows']} diverge ({row_diff['n_divergent_rows']} total), "
                           f"relative deltas [{row_diff['min_relative_delta']:.3e}, "
                           f"{row_diff['max_relative_delta']:.3e}], uniform_delta_pattern="
                           f"{row_diff['uniform_delta_pattern']}: {row_diff['interpretation']}")
            raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE", detail)
        if len(a.get("batches") or []) != len(b["batches"]):
            raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                                   f"{name} batch count differs: {len(a.get('batches') or [])} "
                                   f"recorded vs {len(b['batches'])} recomputed")
        if row_diff:  # hex differs even where sums agree/cancel: forward values are not identical
            summary = (f"{name} batch 0 row-level forward witness: rows "
                       f"{['row ' + str(i) for i in row_diff['first_divergent_rows']]} of "
                       f"{row_diff['n_rows']} diverge ({row_diff['n_divergent_rows']} total), "
                       f"relative deltas [{row_diff['min_relative_delta']:.3e}, "
                       f"{row_diff['max_relative_delta']:.3e}], uniform_delta_pattern="
                       f"{row_diff['uniform_delta_pattern']}: {row_diff['interpretation']}")
            if tolerance_active and row_diff["max_relative_delta"] <=                     MONITOR_TOLERANCE["relative_per_partial_sum"]:
                tolerated.append({"split": name, "batch": 0, "row_forward_diff": row_diff,
                                  "max_relative_delta": row_diff["max_relative_delta"],
                                  "tolerance": MONITOR_TOLERANCE})
            else:
                raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE", summary)
    return tolerated


class MeanMonitorStopper:
    """Monitor ``0.5 * (fit + inner)``; patience vs ``reference - min_delta``; keep lowest.

    Rule identity ``i6dmo.mean_monitor_stop.v1``: an epoch is stale unless its
    monitored loss is below the last significant reference minus ``min_delta``;
    stopping happens when ``stale >= patience`` or at ``max_epochs``. The
    restored checkpoint is the lowest monitored epoch (first tie wins), which
    includes improvements smaller than ``min_delta``.
    """

    RULE = "i6dmo.mean_monitor_stop.v1"

    def __init__(self, patience, min_delta, max_epochs):
        self.patience, self.min_delta, self.max_epochs = patience, min_delta, max_epochs
        self.history, self.best_epoch, self.best_weights = [], None, None
        self.best_monitored, self.reference, self.stale = math.inf, math.inf, 0
        self.stop_reason = None

    def update(self, fit_loss, inner_loss, weights):
        monitored = 0.5 * (fit_loss + inner_loss)
        if not math.isfinite(monitored):
            raise TrainDisposition("NONFINITE_LOSS", f"monitored loss {monitored}")
        epoch = len(self.history) + 1
        if monitored < self.best_monitored:
            self.best_monitored, self.best_epoch = monitored, epoch
            self.best_weights = [np.array(w, copy=True) for w in weights]
        if monitored < self.reference - self.min_delta:
            self.reference, self.stale = monitored, 0
        else:
            self.stale += 1
        self.history.append({"epoch": epoch, "fit_loss": fit_loss, "inner_loss": inner_loss,
                             "monitored": monitored, "stale": self.stale, "best_epoch": self.best_epoch})
        if self.stale >= self.patience:
            self.stop_reason = "patience"
        elif epoch >= self.max_epochs:
            self.stop_reason = "max_epochs"
        return self.stop_reason is not None

    def identity(self):
        return {"rule": self.RULE, "patience": self.patience, "min_delta": self.min_delta,
                "max_epochs": self.max_epochs, "monitor": "0.5*(fit_loss+inner_loss)",
                "restore": "lowest_monitored_first_tie"}


def _monitor_model(bundle, workdir):
    """A FRESH canonical-loader-path instance holding the training model's current weights.

    Monitoring (and therefore early stopping) never reads the trained in-session
    instance: training on GPU acquires cuDNN/compilation state that can alter
    subsequent in-process forwards, so the monitored value is computed on the
    same save -> build_dmo -> load_weights instance the replay uses.
    """
    from tools.i6dmo import model as M
    path = Path(workdir) / "monitor.weights.h5"
    bundle.forecast_model.save_weights(path)
    fresh = M.build_dmo(bundle.config)
    fresh.forecast_model.load_weights(path)
    return fresh.forecast_model


def fit_dmo(bundle, fit, inner, training, seed, progress=None, ids=None):
    """One fit with AdamW; returns the audit dict. Restores the selected weights."""
    import tempfile
    import tensorflow as tf
    from predictor_plugins.modular_temporal import weights_hash
    from tools.i6dmo.model import predict_matrix

    model = bundle.forecast_model
    tf.keras.utils.set_random_seed(seed)
    order_rng = np.random.default_rng(seed)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=training["learning_rate"],
                                          weight_decay=training["weight_decay"])
    x_fit, y_fit, m_fit = fit
    x_in, y_in, m_in = inner
    variables = model.trainable_variables
    optimizer.build(variables)  # optimizer slots exist before the traced step

    @tf.function(reduce_retracing=True)
    def step(x, y, m):
        with tf.GradientTape() as tape:
            pred = tf.concat(model(x, training=True), axis=1)
            loss, _, _ = masked_normalized_loss(y, pred, m)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    stopper = MeanMonitorStopper(training["patience"], training["min_delta"], training["max_epochs"])
    batch, updates, rows = training["batch_size"], 0, []
    best_witness, loader_overhead = None, []
    monitor_dir = tempfile.TemporaryDirectory(prefix="i6dmo_monitor_")
    initial = weights_hash(model)
    started = time.monotonic()
    while True:
        perm = order_rng.permutation(len(x_fit))
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            loss = float(step(tf.constant(x_fit[idx]), tf.constant(y_fit[idx], tf.float32),
                              tf.constant(m_fit[idx])))
            if not math.isfinite(loss):
                raise TrainDisposition("NONFINITE_LOSS", f"training batch loss {loss}")
            updates += 1
        # The monitored value early stopping reads comes ONLY from the shared inference path on
        # the canonical sequential partition AND the canonical loader-path instance; the
        # per-epoch permutation above is gradient-only.
        overhead_start = time.monotonic()
        monitor_model = _monitor_model(bundle, monitor_dir.name)
        loader_overhead.append(time.monotonic() - overhead_start)
        fm, im, _, epoch_witness = evaluate_monitored(
            monitor_model, (x_fit, y_fit, m_fit), (x_in, y_in, m_in), batch, ids=ids)
        was_best = stopper.best_monitored
        stop = stopper.update(fm["loss"], im["loss"], model.get_weights())
        if stopper.best_monitored < was_best:
            from tools.i6dmo.model import weight_tensor_digests
            # Weights-at-witness: the exact tensors of the instance the witness evaluated.
            epoch_witness["weights_at_witness"] = weight_tensor_digests(monitor_model)
            best_witness = epoch_witness
        row = dict(stopper.history[-1], updates=updates, weights_sha256=weights_hash(model),
                   fit=fm, inner=im, elapsed_s=time.monotonic() - started)
        rows.append(row)
        if progress is not None:
            progress({k: row[k] for k in ("epoch", "fit_loss", "inner_loss", "monitored", "stale",
                                          "best_epoch", "updates", "elapsed_s")})
        if stop:
            break
    last = weights_hash(model)
    model.set_weights(stopper.best_weights)
    restored = weights_hash(model)
    if restored != rows[stopper.best_epoch - 1]["weights_sha256"]:
        raise TrainDisposition("RESTORE_MISMATCH", "restored weights differ from the selected epoch")
    record = {"procedure": MONITOR_EVAL, "batch_size": batch, "graph": "inference",
              "partition": MONITOR_PARTITION, "monitor_instance": MONITOR_INSTANCE,
              "input_materialization": MONITOR_INPUT,
              "determinism_flags": current_determinism_flags(),
              "declared_tolerance": dict(MONITOR_TOLERANCE),
              "witness": best_witness,
              "loader_overhead_s": {"epochs": len(loader_overhead),
                                    "mean_s": math.fsum(loader_overhead) / len(loader_overhead),
                                    "max_s": max(loader_overhead),
                                    "total_s": math.fsum(loader_overhead)},
              "prediction_dtype": "float32 inputs, float64 per-batch outputs",
              "accumulation": "global per-head float64 math.fsum sums/counts, "
                              "equal-weight mean over supported heads"}
    best_row = rows[stopper.best_epoch - 1]
    fm_best, im_best, monitored_best = monitor_self_check(
        _monitor_model(bundle, monitor_dir.name), (x_fit, y_fit, m_fit), (x_in, y_in, m_in),
        record, best_row["monitored"], ids=ids)
    monitor_dir.cleanup()
    record.update(self_check="PASS", best_epoch=stopper.best_epoch, monitored=monitored_best,
                  fit={"sums": fm_best["sums"], "counts": fm_best["counts"]},
                  inner={"sums": im_best["sums"], "counts": im_best["counts"]})
    if int(optimizer.iterations.numpy()) != updates:
        raise TrainDisposition("UPDATE_COUNT_MISMATCH", "optimizer iterations disagree with updates")
    return {"history": rows, "best_epoch": stopper.best_epoch, "best_monitored": stopper.best_monitored,
            "monitor_evaluation": record,
            "stop_reason": stopper.stop_reason, "epochs_completed": len(rows), "updates": updates,
            "initial_weights_sha256": initial, "last_weights_sha256": last,
            "restored_weights_sha256": restored, "early_stop": stopper.identity(),
            "fit_seconds": time.monotonic() - started}


# --------------------------------------------------------------------- unit

def _cgroup_peak():
    try:
        line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        return int(Path("/sys/fs/cgroup" + line[3:].strip(), "memory.peak").read_text().strip())
    except (OSError, StopIteration, ValueError):
        return None


def _resources(device):
    import tensorflow as tf
    usage = resource.getrusage(resource.RUSAGE_SELF)
    gpu_peak = None
    if device == "gpu" and tf.config.list_physical_devices("GPU"):
        try:
            gpu_peak = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
        except (ValueError, RuntimeError, KeyError):
            gpu_peak = None
    return {"tf_gpu_memory_peak_bytes": gpu_peak, "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime, "max_rss_kib": usage.ru_maxrss,
            "cgroup_memory_peak_bytes": _cgroup_peak(), "device": device,
            "visible_gpus": len(tf.config.list_physical_devices("GPU")),
            "tensorflow_version": tf.__version__}


def _code_identity():
    out = {}
    for rel in CODE_FILES:
        path = REPO / rel
        out[rel] = file_sha256(path) if path.exists() else None
    return out


def _summary(arrays, pred_price):
    """Model-side pooled original-price MAE vs naive on identical valid rows (not authoritative)."""
    rows = []
    for h in range(arrays["mask"].shape[1]):
        sel = arrays["mask"][:, h]
        n = int(sel.sum())
        if not n:
            rows.append({"count": 0, "mae": None, "naive": None, "relative_error": None})
            continue
        target = arrays["target_price"][sel, h]
        mae = math.fsum(np.abs(pred_price[sel, h] - target).tolist()) / n
        naive = math.fsum(np.abs(arrays["origin_price"][sel] - target).tolist()) / n
        rows.append({"count": n, "mae": mae, "naive": naive,
                     "relative_error": (mae - naive) / naive if naive > 0 else None})
    return rows


def train_unit(design_path, bundle_dir, out_dir, device, *, training_overrides=None):
    """Run one unit end to end; returns the receipt (also written as RECEIPT.json).

    ``training_overrides`` (``max_epochs``/``patience`` only) exist for software
    tests; any override stamps the receipt ``scientific: false``.
    """
    started, wall0 = datetime.now(timezone.utc), time.monotonic()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ContractError("OUT_DIR_NOT_EMPTY", str(out))
    if device not in ("cpu", "gpu"):
        raise ContractError("DEVICE", "device must be cpu or gpu")
    overrides = dict(training_overrides or {})
    if set(overrides) - set(OVERRIDABLE):
        raise ContractError("OVERRIDE", f"only {OVERRIDABLE} may be overridden in software tests")
    effective = effective_from_design(design_path)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": RECEIPT_SCHEMA, "experiment_id": effective["experiment_id"],
               "started_at": started.isoformat(), "scientific": not overrides, "overrides": overrides,
               "identity": {"design_sha256": effective["design_sha256"],
                            "effective_config_sha256": effective["effective_config_sha256"],
                            "code_files_sha256": _code_identity()},
               "artifacts": {}}
    try:
        from tools.i6dmo.model import enable_determinism
        receipt["determinism"] = enable_determinism()
        import tensorflow as tf
        if device == "gpu" and not tf.config.list_physical_devices("GPU"):
            raise TrainDisposition("NO_GPU_VISIBLE", "a GPU fit was requested; CPU fallback is forbidden")
        receipt["device"] = device
        data = load_data_bundle(bundle_dir, effective)
        receipt["identity"].update(bundle_manifest_sha256=data["manifest_sha256"],
                                   feature_manifest_sha256=data["feature_identity"]["manifest_sha256"])
        fit_rows, fit_dropped = trainable_rows(data["fit"])
        inner_rows, inner_dropped = trainable_rows(data["inner"])
        scales = fit_scales(fit_rows)
        training = {**effective["training"], **overrides}
        from tools.i6dmo import model as M
        placement = "/GPU:0" if device == "gpu" else "/CPU:0"

        def progress(event):
            _atomic_json(out / "PROGRESS.json", {**event, "max_epochs": training["max_epochs"],
                                                 "updated_at": datetime.now(timezone.utc).isoformat()})

        with tf.device(placement):
            bundle = M.build_dmo(effective["model"])
            result = fit_dmo(bundle, materialize_monitor_inputs(fit_rows, scales),
                             materialize_monitor_inputs(inner_rows, scales),
                             training, effective["seed"], progress,
                             ids=(fit_rows["row_id"], inner_rows["row_id"]))
            t0 = time.monotonic()
            predictions = {}
            for name in required_splits(effective):
                x, _, _ = transform(data[name], scales)
                predictions[name] = to_price(M.predict_matrix(bundle.forecast_model, x), data[name], scales)
            predict_s = time.monotonic() - t0
        model_doc = M.save_dmo(bundle, out / "model", extra={"scales_sha256": scales["scales_sha256"]})
        _atomic_json(out / "SCALES.json", scales)
        _atomic_json(out / "TRAINING.json", {
            "schema": "i6dmo.training.v1", "settings": training, "rows": {
                "fit_trainable": len(fit_rows["row_id"]), "fit_without_valid_head": fit_dropped,
                "inner_trainable": len(inner_rows["row_id"]), "inner_without_valid_head": inner_dropped},
            **result})
        (out / "predictions").mkdir(exist_ok=True)
        for name in required_splits(effective):
            np.savez(out / "predictions" / f"{name}.npz", row_id=data[name]["row_id"],
                     horizons_hours=np.asarray(effective["horizons_hours"], np.int64),
                     origin_price=data[name]["origin_price"], predicted_price=predictions[name],
                     target_price=data[name]["target_price"], mask=data[name]["mask"])
        files = ["model/forecast_model.keras", "model/MODEL.json", "SCALES.json", "TRAINING.json",
                 *[f"predictions/{n}.npz" for n in required_splits(effective)]]
        receipt["artifacts"] = {rel: file_sha256(out / rel) for rel in files}
        receipt["identity"].update(weights_sha256=model_doc["weights_sha256"],
                                   model_archive_sha256=model_doc["archive_sha256"],
                                   scales_sha256=scales["scales_sha256"])
        receipt["training"] = {k: result[k] for k in ("best_epoch", "best_monitored", "stop_reason",
                                                      "epochs_completed", "updates", "early_stop")}
        receipt["model_side_summary"] = {
            "note": "pooled original-price MAE/naive on identical valid rows; the independent replay governs",
            "horizons_hours": effective["horizons_hours"],
            **{name: _summary(data[name], predictions[name]) for name in required_splits(effective)}}
        receipt["timings"] = {"fit_s": result["fit_seconds"], "predict_s": predict_s}
        receipt.update(status="COMPLETED", code="COMPLETED")
    except TrainDisposition as exc:
        receipt.update(status="FAILED_TYPED", code=exc.code, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - every unit needs a terminal
        receipt.update(status="FAILED_UNTYPED", code="UNHANDLED_EXCEPTION",
                       detail=f"{type(exc).__name__}: {exc}"[:2000])
    receipt.setdefault("timings", {})["wall_s"] = time.monotonic() - wall0
    try:
        receipt["resources"] = _resources(device)
    except Exception:  # noqa: BLE001
        receipt["resources"] = None
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out / "RECEIPT.json", receipt)
    return receipt


EXIT = {"COMPLETED": 0, "FAILED_TYPED": 3, "FAILED_UNTYPED": 4}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.train",
                                     description="One I6-DMO fit from a frozen data bundle (no overrides).")
    parser.add_argument("--design", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    overrides = None
    raw = os.environ.get("I6DMO_TEST_TRAINING_OVERRIDES")
    if raw:  # software tests only: stamps the receipt scientific=false; no CLI flag exists
        try:
            overrides = json.loads(raw)
        except ValueError:
            print(json.dumps({"status": "CONTRACT_ERROR", "code": "OVERRIDE",
                              "detail": "I6DMO_TEST_TRAINING_OVERRIDES is not JSON"}))
            return 2
    try:
        receipt = train_unit(args.design, args.bundle, args.out, args.device,
                             training_overrides=overrides)
    except ContractError as exc:
        print(json.dumps({"status": "CONTRACT_ERROR", "code": exc.code, "detail": str(exc)}))
        return 2
    if args.json:
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    else:
        print(f"{receipt['status']} {receipt['code']}")
    return EXIT[receipt["status"]]


if __name__ == "__main__":
    sys.exit(main())
