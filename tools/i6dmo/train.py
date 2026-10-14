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


MONITOR_EVAL = "i6dmo.monitor_eval.v1"


def evaluate_monitored(model, fit, inner, batch_size):
    """THE monitored-loss evaluation procedure (training and replay share it, DMO10).

    One fixed path: frozen-weight predictions through ``predict_matrix`` at the
    given batch size (float32 inputs, float64 outputs per batch, concatenated in
    order), then :func:`masked_split_metrics` global per-head float64 ``fsum``
    sums/counts, equal-weight mean over supported heads, and
    ``0.5 * (fit_loss + inner_loss)``. Bitwise identity on one device requires
    the identical batch size and order, which the training record pins.
    """
    from tools.i6dmo.model import predict_matrix
    fm = masked_split_metrics(fit[1], predict_matrix(model, fit[0], batch_size=batch_size), fit[2])
    im = masked_split_metrics(inner[1], predict_matrix(model, inner[0], batch_size=batch_size), inner[2])
    return fm, im, 0.5 * (fm["loss"] + im["loss"])


def replay_monitored(model, fit, inner, record, batch_size=None):
    """Recompute the monitored loss through the recorded identical procedure.

    ``record`` is TRAINING.json's ``monitor_evaluation``. A different procedure
    name or batch size is a typed MONITOR_PATH_MISMATCH: a one-pass recompute of
    a batched training monitor is a different reduction order, not a replay.
    """
    if not isinstance(record, dict) or record.get("procedure") != MONITOR_EVAL:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"recorded procedure {record.get('procedure') if isinstance(record, dict) else record!r}"
                               f" is not {MONITOR_EVAL}")
    recorded = record.get("batch_size")
    if type(recorded) is not int or recorded < 1:
        raise TrainDisposition("MONITOR_PATH_MISMATCH", "recorded batch_size missing")
    if batch_size is not None and batch_size != recorded:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"replay batch_size {batch_size} != recorded {recorded}; "
                               "the monitored value is only comparable through the identical path")
    return evaluate_monitored(model, fit, inner, recorded)


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


def fit_dmo(bundle, fit, inner, training, seed, progress=None):
    """One fit with AdamW; returns the audit dict. Restores the selected weights."""
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
        fm, im, _ = evaluate_monitored(model, (x_fit, y_fit, m_fit), (x_in, y_in, m_in), batch)
        stop = stopper.update(fm["loss"], im["loss"], model.get_weights())
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
    if int(optimizer.iterations.numpy()) != updates:
        raise TrainDisposition("UPDATE_COUNT_MISMATCH", "optimizer iterations disagree with updates")
    return {"history": rows, "best_epoch": stopper.best_epoch, "best_monitored": stopper.best_monitored,
            "monitor_evaluation": {"procedure": MONITOR_EVAL, "batch_size": batch,
                                   "prediction_dtype": "float32 inputs, float64 per-batch outputs",
                                   "accumulation": "global per-head float64 math.fsum sums/counts, "
                                                   "equal-weight mean over supported heads"},
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
        import tensorflow as tf
        from tools.i6dmo.model import enable_determinism
        receipt["determinism"] = enable_determinism()
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
            result = fit_dmo(bundle, transform(fit_rows, scales), transform(inner_rows, scales),
                             training, effective["seed"], progress)
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
