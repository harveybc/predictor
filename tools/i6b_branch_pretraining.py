"""Bounded I6-B pretraining for exact production ``causal_conv1d`` branches.

The public trainer accepts TRAIN arrays only.  Its ordered tail is an inner
TRAIN selection split, separated from the fitting prefix by a purge.  It never
accepts, discovers, or reads outer VALIDATION or TEST data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorflow as tf

from predictor_plugins.modular_temporal.artifacts import load_donor, save_donor
from predictor_plugins.modular_temporal.pretraining import branch_autoencoder


keras = tf.keras
REPORT_SCHEMA = "predictor.i6b.branch_pretraining.v1"
STATUS_SCHEMA = "predictor.i6b.branch_pretraining.status.v1"
_PLUGIN = ("causal_conv1d", "2.0.0")
_INPUT_SHAPE = (24, 2)
_OUTPUT_SHAPE = (24, 16)
_SETTINGS = {
    "seed": 42,
    "max_epochs": 50,
    "patience": 5,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "min_delta": 0.0,
    "inner_tail_fraction": 0.2,
    "purge_rows": 24,
    "decoder_channels": 16,
}


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _settings(value):
    value = dict(value or {})
    unknown = set(value) - set(_SETTINGS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    result = {**_SETTINGS, **value}
    for name in ("seed", "max_epochs", "patience", "batch_size", "purge_rows", "decoder_channels"):
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, int) or number < (0 if name == "purge_rows" else 1):
            raise ValueError(f"{name} must be {'nonnegative' if name == 'purge_rows' else 'positive'} integer")
    _finite_number(result["learning_rate"], "learning_rate", positive=True)
    _finite_number(result["min_delta"], "min_delta", nonnegative=True)
    fraction = _finite_number(result["inner_tail_fraction"], "inner_tail_fraction", positive=True)
    if fraction >= 1:
        raise ValueError("inner_tail_fraction must be below one")
    return result


def _atomic_json(path, document):
    path = Path(path)
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _array_digest(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _verified_manifests(bundle):
    """Return exact production manifests or reject before data or output access."""
    branches = getattr(bundle, "branch_models", None)
    if not isinstance(branches, dict) or not branches:
        raise ValueError("verified modular bundle must expose named branch models")
    manifests = {}
    for name, model in branches.items():
        manifest = bundle.donor_manifest("branch", name)
        plugin = manifest.get("plugin", {})
        identity = (plugin.get("name"), plugin.get("version"))
        if identity != _PLUGIN:
            raise ValueError("I6-B requires causal_conv1d v2 branches without fallback")
        if manifest.get("params") != {"channels": 16, "kernel_size": 3}:
            raise ValueError("I6-B requires exact causal_conv1d v2 effective parameters")
        if tuple(manifest.get("input_shape", ())) != _INPUT_SHAPE or tuple(model.input_shape[1:]) != _INPUT_SHAPE:
            raise ValueError("I6-B branch input must be exactly 24x2")
        if tuple(manifest.get("output_shape", ())) != _OUTPUT_SHAPE or tuple(model.output_shape[1:]) != _OUTPUT_SHAPE:
            raise ValueError("I6-B branch output must be exactly 24x16")
        input_grid = tuple(manifest.get("input_grid", ()))
        output_grid = tuple(manifest.get("output_grid", ()))
        if len(input_grid) != 24 or output_grid != input_grid:
            raise ValueError("I6-B branch must preserve the exact 24-step temporal grid")
        manifests[name] = manifest
    return manifests


def _train_arrays(train_windows, branch_names):
    if not isinstance(train_windows, Mapping) or set(train_windows) != set(branch_names):
        raise ValueError("train_windows must name every and only verified bundle branch")
    arrays = {}
    for name in branch_names:
        raw = np.asarray(train_windows[name])
        if raw.dtype.kind not in "fi" or raw.ndim != 3 or tuple(raw.shape[1:]) != _INPUT_SHAPE:
            raise ValueError(f"{name} TRAIN windows must be numeric [N,24,2]")
        if len(raw) < 3 or not np.isfinite(raw).all():
            raise ValueError(f"{name} TRAIN windows must be nonempty and finite")
        arrays[name] = np.ascontiguousarray(raw, dtype="float32")
    return arrays


def _corpus(value):
    if not isinstance(value, dict) or set(value) != {"dataset_id", "support"}:
        raise ValueError("corpus must contain exactly dataset_id and TRAIN support")
    if not isinstance(value["dataset_id"], str) or not value["dataset_id"].strip() or not value["support"]:
        raise ValueError("corpus dataset_id and TRAIN support are required")
    # A strict JSON round trip rejects objects that could not be retained in provenance.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _partition(array, settings):
    tail_rows = max(1, int(math.ceil(len(array) * settings["inner_tail_fraction"])))
    tail_start = len(array) - tail_rows
    fit_end = tail_start - settings["purge_rows"]
    if fit_end < 1:
        raise ValueError("TRAIN windows are insufficient for fitting, purge, and inner tail")
    return array[:fit_end], array[tail_start:], {
        "ordered": True,
        "fit_rows": fit_end,
        "purged_rows": settings["purge_rows"],
        "inner_tail_rows": tail_rows,
        "inner_tail_start": tail_start,
    }


class MeanTrainTailEarlyStopping(keras.callbacks.Callback):
    """Select and restore the epoch minimizing mean(TRAIN, inner TRAIN tail)."""

    def __init__(self, *, patience, min_delta):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = 0
        self.best_score = float("inf")
        self.restored_best_weights = False
        self.history = []
        self._best_weights = None
        self._stale = 0

    def on_train_begin(self, logs=None):
        self.best_epoch = 0
        self.best_score = float("inf")
        self.restored_best_weights = False
        self.history = []
        self._best_weights = None
        self._stale = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = _finite_number(logs.get("loss"), "train_loss", nonnegative=True)
        tail_loss = _finite_number(logs.get("val_loss"), "inner_train_tail_loss", nonnegative=True)
        score = float((train_loss + tail_loss) / 2.0)
        record = {"epoch": epoch + 1, "train_loss": float(train_loss),
                  "inner_train_tail_loss": float(tail_loss), "selection_loss": score}
        self.history.append(record)
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_epoch = epoch + 1
            self._best_weights = [np.asarray(value).copy() for value in self.model.get_weights()]
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self._best_weights is None:
            raise ValueError("training produced no finite selected checkpoint")
        self.model.set_weights(self._best_weights)
        self.restored_best_weights = True


def _fit_branch(bundle, name, array, destination, corpus, settings, index):
    fit, tail, partition = _partition(array, settings)
    seed = settings["seed"] + index
    keras.utils.set_random_seed(seed)
    branch = bundle.branch_models[name]
    autoencoder = branch_autoencoder(bundle, name, channels=settings["decoder_channels"])
    if not any(layer is branch for layer in autoencoder.layers):
        raise ValueError("branch_autoencoder did not retain the production branch object")
    callback = MeanTrainTailEarlyStopping(patience=settings["patience"], min_delta=settings["min_delta"])
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss=keras.losses.MeanSquaredError(),
    )
    history = autoencoder.fit(
        fit,
        fit,
        validation_data=(tail, tail),
        epochs=settings["max_epochs"],
        batch_size=settings["batch_size"],
        shuffle=False,
        callbacks=[callback],
        verbose=0,
    )
    if not callback.restored_best_weights or callback.best_epoch < 1:
        raise ValueError("best branch weights were not restored")
    reconstruction = np.asarray(autoencoder.predict(tail, batch_size=settings["batch_size"], verbose=0))
    if reconstruction.shape != tail.shape or not np.isfinite(reconstruction).all():
        raise ValueError("reconstruction output is nonfinite or has the wrong shape")
    error = reconstruction.astype("float64") - tail.astype("float64")
    measured = {"state": "MEASURED", "mae_z": float(np.mean(np.abs(error))),
                "mse_z": float(np.mean(error ** 2)), "population": "INNER_TRAIN_TAIL"}
    objective = {
        "name": "branch_autoencoder_reconstruction",
        "loss": "mse",
        "monitor": "mean(train_loss,val_loss)",
        "validation_semantics": "PURGED_ORDERED_INNER_TRAIN_TAIL",
        "restore_best_weights": True,
        "ordered_inner_train_tail": True,
        "purge_rows": settings["purge_rows"],
    }
    learned_corpus = {"kind": "TRAIN_ONLY", "dataset_id": corpus["dataset_id"],
                      "data_sha256": _array_digest(array), "support": corpus["support"]}
    path = destination / f"{name}.keras"
    document = save_donor(
        branch,
        path,
        bundle.donor_manifest("branch", name),
        declared_params={"channels": 16, "kernel_size": 3},
        objective=objective,
        provenance={"conditioning_contract": "OPERATIONAL", "learned_corpus": learned_corpus,
                    "reconstruction": measured},
    )
    return {
        "branch": name,
        "artifact": path.name,
        "manifest": path.with_suffix(".manifest.json").name,
        "manifest_sha256": document["manifest_sha256"],
        "model_sha256": document["model_sha256"],
        "weights_sha256": document["weights_sha256"],
        "data_sha256": learned_corpus["data_sha256"],
        "seed": seed,
        "partition": partition,
        "epochs_completed": len(history.history["loss"]),
        "selected_epoch": callback.best_epoch,
        "selection_loss": callback.best_score,
        "restored_best_weights": callback.restored_best_weights,
        "curve": callback.history,
        "reconstruction": measured,
    }


def train_branch_donors(bundle, train_windows, output_dir, corpus, settings=None):
    """Pretrain and save every exact production branch from TRAIN arrays only.

    Parameters are intentionally limited to the verified bundle, TRAIN windows,
    output location, TRAIN corpus identity, and bounded settings.  Outer split
    arrays cannot enter this API.
    """
    manifests = _verified_manifests(bundle)
    resolved = _settings(settings)
    arrays = _train_arrays(train_windows, manifests)
    corpus = _corpus(corpus)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output_dir must be absent or empty; donors are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    branches = []
    for index, name in enumerate(manifests):
        branches.append(_fit_branch(bundle, name, arrays[name], destination, corpus, resolved, index))
    report = {"schema": REPORT_SCHEMA, "status": "COMPLETE", "learned_corpus": "TRAIN_ONLY",
              "branch_count": len(branches), "settings": resolved, "branches": branches}
    _atomic_json(destination / "REPORT.json", report)
    _atomic_json(destination / "STATUS.json", {"schema": STATUS_SCHEMA, "status": "COMPLETE",
                                                "completed": len(branches), "total": len(branches)})
    return report


def load_report(output_dir):
    """Read the retained report with strict schema and finite-number checks."""
    path = Path(output_dir) / "REPORT.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON number {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("I6-B report is missing or invalid") from exc
    if (not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "COMPLETE" or not isinstance(report.get("branches"), list)):
        raise ValueError("I6-B report schema mismatch")
    return report


def pretraining_status(output_dir, bundle):
    """Re-verify every retained donor against the requesting production bundle."""
    manifests = _verified_manifests(bundle)
    report = load_report(output_dir)
    rows = report["branches"]
    names = [row.get("branch") for row in rows]
    if names != list(manifests) or report.get("branch_count") != len(manifests):
        raise ValueError("I6-B report/bundle branch mismatch")
    for row in rows:
        path = Path(output_dir) / row["artifact"]
        loaded = load_donor(path, manifests[row["branch"]], require_contract="OPERATIONAL")
        if tuple(loaded.input_shape[1:]) != _INPUT_SHAPE or tuple(loaded.output_shape[1:]) != _OUTPUT_SHAPE:
            raise ValueError("I6-B loaded donor shape mismatch")
    return {"schema": STATUS_SCHEMA, "status": "COMPLETE", "completed": len(rows),
            "total": len(manifests), "artifacts_verified": True}
