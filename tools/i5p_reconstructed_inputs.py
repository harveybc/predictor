"""I5-P diagnostic: reconstruct selected observations at each forecast origin.

This is deliberately separate from the sealed FS4 queue. The caller supplies an
external SHA-256 for the exact input arrays and each retained terminal. The
trained feature-extractor code is pinned, and neither a model nor a normalizer
is fitted here. Inputs to the decoder end at the scored origin, never after it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.fs4_temporal_predictor import load_extractor

PINNED_EXTRACTOR_ROOT = Path(__file__).resolve().parents[2] / "feature-extractor-i5p-pinned-2b96548"
PINNED_EXTRACTOR_COMMIT = "2b96548ee707457e33f85aeb8acea488ef6a778d"
CODE_FILES = ("app/fs4_task_runner.py", "app/fs4_extractibility.py", "app/univariate_temporal.py",
              "app/univariate_temporal_pilot.py", "app/npz_encoder_adapter.py")
RESULT_SCHEMA = "fs4.extractibility.result.v1"
TASK_SCHEMA = "fs4.extractibility.task.v1"
DIAGNOSTIC_SCHEMA = "i5p.reconstructed_inputs.v1"


class Refusal(ValueError):
    """An identity or causal-support violation; never silently skip an origin."""


@dataclass(frozen=True)
class ReconstructedInputs:
    values: np.ndarray
    row_ids: np.ndarray
    target: np.ndarray
    naive: np.ndarray
    receipt: dict


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extractor_code_sha256(root: str | Path) -> str:
    h = hashlib.sha256()
    for rel in CODE_FILES:
        path = Path(root) / rel
        if not path.is_file():
            raise Refusal(f"EXTRACTOR_CODE_MISSING: {path}")
        h.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _array_hash(h, array, dtype) -> None:
    a = np.ascontiguousarray(array, dtype=dtype)
    h.update(str(a.shape).encode() + b"\0" + a.tobytes() + b"\0")


def data_sha256(timestamps, row_ids, values, columns, scored_rows, target, naive) -> str:
    """Exact diagnostic input identity; ordered columns, rows and targets are included."""
    h = hashlib.sha256()
    h.update(json.dumps(list(columns), separators=(",", ":")).encode() + b"\0")
    for a, dtype in ((timestamps, "<i8"), (row_ids, "<i8"), (values, "<f8"), (scored_rows, "<i8"),
                     (target, "<f8"), (naive, "<f8")):
        _array_hash(h, a, dtype)
    return h.hexdigest()


def load_pinned_extractor(root: str | Path):
    root = Path(root).resolve()
    if not (root / "app" / "fs4_extractibility.py").is_file():
        raise Refusal(f"EXTRACTOR_CODE_MISSING: {root}")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED_EXTRACTOR_COMMIT:
        raise Refusal(f"EXTRACTOR_COMMIT_MISMATCH: {head}")
    X, U = load_extractor(root)
    if Path(X.__file__).resolve() != (root / "app" / "fs4_extractibility.py").resolve():
        raise Refusal("EXTRACTOR_PACKAGE_MISMATCH")
    return X, U


def _terminal(path: Path, expected_sha: str, feature: str, X, code_sha: str) -> dict:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise Refusal(f"TERMINAL_IDENTITY_MISMATCH: {feature}")
    rec = json.loads(path.read_text())
    if rec.get("feature_id") != feature:
        raise Refusal(f"FEATURE_MISMATCH: {feature}")
    if rec.get("architecture", {}).get("id") != X.ARCHITECTURE_ID_V2 or \
            rec["architecture"].get("last_latent_lag_rows") != 0:
        raise Refusal(f"ARCHITECTURE_MISMATCH: {feature}")
    if rec.get("schema") != RESULT_SCHEMA or rec.get("status") != "COMPLETE" or \
            rec.get("arm") != "TRAINED_ENCODER_V2":
        raise Refusal(f"TERMINAL_CONTRACT_MISMATCH: {feature}")
    commit = rec.get("code_commit")
    if not isinstance(commit, str) or len(commit) != 40 or rec.get("code_sha256") != code_sha:
        raise Refusal(f"EXTRACTOR_CODE_MISMATCH: {feature}")
    claim = {k: rec.get(k) for k in ("population_id", "identity", "feature_id", "fold_id", "arm", "seed")}
    claim["schema"] = TASK_SCHEMA
    if X.task_digest(claim) != rec.get("task_id"):
        raise Refusal(f"TASK_IDENTITY_MISMATCH: {feature}")
    source = rec.get("source", {})
    identity = rec.get("identity")
    corpora = X.CORPORA.get(identity)
    if not corpora or rec.get("population_id") != corpora["population_id"] or \
            source.get("file_sha256") != corpora["files"] or \
            source.get("role") not in corpora["files"] or \
            source.get("grid_seconds") != X.GRID_SECONDS or \
            source.get("bar_seconds") != corpora["bar_seconds"]:
        raise Refusal(f"TRAINING_SOURCE_MISMATCH: {feature}")
    fold = rec.get("fold", {})
    hp = rec.get("hyper", {})
    if fold.get("fold_id") != rec.get("fold_id") or \
            rec.get("fold_id") not in X.FOLD_YEARS or hp.get("window") != X.WINDOW:
        raise Refusal(f"TRAINING_CONFIG_MISMATCH: {feature}")
    input_sha = X.digest({"identity": identity, "population_id": rec["population_id"],
                          "feature_id": feature, "file_sha256": source["file_sha256"],
                          "column_sha256": source.get("column_sha256"), "grid_seconds": X.GRID_SECONDS,
                          "window": hp["window"], "fold": {"fold_id": fold["fold_id"],
                                                         "fit": fold.get("fit"), "val": fold.get("val")}})
    if rec.get("input_sha256") != input_sha or rec.get("architecture", {}).get("target_input") is not False:
        raise Refusal(f"TRAINING_INPUT_IDENTITY_MISMATCH: {feature}")
    if rec.get("architecture", {}).get("calendar_context") != list(X.U.CALENDAR_SPEC):
        raise Refusal(f"CALENDAR_CONFIG_MISMATCH: {feature}")
    nm = rec.get("normalization", {})
    if not all(isinstance(nm.get(k), (int, float)) and np.isfinite(nm[k]) for k in ("mean", "std")) or \
            nm["std"] <= 0 or type(nm.get("n")) is not int or nm["n"] < 1 or \
            type(nm.get("constant")) is not bool:
        raise Refusal(f"NORMALIZATION_MISMATCH: {feature}")
    return rec


def reconstruct_selected_inputs(*, timestamps, row_ids, values, columns, scored_rows, target, naive,
                                checkpoints: dict, expected_data_sha256: str,
                                extractor_root: str | Path = PINNED_EXTRACTOR_ROOT) -> ReconstructedInputs:
    """Return one reconstructed observed value per selected feature and scored origin.

    `scored_rows` are positional indices; `row_ids` are the original dataset's
    identifiers. No row is dropped. Missing origin observations and insufficient
    24-hour history refuse.
    The target and naive pass through unchanged and are not model inputs.
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    ids = np.asarray(row_ids, dtype=np.int64)
    raw = np.asarray(values, dtype=np.float64)
    rows = np.asarray(scored_rows, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    naive = np.asarray(naive, dtype=np.float64)
    names = tuple(columns)
    if not names or len(names) != len(set(names)) or set(names) != set(checkpoints) or \
            raw.ndim != 2 or raw.shape != (len(ts), len(names)):
        raise Refusal("COLUMN_MISMATCH")
    if ts.ndim != 1 or ids.shape != ts.shape or len(ts) < 24 or np.any(np.diff(ts) <= 0) or \
            len(np.unique(ids)) != len(ids) or np.any(ts % 3600 != 0) or \
            rows.ndim != 1 or not len(rows) or np.any(np.diff(rows) <= 0) or \
            rows[0] < 0 or rows[-1] >= len(ts) or target.shape != rows.shape or naive.shape != rows.shape:
        raise Refusal("ROW_SUPPORT_MISMATCH")
    if not np.isfinite(target).all() or not np.isfinite(naive).all():
        raise Refusal("NONFINITE_INPUT")
    actual_sha = data_sha256(ts, ids, raw, names, rows, target, naive)
    if actual_sha != expected_data_sha256:
        raise Refusal("DATA_IDENTITY_MISMATCH")
    X, U = load_pinned_extractor(extractor_root)
    code_sha = extractor_code_sha256(extractor_root)
    reconstructed = np.empty((len(rows), len(names)), dtype=np.float64)
    identities = []
    for col, feature in enumerate(names):
        ref = checkpoints[feature]
        path = Path(ref["terminal"])
        rec = _terminal(path, ref["sha256"], feature, X, code_sha)
        weights = path.parent / "chosen.weights.h5"
        if not weights.is_file():
            raise Refusal(f"CHECKPOINT_MISSING: {feature}")
        file_sha = sha256_file(weights)
        if file_sha != rec.get("artifacts", {}).get("chosen_weights_file_sha256"):
            raise Refusal(f"CHECKPOINT_DIGEST_MISMATCH: {feature}")
        hp = X.Hyper(**rec["hyper"])
        encoder, decoder, model = X.build_origin_covering_models(hp)
        model.load_weights(weights)
        model_sha = X.weights_digest([encoder, decoder])
        if model_sha != rec.get("model_sha256") or model_sha != rec.get("weights", {}).get("chosen_weights_sha256"):
            raise Refusal(f"MODEL_IDENTITY_MISMATCH: {feature}")
        norm_data = rec["normalization"]
        norm = U.Normalization(float(norm_data["mean"]), float(norm_data["std"]),
                               int(norm_data["n"]), norm_data["constant"])
        grid = X.to_grid(ts, raw[:, col])
        anchors = grid.row_index[rows]
        if np.any(anchors < hp.window - 1) or not grid.observed[anchors].all():
            raise Refusal(f"ORIGIN_SUPPORT_MISMATCH: {feature}")
        batch = U.make_windows(grid.ts, grid.x, grid.observed, anchors, hp.window, norm)
        prediction = np.asarray(model.predict(batch.as_inputs(), batch_size=512, verbose=0), np.float64)
        if prediction.shape != (len(rows), hp.window, 1) or not np.isfinite(prediction).all():
            raise Refusal(f"RECONSTRUCTION_INVALID: {feature}")
        reconstructed[:, col] = prediction[:, -1, 0] * norm.std + norm.mean
        identities.append({"feature_id": feature, "terminal_sha256": ref["sha256"],
                           "weights_file_sha256": file_sha, "model_sha256": model_sha,
                           "task_id": rec["task_id"], "producer_commit": rec["code_commit"]})
    receipt = {"schema": DIAGNOSTIC_SCHEMA, "diagnostic_only": True,
               "input_sha256": actual_sha, "selected_features": list(names),
               "rows": len(rows), "extractor_commit": PINNED_EXTRACTOR_COMMIT,
               "extractor_code_sha256": code_sha, "models": identities,
               "support": "causal hourly windows [t-23h,t], reconstruction at t only"}
    return ReconstructedInputs(reconstructed, ids[rows].copy(), target.copy(), naive.copy(), receipt)
