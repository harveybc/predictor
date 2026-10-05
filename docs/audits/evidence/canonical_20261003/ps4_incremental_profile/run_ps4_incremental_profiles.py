"""Resumable TRAIN-only PS4 profiles for one series column at a time.

A column is read from a local ``series.npz`` under ``x__<feature_id>`` or the
bare feature id. The population is the train prefix of one of the five existing
inner folds. This runner does not select features, does not train a model, and
does not read a target, an external validation split, or a test split.

The ten transforms already measured by the published PS4 profile are refused
before any series read. A unit counts as MEASURED only after an atomic write of
a complete payload. A partial file is not MEASURED and is recomputed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
TRANSFORM_DIR = HERE.parent / "ps4_transform_profile"
for entry in (str(ROOT), str(TRANSFORM_DIR), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_ps4_transform_profiles import (  # noqa: E402
    ProfileRefusal,
    load_train_folds,
    profile_fold,
    sha256_file,
)

INNER_FOLDS = ("inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023")
CANONICAL_TRAIN_ENDS = {
    "inner_2019": 41329,
    "inner_2020": 47542,
    "inner_2021": 53771,
    "inner_2022": 59966,
    "inner_2023": 66206,
}
CANONICAL_FOLD_SHA256 = "a376bec1e0614db5d93ca601be1a3d16136a6b29d408e9738b0b9e7667337c2c"
ALREADY_MEASURED_TRANSFORMS = (
    "tv.hilbert_amp",
    "tv.kalman_dev",
    "tv.stl_dev",
    "tv.stl_seasonal",
    "tv.wav_d1",
    "tv.wav_d2",
    "tv.wav_d3",
    "tv.wav_d4",
    "tv.wav_d5",
    "tv.mt_band_6_48h",
)
SELECTION_DENOMINATOR = 366
DATASET_ID = "eurusd_ps4_incremental_profile"
UNIT_SCHEMA = "ps4_incremental_unit.v2"
LEGACY_UNIT_SCHEMA = "ps4_incremental_unit.v1"
REPORT_SCHEMA = "ps4_incremental_profile.v2"
TERMINAL_SCHEMA = "ps3r_terminal_identity.v1"
ACCEPTED_INDEX_SCHEMA = "ps4_accepted_units.v1"
_SPLIT_WORDS = {"validation", "external_validation", "outer_validation", "val"}
_TEST_WORDS = {"test", "outer_test"}


def canonical_feature_id(feature_id: str) -> str:
    name = feature_id.strip()
    if name.startswith("x__"):
        name = name[3:]
    return name


def _is_target_name(name: str) -> bool:
    bare = canonical_feature_id(name)
    lowered = bare.lower()
    return bare.startswith("Y_") or lowered.startswith("y_") or lowered.startswith("target")


def assert_not_target(feature_id: str) -> str:
    name = canonical_feature_id(feature_id)
    if not name or _is_target_name(name):
        raise ProfileRefusal("TARGET_COLUMN", feature_id.strip() or "<empty>")
    return name


def assert_split(split: str) -> None:
    token = split.strip().lower()
    if token == "train":
        return
    if token in _TEST_WORDS or "test" in token:
        raise ProfileRefusal("TEST_SPLIT", split)
    if token in _SPLIT_WORDS or "val" in token:
        raise ProfileRefusal("EXTERNAL_VALIDATION_SPLIT", split)
    raise ProfileRefusal("SPLIT_NOT_TRAIN", split)


def assert_fold_name(name: str) -> None:
    if name in INNER_FOLDS:
        return
    lowered = name.lower()
    if lowered in _TEST_WORDS or "test" in lowered:
        raise ProfileRefusal("TEST_SPLIT", name)
    if lowered in _SPLIT_WORDS or "val" in lowered:
        raise ProfileRefusal("EXTERNAL_VALIDATION_SPLIT", name)
    raise ProfileRefusal("FOLD_BOUNDARY_ABSENT", name)


def assert_not_already_measured(feature_id: str) -> str:
    name = assert_not_target(feature_id)
    if name in ALREADY_MEASURED_TRANSFORMS:
        raise ProfileRefusal("ALREADY_MEASURED", name)
    return name


def load_inner_folds(path: Path, n_rows: int, *, canonical_bounds: bool) -> list[dict]:
    folds = load_train_folds(path, n_rows)
    names = [item["name"] for item in folds]
    if names != list(INNER_FOLDS):
        for name in names:
            assert_fold_name(name)
        raise ProfileRefusal("FOLD_BOUNDARY_ABSENT", ",".join(names))
    if canonical_bounds:
        digest = sha256_file(path)
        if digest != CANONICAL_FOLD_SHA256:
            raise ProfileRefusal(
                "FOLD_BOUNDARY_ABSENT",
                "folds digest is not the authenticated inner boundaries",
            )
        for item in folds:
            expected = CANONICAL_TRAIN_ENDS[item["name"]]
            if item["train_end"] != expected:
                raise ProfileRefusal(
                    "FOLD_BOUNDARY_ABSENT",
                    f"{item['name']} {item['train_end']} != {expected}",
                )
    return folds


def assert_canonical_fold_file(path: Path) -> list[dict]:
    """Check the five authenticated boundaries without reading a series."""
    return load_inner_folds(path, max(CANONICAL_TRAIN_ENDS.values()) + 1, canonical_bounds=True)


def load_series_column(path: Path, feature_id: str) -> np.ndarray:
    name = assert_not_target(feature_id)
    path = Path(path)
    if path.name == "targets.npz":
        raise ProfileRefusal("TARGET_COLUMN", "targets.npz")
    if path.is_symlink() or not path.is_file():
        raise ProfileRefusal("SERIES_ABSENT", path.name)
    prefixed = "x__" + name
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        has_prefixed = prefixed in names
        has_bare = name in names
        if not has_prefixed and not has_bare:
            raise ProfileRefusal("FEATURE_ABSENT_FROM_SERIES", name)
        if has_prefixed and has_bare:
            left = np.asarray(archive[prefixed])
            right = np.asarray(archive[name])
            if left.shape != right.shape or not np.array_equal(left, right, equal_nan=True):
                raise ProfileRefusal("AMBIGUOUS_SERIES_COLUMN", name)
            column = left
        else:
            key = prefixed if has_prefixed else name
            if _is_target_name(key):
                raise ProfileRefusal("TARGET_COLUMN", key)
            column = np.asarray(archive[key])
    values = np.asarray(column, dtype=np.float64)
    if values.ndim != 1:
        raise ProfileRefusal("SERIES_COLUMN_RANK", f"{name} ndim={values.ndim}")
    return values


def unit_path(out: Path, feature_id: str, fold: str) -> Path:
    safe = canonical_feature_id(feature_id).replace("/", "_")
    return Path(out) / "units" / f"{safe}__{fold}.json"


def _rows_sha256(rows: list) -> str:
    body = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _json_sha256(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def load_terminal_identity(path: Path | None, feature_id: str, series_sha256: str) -> dict:
    """Authenticate the retained PS3-R terminal that made a PS4 unit eligible."""
    if path is None:
        raise ProfileRefusal("TERMINAL_MANIFEST_ABSENT", feature_id)
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ProfileRefusal("TERMINAL_MANIFEST_ABSENT", str(path))
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileRefusal("TERMINAL_MANIFEST_INVALID", str(error)) from error
    families = manifest.get("families") if isinstance(manifest, dict) else None
    features = manifest.get("features") if isinstance(manifest, dict) else None
    seed = manifest.get("seed") if isinstance(manifest, dict) else None
    results_file = manifest.get("results_file") if isinstance(manifest, dict) else None
    required = (
        manifest.get("schema") == "ut_pilot_run.v1"
        and manifest.get("status") == "COMPLETED"
        and features == [feature_id]
        and isinstance(seed, int) and not isinstance(seed, bool)
        and isinstance(manifest.get("code_commit"), str) and bool(manifest["code_commit"])
        and isinstance(families, list) and bool(families)
        and all(isinstance(family, str) and family for family in families)
        and len(families) == len(set(families))
        and _is_sha256(manifest.get("results_sha256"))
        and isinstance(results_file, str) and results_file == Path(results_file).name
        and manifest.get("series_sha256") == series_sha256
    )
    if not required:
        raise ProfileRefusal("TERMINAL_IDENTITY_MISMATCH", feature_id)
    retained_results = path.parent / results_file
    if retained_results.is_symlink() or not retained_results.is_file():
        raise ProfileRefusal("TERMINAL_RESULTS_ABSENT", str(retained_results))
    actual_results_sha256 = sha256_file(retained_results)
    if actual_results_sha256 != manifest["results_sha256"]:
        raise ProfileRefusal("TERMINAL_RESULTS_DIGEST_MISMATCH", results_file)
    return {
        "schema": TERMINAL_SCHEMA,
        "results_sha256": manifest["results_sha256"],
        "seed": seed,
        "code_revision": manifest["code_commit"],
        "families": families,
        "results_file": results_file,
        "terminal_manifest_sha256": sha256_file(path),
    }


def _identity_payload(payload: dict) -> dict:
    return {
        "feature_id": payload.get("feature_id"),
        "fold_id": payload.get("fold_id"),
        "train_rows": payload.get("train_rows"),
        "source_sha256": payload.get("source_sha256"),
        "terminal_identity": payload.get("terminal_identity"),
        "rows_sha256": payload.get("rows_sha256"),
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def classify_unit(path: Path, *, expected_feature_id: str = "", expected_fold_id: str = "",
                  expected_train_rows: list[int] | None = None,
                  expected_source_sha256: dict | None = None,
                  expected_terminal_identity: dict | None = None,
                  expected_acceptance: dict | None = None) -> str:
    """MEASURED only for a complete atomic unit. Partials are PENDING."""
    path = Path(path)
    if path.name.endswith(".tmp"):
        return "PENDING"
    if not path.is_file():
        return "ABSENT"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "PENDING"
    if not isinstance(payload, dict) or payload.get("schema") != UNIT_SCHEMA:
        return "PENDING"
    acceptance_valid = (
        isinstance(expected_acceptance, dict)
        and expected_acceptance.get("unit") == path.name
        and expected_acceptance.get("unit_sha256") == sha256_file(path)
    )
    feature = payload.get("feature_id")
    fold = payload.get("fold_id")
    train_rows = payload.get("train_rows")
    source = payload.get("source_sha256")
    terminal = payload.get("terminal_identity")
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("rows_sha256") != _rows_sha256(rows):
        return "PENDING"
    valid_identity = (
        isinstance(feature, str) and bool(feature)
        and fold in INNER_FOLDS
        and isinstance(train_rows, list) and len(train_rows) == 2
        and train_rows[0] == 0 and isinstance(train_rows[1], int) and train_rows[1] > 0
        and isinstance(source, dict) and set(source) == {"series.npz", "folds.json"}
        and all(_is_sha256(value) for value in source.values())
        and isinstance(terminal, dict)
        and terminal.get("schema") == TERMINAL_SCHEMA
        and _is_sha256(terminal.get("results_sha256"))
        and isinstance(terminal.get("seed"), int) and not isinstance(terminal.get("seed"), bool)
        and isinstance(terminal.get("code_revision"), str) and bool(terminal["code_revision"])
        and isinstance(terminal.get("families"), list) and bool(terminal["families"])
        and len(terminal["families"]) == len(set(terminal["families"]))
        and _is_sha256(terminal.get("terminal_manifest_sha256"))
    )
    if not valid_identity:
        return "PENDING"
    if payload.get("identity_sha256") != _json_sha256(_identity_payload(payload)):
        return "PENDING"
    if expected_feature_id and feature != expected_feature_id:
        return "PENDING"
    if expected_fold_id and fold != expected_fold_id:
        return "PENDING"
    if expected_train_rows is not None and train_rows != expected_train_rows:
        return "PENDING"
    if expected_source_sha256 is not None and source != expected_source_sha256:
        return "PENDING"
    if expected_terminal_identity is not None and terminal != expected_terminal_identity:
        return "PENDING"
    if acceptance_valid:
        for field, actual in (
            ("feature_id", feature),
            ("fold_id", fold),
            ("source_sha256", source),
            ("terminal_identity", terminal),
        ):
            if expected_acceptance.get(field) != actual:
                acceptance_valid = False
                break
    status = payload.get("unit_status")
    rows_match = rows and all(
        isinstance(row, dict)
        and row.get("feature_id") == feature
        and row.get("fold") == fold
        and row.get("train_rows") == train_rows
        and row.get("source_digests") == source
        for row in rows
    )
    if (status == "MEASURED" and acceptance_valid and rows_match
            and payload.get("metric_rows") == len(rows)):
        return "MEASURED"
    if status == "FAILED" and rows == [] and payload.get("reason") and payload.get("metric_rows") == 0:
        return "FAILED"
    return "PENDING"


def load_accepted_index(path: Path, expected_sha256: str) -> dict[str, dict]:
    """Load an externally retained acceptance index anchored by a caller-owned digest."""
    path = Path(path)
    if not _is_sha256(expected_sha256):
        raise ProfileRefusal("ACCEPTED_INDEX_DIGEST_REQUIRED", str(path))
    if path.is_symlink() or not path.is_file():
        raise ProfileRefusal("ACCEPTED_INDEX_ABSENT", str(path))
    if sha256_file(path) != expected_sha256:
        raise ProfileRefusal("ACCEPTED_INDEX_DIGEST_MISMATCH", str(path))
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileRefusal("ACCEPTED_INDEX_INVALID", str(error)) from error
    rows = payload.get("units") if isinstance(payload, dict) else None
    if payload.get("schema") != ACCEPTED_INDEX_SCHEMA or not isinstance(rows, list):
        raise ProfileRefusal("ACCEPTED_INDEX_INVALID", str(path))
    accepted = {}
    for row in rows:
        name = row.get("unit") if isinstance(row, dict) else None
        valid = (
            isinstance(name, str) and bool(name) and name == Path(name).name
            and _is_sha256(row.get("unit_sha256"))
            and isinstance(row.get("feature_id"), str) and bool(row["feature_id"])
            and row.get("fold_id") in INNER_FOLDS
            and isinstance(row.get("source_sha256"), dict)
            and isinstance(row.get("terminal_identity"), dict)
        )
        if not valid or name in accepted:
            raise ProfileRefusal("ACCEPTED_INDEX_INVALID", str(name))
        accepted[name] = row
    return accepted


def _unit_body(feature: str, fold: str, train_end: int, rows: list, source: dict,
               terminal_identity: dict, status: str, reason: str) -> dict:
    payload = {
        "schema": UNIT_SCHEMA,
        "unit_status": status,
        "feature_id": feature,
        "fold_id": fold,
        "train_rows": [0, int(train_end)],
        "reason": reason,
        "source_sha256": dict(source),
        "terminal_identity": dict(terminal_identity),
        "rows": rows,
        "rows_sha256": _rows_sha256(rows),
        "metric_rows": len(rows),
    }
    payload["identity_sha256"] = _json_sha256(_identity_payload(payload))
    return payload


def _requested_train_rows(folds_path: Path, fold_name: str, canonical_bounds: bool) -> list[int]:
    folds = load_inner_folds(folds_path, 10 ** 18, canonical_bounds=canonical_bounds)
    chosen = next(item for item in folds if item["name"] == fold_name)
    return [0, int(chosen["train_end"])]


def _thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def execute_unit(series: Path, folds_path: Path, feature_id: str, fold_name: str, out: Path, *,
                 split: str = "train", dataset_id: str = DATASET_ID,
                 canonical_bounds: bool = False, expected_digest: str = "",
                 terminal_manifest: Path | None = None,
                 expected_acceptance: dict | None = None) -> dict:
    """Profile one feature-fold, or resume when that unit is already MEASURED."""
    _thread_env()
    assert_split(split)
    assert_fold_name(fold_name)
    feature = assert_not_already_measured(feature_id)
    series = Path(series)
    if series.name == "targets.npz":
        raise ProfileRefusal("TARGET_COLUMN", "targets.npz")
    if series.is_symlink() or not series.is_file():
        raise ProfileRefusal("SERIES_ABSENT", series.name)
    actual_digest = sha256_file(series)
    if expected_digest and actual_digest != expected_digest:
        raise ProfileRefusal("DIGEST_MISMATCH", "series sha256 does not match the declared digest")
    terminal_identity = load_terminal_identity(terminal_manifest, feature, actual_digest)
    train_rows = _requested_train_rows(folds_path, fold_name, canonical_bounds)
    source = {"series.npz": actual_digest, "folds.json": sha256_file(folds_path)}
    destination = unit_path(out, feature, fold_name)
    state = classify_unit(
        destination,
        expected_feature_id=feature,
        expected_fold_id=fold_name,
        expected_train_rows=train_rows,
        expected_source_sha256=source,
        expected_terminal_identity=terminal_identity,
        expected_acceptance=expected_acceptance,
    )
    if state in {"MEASURED", "FAILED"}:
        return {
            "resumed": True,
            "unit_status": state,
            "feature_id": feature,
            "fold": fold_name,
            "metric_rows": json.loads(destination.read_text()).get("metric_rows", 0),
            "unit": destination.name,
        }
    if destination.is_file():
        try:
            retained = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError):
            retained = None
        if (
            isinstance(retained, dict)
            and retained.get("schema") == UNIT_SCHEMA
            and retained.get("unit_status") in {"MEASURED", "FAILED"}
        ):
            raise ProfileRefusal(
                "UNIT_IDENTITY_MISMATCH",
                "retained terminal unit contradicts the requested source or identity",
            )
    try:
        column = load_series_column(series, feature)
        folds = load_inner_folds(folds_path, int(column.size), canonical_bounds=canonical_bounds)
        chosen = next(item for item in folds if item["name"] == fold_name)
        if [0, int(chosen["train_end"])] != train_rows:
            raise ProfileRefusal("FOLD_BOUNDARY_CHANGED", fold_name)
        rows = profile_fold(dataset_id, feature, column, fold_name, chosen["train_end"], source)
        payload = _unit_body(
            feature, fold_name, chosen["train_end"], rows, source,
            terminal_identity, "MEASURED", "",
        )
    except ProfileRefusal:
        raise
    except Exception:
        payload = _unit_body(
            feature, fold_name, train_rows[1], [], source,
            terminal_identity, "FAILED", "PROFILER_EXCEPTION",
        )
        _write_json(destination, payload)
        return {
            "resumed": False,
            "unit_status": "FAILED",
            "feature_id": feature,
            "fold": fold_name,
            "metric_rows": 0,
            "unit": destination.name,
        }
    _write_json(destination, payload)
    return {
        "resumed": False,
        "unit_status": "MEASURED",
        "feature_id": feature,
        "fold": fold_name,
        "metric_rows": len(rows),
        "unit": destination.name,
    }


def bind_legacy_unit(path: Path, terminal_manifest: Path) -> dict:
    """Bind retained v1 scientific rows to a verified PS3-R terminal without recomputing them."""
    path = Path(path)
    try:
        legacy = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileRefusal("LEGACY_UNIT_INVALID", str(error)) from error
    feature = legacy.get("feature_id") if isinstance(legacy, dict) else None
    fold = legacy.get("fold") if isinstance(legacy, dict) else None
    train_rows = legacy.get("train_rows") if isinstance(legacy, dict) else None
    source = legacy.get("source_sha256") if isinstance(legacy, dict) else None
    rows = legacy.get("rows") if isinstance(legacy, dict) else None
    valid = (
        legacy.get("schema") == LEGACY_UNIT_SCHEMA
        and legacy.get("unit_status") == "MEASURED"
        and isinstance(feature, str) and bool(feature)
        and fold in INNER_FOLDS
        and isinstance(train_rows, list) and len(train_rows) == 2
        and train_rows[0] == 0 and isinstance(train_rows[1], int) and train_rows[1] > 0
        and isinstance(source, dict) and set(source) == {"series.npz", "folds.json"}
        and all(_is_sha256(value) for value in source.values())
        and isinstance(rows, list) and bool(rows)
        and legacy.get("rows_sha256") == _rows_sha256(rows)
        and legacy.get("metric_rows") == len(rows)
        and all(
            isinstance(row, dict)
            and row.get("feature_id") == feature
            and row.get("fold") == fold
            and row.get("train_rows") == train_rows
            and row.get("source_digests") == source
            for row in rows
        )
    )
    if not valid:
        raise ProfileRefusal("LEGACY_UNIT_INVALID", path.name)
    terminal = load_terminal_identity(terminal_manifest, feature, source["series.npz"])
    rebound = _unit_body(
        feature, fold, train_rows[1], rows, source, terminal, "MEASURED", "",
    )
    _write_json(path, rebound)
    generated_acceptance = {
        "unit": path.name,
        "unit_sha256": sha256_file(path),
        "feature_id": feature,
        "fold_id": fold,
        "source_sha256": source,
        "terminal_identity": terminal,
    }
    if classify_unit(path, expected_acceptance=generated_acceptance) != "MEASURED":
        raise ProfileRefusal("LEGACY_BIND_FAILED", path.name)
    return {
        "unit": path.name,
        "unit_status": "MEASURED",
        "terminal_manifest_sha256": terminal["terminal_manifest_sha256"],
        "scientific_rows_recomputed": False,
    }


def drive(features: list[str], series: Path, folds_path: Path, out: Path, *,
          fold: str, split: str = "train", dataset_id: str = DATASET_ID,
          canonical_bounds: bool = False, expected_digest: str = "",
          terminal_manifest: Path | None = None) -> dict:
    """Profile features that are not among the ten already measured transforms."""
    skipped = []
    units = []
    for feature in features:
        name = canonical_feature_id(feature)
        if name in ALREADY_MEASURED_TRANSFORMS:
            skipped.append(name)
            continue
        units.append(execute_unit(
            series, folds_path, name, fold, out, split=split, dataset_id=dataset_id,
            canonical_bounds=canonical_bounds, expected_digest=expected_digest,
            terminal_manifest=terminal_manifest,
        ))
    return {"skipped_already_measured": skipped, "units": units}


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def scan_evidence(evidence: Path) -> dict:
    """Find declared series.npz files. Symlinks are ignored. Bytes are not profiled."""
    evidence = Path(evidence)
    present = []
    missing = []
    declared = set()
    for manifest in sorted(evidence.rglob("batch_manifest.json")):
        if manifest.is_symlink() or not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text())
        series = payload.get("series") if isinstance(payload, dict) else None
        if not isinstance(series, dict) or series.get("file") != "series.npz":
            continue
        path = manifest.parent / "series.npz"
        item = {
            "manifest": _rel(manifest, evidence),
            "series": _rel(path, evidence) if path.is_file() else (manifest.parent.name + "/series.npz"),
            "declared_sha256": series.get("sha256"),
            "features": list(payload.get("features") or []),
        }
        if not path.is_file():
            item["series"] = _rel(manifest.parent, evidence) + "/series.npz"
        declared.add(item["series"])
        if path.is_file() and not path.is_symlink():
            present.append(item)
        else:
            missing.append(item)
    orphans = []
    for path in sorted(evidence.rglob("series.npz")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = _rel(path, evidence)
        if rel not in declared:
            orphans.append(rel)
    return {"present": present, "missing": missing, "orphans": orphans}


def _done_rows(evidence: Path) -> list[dict]:
    rows = []
    for path in sorted(evidence.rglob("feature_cost.csv")):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "feature" not in fields or "status" not in fields or "series_sha256" not in fields:
                continue
            for row in reader:
                if (row.get("status") or "").strip() != "DONE":
                    continue
                feature = (row.get("feature") or "").strip()
                digest = (row.get("series_sha256") or "").strip()
                if feature and digest:
                    rows.append({
                        "feature": feature,
                        "series_sha256": digest,
                        "cost_table": _rel(path, evidence),
                    })
    return rows


def select_one_feature_fold(evidence: Path) -> dict:
    """One unprofiled terminal column, or PENDING. Does not read series bytes."""
    scan = scan_evidence(evidence)
    if not scan["present"]:
        return {
            "state": "PENDING",
            "reason": "NO_LOCAL_TRAIN_SERIES_IN_WORKTREE_EVIDENCE",
            "missing_series": scan["missing"],
            "orphans": scan["orphans"],
        }
    done = _done_rows(evidence)
    for item in scan["present"]:
        declared = item.get("declared_sha256") or ""
        for feature in item["features"]:
            name = canonical_feature_id(feature)
            if name in ALREADY_MEASURED_TRANSFORMS or _is_target_name(name):
                continue
            if any(row["feature"] == name and row["series_sha256"] == declared for row in done):
                return {
                    "state": "READY",
                    "reason": "",
                    "feature_id": name,
                    "fold": INNER_FOLDS[0],
                    "series": item["series"],
                    "manifest": item["manifest"],
                    "declared_sha256": declared,
                }
    return {
        "state": "PENDING",
        "reason": "NO_AUTHENTIC_TERMINAL_SERIES_COLUMN",
        "missing_series": scan["missing"],
        "orphans": scan["orphans"],
    }


def _names_sha256() -> str:
    body = ("\n".join(ALREADY_MEASURED_TRANSFORMS) + "\n").encode()
    return hashlib.sha256(body).hexdigest()


def count_units(units_dir: Path, *, accepted_index: Path | None = None,
                expected_accepted_index_sha256: str = "") -> dict:
    counts = {"MEASURED": 0, "PENDING": 0, "FAILED": 0}
    measured = []
    failed = []
    units_dir = Path(units_dir)
    if not units_dir.is_dir():
        return {"counts": counts, "measured": measured, "failed": failed}
    accepted = (
        load_accepted_index(accepted_index, expected_accepted_index_sha256)
        if accepted_index is not None else {}
    )
    for path in sorted(units_dir.glob("*.json")):
        status = classify_unit(path, expected_acceptance=accepted.get(path.name))
        if status not in counts:
            status = "PENDING"
        counts[status] += 1
        if status == "MEASURED":
            measured.append(path.name)
        elif status == "FAILED":
            failed.append(path.name)
    counts["PENDING"] += sum(1 for _ in units_dir.glob("*.json.tmp"))
    return {"counts": counts, "measured": measured, "failed": failed}


def _opened_units(units_dir: Path, accepted: dict[str, dict]) -> list[dict]:
    opened = []
    units_dir = Path(units_dir)
    if not units_dir.is_dir():
        return opened
    for path in sorted(units_dir.glob("*.json")):
        status = classify_unit(path, expected_acceptance=accepted.get(path.name))
        if status not in {"MEASURED", "FAILED"}:
            continue
        payload = json.loads(path.read_text())
        source = payload.get("source_sha256") or {}
        opened.append({
            "unit": path.name,
            "unit_status": status,
            "feature_id": payload.get("feature_id"),
            "fold": payload.get("fold_id"),
            "metric_rows": payload.get("metric_rows"),
            "rows_sha256": payload.get("rows_sha256"),
            "series_sha256": source.get("series.npz", ""),
            "terminal_identity": payload.get("terminal_identity"),
        })
    return opened


def _next_opened_fold(opened: list[dict], selected: dict) -> dict:
    by_feature: dict[str, set[str]] = {}
    for item in opened:
        if item["unit_status"] != "MEASURED" or not item.get("feature_id"):
            continue
        by_feature.setdefault(item["feature_id"], set()).add(item.get("fold") or "")
    for feature, folds in by_feature.items():
        for fold in INNER_FOLDS:
            if fold not in folds:
                return {
                    "state": "PENDING",
                    "reason": "NEXT_FOLD_OF_OPEN_FEATURE",
                    "feature_id": feature,
                    "fold": fold,
                }
    return {
        "state": "PENDING",
        "reason": "OTHER_TERMINALS_STILL_UNPROFILED",
        "feature_id": None,
        "fold": None,
        "missing_series": selected.get("missing_series", []),
    }


def publish_report(evidence: Path, folds_path: Path, units_dir: Path, destination: Path, *,
                   canonical_bounds: bool = False, accepted_index: Path | None = None,
                   expected_accepted_index_sha256: str = "") -> dict:
    evidence = Path(evidence)
    if canonical_bounds:
        assert_canonical_fold_file(folds_path)
    scan = scan_evidence(evidence)
    selected = select_one_feature_fold(evidence)
    accepted = (
        load_accepted_index(accepted_index, expected_accepted_index_sha256)
        if accepted_index is not None else {}
    )
    tallied = count_units(
        units_dir, accepted_index=accepted_index,
        expected_accepted_index_sha256=expected_accepted_index_sha256,
    )
    counts = dict(tallied["counts"])
    opened = _opened_units(units_dir, accepted)
    if opened:
        selected = _next_opened_fold(opened, selected)
    if selected.get("state") in {"READY", "PENDING"}:
        counts["PENDING"] += 1
    profiler = Path(__file__).resolve()
    published = evidence / "ps4_transform_profile" / "REPORT.json"
    hashes = {
        "already_measured_names": _names_sha256(),
        "folds.json": sha256_file(folds_path),
        "profiler": sha256_file(profiler),
    }
    if accepted_index is not None:
        hashes["accepted_units.json"] = expected_accepted_index_sha256
    if published.is_file() and not published.is_symlink():
        hashes["published_transform_profile_report"] = sha256_file(published)
    missing = [
        {
            "manifest": item["manifest"],
            "series": item["series"],
            "declared_sha256": item.get("declared_sha256"),
            "feature_count": len(item.get("features") or []),
        }
        for item in scan["missing"]
    ]
    hashes["declared_missing_series"] = {
        item["series"]: item.get("declared_sha256") for item in missing
    }
    report = {
        "schema": REPORT_SCHEMA,
        "counts": counts,
        "count_scope": (
            "incremental feature-fold units opened or withheld by this profiler; "
            "not the selection denominator"
        ),
        "selection_denominator": SELECTION_DENOMINATOR,
        "selection_closed": False,
        "ps4_selects_features": False,
        "selection_note": (
            "PS4 does not select a feature. The selection denominator is 366 and is not closed."
        ),
        "already_measured_transforms_not_rerun": list(ALREADY_MEASURED_TRANSFORMS),
        "measured_units": tallied["measured"],
        "measured_unit_hashes": [
            item for item in opened if item["unit_status"] == "MEASURED"
        ],
        "failed_units": tallied["failed"],
        "candidate": {
            "state": selected["state"],
            "reason": selected.get("reason", ""),
            "feature_id": selected.get("feature_id"),
            "fold": selected.get("fold"),
        },
        "missing_series": missing,
        "orphan_series": scan["orphans"],
        "execution_receipt": {
            "status": "NOT_RETAINED",
            "reason": (
                "No attempt receipt was retained for the original governed VIX execution; "
                "the report does not infer its command, exit status, wall time, or resource peak."
            ),
        },
        "hashes": hashes,
        "flags": [
            "NO_TARGET_READ",
            "NO_OUTER_VALIDATION_READ",
            "NO_TEST_READ",
            "NO_FEATURE_SELECTION_DECISION",
            *(["NO_NEW_MODEL_MEASUREMENT"] if not tallied["measured"] else []),
            "SELECTION_DENOMINATOR_NOT_CLOSED",
        ],
    }
    _write_json(destination, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="resumable TRAIN-only PS4 profiles for series columns")
    sub = parser.add_subparsers(dest="command", required=True)
    unit = sub.add_parser("unit", help="profile one feature-fold or resume it")
    unit.add_argument("--series", type=Path, required=True)
    unit.add_argument("--folds", type=Path, required=True)
    unit.add_argument("--feature", required=True)
    unit.add_argument("--fold", required=True)
    unit.add_argument("--out", type=Path, required=True)
    unit.add_argument("--split", default="train")
    unit.add_argument("--dataset-id", default=DATASET_ID)
    unit.add_argument("--canonical-bounds", action="store_true")
    unit.add_argument("--expected-digest", default="")
    unit.add_argument("--terminal-manifest", type=Path, required=True)
    report = sub.add_parser("report", help="write MEASURED/PENDING/FAILED counts without profiling")
    report.add_argument("--evidence", type=Path, required=True)
    report.add_argument("--folds", type=Path, required=True)
    report.add_argument("--units", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--canonical-bounds", action="store_true")
    report.add_argument("--accepted-index", type=Path, required=True)
    report.add_argument("--accepted-index-sha256", required=True)
    bind = sub.add_parser("bind-legacy", help="bind retained v1 rows to an authenticated PS3-R terminal")
    bind.add_argument("--unit", type=Path, required=True)
    bind.add_argument("--terminal-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    _thread_env()
    try:
        if args.command == "unit":
            result = execute_unit(
                args.series, args.folds, args.feature, args.fold, args.out,
                split=args.split, dataset_id=args.dataset_id,
                canonical_bounds=args.canonical_bounds, expected_digest=args.expected_digest,
                terminal_manifest=args.terminal_manifest,
            )
        elif args.command == "report":
            result = publish_report(
                args.evidence, args.folds, args.units, args.out,
                canonical_bounds=args.canonical_bounds,
                accepted_index=args.accepted_index,
                expected_accepted_index_sha256=args.accepted_index_sha256,
            )
        else:
            result = bind_legacy_unit(args.unit, args.terminal_manifest)
    except ProfileRefusal as refusal:
        print(f"{refusal.code}: {refusal.detail}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in result if key != "hashes"}, sort_keys=True)
          if args.command == "report" else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
