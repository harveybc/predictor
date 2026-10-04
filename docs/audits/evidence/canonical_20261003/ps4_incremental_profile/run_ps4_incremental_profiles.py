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
UNIT_SCHEMA = "ps4_incremental_unit.v1"
REPORT_SCHEMA = "ps4_incremental_profile.v1"
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def classify_unit(path: Path) -> str:
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
    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("rows_sha256") != _rows_sha256(rows):
        return "PENDING"
    status = payload.get("unit_status")
    if status == "MEASURED" and rows and all(isinstance(row, dict) for row in rows):
        return "MEASURED"
    if status == "FAILED" and rows == [] and payload.get("reason"):
        return "FAILED"
    return "PENDING"


def _unit_body(feature: str, fold: str, train_end: int | None, rows: list, source: dict,
               status: str, reason: str) -> dict:
    return {
        "schema": UNIT_SCHEMA,
        "unit_status": status,
        "feature_id": feature,
        "fold": fold,
        "train_rows": None if train_end is None else [0, int(train_end)],
        "reason": reason,
        "source_sha256": source,
        "rows": rows,
        "rows_sha256": _rows_sha256(rows),
        "metric_rows": len(rows),
    }


def _thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def execute_unit(series: Path, folds_path: Path, feature_id: str, fold_name: str, out: Path, *,
                 split: str = "train", dataset_id: str = DATASET_ID,
                 canonical_bounds: bool = False, expected_digest: str = "") -> dict:
    """Profile one feature-fold, or resume when that unit is already MEASURED."""
    _thread_env()
    assert_split(split)
    assert_fold_name(fold_name)
    feature = assert_not_already_measured(feature_id)
    series = Path(series)
    if series.name == "targets.npz":
        raise ProfileRefusal("TARGET_COLUMN", "targets.npz")
    destination = unit_path(out, feature, fold_name)
    state = classify_unit(destination)
    if state in {"MEASURED", "FAILED"}:
        return {
            "resumed": True,
            "unit_status": state,
            "feature_id": feature,
            "fold": fold_name,
            "metric_rows": json.loads(destination.read_text()).get("metric_rows", 0),
            "unit": destination.name,
        }
    if expected_digest:
        actual = sha256_file(series) if series.is_file() and not series.is_symlink() else ""
        if actual != expected_digest:
            raise ProfileRefusal("DIGEST_MISMATCH", "series sha256 does not match the declared digest")
    try:
        column = load_series_column(series, feature)
        folds = load_inner_folds(folds_path, int(column.size), canonical_bounds=canonical_bounds)
        chosen = next(item for item in folds if item["name"] == fold_name)
        source = {"series.npz": sha256_file(series), "folds.json": sha256_file(folds_path)}
        rows = profile_fold(dataset_id, feature, column, fold_name, chosen["train_end"], source)
        payload = _unit_body(feature, fold_name, chosen["train_end"], rows, source, "MEASURED", "")
    except ProfileRefusal:
        raise
    except Exception:
        source = {}
        if series.is_file() and not series.is_symlink():
            source["series.npz"] = sha256_file(series)
        payload = _unit_body(feature, fold_name, None, [], source, "FAILED", "PROFILER_EXCEPTION")
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


def drive(features: list[str], series: Path, folds_path: Path, out: Path, *,
          fold: str, split: str = "train", dataset_id: str = DATASET_ID,
          canonical_bounds: bool = False, expected_digest: str = "") -> dict:
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


def count_units(units_dir: Path) -> dict:
    counts = {"MEASURED": 0, "PENDING": 0, "FAILED": 0}
    measured = []
    failed = []
    units_dir = Path(units_dir)
    if not units_dir.is_dir():
        return {"counts": counts, "measured": measured, "failed": failed}
    for path in sorted(units_dir.glob("*.json")):
        status = classify_unit(path)
        if status not in counts:
            status = "PENDING"
        counts[status] += 1
        if status == "MEASURED":
            measured.append(path.name)
        elif status == "FAILED":
            failed.append(path.name)
    counts["PENDING"] += sum(1 for _ in units_dir.glob("*.json.tmp"))
    return {"counts": counts, "measured": measured, "failed": failed}


def publish_report(evidence: Path, folds_path: Path, units_dir: Path, destination: Path, *,
                   canonical_bounds: bool = False) -> dict:
    evidence = Path(evidence)
    if canonical_bounds:
        assert_canonical_fold_file(folds_path)
    scan = scan_evidence(evidence)
    selected = select_one_feature_fold(evidence)
    tallied = count_units(units_dir)
    counts = dict(tallied["counts"])
    if not tallied["measured"] and not tallied["failed"] and selected["state"] != "READY":
        counts["PENDING"] += 1
    elif selected["state"] == "READY" and not tallied["measured"] and not tallied["failed"]:
        counts["PENDING"] += 1
    profiler = Path(__file__).resolve()
    published = evidence / "ps4_transform_profile" / "REPORT.json"
    hashes = {
        "already_measured_names": _names_sha256(),
        "folds.json": sha256_file(folds_path),
        "profiler": sha256_file(profiler),
    }
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
        "failed_units": tallied["failed"],
        "candidate": {
            "state": selected["state"],
            "reason": selected.get("reason", ""),
            "feature_id": selected.get("feature_id"),
            "fold": selected.get("fold"),
        },
        "missing_series": missing,
        "orphan_series": scan["orphans"],
        "hashes": hashes,
        "flags": [
            "NO_TARGET_READ",
            "NO_OUTER_VALIDATION_READ",
            "NO_TEST_READ",
            "NO_FEATURE_SELECTION_DECISION",
            "NO_NEW_MODEL_MEASUREMENT",
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
    report = sub.add_parser("report", help="write MEASURED/PENDING/FAILED counts without profiling")
    report.add_argument("--evidence", type=Path, required=True)
    report.add_argument("--folds", type=Path, required=True)
    report.add_argument("--units", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--canonical-bounds", action="store_true")
    args = parser.parse_args(argv)
    _thread_env()
    try:
        if args.command == "unit":
            result = execute_unit(
                args.series, args.folds, args.feature, args.fold, args.out,
                split=args.split, dataset_id=args.dataset_id,
                canonical_bounds=args.canonical_bounds, expected_digest=args.expected_digest,
            )
        else:
            result = publish_report(
                args.evidence, args.folds, args.units, args.out,
                canonical_bounds=args.canonical_bounds,
            )
    except ProfileRefusal as refusal:
        print(f"{refusal.code}: {refusal.detail}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in result if key != "hashes"}, sort_keys=True)
          if args.command == "report" else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
