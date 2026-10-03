"""Measured PS4 profiles for the ten emitted causal transform features.

The population is the authenticated train prefix of each fold. One parquet
column is read at a time. Estimators are the pinned ``variable_rows``
implementation. This runner does not select features and does not read a
target, a validation block, or an outer test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.df_profile_information import blas_single_thread, variable_rows  # noqa: E402

REQUIRED_PARQUET_SHA256 = "fd0b4423db991cfb05cd4f9357a378648579d4c11552a1cac413ed711c3520bc"
EXPECTED_FEATURES = 10
EXPECTED_FOLDS = 5
DATASET_ID = "eurusd_ps4_transform_profile"
SCIENTIFIC_FILES = ("profile_rows.jsonl", "REPORT.json", "input_digests.json")


class ProfileRefusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_parquet_digest(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ProfileRefusal("DIGEST_MISMATCH", f"parquet sha256 {actual} != {expected}")
    return actual


def emitted_feature_ids(join_csv: Path) -> list[str]:
    """Ten unique emitted identifiers. Metric rows of one variant repeat the same id."""
    variants: dict[str, str] = {}
    with Path(join_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "emitted_feature_id" not in (reader.fieldnames or []):
            raise ProfileRefusal("EMITTED_FEATURE_COUNT", "join has no emitted_feature_id column")
        for row in reader:
            feature = (row.get("emitted_feature_id") or "").strip()
            if not feature:
                continue
            variant = (row.get("variant_id") or "").strip()
            previous = variants.get(feature)
            if previous is None:
                variants[feature] = variant
            elif previous != variant:
                raise ProfileRefusal("DUPLICATE_EMITTED_FEATURE", f"{feature} {previous} and {variant}")
    if len(variants) != EXPECTED_FEATURES:
        raise ProfileRefusal("EMITTED_FEATURE_COUNT", f"{len(variants)} unique emitted features")
    return list(variants)


def assert_columns_present(path: Path, features: list[str]) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [feature for feature in features if feature not in names]
    if missing:
        raise ProfileRefusal("FEATURE_ABSENT_FROM_PARQUET", ",".join(missing))


def load_train_folds(path: Path, n_rows: int) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    raw = payload.get("folds") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or len(raw) != EXPECTED_FOLDS:
        raise ProfileRefusal("FOLD_BOUNDARY_ABSENT", "five fold records are required")
    folds = []
    previous_end = 0
    for item in raw:
        bounds = item.get("train_rows") if isinstance(item, dict) else None
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name or not isinstance(bounds, list) or len(bounds) != 2:
            raise ProfileRefusal("FOLD_BOUNDARY_ABSENT", str(item))
        start, end = int(bounds[0]), int(bounds[1])
        if start < 0 or end > n_rows:
            raise ProfileRefusal("FOLD_OUT_OF_RANGE", f"{name} [{start}, {end}) rows={n_rows}")
        if start != 0:
            raise ProfileRefusal("FOLD_OVERLAP", f"{name} train does not start at 0")
        if end <= start or end <= previous_end:
            raise ProfileRefusal("FOLD_NONCHRONOLOGICAL", f"{name} [{start}, {end})")
        previous_end = end
        folds.append({"name": name, "train_end": end})
    return folds


def train_population(parts, n_train: int):
    if parts != [("train", 0, n_train)]:
        raise ProfileRefusal("PARTITION_NOT_TRAIN", repr(parts))
    return parts


def _wrap(feature: str, fold: str, train_end: int, row: dict, source_digests: dict) -> dict:
    return {
        "feature_id": feature,
        "fold": fold,
        "train_rows": [0, int(train_end)],
        "dataset_id": row["dataset_id"],
        "source_digests": source_digests,
        "profiler_code_digest": row["code_sha256"],
        "metric": row["metric"],
        "estimator": row["estimator"],
        "status": row["status"],
        "reason": row["reason"],
        "value": row["value"],
        "cpu_seconds": row["cpu_seconds"],
    }


def profile_fold(dataset_id: str, feature_id: str, column, fold: str, train_end: int,
                 source_digests: dict | None = None) -> list[dict]:
    """Profile ``column[:train_end]``. Rows at or after ``train_end`` are not passed in."""
    values = np.asarray(column[:train_end], dtype=np.float64)
    parts = train_population([("train", 0, int(values.size))], int(values.size))
    with blas_single_thread():
        produced = variable_rows(dataset_id, feature_id, values, parts)
    digests = source_digests or {}
    return [_wrap(feature_id, fold, train_end, row, digests) for row in produced]


def _identity(row: dict) -> str:
    estimator = json.dumps(row.get("estimator"), sort_keys=True, default=str)
    return f"{row.get('feature_id')}|{row.get('fold')}|{row.get('metric')}|{estimator}"


def assert_unique(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        key = _identity(row)
        if key in seen:
            raise ProfileRefusal("DUPLICATE_OUTPUT_IDENTITY", key)
        seen.add(key)


def assert_finite_completed(rows: list[dict]) -> None:
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        value = row.get("value")
        if value is None or not np.isfinite(float(value)):
            raise ProfileRefusal(
                "NON_FINITE_COMPLETED",
                f"{row.get('feature_id')}/{row.get('fold')}/{row.get('metric')}",
            )


def assert_complete_units(rows: list[dict], features: list[str], folds: list[str]) -> None:
    present = {(row["feature_id"], row["fold"]) for row in rows}
    missing = [f"{feature}/{fold}" for feature in features for fold in folds
               if (feature, fold) not in present]
    if missing:
        raise ProfileRefusal("MISSING_UNIT", ",".join(missing))


def scientific_bytes(rows: list[dict]) -> bytes:
    cleaned = [{key: value for key, value in row.items() if key != "cpu_seconds"} for row in rows]
    cleaned.sort(key=_identity)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode()


def _unit_path(out: Path, feature: str, fold: str) -> Path:
    safe = feature.replace("/", "_")
    return out / "units" / f"{safe}__{fold}.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_column(path: Path, feature: str) -> np.ndarray:
    table = pq.read_table(path, columns=[feature])
    return np.asarray(table.column(feature).to_numpy(zero_copy_only=False), dtype=np.float64)


def _status_counts(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return dict(sorted(counts.items()))


def _report(rows: list[dict], features: list[str], folds: list[dict]) -> dict:
    return {
        "schema": "ps4_transform_profile.v1",
        "denominator_feature_fold_units": len(features) * len(folds),
        "features": features,
        "folds": [item["name"] for item in folds],
        "metric_rows": len(rows),
        "status_counts": _status_counts(rows),
        "flags": [
            "NO_TARGET_READ",
            "NO_OUTER_VALIDATION_READ",
            "NO_TEST_READ",
            "NO_FEATURE_SELECTION_DECISION",
            "NO_NEW_MODEL_MEASUREMENT",
        ],
    }


def _assemble(out: Path, rows: list[dict], features: list[str], folds: list[dict],
              digests: dict) -> None:
    ordered = sorted(rows, key=lambda row: (row["feature_id"], row["fold"], row["metric"], _identity(row)))
    assert_unique(ordered)
    assert_finite_completed(ordered)
    assert_complete_units(ordered, features, [item["name"] for item in folds])
    body = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in ordered)
    target = out / "profile_rows.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text(body)
    os.replace(temporary, target)
    _write_json(out / "REPORT.json", _report(ordered, features, folds))
    _write_json(out / "input_digests.json", digests)


def _load_rows(out: Path) -> list[dict]:
    rows = []
    for line in (out / "profile_rows.jsonl").read_text().splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def execute(parquet: Path, folds_path: Path, join_csv: Path, out: Path, *,
            expected_digest: str, dataset_id: str = DATASET_ID,
            extra_digests: dict | None = None) -> dict:
    parquet, folds_path, join_csv, out = map(Path, (parquet, folds_path, join_csv, out))
    out.mkdir(parents=True, exist_ok=True)
    digest = assert_parquet_digest(parquet, expected_digest)
    features = emitted_feature_ids(join_csv)
    assert_columns_present(parquet, features)
    n_rows = pq.ParquetFile(parquet).metadata.num_rows
    folds = load_train_folds(folds_path, n_rows)
    source = {
        "features_train.parquet": digest,
        "folds.json": sha256_file(folds_path),
        "transform_feature_join.csv": sha256_file(join_csv),
    }
    if extra_digests:
        source.update(extra_digests)
    digests = {"schema": "ps4_transform_profile_digests.v1", "sha256": source}

    def compute(feature: str, fold: dict) -> list[dict]:
        column = _read_column(parquet, feature)
        return profile_fold(dataset_id, feature, column, fold["name"], fold["train_end"], source)

    jsonl = out / "profile_rows.jsonl"
    if jsonl.exists() and all(_unit_path(out, feature, fold["name"]).exists()
                               for feature in features for fold in folds):
        fresh = [row for feature in features for fold in folds for row in compute(feature, fold)]
        if scientific_bytes(fresh) != scientific_bytes(_load_rows(out)):
            raise ProfileRefusal("REPLAY_MISMATCH", "scientific rows changed on replay")
        receipt = {
            "schema": "ps4_transform_profile_attempt.v1",
            "replay": True,
            "metric_rows": len(fresh),
            "cpu_seconds": round(sum(row["cpu_seconds"] for row in fresh), 6),
        }
        attempt = out / "attempts" / f"replay-{time.time_ns()}.json"
        _write_json(attempt, receipt)
        return {"replay": True, "metric_rows": len(fresh)}

    rows: list[dict] = []
    for feature in features:
        for fold in folds:
            unit = _unit_path(out, feature, fold["name"])
            if unit.exists():
                rows.extend(json.loads(unit.read_text()))
                continue
            produced = compute(feature, fold)
            _write_json(unit, produced)
            rows.extend(produced)
    _assemble(out, rows, features, folds, digests)
    return {"replay": False, "metric_rows": len(rows), "units": len(features) * len(folds)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure PS4 profiles for ten transform features")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-digest", default=REQUIRED_PARQUET_SHA256)
    parser.add_argument("--admissible", type=Path, default=None)
    parser.add_argument("--digests", type=Path, default=None)
    parser.add_argument("--ready", type=Path, default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    extra = {}
    for label, path in (("admissible_features.json", args.admissible),
                        ("digests.json", args.digests), ("READY", args.ready)):
        if path is not None:
            extra[label] = sha256_file(path)
    try:
        result = execute(
            args.parquet, args.folds, args.join, args.out,
            expected_digest=args.expected_digest, extra_digests=extra,
        )
    except ProfileRefusal as refusal:
        print(f"{refusal.code}: {refusal.detail}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
