"""Build deterministic dataset and variable profiles for CRISP-DM discovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SPEC_SCHEMA = "predictor.crispdm_dataset_registry.v1"
INVENTORY_SCHEMA = "predictor.crispdm_dataset_inventory.v1"
SOURCE_CLASSES = {"financial", "public", "synthetic"}
FORMATS = {"csv"}

_DATASET_REQUIRED = {
    "dataset_id", "source_class", "domain", "provider", "version", "license_id",
    "availability_policy", "exposure_status", "root_id", "relative_path", "format",
    "timestamp_column", "timestamp_semantics", "default_variable_role", "default_unit",
    "variable_roles", "units",
}
_DATASET_OPTIONAL = {"declared_sha256", "declared_rows", "notes"}


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON value: {value}")),
    )


def parse_roots(items: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"root must have ROOT_ID=PATH form: {item}")
        root_id, raw_path = item.split("=", 1)
        if not root_id or root_id in roots:
            raise ValueError(f"invalid or duplicate root id: {root_id}")
        root = Path(raw_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"root is not a directory: {root_id}")
        roots[root_id] = root
    return roots


def resolve_dataset_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or relative.as_posix() != relative_path or ".." in relative.parts:
        raise ValueError(f"relative_path must be normalized and relative: {relative_path}")
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError(f"relative_path escapes its root: {relative_path}")
    if not path.is_file():
        raise ValueError(f"dataset path is not a regular file: {relative_path}")
    return path


def validate_registry(registry: dict, roots: dict[str, Path]) -> None:
    if set(registry) != {"schema", "scope", "datasets"}:
        raise ValueError("registry must contain exactly schema, scope, datasets")
    if registry["schema"] != SPEC_SCHEMA:
        raise ValueError(f"registry.schema must be {SPEC_SCHEMA}")
    if not isinstance(registry["scope"], dict):
        raise ValueError("registry.scope must be an object")
    if not isinstance(registry["datasets"], list) or not registry["datasets"]:
        raise ValueError("registry.datasets must be a non-empty list")

    seen: set[str] = set()
    for index, dataset in enumerate(registry["datasets"]):
        path = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            raise ValueError(f"{path} must be an object")
        missing = sorted(_DATASET_REQUIRED - set(dataset))
        extra = sorted(set(dataset) - _DATASET_REQUIRED - _DATASET_OPTIONAL)
        if missing or extra:
            raise ValueError(f"{path} schema mismatch; missing={missing}, extra={extra}")
        dataset_id = dataset["dataset_id"]
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in seen:
            raise ValueError(f"{path}.dataset_id must be unique and non-empty")
        seen.add(dataset_id)
        if dataset["source_class"] not in SOURCE_CLASSES:
            raise ValueError(f"{path}.source_class must be one of {sorted(SOURCE_CLASSES)}")
        if dataset["format"] not in FORMATS:
            raise ValueError(f"{path}.format must be one of {sorted(FORMATS)}")
        if dataset["root_id"] not in roots:
            raise ValueError(f"{path}.root_id is not present in --root arguments")
        for field in (
            "domain", "provider", "version", "license_id", "availability_policy",
            "exposure_status", "relative_path", "timestamp_column", "timestamp_semantics",
            "default_variable_role", "default_unit",
        ):
            if not isinstance(dataset[field], str) or not dataset[field]:
                raise ValueError(f"{path}.{field} must be a non-empty string")
        if not isinstance(dataset["variable_roles"], dict) or not isinstance(dataset["units"], dict):
            raise ValueError(f"{path}.variable_roles and units must be objects")
        if "declared_rows" in dataset and (
            isinstance(dataset["declared_rows"], bool)
            or not isinstance(dataset["declared_rows"], int)
            or dataset["declared_rows"] < 0
        ):
            raise ValueError(f"{path}.declared_rows must be a non-negative integer")
        if "declared_sha256" in dataset and not _is_sha256(dataset["declared_sha256"]):
            raise ValueError(f"{path}.declared_sha256 must be lowercase SHA-256")


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_csv_strict(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
    if not header or any(not name for name in header):
        raise ValueError("CSV has empty column names")
    if len(header) != len(set(header)):
        raise ValueError("CSV has duplicate column names")
    return pd.read_csv(path, low_memory=False)


def _json_number(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _profile_time(frame: pd.DataFrame, timestamp_column: str) -> dict:
    if timestamp_column not in frame.columns:
        raise ValueError(f"timestamp column is absent: {timestamp_column}")
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    valid = parsed.dropna()
    invalid_count = int(parsed.isna().sum())
    duplicate_count = int(valid.duplicated().sum())
    monotonic = bool(valid.is_monotonic_increasing)
    start_at = valid.min().isoformat().replace("+00:00", "Z") if len(valid) else None
    end_at = valid.max().isoformat().replace("+00:00", "Z") if len(valid) else None
    median_sampling_seconds = None
    irregular_fraction = None
    if len(valid) >= 2:
        ordered_unique = valid.drop_duplicates().sort_values()
        differences = ordered_unique.diff().dropna().dt.total_seconds()
        if len(differences):
            median_sampling_seconds = _json_number(differences.median())
            tolerance = max(1e-9, abs(median_sampling_seconds) * 1e-9)
            irregular_fraction = _json_number((np.abs(differences - median_sampling_seconds) > tolerance).mean())
    return {
        "timestamp_column": timestamp_column,
        "start_at": start_at,
        "end_at": end_at,
        "invalid_timestamp_count": invalid_count,
        "duplicate_timestamp_count": duplicate_count,
        "monotonic_non_decreasing": monotonic,
        "median_sampling_seconds": median_sampling_seconds,
        "irregular_interval_fraction": irregular_fraction,
    }


def _profile_variable(series: pd.Series, name: str, role: str, unit: str) -> dict:
    missing = series.isna()
    non_missing = series[~missing]
    numeric = pd.to_numeric(non_missing, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float, na_value=np.nan)
    finite_mask = np.isfinite(numeric_array)
    finite_values = numeric_array[finite_mask]
    finite_count = int(finite_mask.sum())
    numeric_count = int(numeric.notna().sum())
    unique_count = int(non_missing.nunique(dropna=True))
    numeric_fraction = numeric_count / len(non_missing) if len(non_missing) else None
    profile = {
        "name": name,
        "role": role,
        "unit": unit,
        "physical_dtype": str(series.dtype),
        "observation_count": int(len(series)),
        "missing_count": int(missing.sum()),
        "finite_count": finite_count,
        "unique_count": unique_count,
        "numeric_fraction": _json_number(numeric_fraction),
        "constant": bool(len(non_missing) > 0 and unique_count <= 1),
        "minimum": _json_number(finite_values.min()) if finite_count else None,
        "maximum": _json_number(finite_values.max()) if finite_count else None,
        "mean": _json_number(finite_values.mean()) if finite_count else None,
        "std": _json_number(finite_values.std(ddof=0)) if finite_count else None,
    }
    profile["profile_sha256"] = sha256_bytes(canonical_json_bytes(profile))
    return profile


def profile_dataset(dataset: dict, roots: dict[str, Path]) -> dict:
    path = resolve_dataset_path(roots[dataset["root_id"]], dataset["relative_path"])
    physical_sha256 = sha256_file(path)
    frame = _read_csv_strict(path)
    if dataset["timestamp_column"] not in frame.columns:
        raise ValueError(f"{dataset['dataset_id']}: timestamp column is absent")

    variables = []
    for name in frame.columns:
        role = dataset["variable_roles"].get(name, dataset["default_variable_role"])
        unit = dataset["units"].get(name, dataset["default_unit"])
        variables.append(_profile_variable(frame[name], name, role, unit))

    issues = []
    if dataset["license_id"] == "UNDECLARED":
        issues.append("LICENSE_UNDECLARED")
    if dataset["availability_policy"] == "UNDECLARED":
        issues.append("AVAILABILITY_UNDECLARED")
    undeclared_units = sorted(v["name"] for v in variables if v["unit"] == "UNDECLARED")
    if undeclared_units:
        issues.append(f"UNITS_UNDECLARED:{len(undeclared_units)}")
    if "declared_sha256" in dataset and dataset["declared_sha256"] != physical_sha256:
        issues.append("DECLARED_SHA256_MISMATCH")
    if "declared_rows" in dataset and dataset["declared_rows"] != len(frame):
        issues.append("DECLARED_ROW_COUNT_MISMATCH")

    record = {
        "dataset_id": dataset["dataset_id"],
        "source_class": dataset["source_class"],
        "domain": dataset["domain"],
        "provider": dataset["provider"],
        "version": dataset["version"],
        "license_id": dataset["license_id"],
        "availability_policy": dataset["availability_policy"],
        "exposure_status": dataset["exposure_status"],
        "root_id": dataset["root_id"],
        "relative_path": dataset["relative_path"],
        "format": dataset["format"],
        "physical_sha256": physical_sha256,
        "row_count": int(len(frame)),
        "variable_count": int(len(frame.columns)),
        "time_profile": _profile_time(frame, dataset["timestamp_column"]),
        "variables": variables,
        "profile_status": "PROFILED_COMPLETE" if not issues else "PROFILED_WITH_METADATA_GAPS",
        "metadata_issues": issues,
    }
    record["profile_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def build_inventory(registry: dict, roots: dict[str, Path], inventoried_at: str) -> dict:
    validate_registry(registry, roots)
    try:
        parsed_as_of = datetime.fromisoformat(inventoried_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("inventoried_at must be an RFC3339 timestamp") from exc
    if parsed_as_of.tzinfo is None:
        raise ValueError("inventoried_at must include a timezone")

    datasets = [profile_dataset(dataset, roots) for dataset in registry["datasets"]]
    datasets.sort(key=lambda value: value["dataset_id"])
    status_counts: dict[str, int] = {}
    for dataset in datasets:
        status_counts[dataset["profile_status"]] = status_counts.get(dataset["profile_status"], 0) + 1
    document = {
        "schema": INVENTORY_SCHEMA,
        "inventoried_at": parsed_as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": registry["scope"],
        "datasets": datasets,
        "summary": {
            "dataset_count": len(datasets),
            "variable_count": sum(dataset["variable_count"] for dataset in datasets),
            "row_count": sum(dataset["row_count"] for dataset in datasets),
            "status_counts": dict(sorted(status_counts.items())),
        },
    }
    document["inventory_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return document


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--root", action="append", default=[], metavar="ROOT_ID=PATH", required=True)
    parser.add_argument("--inventoried-at", required=True, help="RFC3339 timestamp fixed by the run contract")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    roots = parse_roots(args.root)
    registry = load_json(args.registry)
    inventory = build_inventory(registry, roots, args.inventoried_at)
    write_json_atomic(args.output, inventory)
    print(json.dumps({
        "output": str(args.output),
        "inventory_sha256": inventory["inventory_sha256"],
        **inventory["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
