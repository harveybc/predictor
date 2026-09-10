"""Reconcile the financial-data feature manifest without reading data values."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from olap.crispdm_inventory import (
    canonical_json_bytes,
    load_json,
    resolve_dataset_path,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)


SCHEMA = "predictor.financial_data_manifest_reconciliation.v1"


def _rfc3339(value, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: expected RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path}: expected RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{path}: timezone is required")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_record(root: Path, family: str, member: str, timeframe: str, declared: dict) -> dict:
    required = {"path", "status", "rows", "start", "end", "columns"}
    missing = sorted(required - set(declared))
    if missing:
        raise ValueError(f"{family}.{member}.{timeframe}: missing fields {missing}")
    prefix = f"{family}.{member}.{timeframe}"
    if not isinstance(declared["path"], str) or not declared["path"]:
        raise ValueError(f"{prefix}.path: expected a non-empty string")
    if not isinstance(declared["status"], str) or not declared["status"]:
        raise ValueError(f"{prefix}.status: expected a non-empty string")
    if isinstance(declared["rows"], bool) or not isinstance(declared["rows"], int) or declared["rows"] < 0:
        raise ValueError(f"{prefix}.rows: expected a non-negative integer")
    if not isinstance(declared["columns"], list) or not declared["columns"]:
        raise ValueError(f"{prefix}.columns: expected a non-empty list")
    if any(not isinstance(column, str) or not column for column in declared["columns"]):
        raise ValueError(f"{prefix}.columns: expected non-empty strings")
    if len(declared["columns"]) != len(set(declared["columns"])):
        raise ValueError(f"{prefix}.columns: duplicate column")
    start = _rfc3339(declared["start"], f"{prefix}.start")
    end = _rfc3339(declared["end"], f"{prefix}.end")
    parsed_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    parsed_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if parsed_end < parsed_start:
        raise ValueError(f"{prefix}: end precedes start")
    relative_path = declared["path"]
    record = {
        "dataset_id": f"financial_data.{family}.{member}.{timeframe}",
        "family": family,
        "member": member,
        "timeframe": timeframe,
        "relative_path": relative_path,
        "declared_status": declared["status"],
        "declared_rows": declared["rows"],
        "declared_start": start,
        "declared_end": end,
        "declared_columns": declared["columns"],
        "physical_exists": False,
        "physical_size_bytes": None,
        "reconciliation_status": "DECLARED_PHYSICAL_MISSING",
        "metadata_gaps": [
            "CONTENT_DIGEST_NOT_PRESENT_IN_FEATURE_MANIFEST",
            "LICENSE_NOT_PRESENT_IN_FEATURE_MANIFEST",
            "AVAILABILITY_INSTANCE_NOT_PRESENT_IN_FEATURE_MANIFEST",
            "VARIABLE_UNITS_NOT_PRESENT_IN_FEATURE_MANIFEST",
        ],
    }
    try:
        physical = resolve_dataset_path(root, relative_path)
    except (FileNotFoundError, ValueError):
        return record
    record["physical_exists"] = True
    record["physical_size_bytes"] = physical.stat().st_size
    record["reconciliation_status"] = "DECLARED_PHYSICAL_PRESENT_METADATA_JOIN_REQUIRED"
    return record


def reconcile_manifest(manifest_path: Path, financial_data_root: Path, inventoried_at: str) -> dict:
    root = financial_data_root.expanduser().resolve(strict=True)
    inventoried_at = _rfc3339(inventoried_at, "inventoried_at")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    for key in ("trading_assets", "cross_source_features"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"manifest.{key} must be an object")

    records = []
    for asset, asset_record in manifest["trading_assets"].items():
        timeframes = asset_record.get("timeframes") if isinstance(asset_record, dict) else None
        if not isinstance(timeframes, dict):
            raise ValueError(f"manifest.trading_assets.{asset}.timeframes must be an object")
        for timeframe, declared in timeframes.items():
            records.append(_manifest_record(root, "trading_asset", asset, timeframe, declared))

    for timeframe, sources in manifest["cross_source_features"].items():
        if not isinstance(sources, dict):
            raise ValueError(f"manifest.cross_source_features.{timeframe} must be an object")
        for source_id, declared in sources.items():
            records.append(_manifest_record(root, "cross_source", source_id, timeframe, declared))

    records.sort(key=lambda value: value["dataset_id"])
    status_counts: dict[str, int] = {}
    for record in records:
        status = record["reconciliation_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    document = {
        "schema": SCHEMA,
        "inventoried_at": inventoried_at,
        "manifest_relative_path": manifest_path.resolve(strict=True).relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "records": records,
        "summary": {
            "dataset_slice_count": len(records),
            "trading_asset_slice_count": sum(r["family"] == "trading_asset" for r in records),
            "cross_source_slice_count": sum(r["family"] == "cross_source" for r in records),
            "declared_variable_count": sum(len(r["declared_columns"]) for r in records),
            "physical_present_count": sum(r["physical_exists"] for r in records),
            "physical_missing_count": sum(not r["physical_exists"] for r in records),
            "physical_size_bytes": sum(r["physical_size_bytes"] or 0 for r in records),
            "status_counts": dict(sorted(status_counts.items())),
        },
    }
    document["reconciliation_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-data-root", required=True, type=Path)
    parser.add_argument("--manifest", default="features/MANIFEST.json")
    parser.add_argument("--inventoried-at", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    root = args.financial_data_root.expanduser().resolve(strict=True)
    manifest = resolve_dataset_path(root, args.manifest)
    document = reconcile_manifest(manifest, root, args.inventoried_at)
    write_json_atomic(args.output, document)
    print(json.dumps({"output": str(args.output), **document["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
