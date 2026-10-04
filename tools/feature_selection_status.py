#!/usr/bin/env python3
"""Summarize a feature-selection campaign from a TSV plan and result folders.

The plan must contain a header and the columns ``cell_id`` and ``result_dir``.
Result directories are relative to ``--results-root``. By default a completed
cell contains ``run_manifest.json`` and ``results.jsonl``; the manifest declares
``status``, ``results_sha256`` and ``wall_seconds``. File and field names are
configurable so the command can inspect campaigns without rewriting evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class CampaignError(ValueError):
    """Raised when the campaign plan or command configuration is invalid."""


@dataclass(frozen=True)
class PlanCell:
    """One immutable cell declared by a campaign plan."""

    cell_id: str
    result_dir: str


@dataclass(frozen=True)
class StatusConfig:
    """Names used to interpret a campaign's result directories."""

    manifest_name: str = "run_manifest.json"
    artifact_name: str = "results.jsonl"
    running_name: str = "RUNNING"
    failed_name: str = "FAILED"
    status_field: str = "status"
    digest_field: str = "results_sha256"
    duration_field: str = "wall_seconds"
    cell_field: str = "cell_id"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_plan(plan_path: Path, results_root: Path) -> list[PlanCell]:
    """Read and validate the stable identities and paths in a TSV plan."""

    try:
        stream = plan_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise CampaignError(f"cannot read plan {plan_path}: {exc}") from exc

    root = results_root.resolve()
    cells: list[PlanCell] = []
    seen_ids: set[str] = set()
    seen_dirs: set[Path] = set()
    with stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"cell_id", "result_dir"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise CampaignError("plan TSV requires cell_id and result_dir columns")
        for line_number, row in enumerate(reader, start=2):
            cell_id = (row.get("cell_id") or "").strip()
            result_dir = (row.get("result_dir") or "").strip()
            if not cell_id or not result_dir:
                raise CampaignError(f"plan line {line_number} has an empty identity or path")
            if cell_id in seen_ids:
                raise CampaignError(f"duplicate cell_id: {cell_id}")
            relative = Path(result_dir)
            resolved = (root / relative).resolve()
            if relative.is_absolute() or not resolved.is_relative_to(root):
                raise CampaignError(f"result_dir escapes results root: {result_dir}")
            if resolved in seen_dirs:
                raise CampaignError(f"duplicate result_dir: {result_dir}")
            seen_ids.add(cell_id)
            seen_dirs.add(resolved)
            cells.append(PlanCell(cell_id=cell_id, result_dir=result_dir))
    if not cells:
        raise CampaignError("plan TSV contains no cells")
    return cells


def _integrity_error(cell: PlanCell, code: str, detail: str) -> dict[str, str]:
    return {"cell_id": cell.cell_id, "code": code, "detail": detail}


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError(str(exc)) from exc
    if not isinstance(value, dict):
        raise CampaignError("manifest root is not an object")
    return value


def _completed_cell(
    cell: PlanCell,
    result_dir: Path,
    manifest: dict[str, Any],
    config: StatusConfig,
) -> tuple[float | None, dict[str, str] | None]:
    """Authenticate one completed cell and return its measured duration."""

    manifest_cell = manifest.get(config.cell_field)
    if manifest_cell is not None and manifest_cell != cell.cell_id:
        return None, _integrity_error(
            cell,
            "CELL_ID_MISMATCH",
            f"manifest declares {manifest_cell!r}",
        )

    artifact = result_dir / config.artifact_name
    if not artifact.is_file():
        return None, _integrity_error(cell, "MISSING_ARTIFACT", str(artifact))
    expected = manifest.get(config.digest_field)
    if not isinstance(expected, str) or len(expected) != 64:
        return None, _integrity_error(
            cell, "INVALID_ARTIFACT_DIGEST", f"field {config.digest_field!r}"
        )
    actual = _sha256(artifact)
    if not hmac.compare_digest(expected.lower(), actual):
        return None, _integrity_error(
            cell,
            "ARTIFACT_DIGEST_MISMATCH",
            f"expected {expected.lower()}, observed {actual}",
        )

    duration = manifest.get(config.duration_field)
    if isinstance(duration, bool):
        duration = None
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        seconds = math.nan
    if not math.isfinite(seconds) or seconds < 0:
        return None, _integrity_error(
            cell, "INVALID_WALL_SECONDS", f"value {duration!r}"
        )
    return seconds, None


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def _round(value: float) -> float:
    return round(float(value), 6)


def summarize_campaign(
    plan_path: str | Path,
    results_root: str | Path,
    *,
    workers: int = 1,
    config: StatusConfig | None = None,
) -> dict[str, Any]:
    """Return an integrity-aware campaign summary suitable for JSON output.

    Invalid terminal evidence is classified as failed and excluded from timing
    statistics. ETA includes running and pending cells, assumes homogeneous
    workers, and reports both median and nearest-rank p90 scenarios.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise CampaignError("workers must be a positive integer")
    config = config or StatusConfig()
    root = Path(results_root)
    cells = _read_plan(Path(plan_path), root)
    counts = {key: 0 for key in ("completed", "running", "failed", "pending")}
    durations: list[float] = []
    errors: list[dict[str, str]] = []
    cell_rows: list[dict[str, Any]] = []

    for cell in cells:
        result_dir = root / cell.result_dir
        manifest_path = result_dir / config.manifest_name
        state = "pending"
        error: dict[str, str] | None = None
        duration: float | None = None
        if manifest_path.is_file():
            try:
                manifest = _read_manifest(manifest_path)
            except CampaignError as exc:
                state = "failed"
                error = _integrity_error(cell, "INVALID_MANIFEST", str(exc))
            else:
                status = manifest.get(config.status_field)
                if status == "COMPLETED":
                    duration, error = _completed_cell(cell, result_dir, manifest, config)
                    state = "completed" if error is None else "failed"
                elif status == "RUNNING":
                    state = "running"
                elif status == "FAILED":
                    state = "failed"
                else:
                    state = "failed"
                    error = _integrity_error(
                        cell,
                        "INVALID_STATUS",
                        f"field {config.status_field!r} has value {status!r}",
                    )
        elif (result_dir / config.running_name).exists():
            state = "running"
        elif (result_dir / config.failed_name).exists():
            state = "failed"

        counts[state] += 1
        if duration is not None:
            durations.append(duration)
        if error is not None:
            errors.append(error)
        cell_rows.append(
            {
                "cell_id": cell.cell_id,
                "result_dir": cell.result_dir,
                "state": state,
                "wall_seconds": _round(duration) if duration is not None else None,
            }
        )

    median = statistics.median(durations) if durations else None
    p90 = _nearest_rank(durations, 0.9) if durations else None
    remaining = counts["running"] + counts["pending"]
    waves = math.ceil(remaining / workers)
    eta_median = waves * median if median is not None else None
    eta_p90 = waves * p90 if p90 is not None else None
    return {
        "schema": "feature_selection_status.v1",
        "plan": str(Path(plan_path)),
        "results_root": str(root),
        "workers": workers,
        "counts": {"total": len(cells), **counts},
        "durations_seconds": {
            "sample_size": len(durations),
            "median": _round(median) if median is not None else None,
            "p90": _round(p90) if p90 is not None else None,
        },
        "eta_seconds": {
            "median": _round(eta_median) if eta_median is not None else None,
            "p90": _round(eta_p90) if eta_p90 is not None else None,
        },
        "integrity_errors": errors,
        "cells": cell_rows,
    }


def _atomic_write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="TSV campaign plan")
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, help="also atomically write this JSON file")
    parser.add_argument("--manifest-name", default="run_manifest.json")
    parser.add_argument("--artifact-name", default="results.jsonl")
    parser.add_argument("--running-name", default="RUNNING")
    parser.add_argument("--failed-name", default="FAILED")
    parser.add_argument("--status-field", default="status")
    parser.add_argument("--digest-field", default="results_sha256")
    parser.add_argument("--duration-field", default="wall_seconds")
    parser.add_argument("--cell-field", default="cell_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = StatusConfig(
        manifest_name=args.manifest_name,
        artifact_name=args.artifact_name,
        running_name=args.running_name,
        failed_name=args.failed_name,
        status_field=args.status_field,
        digest_field=args.digest_field,
        duration_field=args.duration_field,
        cell_field=args.cell_field,
    )
    try:
        report = summarize_campaign(
            args.plan, args.results_root, workers=args.workers, config=config
        )
    except CampaignError as exc:
        print(f"feature-selection-status: {exc}", file=sys.stderr)
        return 2
    if args.output:
        _atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
