#!/usr/bin/env python3
"""Authenticate and reconcile host-local I6-D weekly campaign shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import i6d_weekly_walk_forward as weekly


def _load(path):
    return json.loads(Path(path).read_text())


def _expected_keys(design):
    return {
        (seed, week["ordinal"], arm)
        for seed in design["seeds"]
        for week in design["weeks"]
        for arm in design["arms"]
    }


def _cell_key(cell):
    return cell["seed"], cell["week_ordinal"], cell["arm"]


def _verified_source_cells(source, design):
    source_design = weekly.read_weekly_design(source)
    if source_design["design_sha256"] != design["design_sha256"]:
        raise ValueError(f"design identity mismatch: {source}")
    expected = _expected_keys(design)
    cells = []
    for path in sorted((Path(source) / "weekly_cells").glob("seed_*/week_*/*.json")):
        cell = weekly._verify_seal(_load(path), "cell_sha256")
        if cell.get("schema") != weekly.CELL_SCHEMA:
            raise ValueError(f"cell schema mismatch: {path}")
        if cell.get("design_sha256") != design["design_sha256"]:
            raise ValueError(f"cell design mismatch: {path}")
        try:
            key = _cell_key(cell)
        except KeyError as exc:
            raise ValueError(f"cell identity missing: {path}") from exc
        if key not in expected:
            raise ValueError(f"cell is outside sealed population: {path}")
        cells.append((key, cell))
    return cells


def merge_shards(destination, sources, *, close=False):
    """Copy only authenticated cells; repeated identical merges are idempotent."""
    destination = Path(destination)
    sources = [Path(source) for source in sources]
    design = weekly.read_weekly_design(destination)
    candidates = {}
    for source in sources:
        for key, cell in _verified_source_cells(source, design):
            digest = cell["cell_sha256"]
            if key in candidates and candidates[key]["cell_sha256"] != digest:
                raise ValueError(f"conflicting cell across sources: {key}")
            candidates[key] = cell
    for key, cell in candidates.items():
        path = weekly._cell_path(destination, *key)
        if path.exists():
            retained = weekly._verify_seal(_load(path), "cell_sha256")
            if retained["cell_sha256"] != cell["cell_sha256"]:
                raise ValueError(f"conflicting cell at destination: {key}")

    copied = existing = 0
    for key, cell in sorted(candidates.items()):
        path = weekly._cell_path(destination, *key)
        if path.exists():
            existing += 1
        else:
            weekly._atomic_json(path, cell)
            copied += 1
    status = weekly.weekly_status(destination)
    weekly._atomic_json(destination / "WEEKLY_STATUS.json", status)
    result = {
        "schema": "predictor.i6d.weekly_shard_merge.v1",
        "design_sha256": design["design_sha256"],
        "sources": [str(source) for source in sources],
        "copied": copied,
        "existing": existing,
        "complete_cells": status["complete_cells"],
        "expected_cells": status["expected_cells"],
        "complete": status["state"] == "COMPLETE",
    }
    if close:
        closure = weekly.close_weekly_campaign(destination)
        result["closure_state"] = closure["state"]
        result["closure_sha256"] = closure["closure_sha256"]
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--close", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = merge_shards(args.destination, args.source, close=args.close)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
