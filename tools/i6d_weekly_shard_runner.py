#!/usr/bin/env python3
"""Unattended deterministic shard runner for an initialized I6-D campaign."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from tools import fs4_weekly_wrapper as weekly_data
from tools import i6d_weekly_walk_forward as weekly


def enumerate_cells(design):
    """Return the complete sealed cell population in stable execution order."""
    return [
        (seed, week["ordinal"], arm)
        for seed in design["seeds"]
        for week in design["weeks"]
        for arm in design["arms"]
    ]


def shard_cells(design, shard_index, shard_count):
    """Partition cells by stable ordinal without overlap or omission."""
    if (isinstance(shard_index, bool) or isinstance(shard_count, bool)
            or not isinstance(shard_index, int) or not isinstance(shard_count, int)
            or shard_count <= 0 or not 0 <= shard_index < shard_count):
        raise ValueError("shard index/count are invalid")
    return [
        cell for ordinal, cell in enumerate(enumerate_cells(design))
        if ordinal % shard_count == shard_index
    ]


def progress_report(*, worker_id, assigned, completed, failures, durations, active):
    """Build a monitorable status whose ETA is based only on terminal cells."""
    failed = len(failures)
    pending = max(0, assigned - completed - failed)
    median = float(statistics.median(durations)) if durations else None
    if active is not None:
        state = "RUNNING"
    elif pending:
        state = "READY"
    elif failures:
        state = "COMPLETE_WITH_FAILURES"
    else:
        state = "COMPLETE"
    return {
        "schema": "predictor.i6d.weekly_shard_status.v1",
        "worker_id": worker_id,
        "state": state,
        "assigned": assigned,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "active": None if active is None else {
            "seed": active[0], "week_ordinal": active[1], "arm": active[2]
        },
        "durations_seconds": [float(value) for value in durations],
        "median_cell_seconds": median,
        "eta_seconds": None if median is None else median * pending,
        "failures": failures,
        "updated_unix": time.time(),
    }


def _status_path(output, worker_id):
    return Path(output) / "shard_status" / f"{worker_id}.json"


def _read_prior(path):
    if not path.exists():
        return {"failures": [], "durations_seconds": []}
    value = json.loads(path.read_text())
    if value.get("schema") != "predictor.i6d.weekly_shard_status.v1":
        raise ValueError("existing shard status schema mismatch")
    return value


def _failure_key(failure):
    return failure["seed"], failure["week_ordinal"], failure["arm"]


def run_shard(*, output, worker_id, shard_index, shard_count,
              feature_parquets, target_parquet, validation_feature_parquets,
              validation_target_parquet):
    """Run every assigned cell once, retaining failures and resumable progress."""
    design = weekly.read_weekly_design(output)
    assigned_cells = shard_cells(design, shard_index, shard_count)
    status_path = _status_path(output, worker_id)
    prior = _read_prior(status_path)
    failures = list(prior.get("failures", []))
    failed_keys = {_failure_key(item) for item in failures}
    durations = [float(value) for value in prior.get("durations_seconds", [])]
    store = weekly_data.DataStore.from_paths(
        "EURUSD", feature_parquets, target_parquet,
        validation_feature_parquets, validation_target_parquet, bar_hours=1,
    )

    def completed_count():
        return sum(
            weekly._cell_path(output, seed, week, arm).exists()
            for seed, week, arm in assigned_cells
        )

    for cell in assigned_cells:
        seed, week_ordinal, arm = cell
        if weekly._cell_path(output, seed, week_ordinal, arm).exists() or cell in failed_keys:
            continue
        weekly._atomic_json(status_path, progress_report(
            worker_id=worker_id, assigned=len(assigned_cells),
            completed=completed_count(), failures=failures,
            durations=durations, active=cell,
        ))
        started = time.monotonic()
        try:
            weekly.run_weekly_cell_to_disk(
                output, arm, week_ordinal, seed, store
            )
            durations.append(time.monotonic() - started)
        except Exception as exc:  # retained terminal failure; independent cells continue
            failure = {
                "seed": seed,
                "week_ordinal": week_ordinal,
                "arm": arm,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            failed_keys.add(cell)
        weekly._atomic_json(status_path, progress_report(
            worker_id=worker_id, assigned=len(assigned_cells),
            completed=completed_count(), failures=failures,
            durations=durations, active=None,
        ))

    final = progress_report(
        worker_id=worker_id, assigned=len(assigned_cells),
        completed=completed_count(), failures=failures,
        durations=durations, active=None,
    )
    weekly._atomic_json(status_path, final)
    return final


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--feature-parquet", action="append", required=True)
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--validation-feature-parquet", action="append", required=True)
    parser.add_argument("--validation-target-parquet", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = run_shard(
        output=args.output,
        worker_id=args.worker_id,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        feature_parquets=args.feature_parquet,
        target_parquet=args.target_parquet,
        validation_feature_parquets=args.validation_feature_parquet,
        validation_target_parquet=args.validation_target_parquet,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
