#!/usr/bin/env python3
"""Run an I6-D shard with a fresh TensorFlow process for every cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools import i6d_weekly_shard_runner as shard
from tools import i6d_weekly_walk_forward as weekly


CELL_TOOL = Path(__file__).with_name("i6d_matched_control.py")


def child_environment(gpu_uuid):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return environment


def cell_command(*, output, seed, week_ordinal, arm, feature_parquets,
                 target_parquet, validation_feature_parquets,
                 validation_target_parquet):
    command = [
        sys.executable, str(CELL_TOOL), "weekly-run-cell",
        "--output", str(output), "--arm", arm,
        "--week-ordinal", str(week_ordinal), "--seed", str(seed),
    ]
    for path in feature_parquets:
        command.extend(("--feature-parquet", str(path)))
    command.extend(("--target-parquet", str(target_parquet)))
    for path in validation_feature_parquets:
        command.extend(("--validation-feature-parquet", str(path)))
    command.extend(("--validation-target-parquet", str(validation_target_parquet)))
    return command


def run_isolated_shard(*, output, worker_id, shard_index, shard_count, gpu_uuid,
                       feature_parquets, target_parquet,
                       validation_feature_parquets, validation_target_parquet):
    """Resume a shard while returning all TensorFlow allocations after each cell."""
    output = Path(output)
    design = weekly.read_weekly_design(output)
    assigned = shard.shard_cells(design, shard_index, shard_count)
    status_path = shard._status_path(output, worker_id)
    prior = shard._read_prior(status_path)
    failures = list(prior.get("failures", []))
    failed_keys = {shard._failure_key(item) for item in failures}
    durations = [float(value) for value in prior.get("durations_seconds", [])]
    environment = child_environment(gpu_uuid)

    def completed_count():
        return sum(
            weekly._cell_path(output, seed, week, arm).exists()
            for seed, week, arm in assigned
        )

    for cell in assigned:
        seed, week_ordinal, arm = cell
        if weekly._cell_path(output, seed, week_ordinal, arm).exists() or cell in failed_keys:
            continue
        weekly._atomic_json(status_path, shard.progress_report(
            worker_id=worker_id, assigned=len(assigned),
            completed=completed_count(), failures=failures,
            durations=durations, active=cell,
        ))
        started = time.monotonic()
        try:
            subprocess.run(
                [sys.executable, "-c",
                 "from tools.i6d_weekly_shard_runner import gpu_preflight; gpu_preflight()"],
                check=True, env=environment,
            )
            subprocess.run(cell_command(
                output=output, seed=seed, week_ordinal=week_ordinal, arm=arm,
                feature_parquets=feature_parquets,
                target_parquet=target_parquet,
                validation_feature_parquets=validation_feature_parquets,
                validation_target_parquet=validation_target_parquet,
            ), check=True, env=environment)
            durations.append(time.monotonic() - started)
        except subprocess.CalledProcessError as exc:
            failure = {
                "seed": seed, "week_ordinal": week_ordinal, "arm": arm,
                "reason": f"CHILD_EXIT:{exc.returncode}",
            }
            failures.append(failure)
            failed_keys.add(cell)
        weekly._atomic_json(status_path, shard.progress_report(
            worker_id=worker_id, assigned=len(assigned),
            completed=completed_count(), failures=failures,
            durations=durations, active=None,
        ))

    final = shard.progress_report(
        worker_id=worker_id, assigned=len(assigned),
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
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--feature-parquet", action="append", required=True)
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--validation-feature-parquet", action="append", required=True)
    parser.add_argument("--validation-target-parquet", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = run_isolated_shard(
        output=args.output, worker_id=args.worker_id,
        shard_index=args.shard_index, shard_count=args.shard_count,
        gpu_uuid=args.gpu_uuid, feature_parquets=args.feature_parquet,
        target_parquet=args.target_parquet,
        validation_feature_parquets=args.validation_feature_parquet,
        validation_target_parquet=args.validation_target_parquet,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
