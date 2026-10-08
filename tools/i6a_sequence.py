"""Run remaining I6-A horizons unattended, one target at a time per host.

The worker sequence waits for an already-running predecessor and reuses
verified cell receipts. The collector sequence closes each complete target,
publishes it to the warehouse, and updates the compact results catalog.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tools.i6a_campaign import atomic_json
from tools.i6a_weekly_arch_pilot import target_horizon_hours


def target_list(values):
    if not values or len(set(values)) != len(values):
        raise ValueError("EMPTY_OR_DUPLICATE_TARGETS")
    for value in values:
        target_horizon_hours(value)
    return values


def waiting_state(path, expected):
    if not path.is_file():
        return "WAITING"
    status = json.loads(path.read_text())
    if status.get("failed") or status.get("state") in ("FAILED", "INCOMPLETE_EVIDENCE", "CANCELLED"):
        return "FAILED"
    if status.get("state") == expected:
        if expected == "COMPLETE":
            return "READY" if (type(status.get("completed")) is int and type(status.get("total")) is int
                               and status["total"] > 0 and status["completed"] == status["total"]) else "FAILED"
        return "READY" if status.get("reports") == 208 else "FAILED"
    return "WAITING"


def wait_for(path, expected, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = waiting_state(path, expected)
        if state == "READY":
            return
        if state == "FAILED":
            raise RuntimeError(f"PREDECESSOR_FAILED: {path}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"PREDECESSOR_TIMEOUT: {path}")
        time.sleep(30)


def sequence_status(path, mode, targets, completed, current, state, **extra):
    atomic_json(path, {"schema": "i6a.sequence_status.v1", "mode": mode,
                       "targets": targets, "completed": completed, "total": len(targets),
                       "current": current, "state": state, **extra})


def worker(args):
    targets = target_list(args.target)
    sequence_status(args.sequence_status, "worker", targets, 0, targets[0], "WAITING_FOR_PREDECESSOR",
                    predecessor_status_path=str(args.wait_status))
    try:
        wait_for(args.wait_status, "COMPLETE", args.wait_timeout)
    except (RuntimeError, TimeoutError):
        sequence_status(args.sequence_status, "worker", targets, 0, targets[0], "FAILED",
                        predecessor_status_path=str(args.wait_status))
        raise
    completed = 0
    for target in targets:
        result_dir = args.results_root / f"{target}_{args.host_label}"
        child_status = result_dir / f"STATUS_{args.host_label}.json"
        sequence_status(args.sequence_status, "worker", targets, completed, target, "RUNNING",
                        current_status_path=str(child_status))
        command = [sys.executable, "-m", "tools.i6a_campaign", "--target", target,
                   "--expected-gpu-uuid", args.expected_gpu_uuid, "--freeze", str(args.freeze),
                   "--target-parquet", str(args.target_parquet),
                   "--val-target-parquet", str(args.val_target_parquet),
                   "--results-dir", str(result_dir), "--week-modulus", "1", "--week-remainder", "0",
                   "--host-label", args.host_label]
        for path in args.feature_parquet:
            command.extend(("--feature-parquet", str(path)))
        for path in args.val_feature_parquet:
            command.extend(("--val-feature-parquet", str(path)))
        child = subprocess.run(command, check=False)
        if child.returncode != 0 or waiting_state(child_status, "COMPLETE") != "READY":
            sequence_status(args.sequence_status, "worker", targets, completed, target, "FAILED",
                            exit_code=child.returncode, current_status_path=str(child_status))
            return 2
        completed += 1
    sequence_status(args.sequence_status, "worker", targets, completed, None, "COMPLETE")
    return 0


def collector(args):
    from tools.i6a_collect import collect
    from tools.i6a_fleet_status import campaign_status

    targets = target_list(args.target)
    sequence_status(args.sequence_status, "collector", targets, 0, targets[0], "WAITING_FOR_PREDECESSOR",
                    predecessor_status_path=str(args.wait_status))
    try:
        wait_for(args.wait_status, "PUBLISHED", args.wait_timeout)
        predecessor = json.loads(args.wait_status.read_text())
        if campaign_status(predecessor["target"], args.wait_status.parent).get("state") != "PUBLISHED":
            raise ValueError("PREDECESSOR_CLOSURE_MISMATCH")
    except (RuntimeError, TimeoutError, ValueError, KeyError):
        sequence_status(args.sequence_status, "collector", targets, 0, targets[0], "FAILED",
                        predecessor_status_path=str(args.wait_status))
        raise
    completed = 0
    for target in targets:
        root = args.local_root / f"{target}_{args.host_label}"
        sequence_status(args.sequence_status, "collector", targets, completed, target, "RUNNING",
                        current_status_path=str(root / "STATUS.json"))
        if campaign_status(target, root).get("state") != "PUBLISHED":
            try:
                code = collect(["--source", f"{args.host_label}={args.host}:{args.remote_root}/{target}_{args.host_label}",
                                "--destination", str(root), "--freeze", str(args.freeze),
                                "--target", target, "--closure", str(root / "CLOSURE.json"),
                                "--warehouse-url", args.warehouse_url, "--token-file", str(args.token_file),
                                "--code-commit", args.code_commit, "--catalog-root", str(args.catalog_root),
                                "--poll-seconds", "60", "--timeout-seconds", str(args.target_timeout)])
            except Exception:
                sequence_status(args.sequence_status, "collector", targets, completed, target, "FAILED",
                                current_status_path=str(root / "STATUS.json"))
                raise
            if code != 0:
                sequence_status(args.sequence_status, "collector", targets, completed, target, "FAILED",
                                current_status_path=str(root / "STATUS.json"))
                return code
        if campaign_status(target, root).get("state") != "PUBLISHED":
            raise ValueError(f"TARGET_NOT_PUBLISHED: {target}")
        completed += 1
    sequence_status(args.sequence_status, "collector", targets, completed, None, "COMPLETE")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("worker", "collector"):
        p = sub.add_parser(name)
        p.add_argument("--target", action="append", required=True)
        p.add_argument("--host-label", required=True)
        p.add_argument("--freeze", required=True, type=Path)
        p.add_argument("--wait-status", required=True, type=Path)
        p.add_argument("--wait-timeout", type=int, default=14400)
        p.add_argument("--sequence-status", required=True, type=Path)
    p = sub.choices["worker"]
    p.add_argument("--expected-gpu-uuid", required=True)
    p.add_argument("--feature-parquet", action="append", required=True, type=Path)
    p.add_argument("--target-parquet", required=True, type=Path)
    p.add_argument("--val-feature-parquet", action="append", required=True, type=Path)
    p.add_argument("--val-target-parquet", required=True, type=Path)
    p.add_argument("--results-root", required=True, type=Path)
    p = sub.choices["collector"]
    p.add_argument("--host", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--local-root", required=True, type=Path)
    p.add_argument("--warehouse-url", required=True)
    p.add_argument("--token-file", required=True, type=Path)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--catalog-root", required=True, type=Path)
    p.add_argument("--target-timeout", type=int, default=14400)
    args = ap.parse_args(argv)
    if args.wait_timeout < 1:
        raise ValueError("INVALID_WAIT_TIMEOUT")
    return worker(args) if args.mode == "worker" else collector(args)


if __name__ == "__main__":
    raise SystemExit(main())
