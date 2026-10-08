"""Resumable 2024 weekly architecture campaign, sharded by complete weeks.

Run one instance per host with disjoint --week-remainders. Every week runs all
four arms on the same host; no cell is retrained when its digest verifies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools import fs4_weekly_wrapper as W
from tools.i6a_architectures import ARMS
from tools.i6a_weekly_arch_pilot import make_task


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(tmp, path)


def verify_result(path: Path, arm: str, task: dict):
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("sha256") != W.digest({k: v for k, v in record.items() if k != "sha256"}):
        raise ValueError(f"RESULT_DIGEST_MISMATCH: {path}")
    if record.get("arm") != arm or record.get("task") != task or record.get("result", {}).get("disposition") != "COMPLETED":
        raise ValueError(f"RESULT_IDENTITY_MISMATCH: {path}")
    result = record["result"]
    if result.get("result_sha256") != W.digest({k: v for k, v in result.items() if k != "result_sha256"}):
        raise ValueError(f"INNER_RESULT_DIGEST_MISMATCH: {path}")
    return record


def paired_week(records):
    if len(records) != len(ARMS):
        raise ValueError("INCOMPLETE_PAIRED_WEEK")
    results = [r["result"] for r in records]
    keys = ("rows_sha256", "n_scored", "fit_population_digest", "inner_population_digest", "fit_rows", "inner_rows", "naive")
    for key in keys:
        if any(r.get(key) != results[0].get(key) for r in results[1:]):
            raise ValueError(f"PAIRED_WEEK_MISMATCH: {key}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--feature-parquet", action="append", required=True)
    ap.add_argument("--target-parquet", required=True)
    ap.add_argument("--val-feature-parquet", action="append", required=True)
    ap.add_argument("--val-target-parquet", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--week-modulus", type=int, default=1)
    ap.add_argument("--week-remainder", type=int, action="append", required=True)
    ap.add_argument("--host-label", required=True)
    args = ap.parse_args(argv)
    if args.week_modulus <= 0 or len(set(args.week_remainder)) != len(args.week_remainder) or any(
            not 0 <= r < args.week_modulus for r in args.week_remainder):
        raise ValueError("INVALID_WEEK_PARTITION")
    freeze = json.loads(Path(args.freeze).read_text())
    probe = make_task(freeze, "Y_s_1h", 0, 2024)
    weeks = [w for w in W.W.build_protocol(2024, probe["plan_sha256"]).weeks()
             if w.split is W.EvaluationSplit.VALIDATION]
    assigned = [i for i in range(len(weeks)) if i % args.week_modulus in args.week_remainder]
    tasks = [(i, arm, make_task(freeze, "Y_s_1h", i, 2024)) for i in assigned for arm in ARMS]
    root = Path(args.results_dir)
    status_file = root / f"STATUS_{args.host_label}.json"
    durations = []
    base_args = ["--freeze", args.freeze, "--target-parquet", args.target_parquet,
                 "--val-target-parquet", args.val_target_parquet]
    for p in args.feature_parquet:
        base_args += ["--feature-parquet", p]
    for p in args.val_feature_parquet:
        base_args += ["--val-feature-parquet", p]
    total = len(tasks)
    for position, (week, arm, task) in enumerate(tasks):
        path = root / f"{arm}_val2024_week{week}.json"
        if verify_result(path, arm, task) is not None:
            continue
        atomic_json(status_file, {"schema": "i6a.campaign_status.v1", "host": args.host_label,
                                  "completed": position, "total": total, "running": {"week": week, "arm": arm},
                                  "eta_seconds": int((total - position) * sorted(durations)[len(durations) // 2]) if durations else None})
        command = [sys.executable, "-m", "tools.i6a_weekly_arch_pilot", *base_args,
                   "--arm", arm, "--validation-year", "2024", "--week-ordinal", str(week), "--output", str(path)]
        start = time.monotonic()
        with (root / f"{arm}_val2024_week{week}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            atomic_json(status_file, {"schema": "i6a.campaign_status.v1", "host": args.host_label,
                                      "completed": position, "total": total, "failed": {"week": week, "arm": arm,
                                                                                          "exit_code": completed.returncode}})
            return completed.returncode
        verify_result(path, arm, task)
        durations.append(time.monotonic() - start)
        if arm == ARMS[-1]:
            paired_week([verify_result(root / f"{a}_val2024_week{week}.json", a, make_task(freeze, "Y_s_1h", week, 2024))
                         for a in ARMS])
        atomic_json(status_file, {"schema": "i6a.campaign_status.v1", "host": args.host_label,
                                  "completed": position + 1, "total": total, "last": {"week": week, "arm": arm},
                                  "eta_seconds": int((total - position - 1) * sorted(durations)[len(durations) // 2])})
    atomic_json(status_file, {"schema": "i6a.campaign_status.v1", "host": args.host_label,
                              "completed": total, "total": total, "state": "COMPLETE", "eta_seconds": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
