"""Read-only I6-A progress and ETA from verified receipts, including resumed runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tools import fs4_weekly_wrapper as W
from tools.i6a_architectures import ARMS
from tools.i6a_campaign import verify_result
from tools.i6a_weekly_arch_pilot import make_task


def snapshot(freeze, root, modulus, remainders, host_label):
    first = make_task(freeze, "Y_s_1h", 0, 2024)
    weeks = [w for w in W.W.build_protocol(2024, first["plan_sha256"]).weeks()
             if w.split is W.EvaluationSplit.VALIDATION]
    cells = [(i, arm, make_task(freeze, "Y_s_1h", i, 2024)) for i in range(len(weeks))
             if i % modulus in remainders for arm in ARMS]
    verified, durations, problems = 0, [], []
    for week, arm, task in cells:
        path = root / f"{arm}_val2024_week{week}.json"
        if not path.is_file():
            continue
        try:
            rec = verify_result(path, arm, task)
            verified += 1
            durations.append(float(rec["result"]["cost"]["wall_seconds"]))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            problems.append({"week": week, "arm": arm, "reason": str(exc)})
    controller = root / f"STATUS_{host_label}.json"
    state = json.loads(controller.read_text()) if controller.is_file() else {}
    pending = len(cells) - verified
    return {"schema": "i6a.readonly_status.v1", "host": host_label,
            "verified": verified, "total": len(cells), "pending_fits": pending,
            "eta_seconds": round(pending * statistics.median(durations[-12:])) if durations else None,
            "eta_basis": "median of last 12 verified per-cell wall readings; excludes interpreter startup and is a lower bound",
            "running": state.get("running"), "failed": state.get("failed"), "problems": problems}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--week-modulus", type=int, default=1)
    ap.add_argument("--week-remainder", type=int, action="append", required=True)
    ap.add_argument("--host-label", required=True)
    args = ap.parse_args(argv)
    if args.week_modulus <= 0 or any(r < 0 or r >= args.week_modulus for r in args.week_remainder):
        raise ValueError("INVALID_WEEK_PARTITION")
    print(json.dumps(snapshot(json.loads(args.freeze.read_text()), args.results_dir,
                              args.week_modulus, set(args.week_remainder), args.host_label), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
