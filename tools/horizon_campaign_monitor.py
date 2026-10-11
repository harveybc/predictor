"""Cheap scoped status/ETA from retained evidence, without loading ML libraries.

The monitor is intentionally outside the sealed execution identity: its only
inputs are receipts and the read-only queue. It cannot claim, fit, or publish.
Partial-year errors never become annual eligibility. No log-watching agent is
needed to query a model, physical horizon, arm, or week.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import time


def status(root, *, model=None, horizon=None, arm=None, week=None, live=False):
    """Report counters and class-specific timing; unknown classes stay unknown."""
    with sqlite3.connect(f"file:{root/'QUEUE.db'}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT spec,state,terminal,published,slot FROM jobs ORDER BY ordinal").fetchall()
    workers = {w["role"]:w for w in json.loads((root/"WORKERS.private.json").read_text())} if live else {}
    counts, classes, groups, active = defaultdict(int), defaultdict(list), defaultdict(list), []
    for spec, state, text, published, slot in rows:
        job = json.loads(spec)
        if ((model and job["model"] != model) or (horizon and job["arm"]["horizon_hours"] != horizon)
                or (arm and job["arm"]["arm_id"] != arm) or (week is not None and job["week"]["ordinal"] != week)):
            continue
        counts[state] += 1
        counts["PUBLISHED"] += published
        key = (job["model"], job["arm"]["samples"])
        classes[key].append((state, json.loads(text) if text else None))
        if text:
            groups[(job["model"], job["arm"]["arm_id"])].append(json.loads(text))
        if state == "RUNNING":
            heartbeat = root/"cells"/job["id"]/"STATUS.json"
            current = json.loads(heartbeat.read_text()) if heartbeat.exists() else None
            if live and slot in workers and not workers[slot].get("local"):
                import subprocess
                worker = workers[slot]
                try:
                    remote = worker["root"]+"/cells/"+job["id"]+"/STATUS.json"
                    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker["host"], "cat", remote], text=True, capture_output=True, timeout=8, check=True)
                    current = json.loads(response.stdout)
                except (subprocess.SubprocessError, ValueError) as exc:
                    current = dict(status="REMOTE_HEARTBEAT_UNAVAILABLE", reason=str(exc))
            active.append(dict(job_id=job["id"], model=job["model"], horizon_hours=job["arm"]["horizon_hours"],
                               samples=job["arm"]["samples"], week=job["week"]["ordinal"], slot=slot,
                               retained_heartbeat=current))
    timing = []
    for (name, samples), cells in sorted(classes.items()):
        # Score-only adoption is NOT a timing measurement of training a week.
        durations = [c["elapsed_seconds"] for state, c in cells if state == "COMPLETED" and not c.get("reused_fit")]
        jobs_left = sum(state in ("QUEUED", "RUNNING") for state, _ in cells)
        median = statistics.median(durations) if durations else None
        timing.append(dict(model=name, samples=samples, measured_cells=len(durations), remaining=jobs_left,
                           median_child_seconds=median, estimated_serial_seconds=None if median is None else median*jobs_left))
    summaries = []
    for (name, arm_id), cells in sorted(groups.items()):
        measured = [c["metrics"] for c in cells if c["status"] == "COMPLETED"]
        n = sum(m["n"] for m in measured)
        if not n:
            continue
        mae = sum(m["mae"]*m["n"] for m in measured)/n
        naive = sum(m["naive_mae"]*m["n"] for m in measured)/n
        summaries.append(dict(model=name, arm_id=arm_id, horizon_hours=cells[0]["arm"]["horizon_hours"],
                              dispositions=len(cells), expected_dispositions=53, n=n, mae=mae, naive_mae=naive,
                              skill_percent=None if naive == 0 else 100*(mae-naive)/naive,
                              scope="FULL_YEAR" if len(cells) == 53 and not any(c["status"] == "FAILED" for c in cells) else "PARTIAL_YEAR_NOT_ELIGIBLE"))
    return dict(timestamp=time.time(), counts=dict(counts), active=active, timing_classes=timing,
                unknown_cost_classes=[dict(model=r["model"], samples=r["samples"], remaining=r["remaining"]) for r in timing if r["median_child_seconds"] is None and r["remaining"]],
                annual_completion_eta_seconds=None if any(r["median_child_seconds"] is None and r["remaining"] for r in timing) else sum(r["estimated_serial_seconds"] or 0 for r in timing),
                eta_scope="serial class-cost projection; not divided by inactive or unmeasured GPUs",
                results=summaries, skill_formula="(MAE_h - naive_h) / naive_h", skill_negative_is_better=True,
                unit="USD_per_EUR", test_read=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=["NEURAL", "ARIMA"])
    p.add_argument("--horizon", type=int)
    p.add_argument("--arm")
    p.add_argument("--week", type=int)
    p.add_argument("--live", action="store_true", help="bounded read-only remote heartbeat probes")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    value = status(args.root, model=args.model, horizon=args.horizon, arm=args.arm, week=args.week, live=args.live)
    if args.output:
        from tools.i6m_queue import atomic_json
        atomic_json(args.output, value)
    else:
        print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
