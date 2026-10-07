#!/usr/bin/env python3
"""TRAIN-only cost pilot for the phase-4 weekly wrapper (order item 4). No VALIDATION byte is opened.

Builds a throwaway plan with ``validation_year=2023`` from the sealed consolidated candidates and
frontier seal, picks one SMALL and one LARGE set of the frontier and three December-2023 weeks whose
scored week ends inside TRAIN, and runs each (set, week) as its own ``crispdm-run`` job through
``fs4_weekly_wrapper.py run-task --pilot-train-only`` (the production unit of work). Per fit it
records wall seconds, CPU seconds and the cgroup memory peak from the task's own cost block, then
writes ``cost_pilot_<population>.json`` with the measured cap (1.25 x the largest peak) and the
projected totals for the sealed frontier. The pilot decides cost only, never selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import fs4_frontier as F  # noqa: E402
from tools import fs4_weekly_wrapper as WW  # noqa: E402
from tools.fs4_candidates import canonical  # noqa: E402

PILOT_WEEK_STARTS = ("2023-12-04T00:00:00Z", "2023-12-11T00:00:00Z", "2023-12-18T00:00:00Z")


def pick_sets(plan: dict, population: str) -> dict:
    sets = [s for s in plan["sets"] if s["population_id"] == population]
    small = min(sets, key=lambda s: (s["n_features"], s["set_id"]))
    large = max(sets, key=lambda s: (s["n_features"], s["set_id"]))
    mid = sorted(sets, key=lambda s: (s["n_features"], s["set_id"]))[len(sets) // 2]
    return {"small": small, "median": mid, "large": large}


def build_pilot_tasks(consolidated_path, seal_path, population: str, bar_hours: int) -> tuple[dict, list[dict]]:
    cons = json.loads(Path(consolidated_path).read_text())
    seal = F.load_seal(seal_path)
    plan = WW.build_plan([cons], [seal], validation_year=2023, input_modes=("RAW",), bar_hours={population: bar_hours})
    chosen = pick_sets(plan, population)
    ids = {s["set_id"] for s in chosen.values()}
    tasks = [t for t in WW.enumerate_tasks(plan, "validation", set_ids=ids) if t["week"]["start"] in PILOT_WEEK_STARTS]
    for t in tasks:
        t["pilot_size_class"] = next(k for k, s in chosen.items() if s["set_id"] == t["set_id"])
    return plan, tasks


def summarize(rows: list[dict], frontier_sets: list[dict], weeks: int, modes_fits: dict) -> dict:
    ok = [r for r in rows if r.get("cost")]
    peak = max((r["cost"].get("peak_cgroup_bytes") or r["cost"]["peak_rss_bytes"]) for r in ok) if ok else None
    by_n = sorted({r["n_features"] for r in ok})
    # linear interpolation of per-fit wall seconds on the number of features (measured sizes only)
    pts = {n: [r["cost"]["wall_seconds"] for r in ok if r["n_features"] == n] for n in by_n}
    mean = {n: sum(v) / len(v) for n, v in pts.items()}

    def est(n):
        if not mean:
            return None
        xs = sorted(mean)
        if n <= xs[0]:
            return mean[xs[0]]
        if n >= xs[-1]:
            return mean[xs[-1]] * (n / xs[-1])      # beyond the largest measured size: proportional (documented)
        for lo, hi in zip(xs, xs[1:]):
            if lo <= n <= hi:
                return mean[lo] + (mean[hi] - mean[lo]) * (n - lo) / (hi - lo)
    total = sum((est(s["n_features"]) or 0) * weeks for s in frontier_sets)
    return {"measured_peak_bytes": peak, "measured_cap_bytes": int(peak * 1.25) if peak else None,
            "mean_wall_seconds_by_n_features": mean, "frontier_sets": len(frontier_sets), "weeks": weeks,
            "projected_fit_seconds_stage1_raw": total, "projected_fits_stage1_raw": len(frontier_sets) * weeks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--population", required=True)
    ap.add_argument("--consolidated", required=True)
    ap.add_argument("--frontier-seal", required=True)
    ap.add_argument("--bar-hours", type=int, required=True)
    ap.add_argument("--train-features", nargs="+", required=True)
    ap.add_argument("--train-targets", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--crispdm-run", default=str(Path.home() / ".local/bin/crispdm-run"))
    ap.add_argument("--cap", default="8G", help="admission cap for the pilot jobs (the measured peak sets the production cap)")
    ap.add_argument("--wall", default="2h")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sizes", default="small,median,large")
    a = ap.parse_args(argv)
    plan, tasks = build_pilot_tasks(a.consolidated, a.frontier_seal, a.population, a.bar_hours)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = set(a.sizes.split(","))
    rows = []
    for t in tasks:
        if t["pilot_size_class"] not in want:
            continue
        tf = out / f"task_{t['task_id'][:12]}.json"
        tf.write_text(canonical(t))
        rf = out / f"result_{t['task_id'][:12]}.json"
        if rf.is_file():
            rows.append(json.loads(rf.read_text()) | {"pilot_size_class": t["pilot_size_class"]})
            continue
        cmd = [a.crispdm_run, "-q", "-m", a.cap, "-t", a.wall, "-n", f"fs4-pilot-{t['pilot_size_class']}", "--",
               a.python, str(HERE / "fs4_weekly_wrapper.py"), "run-task", "--population", a.population,
               "--train-features", *a.train_features, "--train-targets", a.train_targets, "--bar-hours", str(a.bar_hours),
               "--pilot-train-only", "--task-file", str(tf)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if proc.returncode:
            rows.append({"task_id": t["task_id"], "n_features": t["n_features"], "pilot_size_class": t["pilot_size_class"], "week_start": t["week"]["start"],
                         "terminal_failure": f"rc={proc.returncode}: {proc.stderr[-400:]}", "launcher_wall_seconds": wall})
            continue
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        res["launcher_wall_seconds"] = wall
        rf.write_text(canonical(res))
        rows.append(res | {"pilot_size_class": t["pilot_size_class"]})
    seal = F.load_seal(a.frontier_seal)
    frontier_sets = [s for s in plan["sets"]]
    summary = summarize(rows, frontier_sets, 52, {})
    doc = {"schema": "fs4.weekly_cost_pilot.v1", "population": a.population, "weeks_pilot": list(PILOT_WEEK_STARTS),
           "validation_read": 0, "frontier_seal_sha256": seal["seal_sha256"],
           "fits": [{k: r.get(k) for k in ("pilot_size_class", "n_features", "week_start", "disposition", "reason", "fit_rows", "n_scored",
                                           "cost", "terminal_failure", "launcher_wall_seconds", "skill_mae")} for r in rows],
           "summary": summary}
    (out / f"cost_pilot_{a.population}.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(canonical(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
