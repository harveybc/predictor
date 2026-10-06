#!/usr/bin/env python3
"""STATUS.json for phases 2 and 3, computed from evidence on disk (never from memory).

Per population x method x host: expected, complete, failed, active, rate and ETA.
``complete`` counts terminals whose sidecar ``.meta.json`` names a unit of the plan;
``failed`` counts units with a failure record and no terminal; ``active`` counts fresh
claims without a terminal.  Rate is completed units in the trailing window; ETA is the
remaining estimated cost divided by the completed cost per second in that window.

    feature_selection_phase23_status.py --plan PLAN.json [...] --terminals DIR [...] --state-root DIR [...] --out STATUS.json [--every 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "fs_phase23.status.v1"
RATE_WINDOW_SECONDS = 3600.0
CLAIM_STALE_SECONDS = 900.0


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write_atomic(path: Path, doc: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _terminal_metas(dirs: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for meta in d.glob("*.meta.json"):
            doc = _read(meta)
            if doc and doc.get("unit_id") and doc["unit_id"] not in out:
                out[doc["unit_id"]] = doc
        for gz in d.glob("*.json.gz"):
            uid = gz.name[: -len(".json.gz")]
            if uid not in out:
                out[uid] = {"unit_id": uid, "finished_at": gz.stat().st_mtime, "wall_seconds": None, "host_id": None, "meta_missing": True}
    return out


def _records(roots: list[Path], sub: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for root in roots:
        d = Path(root) / sub
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            doc = _read(path)
            if not doc or not doc.get("unit_id"):
                continue
            prev = out.get(doc["unit_id"])
            if prev is None or (doc.get("at") or doc.get("claimed_at") or 0) >= (prev.get("at") or prev.get("claimed_at") or 0):
                out[doc["unit_id"]] = doc
    return out


def _cell() -> dict:
    return {"expected": 0, "complete": 0, "failed": 0, "active": 0, "pending": 0, "expected_cost": 0.0, "complete_cost": 0.0,
            "rate_units_per_hour": None, "rate_cost_per_second": None, "eta_seconds": None, "failures": [], "by_host": {}}


def build_status(*, plan_paths: list[Path], terminal_dirs: list[Path], state_roots: list[Path], out_path: Path,
                 follower_result: dict | None = None, now: float | None = None) -> dict:
    now = now or time.time()
    metas = _terminal_metas([Path(d) for d in terminal_dirs] + [Path(r) / "terminals" for r in state_roots] + [Path(r) / "adopted" for r in state_roots])
    failures = _records([Path(r) for r in state_roots], "failures")
    claims = _records([Path(r) for r in state_roots], "claims")
    quarantine = {}
    for root in state_roots:
        for path in (Path(root) / "quarantine").glob("*.reason.json"):
            doc = _read(path)
            if doc:
                quarantine[doc.get("unit_id")] = doc
    status = {"schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "generated_epoch": now,
              "populations": {}, "phase": "PHASE_2_RUNNING", "follower": follower_result}
    phase_rank = {"PHASE_2_RUNNING": 0, "PHASE_2_COMPLETE": 1, "PHASE_3_RUNNING": 2, "PHASE_3_COMPLETE": 3}
    phases = []
    for plan_path in plan_paths:
        plan = _read(plan_path)
        if not plan:
            continue
        pop = plan["population_id"]
        cell = _cell()
        window_done = []
        for shard in plan["shards"]:
            uid = shard["unit_id"]
            host = shard.get("host_id") or "unassigned"
            h = cell["by_host"].setdefault(host, {"expected": 0, "complete": 0, "failed": 0, "active": 0, "pending": 0})
            cell["expected"] += 1
            h["expected"] += 1
            cell["expected_cost"] += float(shard.get("estimated_cost", 0.0))
            meta = metas.get(uid)
            if meta is not None:
                cell["complete"] += 1
                h["complete"] += 1
                cell["complete_cost"] += float(shard.get("estimated_cost", 0.0))
                fin = meta.get("finished_at")
                if fin and now - float(fin) <= RATE_WINDOW_SECONDS:
                    window_done.append((float(fin), float(shard.get("estimated_cost", 0.0))))
                done_host = meta.get("host_id")
                if done_host and done_host != host:
                    cell["by_host"].setdefault(done_host, {"expected": 0, "complete": 0, "failed": 0, "active": 0, "pending": 0, "stolen": 0})
                    cell["by_host"][done_host]["stolen"] = cell["by_host"][done_host].get("stolen", 0) + 1
                continue
            claim = claims.get(uid)
            if claim and now - float(claim.get("claimed_at", 0)) < CLAIM_STALE_SECONDS:
                cell["active"] += 1
                h["active"] += 1
                continue
            fail = failures.get(uid) or quarantine.get(uid)
            if fail:
                cell["failed"] += 1
                h["failed"] += 1
                cell["failures"].append({"unit_id": uid, "host_id": fail.get("host_id"), "reason": fail.get("reason"), "shard_index": shard["shard_index"]})
                continue
            cell["pending"] += 1
            h["pending"] += 1
        if window_done:
            first = min(f for f, _ in window_done)
            span = max(now - first, 1.0)
            cell["rate_units_per_hour"] = len(window_done) / span * 3600.0
            cost_rate = sum(c for _, c in window_done) / span
            cell["rate_cost_per_second"] = cost_rate
            remaining = cell["expected_cost"] - cell["complete_cost"]
            cell["eta_seconds"] = (remaining / cost_rate) if cost_rate > 0 else None
        elif cell["complete"] >= cell["expected"]:
            cell["eta_seconds"] = 0.0
        cell["failures"] = cell["failures"][:50]
        cell["pairs_expected"] = plan.get("expected_pairs")
        cell["pairs_complete"] = sum(s["pair_count"] for s in plan["shards"] if s["unit_id"] in metas)
        pop_status = status["populations"].setdefault(pop, {})
        pop_status[plan.get("method", "pairwise_v1")] = cell
        # phase 3 cell from the state roots
        p3 = {"expected": len(plan.get("targets", [])), "complete": 0, "failed": 0, "active": 0, "eta_seconds": None}
        phase = "PHASE_2_RUNNING"
        for root in state_roots:
            root = Path(root)
            if (root / "PHASE_2_COMPLETE.json").exists():
                phase = "PHASE_2_COMPLETE"
            if (root / "PHASE_3_FILTER_COMPLETE.json").exists():
                phase = "PHASE_3_COMPLETE"
            t3 = root / "phase3" / "terminals"
            if t3.exists():
                p3["complete"] = max(p3["complete"], len(list(t3.glob("*.json.gz"))))
                if phase == "PHASE_2_COMPLETE" and 0 < p3["complete"] < p3["expected"]:
                    phase = "PHASE_3_RUNNING"
        if follower_result and follower_result.get("phase3_error"):
            p3["failed"] = 1
            p3["error"] = follower_result["phase3_error"]
        if follower_result and follower_result.get("closure_error"):
            cell["closure_error"] = follower_result["closure_error"]
        if p3["complete"] >= p3["expected"] > 0:
            p3["eta_seconds"] = 0.0
        pop_status["filter_v1"] = p3
        pop_status["phase"] = phase
        phases.append(phase)
    if phases:
        status["phase"] = min(phases, key=lambda ph: phase_rank[ph])
    _write_atomic(out_path, status)
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="append", required=True, type=Path)
    p.add_argument("--terminals", action="append", default=[], type=Path)
    p.add_argument("--state-root", action="append", default=[], type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--every", type=float, default=None)
    args = p.parse_args(argv)
    while True:
        build_status(plan_paths=args.plan, terminal_dirs=args.terminals, state_roots=args.state_root, out_path=args.out)
        if args.every is None:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
