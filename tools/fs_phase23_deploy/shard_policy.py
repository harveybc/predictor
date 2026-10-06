#!/usr/bin/env python3
"""Shard-assignment policy for the automated phase 2/3 selection campaign (§E.4, plan §5).

Input: the campaign plan's shard list (each shard with an estimated cost) and one host-health
document per role (tools/fs_phase23_deploy/host_health.sh). Output: a deterministic, exclusive
assignment of every shard to exactly one role, plus the per-host worker slot count and memory
cap the units must use.

Rules (all documented in the output so the receipt explains itself):

* ``coordinator`` receives the CHEAPEST THIRD of the shards: shards sorted by estimated cost
  ascending, the first ``ceil(N * coordinator_fraction)`` of them. The coordinator is the
  owner's desktop and the standing rule forbids heavy batch there; this order assigns it the
  smallest third explicitly, so its footprint is pinned: ONE worker slot, memory cap at most
  ``coordinator_cap_bytes`` (default 1 GiB) per process, nice/ionice idle, under the deployed
  admission monitor. Its share of the COST is reported and is well under a third.
* The remaining shards go to ``worker_a`` / ``worker_b`` by weighted LPT (largest cost first,
  each shard to the host with the lowest ``assigned_cost / capacity``), where capacity is the
  number of worker slots the host can run inside its admission headroom:
  ``slots = min(floor(free_cores / threads_per_slot), floor(headroom * headroom_safety / cap))``
  with headroom = the admission monitor's ``host_free_for_new_bytes`` (what it would admit
  now), never raw MemAvailable. A host whose slots come out at zero gets nothing.
* Costs are whatever the plan declares (``est_cost``); when absent, the pair count is the cost.

Only the standard library is used so the module runs on every host's venv or system python.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCHEMA = "fs_phase23_shard_assignment.v1"
ROLES = ("coordinator", "worker_a", "worker_b")
WORKER_ROLES = ("worker_a", "worker_b")
GIB = 1024 ** 3

DEFAULTS = {
    "coordinator_fraction": 1.0 / 3.0,
    "coordinator_cap_bytes": 1 * GIB,
    "coordinator_slots": 1,
    "worker_cap_bytes": 2 * GIB,
    "threads_per_slot": 2,
    "headroom_safety": 0.5,
    "max_slots_per_worker": 8,
}


def shard_cost(shard: Dict[str, Any]) -> float:
    for key in ("est_cost", "estimated_cost", "cost"):
        if key in shard and shard[key] is not None:
            return float(shard[key])
    for key in ("pair_count", "n_pairs", "pairs"):
        if key in shard and shard[key] is not None:
            return float(shard[key])
    raise ValueError(f"shard {shard.get('shard_id')!r} declares no est_cost or pair_count")


def shard_id(shard: Dict[str, Any]) -> str:
    for key in ("shard_id", "id", "shard"):
        if key in shard and shard[key] is not None:
            return str(shard[key])
    raise ValueError("shard without shard_id")


def load_shards(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    shards = plan.get("shards")
    if shards is None and isinstance(plan.get("populations"), list):
        shards = [s for pop in plan["populations"] for s in pop.get("shards", [])]
    if not isinstance(shards, list) or not shards:
        raise ValueError("plan has no shards")
    ids = [shard_id(s) for s in shards]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate shard_id in plan")
    return shards


def worker_slots(health: Dict[str, Any], cap_bytes: int, threads_per_slot: int, headroom_safety: float, max_slots: int) -> Dict[str, Any]:
    cores = int(health["cpu"]["cores"])
    load1 = float(health["cpu"]["loadavg_1_5_15"][0])
    free_cores = max(0.0, cores - load1)
    adm = health.get("admission") or {}
    headroom = adm.get("host_free_for_new_bytes")
    if headroom is None:  # no admission monitor on the host: do not guess, admit nothing
        headroom = 0
    by_cpu = int(free_cores // threads_per_slot)
    by_mem = int((headroom * headroom_safety) // cap_bytes)
    slots = max(0, min(by_cpu, by_mem, max_slots))
    return {"slots": slots, "by_cpu": by_cpu, "by_mem": by_mem, "free_cores": round(free_cores, 2),
            "admission_headroom_bytes": int(headroom), "cap_bytes": cap_bytes, "threads_per_slot": threads_per_slot}


def assign(plan: Dict[str, Any], health: Dict[str, Dict[str, Any]], **options: Any) -> Dict[str, Any]:
    opts = dict(DEFAULTS, **{k: v for k, v in options.items() if v is not None})
    shards = sorted(load_shards(plan), key=lambda s: (shard_cost(s), shard_id(s)))
    total_cost = sum(shard_cost(s) for s in shards)
    n = len(shards)
    n_coord = int(math.ceil(n * opts["coordinator_fraction"])) if n else 0
    coord_shards = shards[:n_coord]
    rest = shards[n_coord:]

    capacity: Dict[str, Dict[str, Any]] = {}
    for role in WORKER_ROLES:
        if role not in health:
            raise ValueError(f"missing health document for {role}")
        capacity[role] = worker_slots(health[role], int(opts["worker_cap_bytes"]), int(opts["threads_per_slot"]),
                                      float(opts["headroom_safety"]), int(opts["max_slots_per_worker"]))
    live = [r for r in WORKER_ROLES if capacity[r]["slots"] > 0]
    if rest and not live:
        raise ValueError("no worker role has admission headroom for even one slot; nothing assigned")

    assigned: Dict[str, List[str]] = {r: [] for r in ROLES}
    cost: Dict[str, float] = {r: 0.0 for r in ROLES}
    for s in coord_shards:
        assigned["coordinator"].append(shard_id(s)); cost["coordinator"] += shard_cost(s)
    # weighted LPT: largest first, to the live worker with the lowest normalised load; ties by role name
    for s in sorted(rest, key=lambda s: (-shard_cost(s), shard_id(s))):
        target = min(live, key=lambda r: (cost[r] / capacity[r]["slots"], r))
        assigned[target].append(shard_id(s)); cost[target] += shard_cost(s)
    for r in ROLES:
        assigned[r].sort()

    coord_health = health.get("coordinator") or {}
    return {
        "schema": SCHEMA,
        "plan_digest": plan.get("plan_digest") or plan.get("digest"),
        "shard_count": n,
        "total_cost": total_cost,
        "options": {k: (float(v) if isinstance(v, float) else v) for k, v in opts.items()},
        "roles": {
            "coordinator": {
                "shards": assigned["coordinator"], "shard_count": len(assigned["coordinator"]),
                "cost": cost["coordinator"], "cost_share": (cost["coordinator"] / total_cost) if total_cost else 0.0,
                "slots": int(opts["coordinator_slots"]), "cap_bytes": int(opts["coordinator_cap_bytes"]),
                "threads_per_slot": 1, "nice": 19, "ionice_class": "idle",
                "rule": "cheapest third by count (cost ascending); desktop-safe: one slot, cap <= coordinator_cap_bytes, under the admission monitor",
                "admission_headroom_bytes": ((coord_health.get("admission") or {}).get("host_free_for_new_bytes")),
            },
            **{
                r: {
                    "shards": assigned[r], "shard_count": len(assigned[r]), "cost": cost[r],
                    "cost_share": (cost[r] / total_cost) if total_cost else 0.0, **capacity[r],
                    "rule": "weighted LPT over the remaining shards; weight = slots inside admission headroom",
                } for r in WORKER_ROLES
            },
        },
        "exclusive": True,
    }


def check_exclusive(assignment: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    """Every plan shard appears exactly once across roles; returns the list of violations."""
    expected = {shard_id(s) for s in load_shards(plan)}
    seen: Dict[str, str] = {}
    problems: List[str] = []
    for role, body in assignment["roles"].items():
        for sid in body["shards"]:
            if sid in seen:
                problems.append(f"{sid} assigned to both {seen[sid]} and {role}")
            seen[sid] = role
            if sid not in expected:
                problems.append(f"{sid} assigned to {role} but not in plan")
    for sid in sorted(expected - set(seen)):
        problems.append(f"{sid} in plan but unassigned")
    return problems


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--health", action="append", required=True, help="role=path to a host_health.v1 document (three roles)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--coordinator-fraction", type=float)
    p.add_argument("--coordinator-cap-bytes", type=int)
    p.add_argument("--worker-cap-bytes", type=int)
    p.add_argument("--threads-per-slot", type=int)
    p.add_argument("--headroom-safety", type=float)
    p.add_argument("--max-slots-per-worker", type=int)
    a = p.parse_args(argv)
    health = {}
    for item in a.health:
        role, _, path = item.partition("=")
        if role not in ROLES or not path:
            p.error(f"--health expects role=path with role in {ROLES}: {item!r}")
        doc = _load(Path(path))
        if doc.get("role") != role:
            p.error(f"health document {path} declares role {doc.get('role')!r}, expected {role!r}")
        health[role] = doc
    plan = _load(a.plan)
    assignment = assign(plan, health, coordinator_fraction=a.coordinator_fraction, coordinator_cap_bytes=a.coordinator_cap_bytes,
                        worker_cap_bytes=a.worker_cap_bytes, threads_per_slot=a.threads_per_slot,
                        headroom_safety=a.headroom_safety, max_slots_per_worker=a.max_slots_per_worker)
    problems = check_exclusive(assignment, plan)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in ROLES:
        b = assignment["roles"][r]
        print(f"{r}: {b['shard_count']} shards, cost {b['cost']:.0f} ({b['cost_share']:.1%}), slots {b['slots']}, cap {b['cap_bytes'] // (1024**2)} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
