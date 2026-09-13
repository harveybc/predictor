#!/usr/bin/env python3
"""C162 (order 2026-09-13): distribute the profile campaign across roles by
planned memory, not by dataset count.

For every dataset of the three banks, the planned peak is computed from
metadata only, exactly as the runner's parent preflight does (Source without
reading data columns, df_memory_plan.preflight), under one declared planning
budget. Bytes are not re-hashed here: the runner re-verifies them on the host
that runs the dataset, and a host without the exact bytes refuses.

Assignment (declared, deterministic):

* each role has a per-task cap and a campaign capacity (ROLE_CAPS);
* datasets are taken heaviest first (planned peak, then selector) and given to
  the role whose planned load divided by capacity is lowest among the roles
  whose task cap admits the dataset;
* the COORDINATOR also keeps the OLAP database and interactive work, so it
  only admits small tasks, and its runner is launched with at most two
  concurrent datasets;
* a dataset no task cap admits goes to the role with the largest cap and is
  marked OVER_EVERY_TASK_CAP: its runner bounds or refuses it and records why.
  Nothing is dropped.

Outputs (write-once directory): CAMPAIGN_PLAN.json with every dataset, its
planned peak, decision counts and role, and one jobs file per role, as
BANK:selector lines for `df_profile_run.py --jobs-file`. Hosts appear by role only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 1 << 30
# role -> (task cap bytes, campaign capacity bytes used for load balancing, max concurrent datasets)
ROLE_CAPS = {"WORKER_A": (10 * GIB, 12 * GIB, 3), "WORKER_B": (5 * GIB, 6 * GIB, 2),
             "COORDINATOR": (2 * GIB, 3 * GIB, 2)}
PLANNING_TASK_BYTES = 10 * GIB


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def selector(job: dict, contracts: list) -> str:
    if job["bank"] == "FINANCIAL":
        return f"FINANCIAL:{contracts[job['index']]['dataset_id']}"
    return f"{job['bank']}:{Path(job['dir']).name}"


def planned_peak(job: dict, budget: int) -> dict:
    R, P = _load("df_profile_run"), _load("df_memory_plan")
    try:
        src = R.Source(job, verify_bytes=False)
    except Exception as exc:  # noqa: BLE001 - recorded; the host run will refuse it with its own terminal
        return {"planned_peak_bytes": None, "reader_decision": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    meta = src.meta()
    if not meta["variables"]:
        return {"planned_peak_bytes": int(P.BASE_PROCESS_BYTES), "reader_decision": "RUN_EXACT", "decisions": {},
                "dataset_id": src.contract["dataset_id"], "T": src.T, "variables": 0}
    k, vm = P.pair_counts(len(src.numeric))
    rows = []
    planner = P.Planner(run_id="plan", bank=job["bank"], dataset_id=src.contract["dataset_id"], budget_bytes=budget,
                        code_sha256="0" * 64, context={"T": src.T, "n_train": meta["partitions"]["train"][1],
                                                       "has_ts": meta["has_ts"], "rg_rows": meta["rg_rows"],
                                                       "k_pairs": k, "V_matrix": vm},
                        sink=rows.append, stage="CAMPAIGN_PLAN_METADATA_UPPER_BOUND")
    plan = P.preflight(meta, planner)
    decisions = {}
    for r in rows:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
    return {"planned_peak_bytes": int(plan["planned_peak_bytes"]), "reader_decision": plan["reader_decision"],
            "decisions": decisions, "dataset_id": src.contract["dataset_id"], "T": src.T,
            "variables": len(src.numeric)}


def assign(items: list) -> dict:
    load = {r: 0 for r in ROLE_CAPS}
    for it in sorted(items, key=lambda x: (-(x["planned_peak_bytes"] or 0), x["selector"])):
        peak = it["planned_peak_bytes"] or 0
        fits = [r for r, (cap, _, _) in ROLE_CAPS.items() if peak <= cap]
        if not fits:
            role = max(ROLE_CAPS, key=lambda r: ROLE_CAPS[r][0])
            it["assignment_note"] = "OVER_EVERY_TASK_CAP"
        else:
            role = min(fits, key=lambda r: ((load[r] + peak) / ROLE_CAPS[r][1], r))
        it["role"] = role
        load[role] += peak
    return load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--public-panels", type=Path, required=True)
    ap.add_argument("--synthetic-bank", type=Path, required=True)
    ap.add_argument("--financial-contracts", type=Path, required=True)
    ap.add_argument("--financial-root", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"REFUSED: {a.out.name} exists; a campaign plan is write-once")
    R, IR = _load("df_profile_run"), _load("df_isolated_runner")
    contracts = json.loads(a.financial_contracts.read_text())["contracts"]
    jobs = R.public_jobs(a.public_panels) + R.synthetic_jobs(a.synthetic_bank) + \
        R.financial_jobs(a.financial_contracts, a.financial_root)
    budget = IR.limits_for(PLANNING_TASK_BYTES)["budget_bytes"]
    items = []
    for job in jobs:
        it = {"bank": job["bank"], "selector": selector(job, contracts)}
        it.update(planned_peak(job, budget))
        items.append(it)
    load = assign(items)
    code = {n: hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
            for n in ("df_campaign_plan", "df_profile_run", "df_memory_plan", "df_isolated_runner")}
    doc = {"schema": "crispdm.data_foundation.c162_campaign_plan.v1",
           "rule": "heaviest first to the least-loaded admitting role; planned peaks from metadata only",
           "role_caps": {r: {"task_cap_bytes": c, "capacity_bytes": cap, "max_concurrent": w}
                         for r, (c, cap, w) in ROLE_CAPS.items()},
           "planning_budget_bytes": budget, "code_sha256": code,
           "counts_by_role": {r: sum(1 for i in items if i["role"] == r) for r in ROLE_CAPS},
           "planned_load_bytes_by_role": load,
           "over_every_task_cap": [i["selector"] for i in items if i.get("assignment_note")],
           "preflight_errors": [(i["selector"], i["error"]) for i in items if i.get("error")],
           "datasets": items}
    a.out.mkdir(parents=True)
    for r in ROLE_CAPS:
        sel = sorted(i["selector"] for i in items if i["role"] == r)
        (a.out / f"JOBS_{r}.txt").write_text("".join(s + "\n" for s in sel))
    (a.out / "CAMPAIGN_PLAN.json").write_text(json.dumps(doc, indent=1, sort_keys=True).replace(str(Path.home()), "~")
                                              + "\n")
    print(json.dumps({k: doc[k] for k in ("counts_by_role", "planned_load_bytes_by_role", "over_every_task_cap")},
                     indent=1))
    print("preflight_errors:", len(doc["preflight_errors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
