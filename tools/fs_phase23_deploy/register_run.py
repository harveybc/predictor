#!/usr/bin/env python3
"""Register a campaign run in the warehouse with its expected row counts (idempotent).

The driver's follower submits rows and the store auto-registers the run on first submission,
but without ``expected_json`` the store's ``reconcile`` cannot say whether the run is complete.
This glue derives the expected counts from the plan (sum over shards of
``expected_row_counts_per_shard``) and registers the run as FIRST_SUBMISSION with them before
the follower starts; re-registering an identical document is a no-op, a different document under
the same run_id is refused by the store (and reported here). The contract-bound upgrade (contract,
campaign, code and input digests) belongs to the DATA agent's closure procedure and is not
invented here.

    register_run.py --plan PLAN.json --warehouse <path-or-url> [--token-env WAREHOUSE_TOKEN]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def expected_from_plan(plan: dict) -> dict:
    expected: dict[str, int] = {}
    for counts in plan.get("expected_row_counts_per_shard", {}).values():
        for table, n in counts.items():
            expected[table] = expected.get(table, 0) + int(n)
    return dict(sorted(expected.items()))


def run_document(plan: dict) -> dict:
    return {"run_id": plan["identity"], "population_id": plan["population_id"], "phase": "PHASE_2_3",
            "registration": "FIRST_SUBMISSION", "expected_json": expected_from_plan(plan)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--warehouse", required=True)
    p.add_argument("--token-env", default="WAREHOUSE_TOKEN")
    a = p.parse_args(argv)
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    doc = run_document(plan)
    sys.path.insert(0, os.getcwd())
    from tools import fs_phase23_warehouse as wh  # noqa: E402  (the DATA agent's module)
    try:
        w = wh.open_warehouse(a.warehouse, token_env=a.token_env)
        if not hasattr(w, "register_run"):  # the service registers on first submission; nothing to do here
            print(json.dumps({"run_id": doc["run_id"], "skipped": "warehouse handle has no register_run (service route)"}))
            return 0
        result = w.register_run(doc)
    except wh.Refusal as exc:
        print(json.dumps({"run_id": doc["run_id"], "refused": str(exc)}))
        return 3
    print(json.dumps({"run_id": doc["run_id"], "expected_json": doc["expected_json"], **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
