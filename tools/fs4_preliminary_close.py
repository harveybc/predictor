#!/usr/bin/env python3
"""Close a pinned FS4 first wave from terminal tasks and retained warehouse readbacks.

This is a scoped evidence object, never the parent campaign's full closure or
a predictive feature-selection decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools import fs4_closure as full


ARMS = ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER")


class Refusal(ValueError):
    """The first-wave evidence cannot be closed."""


def close_wave(tasks: list[dict], manifest: dict, parent_plan_sha256: str,
               manifest_sha256: str, receipts: dict[str, dict]) -> dict:
    """Authenticate all selected feature/fold/arm cells against the parent store."""
    if manifest.get("schema") != "fs3.preliminary_gpu_triage_bundle.v1" or manifest.get("final_selection") is not False:
        raise Refusal("INVALID_PRELIMINARY_MANIFEST")
    selected = {row["population_id"]: set(row["gpu_feature_ids"]) for row in manifest["populations"]}
    if not selected or len(selected) != len(manifest["populations"]) or any(not v for v in selected.values()):
        raise Refusal("INVALID_PRELIMINARY_POPULATION")
    wave = [task for task in tasks if task["payload"]["population_id"] in selected
            and task["payload"]["feature_id"] in selected[task["payload"]["population_id"]]]
    expected = sum(map(len, selected.values())) * 5 * len(ARMS)
    if len(wave) != expected:
        raise Refusal(f"WAVE_DENOMINATOR_MISMATCH:{len(wave)}:{expected}")
    fold_arms: dict[tuple[str, str, str], set[str]] = {}
    complete = []
    unavailable = []
    for task in wave:
        payload = task["payload"]
        key = (payload["population_id"], payload["feature_id"], payload["fold_id"])
        arms = fold_arms.setdefault(key, set())
        if payload["arm"] not in ARMS or payload["arm"] in arms:
            raise Refusal(f"INVALID_ARM_TRIPLE:{key}")
        arms.add(payload["arm"])
        if task["state"] == "COMPLETE":
            problem = full.validate_complete(task)
            if problem:
                raise Refusal(f"INVALID_COMPLETE:{task['task_id']}:{problem}")
            receipt = receipts.get(task["task_id"])
            if not receipt or receipt.get("schema") != full.SCHEMA_RECEIPT_FILE \
                    or receipt.get("task_id") != task["task_id"] \
                    or receipt.get("plan_sha256") != parent_plan_sha256 \
                    or receipt.get("readback_verified") is not True \
                    or receipt.get("terminal_sha256") != full.local_terminal_sha256(task) \
                    or not receipt.get("receipt_sha256"):
                raise Refusal(f"MISSING_OR_MISMATCHED_READBACK:{task['task_id']}")
            embedded = receipt.get("receipt")
            if not isinstance(embedded, dict) or embedded.get("receipt_sha256") != receipt["receipt_sha256"] \
                    or embedded.get("plan_sha256") != parent_plan_sha256 \
                    or task["task_id"] not in embedded.get("task_ids", []) \
                    or full.core.digest({k: v for k, v in embedded.items() if k != "receipt_sha256"}) != receipt["receipt_sha256"]:
                raise Refusal(f"RECEIPT_ENVELOPE_MISMATCH:{task['task_id']}")
            complete.append(task)
        elif full.is_not_available(task):
            unavailable.append(task)
        else:
            raise Refusal(f"NONTERMINAL_OR_TECHNICAL:{task['task_id']}:{task['state']}")
    for pop, features in selected.items():
        for feature in features:
            folds = {fold for p, f, fold in fold_arms if p == pop and f == feature}
            if len(folds) != 5 or any(fold_arms[(pop, feature, fold)] != set(ARMS) for fold in folds):
                raise Refusal(f"INCOMPLETE_FOLD_ARMS:{pop}:{feature}")
    conflicts = full.triple_check(wave) + full.mixed_not_available(wave)
    if conflicts:
        raise Refusal(f"PAIRED_ARM_CONFLICT:{conflicts[:2]}")
    body = {
        "schema": "fs4.extractibility_partial_complete.v1",
        "state": "EXTRACTIBILITY_PARTIAL_COMPLETE", "final_selection": False,
        "scope": "PINNED_PRELIMINARY_FIRST_WAVE_ONLY",
        "custody": "RETAINED_WAREHOUSE_READBACK_RECEIPTS",
        "parent_plan_sha256": parent_plan_sha256, "manifest_sha256": manifest_sha256,
        "denominator": {"admitted": len(wave), "complete": len(complete),
                        "not_available_for_train": len(unavailable),
                        "sum_equals_admitted": len(complete) + len(unavailable) == len(wave)},
        "populations": full.build_populations(wave),
        "terminals_sha256": full.core.terminals_digest(full.local_terminal_sha256(t) for t in complete),
        "retained_readback_sha256": full.digest({t["task_id"]: receipts[t["task_id"]]["terminal_sha256"] for t in complete}),
        "deferred_features": {row["population_id"]: row["deferred_feature_ids"] for row in manifest["populations"]},
    }
    body["closure_sha256"] = full.digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    store = full.TaskStore(args.queue)
    try:
        parent_plan_sha256, _plan = store.plan()
        tasks = store.tasks()
    finally:
        store.close()
    receipt_dir = full.ReceiptDir(args.receipt_root)
    body = close_wave(tasks, json.loads(manifest_bytes), parent_plan_sha256,
                      hashlib.sha256(manifest_bytes).hexdigest(), receipt_dir.all_verified())
    if args.output.is_file() and json.loads(args.output.read_text()) != body:
        raise Refusal("EXISTING_PARTIAL_CLOSURE_DIFFERS")
    full.atomic_json(args.output, body)
    print(json.dumps({"state": body["state"], "closure_sha256": body["closure_sha256"],
                      "denominator": body["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
