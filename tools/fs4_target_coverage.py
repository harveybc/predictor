#!/usr/bin/env python3
"""Plan a second TRAIN-only FS4 wave for two small sets per target."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tools import fs4_frontier as frontier


RULE = ("For each phase-3 target choose two distinct sets with K<=12, excluding "
        "ALL_ADMISSIBLE and any feature already NOT_AVAILABLE_FOR_TRAIN in the first wave. "
        "Order by the existing TRAIN relevance + 0.5*measured-extractibility priority, "
        "then smaller K and set_id. Preserve all first-wave features and add only members "
        "missing from those sets. This is compute prioritization, not final selection.")


def plan_wave(manifest: dict, partial: dict, consolidated: dict[str, dict],
              rankings: dict[str, dict[str, dict[str, int]]]) -> dict:
    if manifest.get("schema") != "fs3.preliminary_gpu_triage_bundle.v1" or manifest.get("final_selection") is not False:
        raise ValueError("INVALID_FIRST_WAVE")
    if partial.get("schema") != "fs4.extractibility_partial_complete.v1" \
            or partial.get("state") != "EXTRACTIBILITY_PARTIAL_COMPLETE":
        raise ValueError("INVALID_PARTIAL_CLOSURE")
    target_sets, added, gaps, successor = {}, {}, [], copy.deepcopy(manifest)
    for row in successor["populations"]:
        pop = row["population_id"]
        cons = consolidated[pop]
        if cons["population_id"] != pop:
            raise ValueError("POPULATION_MISMATCH")
        ranks_by_target = rankings[pop]
        by_feature = frontier.feature_extractibility(partial, pop)
        records = partial["populations"][pop]["features"]
        unavailable = {f for f, record in records.items()
                       if record["status"] == frontier.NAFT
                       or any(fold.get("status") == frontier.NAFT for fold in record.get("folds", {}).values())}
        selected = set(row["gpu_feature_ids"])
        all_features = selected | set(row["deferred_feature_ids"])
        target_sets[pop], added[pop] = {}, []
        for target in sorted({s["target_id"] for s in cons["sets"]}):
            ranks = ranks_by_target.get(target)
            if not ranks:
                raise ValueError(f"NO_TRAIN_RANKS:{pop}:{target}")
            candidates = []
            for s in cons["sets"]:
                if s["target_id"] != target or not 1 <= s["k"] <= 12 or "ALL_ADMISSIBLE" in s["methods"]:
                    continue
                members = set(s["members"])
                if len(members) != s["k"] or not members <= all_features or members & unavailable:
                    continue
                terms = frontier.priority_terms(s["members"], ranks,
                                                max(len(ranks), max(ranks.values())), by_feature, True)
                candidates.append((s, terms["priority"]))
            candidates.sort(key=lambda item: (-item[1], item[0]["k"], item[0]["set_id"]))
            chosen = [s for s, _ in candidates[:2]]
            target_sets[pop][target] = [s["set_id"] for s in chosen]
            if len(chosen) < 2:
                gaps.append({"population_id": pop, "target_id": target, "available_sets": len(chosen)})
            for s in chosen:
                selected.update(s["members"])
        additions = selected - set(row["gpu_feature_ids"])
        added[pop] = sorted(additions)
        row["gpu_feature_ids"] = sorted(selected)
        row["deferred_feature_ids"] = sorted(all_features - selected)
    successor["parent_manifest_sha256"] = partial["manifest_sha256"] if "manifest_sha256" in partial else None
    successor["parent_partial_closure_sha256"] = partial["closure_sha256"]
    successor["selection_rule"] = RULE
    successor["selection_rule_sha256"] = hashlib.sha256(RULE.encode()).hexdigest()
    successor["wave"] = 2
    successor["target_sets"] = target_sets
    return {"successor_manifest": successor, "target_sets": target_sets,
            "added_features": added, "coverage_gaps": gaps, "final_selection": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-wave", type=Path, required=True)
    parser.add_argument("--partial-closure", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="POP|CONSOLIDATED.json|PHASE_3_FILTER_COMPLETE.json|terminals_dir")
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    raw_manifest = args.first_wave.read_bytes()
    partial = json.loads(args.partial_closure.read_text())
    if partial.get("manifest_sha256") != hashlib.sha256(raw_manifest).hexdigest():
        raise ValueError("PARTIAL_CLOSURE_FIRST_WAVE_MISMATCH")
    if frontier.digest({k: v for k, v in partial.items() if k != "closure_sha256"}) != partial.get("closure_sha256"):
        raise ValueError("PARTIAL_CLOSURE_DIGEST_MISMATCH")
    cons, ranks = {}, {}
    for item in args.input:
        pop, cons_path, closure_path, terminals_dir = item.split("|", 3)
        if pop in cons:
            raise ValueError(f"DUPLICATE_POPULATION:{pop}")
        c = json.loads(Path(cons_path).read_text())
        closure = json.loads(Path(closure_path).read_text())
        if c.get("phase3_closure_sha256") != closure.get("closure_sha256"):
            raise ValueError(f"PHASE3_CLOSURE_MISMATCH:{pop}")
        cons[pop] = c
        ranks[pop] = frontier.load_rankings_from_terminals(terminals_dir, closure["units"])
    result = plan_wave(json.loads(raw_manifest), partial, cons, ranks)
    if set(cons) != {row["population_id"] for row in result["successor_manifest"]["populations"]}:
        raise ValueError("MISSING_POPULATION")
    for path, data in ((args.out_manifest, result["successor_manifest"]),
                       (args.out_report, {k: v for k, v in result.items() if k != "successor_manifest"})):
        if path.is_file() and json.loads(path.read_text()) != data:
            raise ValueError(f"OUTPUT_ALREADY_DIFFERS:{path}")
        frontier.write_atomic(path, data)
    print(json.dumps({"added_features": {k: len(v) for k, v in result["added_features"].items()},
                      "coverage_gaps": result["coverage_gaps"], "target_sets": result["target_sets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
