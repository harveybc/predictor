#!/usr/bin/env python3
"""Seal a small weekly comparison against the measured FS4 successor wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools import fs4_frontier as full


Refusal = full.Refusal
SCHEMA = "fs4.frontier_seal.wave.v1"
RULE = ("FS4 wave frontier v1, TRAIN only. Exactly two predeclared phase-3 sets "
        "per target from the pinned target-coverage report enter the weekly RAW comparison. "
        "Every member must have paired RAW, RANDOM_ENCODER and TRAINED_ENCODER "
        "extractibility on all five TRAIN folds in the successor partial closure. "
        "All other consolidated sets remain explicitly deferred. No VALIDATION or TEST "
        "data choose a set; weekly outcomes decide predictive utility.")


def rule_sha256() -> str:
    return hashlib.sha256(RULE.encode()).hexdigest()


def _fully_measured(record: dict | None) -> bool:
    if not record or record.get("status") != "MEASURED":
        return False
    folds = record.get("folds", {})
    return len(folds) == 5 and all(
        fold.get("status") != "NOT_AVAILABLE_FOR_TRAIN"
        and all(isinstance(fold.get(arm), (int, float)) and not isinstance(fold.get(arm), bool)
                and math.isfinite(fold[arm])
                for arm in ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER"))
        for fold in folds.values())


def build_seal(consolidated: dict, partial: dict, target_sets: dict[str, list[str]]) -> dict:
    if partial.get("schema") != "fs4.extractibility_partial_complete.v1" \
            or partial.get("state") != "EXTRACTIBILITY_PARTIAL_COMPLETE":
        raise Refusal("INVALID_PARTIAL_CLOSURE")
    pop = consolidated["population_id"]
    features = partial["populations"][pop]["features"]
    sets = {s["set_id"]: s for s in consolidated["sets"]}
    expected_targets = {s["target_id"] for s in consolidated["sets"]}
    if set(target_sets) != expected_targets:
        raise Refusal("TARGET_COVERAGE_INCOMPLETE")
    selected = set()
    for target, ids in target_sets.items():
        if len(ids) != 2 or len(set(ids)) != 2:
            raise Refusal(f"TARGET_PAIR_MISSING:{target}")
        for set_id in ids:
            s = sets.get(set_id)
            if s is None or s["target_id"] != target:
                raise Refusal(f"TARGET_SET_MISSING:{target}:{set_id}")
            missing = [name for name in s["members"] if not _fully_measured(features.get(name))]
            if missing:
                raise Refusal(f"INCOMPLETE_EXTRACTIBILITY:{target}:{set_id}:{missing}")
            selected.add(set_id)
    dispositions = [{"set_id": s["set_id"], "target_id": s["target_id"],
                     "disposition": "IN_FRONTIER" if s["set_id"] in selected else "DEFERRED_BY_WAVE_RULE",
                     "reason": "pinned target-coverage pair" if s["set_id"] in selected else "outside pinned first comparison"}
                    for s in consolidated["sets"]]
    body = {"schema": SCHEMA, "population_id": pop, "identity": consolidated["identity"],
            "rule_text": RULE, "rule_sha256": rule_sha256(),
            "inputs": {"consolidated_sha256": consolidated["consolidated_sha256"],
                       "phase3_closure_sha256": consolidated.get("phase3_closure_sha256"),
                       "extractibility": {"closure_sha256": partial["closure_sha256"],
                                          "schema": partial["schema"], "denominator": partial["denominator"]}},
            "validation_read": {"count": 0, "statement": "TRAIN-only target coverage and extractibility"},
            "dispositions": dispositions, "frontier_set_ids": sorted(selected),
            "denominator": {"sets": len(dispositions), "in_frontier": len(selected),
                            "deferred": len(dispositions) - len(selected)},
            "final_selection": False}
    body["seal_sha256"] = full.digest(body)
    return body


def validate_seal(seal: dict) -> None:
    body = {k: v for k, v in seal.items() if k not in ("seal_sha256", "sealed_utc")}
    if seal.get("schema") != SCHEMA or seal.get("rule_text") != RULE \
            or seal.get("rule_sha256") != rule_sha256() or full.digest(body) != seal.get("seal_sha256"):
        raise Refusal("WAVE_FRONTIER_SEAL_CORRUPT")


def load_seal(path: Path) -> dict:
    seal = json.loads(Path(path).read_text())
    validate_seal(seal)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--consolidated", type=Path, required=True)
    parser.add_argument("--partial-closure", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cons = json.loads(args.consolidated.read_text())
    partial = json.loads(args.partial_closure.read_text())
    if full.digest({k: v for k, v in partial.items() if k != "closure_sha256"}) != partial.get("closure_sha256"):
        raise Refusal("PARTIAL_CLOSURE_DIGEST_MISMATCH")
    report = json.loads(args.coverage_report.read_text())
    body = build_seal(cons, partial, report["target_sets"][cons["population_id"]])
    if args.out.is_file() and json.loads(args.out.read_text()) != body:
        raise Refusal("EXISTING_WAVE_SEAL_DIFFERS")
    full.write_atomic(args.out, body)
    print(json.dumps({"population_id": body["population_id"], "seal_sha256": body["seal_sha256"],
                      "denominator": body["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
