#!/usr/bin/env python3
"""Freeze a TRAIN-only Phase-3 GPU triage without declaring final selection.

The broad ALL_ADMISSIBLE and RANDOM_K candidates remain controls, not reasons to
train every feature before the first weekly comparison. Deferred features are
retained by identity and can be revisited if the narrow wave fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHODS = (
    "SPEARMAN_CLUSTER", "MRMR", "JMI", "MRMR_CAUSAL", "JMI_CAUSAL",
    "UNIVARIATE_MI", "CAUSAL_SUPPORTED",
)
K_GRID = (4, 8, 12)


def _digest_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen(candidate_path: Path) -> dict:
    source = json.loads(candidate_path.read_text())
    closure_path = candidate_path.with_name("PHASE_3_FILTER_COMPLETE.json")
    closure = json.loads(closure_path.read_text())
    if closure.get("state") != "PHASE_3_FILTER_COMPLETE" or source.get("phase3_closure_sha256") != closure.get("closure_sha256"):
        raise ValueError("PHASE3_CLOSURE_MISMATCH")
    if source.get("final_selection") is not False or closure.get("final_selection") is not False:
        raise ValueError("SOURCE_ALREADY_FINAL")
    population = source["population_id"]
    identity = source["identity"]
    candidates = source["candidates"]
    targets = {entry["target_id"] for entry in closure["units"]}
    if not targets:
        raise ValueError("EMPTY_TARGET_POPULATION")
    seen = {}
    for candidate in candidates:
        if candidate.get("population_id") != population or candidate.get("identity") != identity:
            raise ValueError("MIXED_POPULATION")
        key = (candidate["target_id"], candidate["method"], candidate["k"])
        if key in seen:
            raise ValueError("DUPLICATE_CANDIDATE")
        seen[key] = candidate
    expected = {(target, method, k) for target in targets for method in METHODS for k in K_GRID}
    if not expected.issubset(seen):
        raise ValueError(f"MISSING_PRELIMINARY_CANDIDATES:{len(expected - seen.keys())}")
    selected = [seen[key] for key in sorted(expected)]
    features = sorted({feature for candidate in selected for feature in candidate["members"]})
    broad = [entry for entry in candidates if entry["method"] == "ALL_ADMISSIBLE"]
    if len(broad) != len(targets):
        raise ValueError("MISSING_ALL_ADMISSIBLE_CONTROL")
    all_features = {feature for entry in broad for feature in entry["members"]}
    declared_count = source["admissible_features"].get(population)
    if declared_count != len(all_features):
        raise ValueError("ADMISSIBLE_DENOMINATOR_MISMATCH")
    if not all_features or not set(features).issubset(all_features):
        raise ValueError("FEATURE_OUTSIDE_ADMISSIBLE_POPULATION")
    seasonal = sorted(feature for feature in features if feature.startswith("cal."))
    return {
        "schema": "fs3.preliminary_gpu_triage.v1",
        "population_id": population,
        "identity": identity,
        "phase3_closure_sha256": closure["closure_sha256"],
        "candidate_file_sha256": _digest_bytes(candidate_path),
        "closure_file_sha256": _digest_bytes(closure_path),
        "methods": list(METHODS),
        "k_grid": list(K_GRID),
        "targets": sorted(targets),
        "subset_count": len(selected),
        "subset_sha256": sorted(candidate["subset_sha256"] for candidate in selected),
        "gpu_feature_ids": sorted(set(features) - set(seasonal)),
        "seasonal_context_ids": seasonal,
        "deferred_feature_ids": sorted(all_features - set(features)),
        "all_admissible_control": "RAW_ONLY_PENDING_COMPARISON",
        "random_control": "RETAIN_AS_NEGATIVE_CONTROL_NOT_GPU_UNION",
        "causal_not_identified_policy": "NEUTRAL_NOT_PROOF_OF_NO_EFFECT",
        "final_selection": False,
        "validation_or_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    populations = [screen(path) for path in args.candidates]
    if len({row["population_id"] for row in populations}) != len(populations):
        raise ValueError("DUPLICATE_POPULATION")
    output = {"schema": "fs3.preliminary_gpu_triage_bundle.v1",
              "populations": sorted(populations, key=lambda row: row["population_id"]),
              "final_selection": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
