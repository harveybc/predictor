"""Offline review probe: omissions must not improve a denoising decision.

Uses the repository's small fixture and real adjudicator, not campaign data.
Prints observations only; it grants no experimental eligibility.
"""
import argparse
import copy
import hashlib
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    fixture = runpy.run_path(str(root / "tests/test_df_d2_lab.py"))
    design = fixture["D"].build_design(fixture["inputs"]())
    adjudicator = fixture["A"]
    rows = fixture["_den"](design)

    def decide(records):
        result = fixture["_by"](adjudicator.decide_denoising(records, design), "ewma")
        return {"decision": result["decision"], "checks": result["evidence"]["checks"]}

    result = {"baseline": decide(rows)}
    damaged = copy.deepcopy(rows)
    for row in damaged:
        if row["operator_kind"] == "ewma" and row["metric"] == "extreme_retention":
            row["value"] = 0.2
    result["measured_damage"] = decide(damaged)
    result["omitted_damage_metric"] = decide([
        r for r in damaged if not (r["operator_kind"] == "ewma" and r["metric"] == "extreme_retention")
    ])
    for metric in ("delay_samples", "residual_signal_share"):
        incomplete = copy.deepcopy(rows)
        for row in incomplete:
            if row["operator_kind"] == "ewma" and row["metric"] == metric:
                row.update(value=None, status="INCONCLUSIVE", reason="measurement unavailable")
        result["inconclusive_" + metric] = decide(incomplete)
    published = args.decisions
    physical = hashlib.sha256(published.read_bytes()).hexdigest()
    expected = "f4958c88f8caa6b78d67fb7ff00c2a6a697276aa9b19b6b73411b78d97622510"
    if physical != expected:
        raise ValueError("Published decisions differ from the reviewed evidence")
    affected = []
    count = 0
    passes = 0
    with published.open() as source:
        for line in source:
            row = json.loads(line)
            count += 1
            if row.get("arm_role") != "CANDIDATE" or row["decision"] not in ("LAB_CALIBRATED", "REGIME_LIMITED"):
                continue
            passes += 1
            missing = {metric: sum(s.get(metric) is None for s in row["evidence"]["per_seed"].values())
                       for metric in ("snr_improvement_db", "delay_samples", "residual_signal_share", "cost")}
            if any(missing.values()):
                affected.append({"subject": row["subject"], "params": row["operator_params"],
                                 "regime": row["regime"], "decision": row["decision"],
                                 "n_valid": row["n_seeds_valid"],
                                 "improvement_n": row["evidence"]["checks"]["improvement"]["n"],
                                 "missing_by_metric": missing})
    print(json.dumps({"fixture_outcomes": {k: v["decision"] for k, v in result.items()},
                      "published_sha256": physical, "decisions": count,
                      "candidate_passes_or_limited": passes, "affected": affected},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
