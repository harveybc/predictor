"""Generate additive C2 reporting successors from retained evidence, without fits.

Run: python tools/c2_reconcile.py --evidence docs/audits/evidence/lane_c2_eth_20261001
The fixed reconciliation_v1 directory is new-only. Originals are never written.
D4 counts are the retained STATUS declaration, not a recount of remote artifacts.
Policy is the owner's current instruction: one normal fit, absolute maximum
three replicas. Historical four-seed evidence is retained, not retroactively
made compliant or replaced by a favourable three-seed subset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

DESTINATION = "reconciliation_v1"
SOURCES = (
    "STATUS.json", "RESULTS.json", "RETURN.md",
    "calibration/BATTERY_CALIBRATION.json", "dossiers/DOSSIER_INDEX.json",
    "paired_inference/multiseed/MULTISEED.json",
)


def _progress(deliverables):
    found = [d for d in deliverables if d["id"] == "D4"]
    if len(found) != 1:
        raise ValueError("D4 progress must have exactly one source")
    d4 = copy.deepcopy(found[0])
    completed, planned = d4["completed"], d4["planned"]
    if (type(completed) is not int or type(planned) is not int or
            not 0 <= completed <= planned or planned <= 0):
        raise ValueError("invalid D4 progress")
    d4.update(remaining=planned - completed, state="RETAINED_PROGRESS_NOT_LIVE_VERIFIED",
              basis="STATUS.json retained declaration; remote artifacts not recounted",
              scheduled=False)
    return d4


def build_successors(evidence):
    """Pure report construction: read source bytes once and bind their digests."""
    evidence = Path(evidence)
    raw = {name: (evidence / name).read_bytes() for name in SOURCES}
    docs = {name: json.loads(data) for name, data in raw.items() if name.endswith(".json")}
    status = copy.deepcopy(docs["STATUS.json"])
    results = copy.deepcopy(docs["RESULTS.json"])
    index = docs["dossiers/DOSSIER_INDEX.json"]
    calibration = docs["calibration/BATTERY_CALIBRATION.json"]["aggregate"]
    if calibration != index["battery_calibration"]:
        raise ValueError("calibration differs from dossier index")
    validation = index["schema_validation"]
    if not (calibration["cells"] == len(index["dossiers"]) == validation["files"]):
        raise ValueError("calibration/dossier denominators disagree")
    d4 = _progress(status["deliverables"])
    multi = docs["paired_inference/multiseed/MULTISEED.json"]
    seeds, labels = multi["seeds"], multi["labels"]
    if (not seeds or any(type(s) is not int for s in seeds) or
            len(set(seeds)) != len(seeds) or not labels or len(set(labels)) != len(labels)):
        raise ValueError("invalid seed/architecture identities")
    expected = {f"{label}|{seed}" for label in labels for seed in seeds}
    if set(multi["cells_used"]) != expected or set(multi["per_cell"]) != expected:
        raise ValueError("seed declarations differ from retained cells")
    policy = dict(
        authority="Owner instruction: preserve historical four-seed evidence as noncompliant; no repetitions",
        normal_fits_per_configuration=1, absolute_max_replicas=3,
        seeds=sorted(seeds), seed_count_per_architecture=len(seeds), architectures=labels,
        retained_cells=len(expected), evidence_preserved=True,
        disposition=("HISTORICAL_PRESERVED_NONCOMPLIANT" if len(seeds) > 3
                     else "HISTORICAL_PRESERVED_REQUIRES_POLICY_REVIEW"),
        repetitions_scheduled=0, retroactive_seed_selection=False,
    )
    provenance = dict(
        source_sha256={name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
        generator="tools/c2_reconcile.py",
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_class="DEVELOPMENT", mode="REPORT_ONLY_NO_EXPERIMENTS",
        limitations=["D4 progress is source-declared, not a fresh remote measurement",
                     "No point-in-time availability certification or identified causal effect",
                     "Original STATUS/RESULTS/RETURN preserved byte-for-byte"],
    )
    status.update(schema="lane_c2_status.reconciled.v1", provenance=provenance,
                  seed_policy=policy, scheduled_actions=[], eta=None,
                  source_updated_at=status.pop("updated_at", None),
                  current_work="Reporting reconciliation only; no jobs started or scheduled",
                  next_dependencies="D4 artifact inventory and reconciliation remain outstanding; no dispatch",
                  historical_milestones=status.pop("milestones", None))
    for i, item in enumerate(status["deliverables"]):
        if item["id"] == "D4":
            status["deliverables"][i] = d4
        elif item["id"] == "D2":
            item.update(planned=calibration["cells"], completed=validation["files"] - validation["invalid"],
                        state=calibration["battery_verdict"], identification="NOT_IDENTIFIED")
    dossiers = results["D2_dossiers"]
    dossiers["historical_pre_calibration"] = {
        key: dossiers[key] for key in ("battery", "schema_errors_total") if key in dossiers
    }
    dossiers.update(battery=calibration, schema_errors_total=validation["invalid"],
                    schema_validation=validation, identification="NOT_IDENTIFIED")
    results.update(schema="lane_c2_results.reconciled.v1", provenance=provenance,
                   seed_policy=policy, D4_ps3r_rerun=d4, scheduled_actions=[])
    text = f"""# C2 additive reconciliation

Generated from retained evidence by `tools/c2_reconcile.py`. DEVELOPMENT only.
Original STATUS.json, RESULTS.json and RETURN.md are preserved; these files are
successors, not replacement experiment results. Source hashes are in both JSONs.

## D4

Retained progress: {d4['completed']}/{d4['planned']}; remaining: {d4['remaining']}.
Basis: historical STATUS declaration, not a fresh recount of remote records.
The old ETA is withdrawn. No resumption, repetitions or other jobs are scheduled.
The missing delivery is the artifact inventory and completed PS3-R reconciliation.

## Calibrated battery

Cells: {calibration['cells']}; schema-invalid dossiers: {validation['invalid']}.
Current verdict: {calibration['battery_verdict']}.
Scrambled-label rejections: {calibration['scrambled_reject_block_bootstrap_L']}/{calibration['cells']}
under the calibrated block interval, versus {calibration['scrambled_reject_hac_lag_h6']}/{calibration['cells']}
under the historical HAC rule. Historical values remain explicitly separated.
Identification remains NOT_IDENTIFIED: interval calibration is not causality.

## Historical seeds

Seeds: {', '.join(map(str, sorted(seeds)))}; {len(seeds)} per architecture,
{len(expected)} retained cells across {len(labels)} architectures.
Disposition: {policy['disposition']} under the current maximum of three replicas
(one normal fit). Evidence is preserved, not discarded, selectively reduced to
three seeds, or promoted to compliant evidence. Repetitions scheduled: 0.

## Scope

No training, inference, bootstrap, remote command or experiment was executed by
this generator. Original scientific measurements are retained, not recalculated.
ETH TRAIN findings and historical EURUSD validation inference remain distinct;
no confirmatory, point-in-time certification or financial-performance claim.
"""
    return {"STATUS.json": json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n",
            "RESULTS.json": json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
            "RETURN.md": text}


def write_successors(evidence):
    evidence = Path(evidence)
    contents = build_successors(evidence)
    destination = evidence / DESTINATION
    destination.mkdir(exist_ok=False)
    for name, content in contents.items():
        with (destination / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(write_successors(args.evidence))
