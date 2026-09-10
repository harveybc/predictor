#!/usr/bin/env python3
"""Emit common campaign envelopes for the completed campaigns.

Producers wired here are the ones the order names for the initial
backfill: T2 (public forecasting screen), M4 (residual-capacity
calibration) and B4 (the quarantined RL campaign). Each envelope
carries the producer's OWN identifiers and digests unchanged, and
writes `UNAVAILABLE` wherever the producer genuinely does not
publish a field — nothing is inferred to make a row look
complete.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap.campaign_envelope import (UNAVAILABLE,  # noqa: E402
                                    build_envelope)


def _read(p):
    return json.loads(Path(p).read_text())


def t2_envelope(adjudication_path: Path) -> dict:
    """T2's completion reconstruction + six-panel screen."""
    doc = _read(adjudication_path)
    screen = doc["screen_adjudication"]
    counts = doc["final_adjudication_counts"]
    units = []
    for panel, facts in sorted(screen["panels"].items()):
        # every number the panel publishes becomes its own row;
        # none is collapsed into a single "score" the producer
        # never computed
        for metric in ("effect", "attribution", "extreme_ratio",
                       "coverage_drop", "width_ratio"):
            if metric not in facts:
                continue
            units.append({
                "cell_key": panel,
                "candidate_key": "denoise_operator_D_vs_X",
                "metric_name": (
                    "mase_improvement_X_minus_D"
                    if metric == "effect" else metric),
                "metric_value": float(facts[metric]),
                "uncertainty_kind": UNAVAILABLE,
                "uncertainty_low": UNAVAILABLE,
                "uncertainty_high": UNAVAILABLE,
                "terminal_state": facts.get("extreme_state",
                                            "COMPLETED_VERIFIED"),
                "n_series": facts.get("n_series", UNAVAILABLE),
                "checkpoint_count": UNAVAILABLE,
                "epoch_count": UNAVAILABLE,
            })
    units.append({
        "cell_key": "ALL_PANELS",
        "candidate_key": "denoise_operator_D_vs_X",
        "metric_name": "primary_estimand_unweighted_panel_mean",
        "metric_value": float(screen[
            "primary_estimand_unweighted_mean_of_panel_effects"]),
        "uncertainty_kind": "t_interval_df5_lower_bound",
        "uncertainty_low": float(screen["t_ci_low_df5"]),
        "uncertainty_high": UNAVAILABLE,
        "terminal_state": "COMPLETED_VERIFIED",
        "checkpoint_count": UNAVAILABLE,
        "epoch_count": UNAVAILABLE,
    })
    return build_envelope(
        campaign_key="t2_resource_successor_v1_20260909",
        producer="T2 public forecasting screen",
        result_class="CONFIRMATION",
        identity={
            "run_id": doc["campaign_root_logical"],
            "code_identity": doc["gate_facts"].get(
                "execution_record_sha256", UNAVAILABLE),
            "design_sha256": doc["gate_facts"].get(
                "design_self_sha256", UNAVAILABLE),
            "record_sha256": doc["record_sha256"],
        },
        data_consumed={
            "datasets": [{"id": "t2_public_bank",
                          "digest": doc["gate_facts"].get(
                              "manifest_sha256", UNAVAILABLE),
                          "eligibility_state":
                              "PUBLICLY_EVALUATED"}],
            "variables": [], "operators": [
                {"id": "denoise_operator_D", "digest": UNAVAILABLE,
                 "eligibility_state": "PUBLICLY_EVALUATED"}],
        },
        partitions={
            "exposure": screen["scope"],
            "splits": "rolling origins as sealed in the T2 design",
        },
        budget={
            "device": "cpu",
            "wall_seconds": float(
                doc["wall_ledger"]["charged_state"]),
            "cost_units": "wall_seconds_charged_by_hash_chained_"
                          "ledger",
            "units_verified": counts["COMPLETED_VERIFIED"],
            "units_failed": counts["TERMINAL_FAILED"],
        },
        terminal={
            "state": "COMPLETE",
            "adjudication": screen["verdict"],
            "reason": screen["reason"],
        },
        artifacts={
            "adjudication_sha256": doc["record_sha256"],
            "wall_ledger_sha256": doc["wall_ledger"]["raw_sha256"],
            "release_done_sha256":
                doc["release_sequence"]["release_done_sha256"],
        },
        units=units)


def m4_envelope(adjudication_path: Path) -> dict:
    """M4's governing calibration adjudication."""
    doc = _read(adjudication_path)
    ladder = doc["ladder"]
    units = []
    for cell, disp in sorted(doc["dispersion"].items()):
        complete = disp.get("complete_generators")
        units.append({
            "cell_key": cell,
            "candidate_key": "residual_capacity_intervention",
            "metric_name": "complete_generators",
            "metric_value": float(complete)
            if isinstance(complete, int) else UNAVAILABLE,
            "uncertainty_kind": (
                "chi_square_ucb95"
                if "ucb95" in disp else UNAVAILABLE),
            "uncertainty_low": UNAVAILABLE,
            "uncertainty_high": float(disp["ucb95"])
            if isinstance(disp.get("ucb95"), (int, float))
            else UNAVAILABLE,
            "terminal_state": disp.get(
                "status", "CALIBRATION_COMPLETE"),
            "checkpoint_count": UNAVAILABLE,
            "epoch_count": UNAVAILABLE,
        })
    units.append({
        "cell_key": "LADDER",
        "candidate_key": "M2_vs_M1",
        "metric_name": "m2_minus_m1_paired_gain",
        "metric_value": float(ladder["m2_minus_m1_paired_gain"]),
        "uncertainty_kind": "paired_t",
        "uncertainty_low": UNAVAILABLE,
        "uncertainty_high": UNAVAILABLE,
        "terminal_state": ladder.get("status",
                                     "LADDER_EXECUTED"),
        "checkpoint_count": UNAVAILABLE,
        "epoch_count": UNAVAILABLE,
    })
    return build_envelope(
        campaign_key="m4_v5_calibration_attempt3_20260909",
        producer="M4 residual capacity",
        result_class="CALIBRATION",
        identity={
            "run_id": "m4_v5_calibration_run_attempt3_20260909",
            "code_identity": UNAVAILABLE,
            "design_sha256": doc["design_sha256"],
            "record_sha256": doc["record_sha256"],
        },
        data_consumed={
            "datasets": [{"id": "m4_synthetic_generator_bank",
                          "digest": doc["design_sha256"],
                          "eligibility_state": "LAB_CALIBRATED"}],
            "variables": [], "operators": [],
        },
        partitions={
            "exposure": "CALIBRATION_ONLY_CONFIRMATION_RESERVED",
            "splits": "TRAIN/STOP/EVALUATION byte-disjoint per "
                      "generator",
        },
        budget={"device": "cpu",
                "wall_seconds": UNAVAILABLE,
                "cost_units": "optimization_updates",
                "calibration_margin": doc["calibration_margin"]},
        terminal={
            "state": "COMPLETE",
            "adjudication": doc["authority"],
            "eligible_slots": sum(
                1 for s in doc["confirmation_slots"]
                if s["typed_status"] ==
                "ELIGIBLE_UNDER_PROPOSED_RULE"),
            "total_slots": len(doc["confirmation_slots"]),
        },
        artifacts={"adjudication_sha256": doc["record_sha256"],
                   "design_sha256": doc["design_sha256"]},
        units=units)


def b4_envelope(*, campaign_key: str, run_id: str,
                disposition: str) -> dict:
    """B4 is QUARANTINED: the envelope records exactly that, with
    no metric invented to fill the shape."""
    return build_envelope(
        campaign_key=campaign_key,
        producer="B4 paired RL campaign",
        result_class="NON_GOVERNING",
        identity={"run_id": run_id,
                  "code_identity": UNAVAILABLE,
                  "design_sha256": UNAVAILABLE},
        data_consumed={"datasets": [], "variables": [],
                       "operators": []},
        partitions={"exposure": "QUARANTINED_NOT_EXPOSED",
                    "splits": UNAVAILABLE},
        budget={"device": "gpu", "wall_seconds": UNAVAILABLE,
                "cost_units": UNAVAILABLE},
        terminal={"state": "QUARANTINED_RUNTIME_STALL",
                  "adjudication": disposition},
        artifacts={"note": "no adjudicable artifact — the "
                           "campaign did not complete"},
        units=[])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t2-adjudication", type=Path)
    ap.add_argument("--m4-adjudication", type=Path)
    ap.add_argument("--b4-quarantined", action="store_true")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if a.t2_adjudication:
        made.append(t2_envelope(a.t2_adjudication))
    if a.m4_adjudication:
        made.append(m4_envelope(a.m4_adjudication))
    if a.b4_quarantined:
        made.append(b4_envelope(
            campaign_key="b4_campaign_generation_v7_20260908",
            run_id="b4_campaign_results_v7_20260908",
            disposition="B4_V7_QUARANTINED_AND_EXTERNAL_"
                        "WATCHDOG_RECOVERY_READY_FOR_MUSASHI_"
                        "REVIEW"))
    for doc in made:
        p = a.out_dir / f"envelope-{doc['campaign_key']}.json"
        p.write_text(json.dumps(doc, indent=1, sort_keys=True)
                     + "\n")
    print(json.dumps([{
        "campaign_key": d["campaign_key"],
        "result_class": d["result_class"],
        "units": len(d["units"]),
        "adjudication": d["terminal"]["adjudication"],
        "envelope_sha256": d["envelope_sha256"],
    } for d in made], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
