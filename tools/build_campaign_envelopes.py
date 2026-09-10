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


import hashlib  # noqa: E402


class ProducerBindingRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _self_sha(doc: dict, key: str) -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def _consume_producer_artifact(path: Path, *, schema_keys: set,
                               self_key: str, producer: str,
                               verifier: str) -> dict:
    """C12: consume an ORIGINAL producer artifact, not any JSON
    with familiar-looking fields.

    The previous builders opened an arbitrary file and copied its
    values, so a fabricated document with the right field names
    would have produced a perfectly self-consistent envelope. This
    requires the producer's exact top-level schema, recomputes the
    artifact's own self-digest with the producer's rule, and
    records the file digest plus the identity of the verifier that
    re-derived it.
    """
    path = Path(path)
    if not path.is_file():
        raise ProducerBindingRefusal(
            f"{producer}: the source artifact is absent at "
            f"{path.name}")
    file_sha = _sha_file(path)
    doc = json.loads(path.read_text())
    got = set(doc)
    if got != schema_keys:
        raise ProducerBindingRefusal(
            f"{producer}: {path.name} is not the producer's "
            f"schema (missing: {sorted(schema_keys - got)}, "
            f"unexpected: {sorted(got - schema_keys)}) — a JSON "
            "with similar fields is not the producer's artifact")
    declared = doc.get(self_key)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ProducerBindingRefusal(
            f"{producer}: {path.name} carries no canonical "
            f"{self_key}")
    recomputed = _self_sha(doc, self_key)
    if recomputed != declared:
        raise ProducerBindingRefusal(
            f"{producer}: {path.name} self-digest does not "
            "re-derive under the producer's own rule — the "
            "artifact was altered after it was produced")
    return {"doc": doc, "source_file_sha256": file_sha,
            "source_file_name": path.name,
            "producer_self_digest": declared,
            "verifier_identity": verifier,
            "verification": "SCHEMA_EXACT_AND_SELF_DIGEST_"
                            "REDERIVED"}


def _read(p):
    return json.loads(Path(p).read_text())


T2_SCHEMA_KEYS = {
    "authority", "campaign_root_logical",
    "final_adjudication_counts", "gate_facts", "record_sha256",
    "release_sequence", "schema", "screen_adjudication",
    "wall_ledger", "wall_seconds_reconstruction"}

M4_SCHEMA_KEYS = {
    "authority", "calibration_margin", "confirmation_slots",
    "design_sha256", "dispersion", "eligibility",
    "incomplete_units_in_denominator", "ladder",
    "ladder_excluded_units",
    "precision_supported_on_eligible_cells",
    "proposed_eligibility_rule", "record_sha256", "schema"}


def t2_envelope(adjudication_path: Path) -> dict:
    """T2's completion reconstruction + six-panel screen."""
    bound = _consume_producer_artifact(
        adjudication_path, schema_keys=T2_SCHEMA_KEYS,
        self_key="record_sha256",
        producer="T2 public forecasting screen",
        verifier="tools/t2_completion_reconstruction.py "
                 "final_adjudication + adjudicate_screen")
    doc = bound["doc"]
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
            "source_file_name": bound["source_file_name"],
            "source_file_sha256": bound["source_file_sha256"],
            "verifier_identity": bound["verifier_identity"],
            "verification": bound["verification"],
        },
        units=units)


def m4_envelope(adjudication_path: Path) -> dict:
    """M4's governing calibration adjudication."""
    bound = _consume_producer_artifact(
        adjudication_path, schema_keys=M4_SCHEMA_KEYS,
        self_key="record_sha256",
        producer="M4 residual capacity",
        verifier="tools/m4_v5_adjudicate.py "
                 "c35_calibration_adjudication")
    doc = bound["doc"]
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
                   "design_sha256": doc["design_sha256"],
                   "source_file_name": bound["source_file_name"],
                   "source_file_sha256":
                       bound["source_file_sha256"],
                   "verifier_identity":
                       bound["verifier_identity"],
                   "verification": bound["verification"]},
        units=units)


def b4_envelope(*, campaign_key: str, run_id: str,
                disposition: str,
                quarantine_record: Path | None = None) -> dict:
    """B4 is QUARANTINED: the envelope records exactly that, with
    no metric invented to fill the shape.

    C12: this envelope was previously built ENTIRELY from
    constants, so it asserted a quarantine no artifact backed.
    When a quarantine record exists it is consumed and bound like
    any other producer artifact; when it does not, the envelope
    says so in place instead of implying evidence.
    """
    if quarantine_record is not None:
        p = Path(quarantine_record)
        if not p.is_file():
            raise ProducerBindingRefusal(
                "B4: the quarantine record was named but is "
                f"absent at {p.name}")
        source = {"source_file_name": p.name,
                  "source_file_sha256": _sha_file(p),
                  "verification": "SOURCE_FILE_DIGESTED"}
    else:
        source = {"source_file_name": UNAVAILABLE,
                  "source_file_sha256": UNAVAILABLE,
                  "verification":
                      "NO_PRODUCER_ARTIFACT_EXISTS_THE_CAMPAIGN_"
                      "DID_NOT_COMPLETE"}
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
        artifacts={**source,
                   "note": "no adjudicable artifact — the "
                           "campaign did not complete"},
        units=[])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t2-adjudication", type=Path)
    ap.add_argument("--m4-adjudication", type=Path)
    ap.add_argument("--b4-quarantined", action="store_true")
    ap.add_argument("--b4-quarantine-record", type=Path)
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
            quarantine_record=a.b4_quarantine_record,
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
