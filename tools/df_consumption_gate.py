#!/usr/bin/env python3
"""C144 (order 2026-09-12): I5 consumes nothing without reviewed D0-D4 states.

Feature selection (I5) may only consume a variable or a transformation whose
states for every data-foundation stage, D0 through D4, are recorded in an
EXTERNAL review record. This gate is the executable form of that rule.

  D0 contract     CONTRACT_REVIEWED
  D1 raw profile  PROFILE_REVIEWED
  D2 lab          LAB_CALIBRATED, or REGIME_LIMITED (consumption limited to its regimes)
  D3              D3_REVIEWED_ACCEPTED
  D4              D4_REVIEWED_ACCEPTED

A record written by Satoshi is not a review. A record whose digest does not
re-derive, whose schema differs, or whose state is not one of the above
(PUBLICLY_ELIGIBLE included) counts as absent. The gate never grants public
eligibility; its strongest answer is that a subject may be CONSIDERED by an
I5 design under review. With no records, which is the state today, every
subject is refused with every stage listed as missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

STAGES = ("D0", "D1", "D2", "D3", "D4")
ALLOWED = {"D0": ("CONTRACT_REVIEWED",), "D1": ("PROFILE_REVIEWED",),
           "D2": ("LAB_CALIBRATED", "REGIME_LIMITED"),
           "D3": ("D3_REVIEWED_ACCEPTED",), "D4": ("D4_REVIEWED_ACCEPTED",)}
RECORD_SCHEMA = "crispdm.data_foundation.stage_review_record.v1"
RECORD_KEYS = {"schema", "stage", "reviewer", "reviewed_at_date", "subject_kind", "states", "regimes",
               "grants_public_eligibility", "record_sha256"}
SELF = ("satoshi", "general satoshi")
DEFAULT_AUTHORITY = Path.home() / ".config/agent-multi/reviewer_authority/data_foundation"


def record_digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record_problems(rec) -> list[str]:
    if not isinstance(rec, dict) or set(rec) != RECORD_KEYS:
        return ["record schema differs"]
    p = []
    if rec["schema"] != RECORD_SCHEMA:
        p.append("foreign schema")
    if rec["stage"] not in STAGES:
        p.append("unknown stage")
    if not isinstance(rec["reviewer"], str) or rec["reviewer"].strip().lower() in SELF or not rec["reviewer"].strip():
        p.append("a record by the producer is not a review")
    if rec["grants_public_eligibility"] is not False:
        p.append("no stage record grants public eligibility")
    if rec["subject_kind"] not in ("VARIABLE", "OPERATOR"):
        p.append("unknown subject kind")
    if not isinstance(rec["states"], dict) or not isinstance(rec["regimes"], dict):
        p.append("states and regimes must be objects")
    if rec["record_sha256"] != record_digest(rec):
        p.append("record digest does not re-derive")
    return p


def load_records(authority: Path = DEFAULT_AUTHORITY) -> tuple[list[dict], list[dict]]:
    good, bad = [], []
    if not Path(authority).is_dir():
        return good, bad
    for p in sorted(Path(authority).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            bad.append({"file": p.name, "problems": ["not JSON"]})
            continue
        problems = record_problems(rec)
        (bad if problems else good).append(rec if not problems else {"file": p.name, "problems": problems})
    return good, bad


def decide(subject_id: str, subject_kind: str, records: list[dict]) -> dict:
    """Whether an I5 design may consider this subject, and why not."""
    missing, limits, invalid = [], {}, []
    for stage in STAGES:
        states = [r for r in records if r["stage"] == stage and r["subject_kind"] == subject_kind
                  and subject_id in r["states"]]
        accepted = [r for r in states if r["states"][subject_id] in ALLOWED[stage]]
        invalid += [f"{stage}:{r['states'][subject_id]}" for r in states if r["states"][subject_id] not in ALLOWED[stage]]
        if not accepted:
            missing.append(stage)
        elif stage == "D2" and all(r["states"][subject_id] == "REGIME_LIMITED" for r in accepted):
            limits["regimes"] = sorted({json.dumps(x, sort_keys=True) for r in accepted
                                        for x in r["regimes"].get(subject_id, [])})
            if not limits["regimes"]:
                missing.append("D2")
    return {"subject_id": subject_id, "subject_kind": subject_kind,
            "may_be_considered_by_i5_under_review": not missing,
            "missing_stages": missing, "states_not_consumable": invalid, "limits": limits,
            "public_eligibility": "NEVER_GRANTED_BY_THIS_GATE"}


def require(subject_id: str, subject_kind: str, records: list[dict]) -> dict:
    d = decide(subject_id, subject_kind, records)
    if not d["may_be_considered_by_i5_under_review"]:
        raise PermissionError(f"REFUSED: {subject_kind} {subject_id} lacks reviewed stages {d['missing_stages']}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    ap.add_argument("--subject", action="append", default=[], help="KIND:ID, e.g. OPERATOR:ewma_alpha_0.3")
    a = ap.parse_args(argv)
    good, bad = load_records(a.authority)
    out = {"records_valid": len(good), "records_refused": bad,
           "decisions": [decide(s.split(":", 1)[1], s.split(":", 1)[0], good) for s in a.subject]}
    print(json.dumps(out, indent=1, sort_keys=True).replace(str(Path.home()), "~"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
