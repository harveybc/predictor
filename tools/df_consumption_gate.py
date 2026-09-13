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

C181 test 7 (order 2026-09-13). A D2 record is schema
``stage_review_record.v2_d2`` and carries ``d2_binding``; it counts only when
the binding is CURRENT: the operator name is a kind of ``df_operators.KINDS``
and the subject is that kind (``kind`` or ``kind:<params>``), the fit mode is a
per-timestamp mode declared for the kind, ``operator_code_sha256`` equals the
current ``df_operators.code_sha256()``, ``lab_code_sha256`` equals the current
``df_d2_design.lab_code_sha256()``, ``root_mode`` is ``FRESH_CONFIRMATION`` and
the fresh root's tape and design digests are bound. A historical C137 decision
(retired name, old code, no fit mode, no fresh root) therefore never passes D2.
Every record is re-checked inside ``decide``, not only when loaded from disk.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("D0", "D1", "D2", "D3", "D4")
ALLOWED = {"D0": ("CONTRACT_REVIEWED",), "D1": ("PROFILE_REVIEWED",),
           "D2": ("LAB_CALIBRATED", "REGIME_LIMITED"),
           "D3": ("D3_REVIEWED_ACCEPTED",), "D4": ("D4_REVIEWED_ACCEPTED",)}
RECORD_SCHEMA = "crispdm.data_foundation.stage_review_record.v1"
RECORD_SCHEMA_D2 = "crispdm.data_foundation.stage_review_record.v2_d2"
RECORD_KEYS = {"schema", "stage", "reviewer", "reviewed_at_date", "subject_kind", "states", "regimes",
               "grants_public_eligibility", "record_sha256"}
D2_BINDING_KEYS = {"operator_kind", "spec_sha256", "fit_mode", "operator_code_sha256", "lab_code_sha256", "root_mode",
                   "fresh_root_tape_sha256", "design_sha256", "decision_run_id"}
FRESH_ROOT_MODE = "FRESH_CONFIRMATION"
SELF = ("satoshi", "general satoshi")
DEFAULT_AUTHORITY = Path.home() / ".config/agent-multi/reviewer_authority/data_foundation"
_HEX = re.compile(r"[0-9a-f]{64}")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def record_digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def d2_binding_problems(rec: dict, subject_id: str | None = None) -> list[str]:
    """Why a D2 record does not bind the current operator, code, fit mode and a fresh root."""
    b = rec.get("d2_binding")
    if not isinstance(b, dict) or set(b) != D2_BINDING_KEYS:
        return ["D2 record lacks the current-API binding (operator name, code digest, fit mode, fresh root)"]
    OPS = _load("df_operators")
    p = []
    kind = b["operator_kind"]
    if kind not in OPS.KINDS:
        p.append(f"operator {kind!r} is not a current kind")
    elif b["fit_mode"] not in OPS.KIND_FIT_MODES[kind] or b["fit_mode"] == OPS.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL:
        p.append(f"fit mode {b['fit_mode']!r} is not a per-timestamp mode of {kind}")
    if kind in getattr(OPS, "NON_CAUSAL_KINDS", ()):
        p.append("a non-causal control never passes D2")
    if b["operator_code_sha256"] != OPS.code_sha256():
        p.append("operator code digest is not the current one")
    if b["lab_code_sha256"] != _load("df_d2_design").lab_code_sha256():
        p.append("D2 lab code digest is not the current one")
    if b["root_mode"] != FRESH_ROOT_MODE:
        p.append(f"root mode {b['root_mode']!r} is not a fresh confirmation root")
    for k in ("spec_sha256", "fresh_root_tape_sha256", "design_sha256"):
        if not (isinstance(b[k], str) and _HEX.fullmatch(b[k])):
            p.append(f"{k} must bind a sha256 digest")
    if subject_id is not None and not (subject_id == kind or str(subject_id).startswith(f"{kind}:")):
        p.append(f"subject {subject_id!r} is not the bound operator {kind!r}")
    return p


def record_problems(rec) -> list[str]:
    d2 = isinstance(rec, dict) and rec.get("stage") == "D2"
    keys = RECORD_KEYS | {"d2_binding"} if d2 else RECORD_KEYS
    if not isinstance(rec, dict) or set(rec) != keys:
        return ["record schema differs" + (" (a D2 record must be v2_d2 with d2_binding)" if d2 else "")]
    p = []
    if rec["schema"] != (RECORD_SCHEMA_D2 if d2 else RECORD_SCHEMA):
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
        states = [r for r in records if isinstance(r, dict) and r.get("stage") == stage
                  and r.get("subject_kind") == subject_kind and isinstance(r.get("states"), dict)
                  and subject_id in r["states"]]
        unbound = []
        for r in states:
            why = record_problems(r) + (d2_binding_problems(r, subject_id) if stage == "D2" else [])
            if why:
                unbound.append(r)
                invalid.append(f"{stage}:{r['states'][subject_id]}:NOT_CONSUMABLE[{'; '.join(why[:3])}]")
        states = [r for r in states if r not in unbound]
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
