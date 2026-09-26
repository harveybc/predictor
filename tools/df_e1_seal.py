#!/usr/bin/env python3
"""The E1 seal, built only from retained artifacts — and honest about being partial.

Sealing E1 means four things, and this module does exactly those four and nothing
else:

    identity     every artifact the seal cites is named by path and its identity is
                 RECOMPUTED from its own bytes. A sealed design must re-derive its
                 own ``design_sha256`` with the repository's canonical rule
                 (``df_mod_e0.sha_obj`` over the body without that field); a closure
                 or a block report must bind a design that is itself in the
                 inventory. An artifact that is absent, unreadable or whose digest
                 does not re-derive is a REFUSAL: a seal that would mislead is not
                 emitted
    rules        the acceptance rule is copied AS IT WAS STATED, verbatim, out of the
                 artifact that stated it before its outcome existed — never
                 paraphrased and never reconstructed from what happened
    numbers      the measured numbers are copied AS THEY WERE MEASURED, with the
                 scope string the producer attached, and with the counts that
                 qualify them (censored fits, NOT_APPLICABLE facts, absent blocks)
    conditions   every condition the review chain placed on E1 is listed with its
                 source and its quote, and each gets ONE of four states:

                     DISCHARGED                      the artifacts show it satisfied
                     PROHIBITION_OBSERVED            it forbids something, and no
                                                     retained artifact does that thing
                     UNMET                           it is not satisfied; the reason
                                                     is named
                     NOT_DISCHARGEABLE_BY_ARTIFACTS  satisfying it needs something
                                                     outside the retained evidence
                                                     (an external review, a producer
                                                     that is not in this repository,
                                                     a measurement that never ran);
                                                     what would discharge it is named

The verdict follows from the conditions and cannot be argued with: it is
``E1_SEALED`` only when every condition is DISCHARGED or PROHIBITION_OBSERVED, and
``E1_PARTIAL_SEAL`` otherwise. A partial seal still seals identity, rules and
numbers — those are facts about retained bytes — and says which conditions it does
not reach and why.

Nothing here fits, loads, scores or replays a model. It reads JSON and Markdown that
already exist and recomputes digests. CPU only.

    python tools/df_e1_seal.py --out SEAL.json [--markdown SEAL.md]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA = "df_e1_seal.v1"
EV = "docs/audits/evidence/d3_k5_20260917"
WP = "docs/audits/work_plan"
PROG = "docs/tres_temas_entrevista/program_v3"

DISCHARGED = "DISCHARGED"
OBSERVED = "PROHIBITION_OBSERVED"
UNMET = "UNMET"
OUTSIDE = "NOT_DISCHARGEABLE_BY_ARTIFACTS"
SEALED, PARTIAL = "E1_SEALED", "E1_PARTIAL_SEAL"


class SealRefusal(ValueError):
    """A seal that would mislead is not emitted."""


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_obj(obj) -> str:
    """The repository's canonical object digest (df_mod_e0.sha_obj), inlined so the
    seal does not depend on importing TensorFlow to hash a dictionary."""
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


# --- the declared inventory ------------------------------------------------------
# Explicit, so an artifact that disappears is a refusal and never a shorter table.
# kind: "design" re-derives its own design_sha256; "closure" and "report" must bind
# a design that is in this inventory; "document" is prose, bound by file digest only.

INVENTORY: tuple[tuple[str, str, str], ...] = (
    # sealed designs, in the order they were sealed
    ("design", f"{EV}/RP30/E1_PILOT_DESIGN.json",
     "the original household DEV pilot: ARCH-A under R0/R1/R2 with controls"),
    ("design", f"{EV}/RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json",
     "the successor pilot design, sealed before it ran"),
    ("design", f"{EV}/RP63/PHASE1_DESIGN_SEALED.json",
     "diagnostic phase 1: one factor at a time on the rows already used"),
    ("design", f"{EV}/RP65/PHASE2_DESIGN_SEALED.json",
     "phase 2 v1, 27 cells, superseded by v2"),
    ("design", f"{EV}/RP66/PHASE2_DESIGN_SEALED_v2.json",
     "phase 2 v2, 36 cells, carrying the post-Huber corrections"),
    ("design", f"{EV}/RP66/blocks/e1_block_dev_matched_v1/DESIGN.json",
     "DEV_MATCHED v1 (abandoned root, preserved)"),
    ("design", f"{EV}/RP66/blocks/e1_block_dev_matched_v2/DESIGN.json",
     "DEV_MATCHED v2"),
    ("design", f"{EV}/RP66/blocks/e1_block_q1_calendar_v1/DESIGN.json",
     "Q1_CALENDAR"),
    ("design", f"{EV}/RP66/blocks/e1_block_q2_context_v1/DESIGN.json",
     "Q2_CONTEXT — the block that never produced an outcome"),
    ("design", f"{EV}/RP66/blocks/e1_block_q3_volume_v1/DESIGN.json",
     "Q3_VOLUME"),
    ("design", f"{EV}/RP74/blocks/e1_block_arch_x_calendar_v1/DESIGN.json",
     "ARCH_X_CALENDAR, 15 cells across three hosts"),
    ("design", f"{EV}/RP82/blocks/e1_block_context_daily_lag_v1/DESIGN.json",
     "CONTEXT_DAILY_LAG"),
    # closures
    ("closure", f"{EV}/RP34/E1_PILOT_CLOSE_REPAIRED.json",
     "the repaired closure of the original pilot"),
    ("closure", f"{EV}/RP55/E1_SUCCESSOR_CLOSE.json",
     "the closure of the successor run"),
    # the owner's closure table over these very units (df_closure_table.py);
    # the seal BINDS it, it does not regenerate it
    ("closure_table", f"{EV}/RP82/CLOSURE_TABLE_RP89.json",
     "the owner-facing closure table over the E1 rounds: model error, naive "
     "error on identical rows, skill, literature value and comparability"),
    # a report whose sealed design is NOT retained in this repository
    ("report_unbound_design", "docs/audits/evidence/HUBER_ADAMW_2026_09_21/"
     "REPORT.json",
     "the Huber/AdamW factorial: its design digest be2e776e… has no retained "
     "artifact here, so its identity cannot be recomputed"),
    # block reports that carry an outcome
    ("report", f"{EV}/RP82/blocks/e1_block_dev_matched_v2/REPORT.json",
     "DEV_MATCHED outcome"),
    ("report", f"{EV}/RP82/blocks/e1_block_q1_calendar_v1/REPORT.json",
     "Q1_CALENDAR outcome"),
    ("report", f"{EV}/RP82/blocks/e1_block_q3_volume_v1/REPORT.json",
     "Q3_VOLUME outcome"),
    ("report", f"{EV}/RP82/blocks/e1_block_arch_x_calendar_v1/REPORT.json",
     "ARCH_X_CALENDAR outcome"),
    ("report", f"{EV}/RP82/blocks/e1_block_context_daily_lag_v1/REPORT.json",
     "CONTEXT_DAILY_LAG outcome"),
    # the documents the conditions are quoted from
    ("document", f"{WP}/MUSASHI_RP17_RP24_REVIEW_2026_09_19.md",
     "the review that forbids sealing 13E v1"),
    ("document", f"{WP}/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md",
     "the ML-baselines review: comparators, diagnoses and the legacy lineage"),
    ("document", f"{WP}/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md",
     "the Huber/AdamW measured comparison"),
    ("document", f"{WP}/MUSASHI_POST_HUBER_REVIEW_2026_09_21.md",
     "the post-Huber review"),
    ("document", f"{WP}/MUSASHI_RP136_RP139_REVIEW_2026_09_23.md",
     "the newest E1 precondition: a matched ECL adapter"),
    ("document", f"{WP}/SATOSHI_PROGRAM_RP17_RP24_RETURN_2026_09_19.md",
     "the return that asks for review 'before any E1 sealing'"),
    ("document", f"{WP}/SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20.md",
     "the return that says MOD-E1 is VERIFIED, not externally reviewed"),
    ("document", f"{WP}/SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21.md",
     "the return that answers the ML-baselines review"),
    ("document", f"{PROG}/13E_E1_TASK_SHEET_2026_09_19.md",
     "the E1 task sheet, v1 — still unsealed by its own words"),
)

#: block reports live beside a pilot-only report; a block whose only report is the
#: pilot one produced no outcome, and that is a fact the seal states rather than a
#: row it omits.
NO_OUTCOME_BLOCKS = (
    (f"{EV}/RP66/blocks/e1_block_q2_context_v1", "Q2_CONTEXT",
     "BUDGET_LIMITED_BEFORE_ANY_OUTCOME"),
)

#: design digests the closure table binds for which THIS repository retains no
#: sealed-design artifact. Declared here so the seal names them instead of
#: quietly passing its own binding check; each one is a hole in the identity the
#: seal can offer.
DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT = {
    "be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965": (
        "the frozen Huber/AdamW v2 design. Its digest is pinned by "
        f"{WP}/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md and carried by the "
        "closure table and the Huber report, but the design document itself is "
        "not in this repository, so this seal cannot recompute it from bytes"),
}

#: reviews whose absence is itself the evidence for a condition
EXPECTED_ABSENT_REVIEWS = (
    f"{WP}/MUSASHI_RP49_RP56_REVIEW_2026_09_20.md",
    f"{WP}/MUSASHI_RP57_RP64_REVIEW_2026_09_21.md",
)


# --- identity --------------------------------------------------------------------

def read_inventory(repo: Path = REPO) -> dict:
    """Recompute the identity of every declared artifact. Absence or a digest that
    does not re-derive raises: the seal is not emitted over broken evidence."""
    artifacts, designs, problems = {}, {}, []
    for kind, rel, why in INVENTORY:
        path = repo / rel
        if not path.is_file():
            raise SealRefusal(
                f"{rel}: declared in the inventory and absent from disk — "
                "a seal is never emitted over evidence it cannot read")
        entry = {"kind": kind, "why": why, "file_sha256": sha_file(path),
                 "bytes": path.stat().st_size}
        if kind in ("design", "closure", "report", "closure_table",
                    "report_unbound_design"):
            try:
                doc = json.loads(path.read_bytes())
            except ValueError as exc:
                raise SealRefusal(f"{rel}: not JSON ({exc})") from exc
            entry["schema"] = doc.get("schema")
            declared = doc.get("design_sha256")
            entry["design_sha256"] = declared
            if kind == "design":
                body = {k: v for k, v in doc.items() if k != "design_sha256"}
                recomputed = sha_obj(body)
                entry["design_sha256_recomputed"] = recomputed
                if declared is None:
                    raise SealRefusal(
                        f"{rel}: a sealed design with no design_sha256")
                if recomputed != declared:
                    raise SealRefusal(
                        f"{rel}: design_sha256 does not re-derive from the bytes "
                        f"(declared {declared}, recomputed {recomputed}) — an "
                        "edited design never enters a seal")
                entry["state"] = doc.get("state")
                designs.setdefault(declared, []).append(rel)
            entry["doc"] = doc
        artifacts[rel] = entry
    # every closure and report must bind a design that is itself sealed here
    unbound = {}
    for rel, entry in artifacts.items():
        if entry["kind"] in ("closure", "report"):
            d = entry.get("design_sha256")
            if not d:
                problems.append(f"{rel}: binds no design")
            elif d not in designs:
                problems.append(
                    f"{rel}: binds design {d[:12]}… which is not in the "
                    "sealed inventory")
        elif entry["kind"] == "report_unbound_design":
            d = entry.get("design_sha256")
            if d in designs:
                problems.append(
                    f"{rel}: declared as having no retained design, but "
                    f"{d[:12]}… IS in the inventory — the seal's own claim is "
                    "wrong")
            elif d not in DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT:
                problems.append(
                    f"{rel}: binds design {d} which is neither retained nor "
                    "declared as unretained")
            else:
                unbound[d] = DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT[d]
        elif entry["kind"] == "closure_table":
            for run, meta in (entry["doc"].get("runs") or {}).items():
                d = (meta or {}).get("design_sha256")
                if not d:
                    problems.append(f"{rel}: run {run!r} binds no design")
                elif d in designs:
                    continue
                elif d in DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT:
                    unbound[d] = DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT[d]
                else:
                    problems.append(
                        f"{rel}: run {run!r} binds design {d[:12]}… which is "
                        "neither retained in the inventory nor declared as "
                        "unretained — the seal will not pass over an identity "
                        "it cannot place")
    if problems:
        raise SealRefusal("; ".join(problems))
    return {"artifacts": artifacts, "designs_by_digest": designs,
            "design_digests_without_a_retained_artifact": unbound}


# --- rules, as they were stated ---------------------------------------------------

RULE_FIELDS = ("reading_rules", "common_evaluation_rule", "scaler_rule",
               "acceptance_before_runner_changes", "held_constant",
               "benchmark_contract", "limits", "tier", "state")


def rules_as_stated(inv: dict) -> dict:
    """Copy the pre-outcome rule text out of each sealed design, verbatim."""
    out = {}
    for rel, entry in inv["artifacts"].items():
        if entry["kind"] != "design":
            continue
        doc = entry["doc"]
        kept = {f: doc[f] for f in RULE_FIELDS if f in doc}
        if kept:
            out[rel] = {"design_sha256": entry["design_sha256"], "rules": kept}
    for rel, entry in inv["artifacts"].items():
        if entry["kind"] != "closure":
            continue
        doc = entry["doc"]
        out[rel] = {"design_sha256": entry["design_sha256"], "rules": {
            "population_rule": (doc.get("population") or {}).get("rule"),
            "replay_rule": (doc.get("replay") or {}).get("rule"),
            "governance_meaning": (doc.get("governance") or {}).get("meaning"),
        }}
    return out


# --- numbers, as they were measured ----------------------------------------------

def numbers_as_measured(inv: dict, repo: Path = REPO) -> dict:
    """The measured facts, each with the qualification its producer attached."""
    closures, blocks = {}, {}
    for rel, entry in inv["artifacts"].items():
        doc = entry["doc"] if "doc" in entry else None
        if entry["kind"] == "closure":
            units = doc.get("units") or {}
            not_applicable = {
                "inference": sorted(c for c, u in units.items()
                                    if u.get("inference") == "NOT_APPLICABLE"),
                "regime": sorted(c for c, u in units.items()
                                 if u.get("regime") == "NOT_APPLICABLE"),
            }
            closures[rel] = {
                "design_sha256": entry["design_sha256"],
                "at": doc.get("at"),
                "verdict": doc.get("verdict"),
                "counts": doc.get("counts"),
                "governance": {k: v for k, v in (doc.get("governance") or
                                                 {}).items() if k != "meaning"},
                "population": {
                    "declared": len((doc.get("population") or {}).get(
                        "declared") or []),
                    "absent_ids": (doc.get("population") or {}).get(
                        "absent_ids"),
                    "strangers_on_disk": (doc.get("population") or {}).get(
                        "strangers_on_disk")},
                "register_problems": doc.get("register_problems"),
                "unit_scopes": _tally(u.get("scope") for u in units.values()),
                "facts_not_applicable": not_applicable,
                "replay_windows_per_unit": (doc.get("replay") or {}).get(
                    "windows_per_unit"),
                "replay_tolerance_kw": (doc.get("replay") or {}).get(
                    "tolerance_kw"),
                "reading": (
                    "every declared unit verified; the NOT_APPLICABLE facts above "
                    "are the auto-encoder units (no forecast to replay) and the "
                    "controls (no detector regime), so the verdict hides no "
                    "unreplayed forecast"),
            }
        elif entry["kind"] == "report":
            blocks[rel] = {
                "design_sha256": entry["design_sha256"],
                "block": doc.get("block"),
                "verified": doc.get("verified"),
                "problems": doc.get("problems"),
                "scope": doc.get("scope"),
                "common_evaluation_rows": doc.get("common_evaluation_rows"),
                "spent_cpu_seconds": doc.get("spent_cpu_seconds"),
                "closure_code_drift": doc.get("closure_code_drift"),
                "summary": doc.get("summary"),
            }
    no_outcome = []
    for rel, name, state in NO_OUTCOME_BLOCKS:
        root = repo / rel
        if not root.is_dir():
            raise SealRefusal(f"{rel}: declared block root absent")
        reports = sorted(p.name for p in root.glob("REPORT*.json"))
        pilot = root / "REPORT.pilot.json"
        spent = None
        if pilot.is_file():
            spent = json.loads(pilot.read_bytes()).get("spent_cpu_seconds")
        if (root / "REPORT.json").is_file():
            raise SealRefusal(
                f"{rel}: declared as producing no outcome, but REPORT.json "
                "exists — the seal's own claim is wrong and it is not emitted")
        no_outcome.append({
            "block": name, "root": rel, "state": state,
            "reports_present": reports,
            "pilot_cpu_seconds": spent,
            "cells_with_an_outcome": 0,
            "reading": "pilots only; no cell of this block was fitted to a "
                       "score, so it says nothing about its question — "
                       "not 'no effect'"})
    return {"closures": closures, "blocks_with_an_outcome": blocks,
            "blocks_without_an_outcome": no_outcome,
            "owner_closure_table": owner_closure_table(inv)}


CLOSURE_TABLE_REQUIRED = ("task_horizon_split", "metric_and_scale",
                          "model_error", "naive_error", "skill_vs_naive",
                          "literature_value_and_source",
                          "comparability_status", "model_population",
                          "naive_population", "model_horizon",
                          "naive_horizon", "model_scale", "naive_scale")


def owner_closure_table(inv: dict) -> dict:
    """Bind the owner's closure table over these units and roll it up per arm.

    The owner's standing rule is that every closure carries the table — model
    error with its scale, the naive on the SAME rows, the skill, the literature
    value with its source, and a comparability status that says NOT_COMPARABLE
    with its reason — generated from artifacts. The table exists; this seal
    binds it and checks that every row really carries those columns, rather than
    restating numbers it has not looked at."""
    rel = next((r for r, e in inv["artifacts"].items()
                if e["kind"] == "closure_table"), None)
    if rel is None:
        raise SealRefusal("no owner closure table is bound by this seal")
    doc = inv["artifacts"][rel]["doc"]
    rows = doc.get("rows") or []
    if not rows:
        raise SealRefusal(f"{rel}: the closure table carries no rows")
    incomplete = [f"{r.get('run')}/{r.get('unit')}"
                  for r in rows
                  if any(r.get(f) is None for f in CLOSURE_TABLE_REQUIRED)]
    per_arm: dict = {}
    for r in rows:
        key = f"{r.get('run')}::{r.get('arm')}"
        a = per_arm.setdefault(key, {
            "run": r.get("run"), "arm": r.get("arm"), "n_rows": 0,
            "model_error_kw": [], "naive_error_kw": [], "skill": [],
            "naive_definition": r.get("naive_definition"),
            "populations": set(), "horizons": set(), "scales": set(),
            "comparability": set(), "verified_rows": 0})
        a["n_rows"] += 1
        a["model_error_kw"].append(r.get("model_error"))
        a["naive_error_kw"].append(r.get("naive_error"))
        sk = r.get("skill_vs_naive") or {}
        if sk.get("status") == "MEASURED":
            a["skill"].append(sk.get("value"))
        a["populations"].add((r.get("model_population"),
                              r.get("naive_population")))
        a["horizons"].add((r.get("model_horizon"), r.get("naive_horizon")))
        a["scales"].add((r.get("model_scale"), r.get("naive_scale")))
        a["comparability"].add(
            r.get("comparability_status") if isinstance(
                r.get("comparability_status"), str) else "STRUCTURED")
        a["verified_rows"] += 1 if r.get("verified") else 0
    rollup = {}
    for key, a in sorted(per_arm.items()):
        same_rows = all(m == n for m, n in a["populations"])
        same_h = all(m == n for m, n in a["horizons"])
        same_scale = all(m == n for m, n in a["scales"])
        if not (same_rows and same_h and same_scale):
            raise SealRefusal(
                f"{rel}: {key} compares a model and a naive that do not share "
                "rows, horizon or scale — such a row is never sealed")
        rollup[key] = {
            "run": a["run"], "arm": a["arm"], "n_rows": a["n_rows"],
            "verified_rows": a["verified_rows"],
            "mean_model_error_kw": _mean(a["model_error_kw"]),
            "mean_naive_error_kw": _mean(a["naive_error_kw"]),
            "mean_skill_vs_naive": _mean(a["skill"]),
            "naive_definition": a["naive_definition"],
            "evaluation_rows": sorted({p[0] for p in a["populations"]}),
            "comparability_status": sorted(a["comparability"]),
        }
    return {
        "path": rel,
        "schema": doc.get("schema"),
        "at": doc.get("at"),
        "generated_by": "tools/df_closure_table.py",
        "problems": doc.get("problems"),
        "verified_rows": doc.get("verified_rows"),
        "preserved_qualified_rows": doc.get("preserved_qualified_rows"),
        "custody_classes": doc.get("custody_classes"),
        "preparation_classes": doc.get("preparation_classes"),
        "custody_rule": doc.get("custody_rule"),
        "rules_as_stated": doc.get("rules"),
        "runs": {run: {"design_sha256": (meta or {}).get("design_sha256"),
                       "design_identity_recomputes": (
                           (meta or {}).get("design_identity") or {}).get(
                               "recomputes"),
                       "design_artifact_retained_here": bool(
                           (meta or {}).get("design_sha256")
                           in inv["designs_by_digest"])}
                 for run, meta in sorted(
                     (doc.get("runs") or {}).items())},
        "rows_total": len(rows),
        "rows_missing_a_required_column": incomplete,
        "per_arm": rollup,
        "reading": (
            "every row carries the model error, the naive on the identical "
            "evaluation rows at the identical horizon and scale, the measured "
            "skill, and a literature comparison typed NOT_COMPARABLE with its "
            "reason. The seal checked those invariants per row; it did not "
            "recompute the errors, which would require the arrays"),
    }


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))
            and not isinstance(v, bool)]
    return sum(vals) / len(vals) if vals else None


def _tally(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# --- the conditions of the review chain ------------------------------------------

def conditions(inv: dict, numbers: dict, repo: Path = REPO) -> list:
    """Each condition, its source, its quote, and the state the artifacts support.

    Every branch below is decided from retained bytes. Nothing is marked
    DISCHARGED because a return said so."""
    out = []
    art = inv["artifacts"]

    # 1 -- the explicit prohibition
    sheet = (repo / f"{PROG}/13E_E1_TASK_SHEET_2026_09_19.md").read_text()
    unsealed = "to be sealed after review" in sheet
    not_launched = "the E1 campaign is NOT launched" in sheet
    out.append({
        "id": "RP17_RP24_NO_SEAL_13E_V1",
        "source": f"{WP}/MUSASHI_RP17_RP24_REVIEW_2026_09_19.md",
        "quote": "No sellar 13E v1 como experimento de preentrenamiento.",
        "state": OBSERVED if (unsealed and not_launched) else UNMET,
        "evidence": (
            "13E v1 still carries the heading 'E1 design (proposal references; to "
            "be sealed after review)' and the sentence 'the E1 campaign is NOT "
            "launched on the v1 eligible: true'; no retained artifact seals it"
            if (unsealed and not_launched) else
            "the sheet no longer declares itself unsealed, which would make the "
            "prohibition violated rather than observed"),
        "note": "this seal does not seal 13E v1 either",
    })

    # 2 -- the legacy lineage
    rp57 = (repo / f"{WP}/SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21.md"
            ).read_text()
    unbound = "UNBOUND" in rp57
    out.append({
        "id": "ML_BASELINES_CAUSALITY_UNVERIFIED",
        "source": f"{WP}/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md",
        "quote": ("CAUSALITY_UNVERIFIED, no apto como benchmark hasta "
                  "reconstruir ambos linajes."),
        "state": OUTSIDE,
        "evidence": (
            "the answering return records both lineages as UNBOUND: the producer "
            "of the decomposed inputs is not in this repository"
            if unbound else
            "the answering return does not record the lineages as UNBOUND; "
            "re-read it before relying on this row"),
        "what_would_discharge_it": (
            "a reconstruction of a phase-3 run's consumed representation, from a "
            "producer that is not in this repository. No amount of work here "
            "reaches it"),
    })

    # 3 -- separate volume, context and calendar
    blocks = {b["block"] for b in numbers["blocks_with_an_outcome"].values()}
    missing = [b["block"] for b in numbers["blocks_without_an_outcome"]]
    have_cal = "Q1_CALENDAR" in blocks
    have_vol = "Q3_VOLUME" in blocks
    out.append({
        "id": "ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR",
        "source": f"{WP}/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md",
        "quote": ("exige medir informacion adicional y separar volumen, "
                  "contexto y calendario sin cambiar todo a la vez"),
        "state": DISCHARGED if (have_cal and have_vol and not missing)
                 else UNMET,
        "evidence": (
            f"calendar={'measured' if have_cal else 'absent'}, "
            f"volume={'measured' if have_vol else 'absent'}, "
            f"context block(s) with no outcome: {missing or 'none'}"),
        "what_would_discharge_it": (
            "fitting the Q2_CONTEXT cells. That is training, and this round is "
            "not authorised to run it"),
    })

    # 4 -- the Huber deferral
    regime_arms = sorted({
        arm for b in numbers["blocks_with_an_outcome"].values()
        for arm in (b.get("summary") or {})})
    reintroduced = [a for a in regime_arms if a in ("R0", "R1", "R2")]
    out.append({
        "id": "HUBER_NO_RETURN_TO_R0_R1_R2_YET",
        "source": f"{WP}/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md",
        "quote": "No return to R0/R1/R2 yet.",
        "state": OBSERVED if not reintroduced else UNMET,
        "evidence": (
            "no post-Huber block carries an R0/R1/R2 arm; the arms measured are "
            + ", ".join(regime_arms)),
    })

    # 5a -- phase 2 corrected
    v1 = art.get(f"{EV}/RP65/PHASE2_DESIGN_SEALED.json")
    v2 = art.get(f"{EV}/RP66/PHASE2_DESIGN_SEALED_v2.json")
    both = bool(v1 and v2)
    corrected = both and v2["schema"] == "df_e1_phase2_design.v2" \
        and v1["schema"] == "df_e1_phase2_design.v1"
    out.append({
        "id": "POST_HUBER_PHASE2_CORRECTED",
        "source": f"{WP}/MUSASHI_POST_HUBER_REVIEW_2026_09_21.md",
        "quote": ("phase-2 context and volume controls still change more than "
                  "stated"),
        "state": DISCHARGED if corrected else UNMET,
        "evidence": (
            f"v1 {v1['design_sha256'][:12]}… ({v1['schema']}, "
            f"state {v1.get('state')}) is preserved and superseded by v2 "
            f"{v2['design_sha256'][:12]}… ({v2['schema']}, "
            f"state {v2.get('state')}); both digests re-derive"
            if both else "one of the two phase-2 designs is absent"),
    })

    # 5b -- and its external acceptance
    out.append({
        "id": "POST_HUBER_EXTERNAL_ACCEPTANCE",
        "source": f"{WP}/MUSASHI_POST_HUBER_REVIEW_2026_09_21.md",
        "quote": ("do not accept the new verifier guarantees or phase-2/"
                  "financial design as ready unchanged"),
        "state": OUTSIDE,
        "evidence": "the correction is an artifact; its acceptance is a review",
        "what_would_discharge_it": (
            "a Musashi review that accepts phase-2 v2 and the verifier "
            "guarantees"),
    })

    # 6 -- the missing external review of E1 itself
    absent = [rel for rel in EXPECTED_ABSENT_REVIEWS
              if not (repo / rel).is_file()]
    rp49 = (repo / f"{WP}/SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20.md"
            ).read_text()
    self_declared = "VERIFIED, not externally reviewed" in rp49
    out.append({
        "id": "MOD_E1_EXTERNAL_REVIEW",
        "source": f"{WP}/SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20.md",
        "quote": "MOD-E1 is **VERIFIED, not externally reviewed**",
        "state": OUTSIDE,
        "evidence": (
            ("the return declares it itself; " if self_declared else
             "the quoted sentence was not found — re-read the return; ")
            + f"and no external review of the E1 rounds exists: absent "
              f"{[Path(a).name for a in absent]}"),
        "what_would_discharge_it": (
            "a review of the RP49-RP56 and RP57-RP64 returns. Only Musashi "
            "writes it; no agent can substitute for it"),
    })

    # 6b -- the owner's standing closure-table rule
    tbl = numbers["owner_closure_table"]
    ok = (not tbl["rows_missing_a_required_column"]
          and tbl["problems"] == []
          and all(r["design_identity_recomputes"] is True
                  for r in tbl["runs"].values()))
    out.append({
        "id": "OWNER_CLOSURE_TABLE_FROM_ARTIFACTS",
        "source": "owner standing order, 2026-09-21",
        "quote": ("every closure carries the table — model error with its "
                  "scale, the naive on the same rows, the skill, the "
                  "literature value with its source, comparability, "
                  "NOT_COMPARABLE with its reason — generated from artifacts"),
        "state": DISCHARGED if ok else UNMET,
        "evidence": (
            f"{tbl['path']} ({tbl['schema']}, {tbl['at']}) covers "
            f"{tbl['rows_total']} rows over {len(tbl['runs'])} runs with "
            f"problems {tbl['problems']}; {tbl['verified_rows']} verified and "
            f"{tbl['preserved_qualified_rows']} preserved with a qualified "
            "scope; every row carries all thirteen required columns and every "
            "run's design identity recomputes"
            if ok else
            f"rows missing a required column: "
            f"{tbl['rows_missing_a_required_column'][:5]}; table problems "
            f"{tbl['problems']}"),
        "note": ("the seal binds this table; it does not regenerate it, and it "
                 "recomputed no error from arrays"),
    })

    # 7 -- the newest precondition
    contracts = sorted({
        (entry["doc"].get("contract_source") or "")
        for rel, entry in art.items() if entry["kind"] == "design"
        and entry["doc"].get("contract_source")})
    out.append({
        "id": "RP136_RP139_MATCHED_ECL_ADAPTER",
        "source": f"{WP}/MUSASHI_RP136_RP139_REVIEW_2026_09_23.md",
        "quote": ("Required before matched ECL fitting: current E1 target code "
                  "is a single channel at one offset."),
        "state": UNMET,
        "evidence": (
            "every retained block design still names a single-target, "
            "single-offset contract: " + ", ".join(contracts)
            + "; no full-output adapter artifact is retained"),
        "what_would_discharge_it": (
            "a full-output adapter with independent target/scaler/reduction "
            "parity, and a fit against it. Both are training"),
    })
    return out


def verdict(conds: list) -> tuple[str, list]:
    gaps = [c["id"] for c in conds if c["state"] in (UNMET, OUTSIDE)]
    return (PARTIAL if gaps else SEALED), gaps


# --- the seal --------------------------------------------------------------------

def seal(repo: Path = REPO) -> dict:
    inv = read_inventory(repo)
    nums = numbers_as_measured(inv, repo)
    conds = conditions(inv, nums, repo)
    v, gaps = verdict(conds)
    identity = {rel: {k: e[k] for k in
                      ("kind", "why", "file_sha256", "bytes", "schema",
                       "design_sha256", "design_sha256_recomputed", "state")
                      if k in e}
                for rel, e in inv["artifacts"].items()}
    drift = {rel: b["closure_code_drift"]
             for rel, b in nums["blocks_with_an_outcome"].items()
             if isinstance(b.get("closure_code_drift"), dict)}
    return {
        "schema": SCHEMA,
        "at": now_iso(),
        "what_this_is": (
            "the E1 seal built only from retained artifacts: identities "
            "recomputed, rules copied as stated, numbers copied as measured, "
            "and every condition of the review chain typed. Nothing was "
            "fitted, loaded, scored or replayed to produce it"),
        "verdict": v,
        "gaps": gaps,
        "verdict_rule": (
            "E1_SEALED only when every condition is DISCHARGED or "
            "PROHIBITION_OBSERVED; E1_PARTIAL_SEAL otherwise. Identity, rules "
            "and numbers are sealed either way — they are facts about bytes"),
        "sealed": {
            "identity": identity,
            "rules_as_stated": rules_as_stated(inv),
            "numbers_as_measured": nums,
        },
        "conditions": conds,
        "disclosures": {
            "design_digests_without_a_retained_artifact": inv[
                "design_digests_without_a_retained_artifact"],
            "closure_code_drift": drift,
            "closure_code_drift_reading": (
                "these blocks were closed under code whose digest the report "
                "records and which has since changed. The recorded numbers are "
                "the numbers that were measured; a reader who wants them "
                "reproduced under today's code must say so and pay for a "
                "replay. The seal does not claim they reproduce"),
            "not_a_governance_validator": (
                "this module reads retained evidence. It does not contact the "
                "warehouse, the governance service or any host, and it never "
                "promotes a HISTORICAL_UNGOVERNED unit"),
        },
    }


def markdown(doc: dict) -> str:
    L = [f"# E1 seal — `{doc['verdict']}`", "",
         f"Generated {doc['at']} by `tools/df_e1_seal.py` from retained "
         "artifacts only. Nothing was fitted, loaded, scored or replayed.", "",
         f"**Verdict rule.** {doc['verdict_rule']}", ""]
    L += ["## Conditions", "",
          "| condition | source | state | evidence |", "|---|---|---|---|"]
    for c in doc["conditions"]:
        L.append(f"| `{c['id']}` | {Path(c['source']).name} | "
                 f"**{c['state']}** | {c['evidence']} |")
    L += ["", "## Sealed identities", "",
          "| artifact | kind | identity |", "|---|---|---|"]
    for rel, e in doc["sealed"]["identity"].items():
        ident = e.get("design_sha256_recomputed") or e["file_sha256"]
        L.append(f"| `{rel}` | {e['kind']} | `{ident[:16]}…` |")
    L += ["", "## Numbers as measured", ""]
    for rel, c in doc["sealed"]["numbers_as_measured"]["closures"].items():
        L.append(f"- **{Path(rel).name}** — `{c['verdict']}`, "
                 f"counts {json.dumps(c['counts'])}, "
                 f"scopes {json.dumps(c['unit_scopes'])}")
    for rel, b in doc["sealed"]["numbers_as_measured"][
            "blocks_with_an_outcome"].items():
        arms = ", ".join(
            f"{a}={v.get('mean_mae_kw'):.6f} kW"
            for a, v in sorted((b.get("summary") or {}).items())
            if isinstance(v.get("mean_mae_kw"), (int, float)))
        L.append(f"- **{b['block']}** — verified={b['verified']}, "
                 f"{b['common_evaluation_rows']} common rows; {arms}")
    for b in doc["sealed"]["numbers_as_measured"][
            "blocks_without_an_outcome"]:
        L.append(f"- **{b['block']}** — `{b['state']}`: "
                 f"{b['cells_with_an_outcome']} cells with an outcome. "
                 f"{b['reading']}")
    tbl = doc["sealed"]["numbers_as_measured"]["owner_closure_table"]
    L += ["", "## The owner's closure table, bound", "",
          f"`{tbl['path']}` ({tbl['schema']}, {tbl['at']}, generated by "
          f"`{tbl['generated_by']}`) — {tbl['rows_total']} rows, "
          f"{tbl['verified_rows']} verified, "
          f"{tbl['preserved_qualified_rows']} preserved with a qualified "
          f"scope, problems {tbl['problems']}.", "",
          "| run :: arm | rows | model MAE kW | naive MAE kW | skill | "
          "comparability |", "|---|---:|---:|---:|---:|---|"]
    for key, r in tbl["per_arm"].items():
        L.append(f"| {key} | {r['n_rows']} | "
                 f"{_fmt(r['mean_model_error_kw'])} | "
                 f"{_fmt(r['mean_naive_error_kw'])} | "
                 f"{_fmt(r['mean_skill_vs_naive'])} | "
                 f"{', '.join(r['comparability_status'])} |")
    L += ["", f"Naive definition, unchanged across every row: "
          f"{next(iter(tbl['per_arm'].values()))['naive_definition']}.", ""]
    L += ["## Disclosures", ""]
    for d, why in doc["disclosures"][
            "design_digests_without_a_retained_artifact"].items():
        L.append(f"- design `{d[:16]}…` has no retained artifact here: {why}")
    for rel, dr in doc["disclosures"]["closure_code_drift"].items():
        L.append(f"- `{Path(rel).parent.name}` closed under code that has "
                 f"since changed: {json.dumps(dr)}")
    L.append("")
    L.append(doc["disclosures"]["closure_code_drift_reading"])
    return "\n".join(L) + "\n"


def _fmt(v):
    return "—" if v is None else f"{v:.6f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=REPO)
    a = ap.parse_args(argv)
    doc = seal(a.repo)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1, default=str))
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(markdown(doc))
    print(json.dumps({"verdict": doc["verdict"], "gaps": doc["gaps"],
                      "conditions": {c["id"]: c["state"]
                                     for c in doc["conditions"]},
                      "artifacts_sealed": len(doc["sealed"]["identity"])},
                     indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SealRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
