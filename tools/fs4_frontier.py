#!/usr/bin/env python3
"""Sealed TRAIN-only frontier for the phase-4 weekly comparison (plan section 4; FS4-10, FS4-12).

The 980 consolidated candidates are not 980 weekly trainings. This module applies ONE predeclared
rule (``FRONTIER_RULE``, digested) to phase-3 TRAIN rankings and, when the extractibility closure
exists, to its paired reconstruction evidence, and writes ``FRONTIER_SEAL.json`` BEFORE any
VALIDATION read. Every consolidated set receives a disposition (IN_FRONTIER or
DEFERRED_BY_FRONTIER_RULE with a reason); the denominator is preserved. The seal never selects a
feature set: selection is the weekly predictive comparison (FS4-10).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools.fs4_candidates import CONTROL_METHODS, FILTER_METHODS, Refusal, canonical, digest, write_atomic

FRONTIER_SCHEMA = "fs4.frontier_seal.v1"
EXTRACTIBILITY_SCHEMA = "fs4.extractibility_complete.v1"
FRONTIER_WIDTH = 2
EXTRACTIBILITY_WEIGHT = 0.5
NO_VALIDATION_STATEMENT = ("sealed from TRAIN-only phase-3 rankings and the TRAIN-only extractibility closure; "
                           "no VALIDATION or TEST byte was read before this seal")
FRONTIER_RULE = f"""FS4 frontier rule v1 (predeclared, TRAIN only; sealed before any VALIDATION read).
Universe: the consolidated candidate sets of one population (identical member sets per target already merged).
1. Every set carrying a control method (ALL_ADMISSIBLE, UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K) is IN_FRONTIER at every K.
2. For every target and every filter family (SPEARMAN_CLUSTER, MRMR, JMI, MRMR_CAUSAL, JMI_CAUSAL) the family's sets over the
   K grid are ordered by priority = relevance + {EXTRACTIBILITY_WEIGHT} * extractibility, descending; ties by smaller K, then set_id.
   The first {FRONTIER_WIDTH} are IN_FRONTIER; the rest are DEFERRED_BY_FRONTIER_RULE (kept in the denominator).
   relevance = mean over members of (1 - (rank - 1) / (n_admissible - 1)) using the phase-3 UNIVARIATE_MI TRAIN rank of the
   member for that target (rank 1 = most relevant); a member without a rank contributes 0.
   extractibility = mean over the MEASURED members of clip((RAW_mae - TRAINED_ENCODER_mae) / RAW_mae, 0, 1), each member averaged over its TRAIN
   folds that carry numeric RAW and TRAINED_ENCODER values of the extractibility closure. A member that is NOT_AVAILABLE_FOR_TRAIN (or
   absent from the closure, or without a usable fold) is a typed disposition of its own: it is listed and counted in the seal and in the
   set's disposition, it is excluded from the mean (it is NOT zero skill), and it is never dropped from the denominator. A set with no
   measured member has no extractibility term (priority = relevance, basis recorded). When no extractibility closure exists at sealing
   time, no set has an extractibility term and the seal records it.
3. A set is IN_FRONTIER if any of its methods qualifies under 1 or 2.
4. The seal is a prioritisation of the weekly comparison budget, not a selection: no reconstruction score and no ranking
   chooses a feature set; the winner comes only from the sealed weekly predictive rule over ALL VALIDATION weeks."""


def rule_sha256() -> str:
    return hashlib.sha256(FRONTIER_RULE.encode()).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- inputs
def load_rankings_from_terminals(terminals_dir, units) -> dict[str, dict[str, int]]:
    """target -> feature -> UNIVARIATE_MI TRAIN rank, from the phase-3 terminals named by the closure units."""
    terminals_dir = Path(terminals_dir)
    out: dict[str, dict[str, int]] = {}
    for u in units:
        path = terminals_dir / f"{u['unit_id']}.json.gz"
        if not path.is_file():
            raise Refusal(f"MISSING_TERMINAL: {u['target_id']} {u['unit_id']}")
        with gzip.open(path, "rt") as fh:
            term = json.load(fh)
        if term.get("unit_id") != u["unit_id"] or term.get("target_id") != u["target_id"]:
            raise Refusal(f"TERMINAL_IDENTITY_MISMATCH: {u['unit_id']}")
        rows = term.get("rows", {}).get("feature_filter_rankings", [])
        ranks = {r["feature_id"]: int(r["rank"]) for r in rows if r.get("method") == "UNIVARIATE_MI"}
        if not ranks:
            raise Refusal(f"NO_UNIVARIATE_MI_RANKS: {u['target_id']}")
        out[u["target_id"]] = ranks
    return out


def load_extractibility(path) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"MISSING_EXTRACTIBILITY: {path}")
    return json.loads(path.read_text())


def _validate_extractibility(ext: dict | None) -> None:
    if ext is None:
        return
    if not isinstance(ext, dict) or ext.get("state") != "EXTRACTIBILITY_COMPLETE" or ext.get("schema") != EXTRACTIBILITY_SCHEMA:
        raise Refusal("MALFORMED_EXTRACTIBILITY: state EXTRACTIBILITY_COMPLETE and schema "
                      f"{EXTRACTIBILITY_SCHEMA} required")
    if not isinstance(ext.get("closure_sha256"), str) or len(ext["closure_sha256"]) != 64:
        raise Refusal("MALFORMED_EXTRACTIBILITY: closure_sha256")
    if not isinstance(ext.get("populations"), dict):
        raise Refusal("MALFORMED_EXTRACTIBILITY: populations")
    den = ext.get("denominator")
    if not isinstance(den, dict) or den.get("sum_equals_admitted") is not True:
        raise Refusal("MALFORMED_EXTRACTIBILITY: denominator block missing or admitted != complete + not_available_for_train + typed_refused")


NAFT = "NOT_AVAILABLE_FOR_TRAIN"


def feature_extractibility(ext: dict | None, population: str) -> dict[str, dict]:
    """feature -> {status, value, folds_used, folds_naft}. status: MEASURED (value set), MEASURED_NO_USABLE_FOLD (value None),
    NOT_AVAILABLE_FOR_TRAIN (typed, own disposition), NOT_IN_CLOSURE (typed). Value = mean fold improvement of TRAINED over RAW in [0, 1]."""
    if ext is None:
        return {}
    feats = ext.get("populations", {}).get(population, {}).get("features", {})
    out: dict[str, dict] = {}
    for fid, rec in feats.items():
        if rec.get("status") == NAFT:
            out[fid] = {"status": NAFT, "value": None, "folds_used": 0, "folds_naft": None}
            continue
        vals, naft = [], 0
        for fold in rec.get("folds", {}).values():
            if fold.get("status") == NAFT:
                naft += 1
                continue
            raw, trained = fold.get("RAW"), fold.get("TRAINED_ENCODER")
            if isinstance(raw, (int, float)) and isinstance(trained, (int, float)) and raw > 0:
                vals.append(min(1.0, max(0.0, (raw - trained) / raw)))
        out[fid] = {"status": "MEASURED" if vals else "MEASURED_NO_USABLE_FOLD", "value": (sum(vals) / len(vals)) if vals else None,
                    "folds_used": len(vals), "folds_naft": naft}
    return out


def extractibility_status_counts(ext: dict | None, population: str, admissible) -> dict:
    by = feature_extractibility(ext, population)
    counts: dict[str, int] = {}
    naft = []
    for f in sorted(admissible):
        st = by.get(f, {}).get("status", "NOT_IN_CLOSURE")
        counts[st] = counts.get(st, 0) + 1
        if st == NAFT:
            naft.append(f)
    return {"features": len(set(admissible)), "by_status": dict(sorted(counts.items())), "not_available_for_train": naft}


# ----------------------------------------------------------------------------- rule
def priority_terms(members, ranks: dict[str, int], n_admissible: int, ext_by_feature: dict[str, dict], has_extractibility: bool = False) -> dict:
    rel = []
    missing_rank = 0
    for m in members:
        r = ranks.get(m)
        if r is None:
            missing_rank += 1
            rel.append(0.0)
        else:
            rel.append(1.0 - (r - 1) / (n_admissible - 1) if n_admissible > 1 else 1.0)
    measured = [ext_by_feature[m]["value"] for m in members if m in ext_by_feature and ext_by_feature[m]["value"] is not None]
    naft = sorted(m for m in members if ext_by_feature.get(m, {}).get("status") == NAFT)
    not_in = sorted(m for m in members if has_extractibility and m not in ext_by_feature)
    no_value = sorted(m for m in members if m in ext_by_feature and ext_by_feature[m]["value"] is None and m not in naft)
    relevance = sum(rel) / len(rel)
    extractibility = (sum(measured) / len(measured)) if measured else None
    basis = ("NO_EXTRACTIBILITY_CLOSURE" if not has_extractibility else
             "MEAN_OF_MEASURED_MEMBERS" if measured else "NO_MEASURED_MEMBER")
    return {"relevance": relevance, "extractibility": extractibility, "extractibility_basis": basis,
            "priority": relevance + (EXTRACTIBILITY_WEIGHT * extractibility if extractibility is not None else 0.0),
            "features_without_rank": missing_rank, "members_measured": len(measured), "members_not_available_for_train": naft,
            "members_not_in_closure": not_in, "members_without_usable_fold": no_value}


def apply_rule(consolidated: dict, rankings: dict[str, dict[str, int]], ext: dict | None) -> list[dict]:
    population = consolidated["population_id"]
    ext_by_feature = feature_extractibility(ext, population)
    sets = consolidated["sets"]
    dispositions: dict[str, dict] = {}
    for s in sets:
        ranks = rankings.get(s["target_id"], {})
        n_adm = max(len(ranks), max(ranks.values()) if ranks else 0)
        terms = priority_terms(s["members"], ranks, n_adm, ext_by_feature, ext is not None)
        dispositions[s["set_id"]] = {
            "set_id": s["set_id"], "target_id": s["target_id"], "horizon_hours": s["horizon_hours"], "k": s["k"],
            "n_features": s["n_features"], "methods": list(s["methods"]), "control_methods": list(s["control_methods"]),
            "priority": terms, "disposition": None, "reason": None, "qualifying_methods": [],
        }
        if s["control_methods"]:
            d = dispositions[s["set_id"]]
            d["disposition"] = "IN_FRONTIER"
            d["qualifying_methods"] = list(s["control_methods"])
            d["reason"] = "control method kept at every K (rule 1)"
    targets = sorted({s["target_id"] for s in sets})
    for target in targets:
        for fam in FILTER_METHODS:
            fam_sets = [s for s in sets if s["target_id"] == target and fam in s["methods"]]
            ordered = sorted(fam_sets, key=lambda s: (-dispositions[s["set_id"]]["priority"]["priority"], s["k"], s["set_id"]))
            for i, s in enumerate(ordered):
                d = dispositions[s["set_id"]]
                if i < FRONTIER_WIDTH:
                    d["qualifying_methods"] = sorted(set(d["qualifying_methods"]) | {fam})
                    if d["disposition"] != "IN_FRONTIER":
                        d["disposition"] = "IN_FRONTIER"
                        d["reason"] = f"within the first {FRONTIER_WIDTH} of family {fam} by TRAIN priority (rule 2)"
                elif d["disposition"] is None:
                    d["disposition"] = "DEFERRED_BY_FRONTIER_RULE"
                    d["reason"] = (f"rank {i + 1} of {len(ordered)} in family {fam} for {target} by TRAIN priority "
                                   f"{d['priority']['priority']:.6f}; beyond width {FRONTIER_WIDTH} (rule 2)")
    for d in dispositions.values():
        if d["disposition"] is None:
            d["disposition"] = "DEFERRED_BY_FRONTIER_RULE"
            d["reason"] = "no control method and no filter family qualified under rules 1-2"
    return [dispositions[s["set_id"]] for s in sets]


def build_seal(consolidated: dict, rankings: dict[str, dict[str, int]], ext: dict | None) -> dict:
    _validate_extractibility(ext)
    if consolidated.get("sets") is None or not consolidated.get("consolidated_sha256"):
        raise Refusal("CONSOLIDATED_CANDIDATES_INVALID")
    dispositions = apply_rule(consolidated, rankings, ext)
    in_frontier = sorted(d["set_id"] for d in dispositions if d["disposition"] == "IN_FRONTIER")
    body = {
        "schema": FRONTIER_SCHEMA, "population_id": consolidated["population_id"], "identity": consolidated["identity"],
        "rule_text": FRONTIER_RULE, "rule_sha256": rule_sha256(), "frontier_width": FRONTIER_WIDTH,
        "extractibility_weight": EXTRACTIBILITY_WEIGHT,
        "inputs": {
            "consolidated_sha256": consolidated["consolidated_sha256"],
            "phase3_closure_sha256": consolidated.get("phase3_closure_sha256"),
            "rankings_sha256": digest({t: dict(sorted(r.items())) for t, r in sorted(rankings.items())}),
            "extractibility": ({"closure_sha256": ext["closure_sha256"], "reason": None, "denominator": ext["denominator"],
                                "admissible_features_by_status": extractibility_status_counts(
                                    ext, consolidated["population_id"], {m for s in consolidated["sets"] for m in s["members"]})}
                               if ext is not None else {"closure_sha256": None, "reason": "NOT_AVAILABLE_AT_SEAL"}),
        },
        "validation_read": {"count": 0, "statement": NO_VALIDATION_STATEMENT},
        "dispositions": dispositions, "frontier_set_ids": in_frontier,
        "denominator": {"sets": len(dispositions), "in_frontier": len(in_frontier), "deferred": len(dispositions) - len(in_frontier)},
        "final_selection": False,
    }
    body["seal_sha256"] = digest(body)
    return body


def seal_frontier(consolidated: dict, rankings: dict[str, dict[str, int]], ext: dict | None, out_path) -> dict:
    """Write the seal once. A different seal at the same path is refused: FRONTIER_CHANGED."""
    body = build_seal(consolidated, rankings, ext)
    out_path = Path(out_path)
    if out_path.is_file():
        prior = json.loads(out_path.read_text())
        if prior.get("seal_sha256") != body["seal_sha256"]:
            raise Refusal("FRONTIER_CHANGED: a sealed frontier exists at this path with a different digest")
        return prior
    body["sealed_utc"] = _now()
    write_atomic(out_path, body)
    return body


def load_seal(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"MISSING_FRONTIER_SEAL: {path}")
    seal = json.loads(path.read_text())
    check = {k: v for k, v in seal.items() if k not in ("seal_sha256", "sealed_utc")}
    if seal.get("schema") != FRONTIER_SCHEMA or digest(check) != seal.get("seal_sha256"):
        raise Refusal("FRONTIER_SEAL_CORRUPT")
    if seal.get("rule_sha256") != rule_sha256():
        raise Refusal("FRONTIER_RULE_MISMATCH: the seal was produced by a different rule text")
    return seal


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--consolidated", required=True, help="CONSOLIDATED_CANDIDATES.json (tools/fs4_candidates.py)")
    ap.add_argument("--phase3-closure", required=True, help="PHASE_3_FILTER_COMPLETE.json (its units name the terminals)")
    ap.add_argument("--phase3-terminals", required=True, help="directory with <unit_id>.json.gz phase-3 terminals")
    ap.add_argument("--extractibility", help="EXTRACTIBILITY_COMPLETE.json; omitted = extractibility term 0, recorded")
    ap.add_argument("--out", required=True, help="FRONTIER_SEAL.json")
    a = ap.parse_args(argv)
    try:
        consolidated = json.loads(Path(a.consolidated).read_text())
        closure = json.loads(Path(a.phase3_closure).read_text())
        if closure.get("closure_sha256") != consolidated.get("phase3_closure_sha256"):
            raise Refusal("PHASE3_CLOSURE_MISMATCH")
        rankings = load_rankings_from_terminals(a.phase3_terminals, closure["units"])
        ext = load_extractibility(a.extractibility) if a.extractibility else None
        seal = seal_frontier(consolidated, rankings, ext, a.out)
    except (Refusal, OSError, json.JSONDecodeError, KeyError) as exc:
        print(canonical({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(canonical({"population_id": seal["population_id"], "seal_sha256": seal["seal_sha256"], "rule_sha256": seal["rule_sha256"],
                     "denominator": seal["denominator"], "extractibility": seal["inputs"]["extractibility"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
