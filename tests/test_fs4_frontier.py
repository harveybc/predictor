"""Sealed TRAIN-only frontier for the phase-4 weekly wrapper (plan section 4, FS4-10, FS4-12).

Written before the implementation. The frontier is a predeclared rule over phase-3 rankings and
(when available) extractibility evidence; it is written and digested before any VALIDATION read.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools import fs4_candidates as C
from tools import fs4_frontier as F

POP = "EURUSD"
IDENT = "phase1-synthetic:0000000000000000"
FEATURES = [f"f{i:02d}" for i in range(12)]


def _set(target, methods, members, horizon=1):
    members = sorted(members)
    return {"set_id": C.set_identity(POP, IDENT, target, members), "population_id": POP, "identity": IDENT,
            "target_id": target, "horizon_hours": horizon, "k": len(members), "declared_k": len(members),
            "members": members, "n_features": len(members), "methods": sorted(methods),
            "control_methods": sorted(set(methods) & set(C.CONTROL_METHODS)), "source_count": len(methods),
            "source_subset_sha256": [], "phase3_closure_sha256": "c" * 64, "phase3_unit_id": "u" * 64}


def _consolidated():
    sets = []
    for target in ("Y_s_1h", "Y_s_2h"):
        sets.append(_set(target, ["ALL_ADMISSIBLE"], FEATURES))
        for k in (2, 4, 6, 8):
            sets.append(_set(target, ["MRMR"], FEATURES[:k]))                 # best-ranked prefix
            sets.append(_set(target, ["JMI"], FEATURES[-k:]))                 # worst-ranked suffix
            sets.append(_set(target, ["UNIVARIATE_MI"], FEATURES[:k]) | {"set_id": "dup"})  # fixed below
            sets.append(_set(target, ["CAUSAL_SUPPORTED"], FEATURES[1:k + 1]))
            sets.append(_set(target, ["RANDOM_K"], FEATURES[2:k + 2]))
    # UNIVARIATE_MI shares members with MRMR -> in a real consolidation they are one set with two methods
    merged = {}
    for s in sets:
        key = (s["target_id"], tuple(s["members"]))
        if key in merged:
            merged[key]["methods"] = sorted(set(merged[key]["methods"]) | set(s["methods"]))
            merged[key]["control_methods"] = sorted(set(merged[key]["methods"]) & set(C.CONTROL_METHODS))
        else:
            merged[key] = dict(s, set_id=C.set_identity(POP, IDENT, s["target_id"], s["members"]))
    body = {"schema": C.SCHEMA, "population_id": POP, "identity": IDENT, "phase3_closure_sha256": "c" * 64,
            "sets": sorted(merged.values(), key=lambda s: s["set_id"]), "consolidated_count": len(merged)}
    body["consolidated_sha256"] = hashlib.sha256(json.dumps(body["sets"], sort_keys=True).encode()).hexdigest()
    return body


def _rankings():
    # UNIVARIATE_MI rank 1 = f00 ... rank 12 = f11, identical for both targets
    return {t: {f: i + 1 for i, f in enumerate(FEATURES)} for t in ("Y_s_1h", "Y_s_2h")}


def _extractibility(good=FEATURES[-4:]):
    feats = {}
    for f in FEATURES:
        feats[f] = {"status": "MEASURED", "folds": {"inner_2023": {"RAW": 1.0, "RANDOM_ENCODER": 1.0,
                                                                   "TRAINED_ENCODER": 0.2 if f in good else 0.95}}}
    feats["f05"] = {"status": "NOT_AVAILABLE_FOR_TRAIN", "folds": {}}
    body = {"schema": F.EXTRACTIBILITY_SCHEMA, "state": "EXTRACTIBILITY_COMPLETE",
            "populations": {POP: {"features": feats}}}
    body["closure_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def test_rule_text_is_predeclared_and_digested():
    assert "ALL_ADMISSIBLE" in F.FRONTIER_RULE and "RANDOM_K" in F.FRONTIER_RULE
    assert "VALIDATION" in F.FRONTIER_RULE and "TRAIN" in F.FRONTIER_RULE
    assert F.rule_sha256() == hashlib.sha256(F.FRONTIER_RULE.encode()).hexdigest()


def test_seal_keeps_all_admissible_and_controls_and_cuts_filter_families(tmp_path):
    seal = F.seal_frontier(_consolidated(), _rankings(), None, tmp_path / "FRONTIER_SEAL.json")
    assert seal["schema"] == F.FRONTIER_SCHEMA and seal["rule_sha256"] == F.rule_sha256()
    assert seal["validation_read"] == {"count": 0, "statement": F.NO_VALIDATION_STATEMENT}
    by_id = {d["set_id"]: d for d in seal["dispositions"]}
    sets = {s["set_id"]: s for s in _consolidated()["sets"]}
    assert len(by_id) == len(sets)                                              # denominator preserved
    for sid, s in sets.items():
        if s["control_methods"]:
            assert by_id[sid]["disposition"] == "IN_FRONTIER", s["methods"]
    for target in ("Y_s_1h", "Y_s_2h"):
        for fam in ("MRMR", "JMI"):
            kept = [d for d in seal["dispositions"] if d["target_id"] == target and fam in d["methods"]
                    and d["disposition"] == "IN_FRONTIER" and not d["control_methods"]]
            assert len(kept) == F.FRONTIER_WIDTH, (target, fam, kept)
    deferred = [d for d in seal["dispositions"] if d["disposition"] == "DEFERRED_BY_FRONTIER_RULE"]
    assert deferred and all(d["reason"] for d in deferred)
    assert seal["denominator"] == {"sets": len(sets), "in_frontier": len(sets) - len(deferred), "deferred": len(deferred)}
    assert seal["inputs"]["extractibility"] == {"closure_sha256": None, "reason": "NOT_AVAILABLE_AT_SEAL"}


def test_ranking_priority_prefers_better_ranked_and_smaller_k_on_ties(tmp_path):
    seal = F.seal_frontier(_consolidated(), _rankings(), None, tmp_path / "FRONTIER_SEAL.json")
    mrmr = [d for d in seal["dispositions"] if d["target_id"] == "Y_s_1h" and "MRMR" in d["methods"] and d["disposition"] == "IN_FRONTIER"]
    assert sorted(d["k"] for d in mrmr) == [2, 4]            # prefix sets: smallest K has the best mean rank
    jmi = [d for d in seal["dispositions"] if d["target_id"] == "Y_s_1h" and "JMI" in d["methods"]]
    assert all(d["priority"]["relevance"] < 0.5 for d in jmi)  # suffix sets are the worst-ranked


def test_extractibility_evidence_enters_only_when_available_and_never_selects_alone(tmp_path):
    ext = _extractibility()
    seal = F.seal_frontier(_consolidated(), _rankings(), ext, tmp_path / "FRONTIER_SEAL.json")
    assert seal["inputs"]["extractibility"]["closure_sha256"] == ext["closure_sha256"]
    jmi8 = next(d for d in seal["dispositions"] if d["target_id"] == "Y_s_1h" and "JMI" in d["methods"] and d["k"] == 8)
    assert 0.0 < jmi8["priority"]["extractibility"] <= 1.0
    assert jmi8["priority"]["features_without_extractibility"] == 1          # f05 counted as zero, never dropped
    # FS4-10: a frontier disposition is a prioritisation; no set is SELECTED by the seal
    assert all(d["disposition"] in ("IN_FRONTIER", "DEFERRED_BY_FRONTIER_RULE") for d in seal["dispositions"])
    assert "SELECTED" not in json.dumps(seal["dispositions"])
    assert seal["final_selection"] is False


def test_seal_is_written_before_validation_and_refuses_to_change(tmp_path):
    path = tmp_path / "FRONTIER_SEAL.json"
    first = F.seal_frontier(_consolidated(), _rankings(), None, path)
    again = F.seal_frontier(_consolidated(), _rankings(), None, path)
    assert again["seal_sha256"] == first["seal_sha256"]
    with pytest.raises(F.Refusal, match="FRONTIER_CHANGED"):
        F.seal_frontier(_consolidated(), _rankings(), _extractibility(), path)
    loaded = F.load_seal(path)
    assert loaded["seal_sha256"] == first["seal_sha256"]
    with pytest.raises(F.Refusal, match="MALFORMED_EXTRACTIBILITY"):
        F.seal_frontier(_consolidated(), _rankings(), {"state": "RUNNING"}, tmp_path / "other.json")


def test_rankings_loader_uses_univariate_mi_rows_only(tmp_path):
    import gzip

    unit = "u" * 64
    rows = [{"feature_id": f, "method": m, "rank": (i + 1) if m == "UNIVARIATE_MI" else 99, "target_id": "Y_s_1h"}
            for i, f in enumerate(FEATURES) for m in ("UNIVARIATE_MI", "MRMR")]
    term = {"schema": "fs_phase23.filter_terminal.v1", "unit_id": unit, "target_id": "Y_s_1h", "population_id": POP,
            "rows": {"feature_filter_rankings": rows}}
    d = tmp_path / "terminals"
    d.mkdir()
    with gzip.open(d / f"{unit}.json.gz", "wt") as fh:
        json.dump(term, fh)
    ranks = F.load_rankings_from_terminals(d, [{"target_id": "Y_s_1h", "unit_id": unit}])
    assert ranks == {"Y_s_1h": {f: i + 1 for i, f in enumerate(FEATURES)}}
    with pytest.raises(F.Refusal, match="MISSING_TERMINAL"):
        F.load_rankings_from_terminals(d, [{"target_id": "Y_s_2h", "unit_id": "v" * 64}])
