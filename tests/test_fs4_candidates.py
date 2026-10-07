"""FS4-01 / FS4-02: candidate consolidation for the phase-4 weekly wrapper.

Written before the implementation. Synthetic candidate files only; no phase-3 evidence is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import fs4_candidates as C

IDENTITY = "phase1-synthetic:0000000000000000"


def _closure(path: Path, candidates: list, population="EURUSD", alias=None):
    path.mkdir(parents=True, exist_ok=True)
    body = {"state": "PHASE_3_FILTER_COMPLETE", "population_id": population, "identity": IDENTITY,
            "closure_sha256": "", "units": [{"target_id": t, "unit_id": hashlib.sha256(t.encode()).hexdigest()}
                                            for t in sorted({c["target_id"] for c in candidates})],
            "methods": list(C.FILTER_METHODS + C.CONTROL_METHODS), "k_grid": [2, 3], "final_selection": False}
    body["closure_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    (path / "PHASE_3_FILTER_COMPLETE.json").write_text(json.dumps(body))
    src = {"schema": "fs_phase23.candidates_for_validation.v1", "population_id": population, "identity": IDENTITY,
           "final_selection": False, "uses_test_split": False, "phase3_closure_sha256": body["closure_sha256"],
           "admissible_features": {population: 6}, "k_grid": [2, 3], "candidates": candidates}
    (path / "CANDIDATES_FOR_VALIDATION.json").write_text(json.dumps(src))
    if alias is not None:
        (path / "ALIAS_GROUPS.json").write_text(json.dumps(alias))
    return path / "CANDIDATES_FOR_VALIDATION.json"


def _cand(target, method, k, members, horizon=1):
    body = {"target_id": target, "method": method, "k": k, "members": list(members), "horizon_hours": horizon,
            "label": "FILTER_CANDIDATE", "population_id": "EURUSD", "identity": IDENTITY, "is_final_selection": False}
    body["subset_sha256"] = hashlib.sha256(json.dumps([target, method, k, sorted(members)]).encode()).hexdigest()
    return body


def _fixture(tmp_path, permute=False):
    a, b = (["f_b", "f_a"] if permute else ["f_a", "f_b"])
    cands = [
        _cand("Y_s_1h", "MRMR", 2, [a, b]),
        _cand("Y_s_1h", "JMI", 2, [b, a] if not permute else [a, b]),        # identical member set -> merged
        _cand("Y_s_1h", "UNIVARIATE_MI", 2, ["f_a", "f_c"]),
        _cand("Y_s_1h", "CAUSAL_SUPPORTED", 2, ["f_c", "f_d"]),
        _cand("Y_s_1h", "RANDOM_K", 2, ["f_d", "f_e"]),
        _cand("Y_s_1h", "ALL_ADMISSIBLE", 6, ["f_a", "f_b", "f_c", "f_d", "f_e", "f_alias"]),
        _cand("Y_s_1h", "SPEARMAN_CLUSTER", 2, ["f_alias", "f_c"]),           # alias -> representative f_a
        _cand("Y_l_24h", "MRMR", 2, ["f_a", "f_b"], horizon=24),              # same members, other target: NOT merged
    ]
    alias = [{"alias_group_id": "g1", "disposition": "ALIAS_GROUP", "members": ["f_a", "f_alias"],
              "representative": "f_a", "population_id": "EURUSD", "run_id": IDENTITY}]
    return _closure(tmp_path, cands, alias=alias)


def test_identical_member_sets_merge_once_per_target_and_denominator_is_kept(tmp_path):
    out = C.consolidate(_fixture(tmp_path))
    assert out["schema"] == C.SCHEMA
    assert out["source_candidate_count"] == 8
    sets = out["sets"]
    one_h = [s for s in sets if s["target_id"] == "Y_s_1h"]
    merged = [s for s in one_h if s["members"] == ["f_a", "f_b"]]
    assert len(merged) == 1 and sorted(merged[0]["methods"]) == ["JMI", "MRMR"]
    assert merged[0]["source_count"] == 2 and len(merged[0]["source_subset_sha256"]) == 2
    # the same member set bound to another target is a different candidate (bound to target/horizon)
    assert [s for s in sets if s["target_id"] == "Y_l_24h" and s["members"] == ["f_a", "f_b"]][0]["horizon_hours"] == 24
    # MRMR+JMI merge, and SPEARMAN [f_alias, f_c] becomes [f_a, f_c] after alias replacement and merges with UNIVARIATE_MI
    assert out["merged_duplicates"] == 2
    assert out["consolidated_count"] == len(sets) == 6
    assert out["per_target"]["Y_s_1h"] == {"source": 7, "consolidated": 5}


def test_column_permutation_leaves_identities_unchanged(tmp_path):
    a = C.consolidate(_fixture(tmp_path / "a"))
    b = C.consolidate(_fixture(tmp_path / "b", permute=True))
    assert {s["set_id"] for s in a["sets"]} == {s["set_id"] for s in b["sets"]}
    assert a["consolidated_sha256"] == b["consolidated_sha256"]
    for s in a["sets"]:
        assert s["members"] == sorted(s["members"])


def test_phase2_exact_aliases_are_replaced_by_the_declared_representative(tmp_path):
    out = C.consolidate(_fixture(tmp_path))
    members = {m for s in out["sets"] for m in s["members"]}
    assert "f_alias" not in members and "f_a" in members
    assert out["alias_replacements"] == 2       # ALL_ADMISSIBLE and SPEARMAN_CLUSTER each carried the alias
    all_adm = [s for s in out["sets"] if "ALL_ADMISSIBLE" in s["methods"]][0]
    assert all_adm["members"] == ["f_a", "f_b", "f_c", "f_d", "f_e"] and all_adm["n_features"] == 5
    assert all_adm["k"] == 5 and all_adm["declared_k"] == 6


def test_controls_are_preserved_and_every_set_is_bound(tmp_path):
    out = C.consolidate(_fixture(tmp_path))
    methods = {m for s in out["sets"] for m in s["methods"]}
    assert set(C.CONTROL_METHODS) <= methods
    closure = json.loads((tmp_path / "PHASE_3_FILTER_COMPLETE.json").read_text())
    for s in out["sets"]:
        assert s["population_id"] == "EURUSD" and s["identity"] == IDENTITY
        assert s["target_id"] and isinstance(s["horizon_hours"], int) and s["k"] == len(s["members"])
        assert s["phase3_closure_sha256"] == closure["closure_sha256"]
        assert s["phase3_unit_id"] == hashlib.sha256(s["target_id"].encode()).hexdigest()
        assert s["set_id"] == C.set_identity("EURUSD", IDENTITY, s["target_id"], s["members"])
        assert s["control_methods"] == sorted(set(s["methods"]) & set(C.CONTROL_METHODS))


def test_closure_mismatch_or_final_selection_refuses(tmp_path):
    src = _fixture(tmp_path)
    closure_path = tmp_path / "PHASE_3_FILTER_COMPLETE.json"
    body = json.loads(closure_path.read_text())
    body["closure_sha256"] = "0" * 64
    closure_path.write_text(json.dumps(body))
    with pytest.raises(C.Refusal, match="PHASE3_CLOSURE_MISMATCH"):
        C.consolidate(src)
    _fixture(tmp_path)
    doc = json.loads(src.read_text())
    doc["final_selection"] = True
    src.write_text(json.dumps(doc))
    with pytest.raises(C.Refusal, match="CANDIDATE_IDENTITY_OR_STATE_INVALID"):
        C.consolidate(src)


def test_write_is_atomic_and_idempotent(tmp_path):
    src = _fixture(tmp_path)
    out_path = tmp_path / "out" / "CONSOLIDATED_CANDIDATES.json"
    first = C.consolidate_to_file(src, out_path)
    second = C.consolidate_to_file(src, out_path)
    assert first["consolidated_sha256"] == second["consolidated_sha256"]
    assert json.loads(out_path.read_text())["consolidated_sha256"] == first["consolidated_sha256"]
