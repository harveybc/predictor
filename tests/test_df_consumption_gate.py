"""C144: I5 may consider a subject only with external review records for
every stage D0-D4; today, with no records, everything is refused."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_consumption_gate", ROOT / "tools/df_consumption_gate.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def binding(kind="ewma", **over):
    OPS, D = G._load("df_operators"), G._load("df_d2_design")
    b = {"operator_kind": kind, "spec_sha256": "a" * 64, "fit_mode": OPS.KIND_FIT_MODES[kind][0]
         if kind in OPS.KIND_FIT_MODES else "EXPANDING_PREFIX", "operator_code_sha256": OPS.code_sha256(),
         "lab_code_sha256": D.lab_code_sha256(), "root_mode": "FRESH_CONFIRMATION",
         "fresh_root_tape_sha256": "b" * 64, "design_sha256": "c" * 64, "decision_run_id": "d2v2_fixture"}
    b.update(over)
    return b


def record(stage, states, reviewer="General Musashi", regimes=None, kind="OPERATOR", d2_binding=None):
    r = {"schema": G.RECORD_SCHEMA, "stage": stage, "reviewer": reviewer, "reviewed_at_date": "2026-09-13",
         "subject_kind": kind, "states": states, "regimes": regimes or {}, "grants_public_eligibility": False,
         "record_sha256": ""}
    if stage == "D2":
        r["schema"] = G.RECORD_SCHEMA_D2
        r["d2_binding"] = d2_binding or binding(next(iter(states)).split(":")[0])
    r["record_sha256"] = G.record_digest(r)
    return r


def full(subject="ewma", d2="LAB_CALIBRATED", regimes=None):
    return [record("D0", {subject: "CONTRACT_REVIEWED"}), record("D1", {subject: "PROFILE_REVIEWED"}),
            record("D2", {subject: d2}, regimes=regimes), record("D3", {subject: "D3_REVIEWED_ACCEPTED"}),
            record("D4", {subject: "D4_REVIEWED_ACCEPTED"})]


def test_with_no_records_everything_is_refused(tmp_path):
    good, bad = G.load_records(tmp_path / "absent")
    d = G.decide("ewma", "OPERATOR", good)
    assert not d["may_be_considered_by_i5_under_review"] and d["missing_stages"] == list(G.STAGES)
    with pytest.raises(PermissionError, match="lacks reviewed stages"):
        G.require("ewma", "OPERATOR", good)


def test_complete_reviews_allow_consideration_never_public_eligibility():
    d = G.decide("ewma", "OPERATOR", full())
    assert d["may_be_considered_by_i5_under_review"] and d["public_eligibility"] == "NEVER_GRANTED_BY_THIS_GATE"


def test_a_missing_or_rejected_stage_refuses():
    recs = full()
    assert G.decide("ewma", "OPERATOR", recs[:2] + recs[3:])["missing_stages"] == ["D2"]
    d = G.decide("ewma", "OPERATOR", full(d2="LAB_REJECTED"))
    assert d["missing_stages"] == ["D2"] and d["states_not_consumable"] == ["D2:LAB_REJECTED"]
    d = G.decide("ewma", "OPERATOR", full(d2="PUBLICLY_ELIGIBLE"))
    assert "D2" in d["missing_stages"]


def test_regime_limited_is_limited_to_its_regimes():
    d = G.decide("ewma", "OPERATOR", full(d2="REGIME_LIMITED", regimes={"ewma": [{"perturbation": "white"}]}))
    assert d["may_be_considered_by_i5_under_review"] and d["limits"]["regimes"] == ['{"perturbation": "white"}']
    assert "D2" in G.decide("ewma", "OPERATOR", full(d2="REGIME_LIMITED"))["missing_stages"]


def test_a_historical_c137_decision_never_passes_d2():
    """C181 test 7: a C137-style D2 record (v1 schema, retired name, old code, no fit mode, no fresh root)."""
    # the retired name comes from the naming record, the only place it is written
    OLD = next(d["previous_name"] for d in G._load("df_operators").NAMING_DECISIONS
               if d["subject"] == "trailing_haar_threshold")
    subject = f'{OLD}:{{"levels": 2, "threshold_k": 3.0}}'
    old = {"schema": G.RECORD_SCHEMA, "stage": "D2", "reviewer": "external reviewer", "reviewed_at_date": "2026-09-13",
           "subject_kind": "OPERATOR", "states": {subject: "LAB_CALIBRATED"}, "regimes": {},
           "grants_public_eligibility": False, "record_sha256": ""}
    old["record_sha256"] = G.record_digest(old)
    recs = [record("D0", {subject: "CONTRACT_REVIEWED"}), record("D1", {subject: "PROFILE_REVIEWED"}), old,
            record("D3", {subject: "D3_REVIEWED_ACCEPTED"}), record("D4", {subject: "D4_REVIEWED_ACCEPTED"})]
    d = G.decide(subject, "OPERATOR", recs)
    assert d["missing_stages"] == ["D2"] and any("NOT_CONSUMABLE" in s for s in d["states_not_consumable"])
    # the same record upgraded to the v2 schema but bound to the retired name and old code still refuses
    stale = record("D2", {subject: "LAB_CALIBRATED"}, d2_binding=binding(
        "trailing_haar_threshold", operator_kind=OLD, operator_code_sha256="ca6d" + "0" * 60))
    d = G.decide(subject, "OPERATOR", recs[:2] + [stale] + recs[3:])
    assert d["missing_stages"] == ["D2"]
    why = " ".join(d["states_not_consumable"])
    assert "not a current kind" in why and "not the current one" in why


def test_d2_binding_must_be_current_and_fresh():
    ok = full()
    assert G.decide("ewma", "OPERATOR", ok)["may_be_considered_by_i5_under_review"]
    for over, text in (({"operator_code_sha256": "0" * 64}, "operator code digest"),
                       ({"lab_code_sha256": "0" * 64}, "lab code digest"),
                       ({"root_mode": "HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY"}, "fresh confirmation root"),
                       ({"fit_mode": "OFFLINE_ANALYSIS_ONLY_NON_CAUSAL"}, "per-timestamp mode"),
                       ({"fresh_root_tape_sha256": None}, "sha256"),
                       ({"operator_kind": "trailing_mean"}, "not the bound operator")):
        recs = list(ok)
        recs[2] = record("D2", {"ewma": "LAB_CALIBRATED"}, d2_binding=binding("ewma", **over))
        d = G.decide("ewma", "OPERATOR", recs)
        assert d["missing_stages"] == ["D2"], over
        assert text in " ".join(d["states_not_consumable"]), (over, d["states_not_consumable"])


def test_self_review_tampering_and_grants_are_refused(tmp_path):
    recs = [record("D0", {"ewma": "CONTRACT_REVIEWED"}, reviewer="Satoshi"),
            dict(record("D1", {"ewma": "PROFILE_REVIEWED"}), reviewer="General Musashi II"),
            record("D2", {"ewma": "LAB_CALIBRATED"})]
    recs.append(dict(record("D3", {"ewma": "D3_REVIEWED_ACCEPTED"}), grants_public_eligibility=True))
    for i, r in enumerate(recs):
        (tmp_path / f"r{i}.json").write_text(json.dumps(r))
    good, bad = G.load_records(tmp_path)
    assert [r["stage"] for r in good] == ["D2"] and len(bad) == 3
    assert any("not a review" in p for b in bad for p in b["problems"])
    assert any("does not re-derive" in p for b in bad for p in b["problems"])
