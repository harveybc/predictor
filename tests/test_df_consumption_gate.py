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


def record(stage, states, reviewer="General Musashi", regimes=None, kind="OPERATOR"):
    r = {"schema": G.RECORD_SCHEMA, "stage": stage, "reviewer": reviewer, "reviewed_at_date": "2026-09-13",
         "subject_kind": kind, "states": states, "regimes": regimes or {}, "grants_public_eligibility": False,
         "record_sha256": ""}
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
