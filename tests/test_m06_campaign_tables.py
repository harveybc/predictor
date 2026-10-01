"""Tests for tools/m06_campaign_tables.py: tables come only from queue + receipts, and refuse inconsistency."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import m06_campaign_tables as T  # noqa: E402


def cid(n):
    return f"{n:016x}" + "0" * 48


def receipt(obj, model, skills):
    return {"exact_match": True, "objective": {"rescored_value": obj}, "digests": {"model_sha256": model},
            "metrics": {"MAE": obj, "baseline_MAE": 0.85, "skill_MAE": 1 - obj / 0.85},
            "per_horizon": {str(h): {"skill_MAE": s} for h, s in enumerate(skills, start=1)}}


def fixture():
    cands, atts, rec = [], [], {}
    objs = {("A", 2021): 0.40, ("A", 2022): 0.41, ("B", 2021): 0.39, ("B", 2022): 0.40}
    for n, ((lab, seed), o) in enumerate(objs.items()):
        c = cid(n + 1)
        cands.append({"cid": c, "config_id": lab * 64, "seed": seed, "label": lab, "status": "verified",
                      "objective": o, "flat": json.dumps({"train.loss": "mae"})})
        atts.append({"cid": c, "kind": "train", "status": "completed", "host": "worker_b", "cgroup_peak_bytes": 2**30,
                     "per_update_seconds": 0.03, "observed_updates": 100, "selected_epoch": 3, "stop_reason": "patience"})
        atts.append({"cid": c, "kind": "verify", "status": "completed", "host": "worker_a", "verdict": "VERIFIED",
                     "model_sha256": f"m{n}"})
        rec[c[:16]] = receipt(o, f"m{n}", [-0.4, 0.5, -0.5])
    cands.append({"cid": cid(99), "config_id": "C" * 64, "seed": 2021, "label": "C", "status": "blocked",
                  "objective": None, "flat": "{}"})
    q = {"campaign": "t", "meta": {"campaign_sha256": "s" * 64}, "counts": {"verified": 4, "blocked": 1},
         "objective": {"metric": "MAE", "split": "validation", "unit": "z_train", "higher_is_better": False},
         "attempts": atts, "candidates": cands,
         "standings": [{"config_id": "A" * 64, "label": "A", "eligible": True, "mean_objective": 0.405},
                       {"config_id": "B" * 64, "label": "B", "eligible": True, "mean_objective": 0.395},
                       {"config_id": "C" * 64, "label": "C", "eligible": False, "mean_objective": None}],
         "incumbent": {"config_id": "B" * 64},
         "incumbent_history": [{"seq": 1, "config_id": "A" * 64, "mean_objective": 0.405, "seeds": [2021, 2022],
                                "cids": [cid(1), cid(2)], "reason": "first"},
                               {"seq": 2, "config_id": "B" * 64, "mean_objective": 0.395, "seeds": [2021, 2022],
                                "cids": json.dumps([cid(3), cid(4)]), "reason": "lower"}]}  # the export carries JSON strings
    return q, rec


def test_tables_are_generated_with_negative_horizons_named_and_not_comparable():
    q, rec = fixture()
    t = T.build(q, rec)
    assert len(t["candidates"]) == 4 and t["comparability"]["class"] == "NOT_COMPARABLE"
    _, inc = T.render(t)
    assert "['h1', 'h3']" in inc and "NOT_COMPARABLE" in inc


def test_missing_receipt_is_refused():
    q, rec = fixture()
    rec.pop(cid(2)[:16])
    with pytest.raises(T.CampaignRefusal, match="MISSING_RECEIPT"):
        T.build(q, rec)


def test_objective_disagreeing_with_the_receipt_is_refused():
    q, rec = fixture()
    q["candidates"][0]["objective"] = 0.35
    with pytest.raises(T.CampaignRefusal, match="OBJECTIVE_MISMATCH"):
        T.build(q, rec)


def test_a_receipt_without_exact_match_is_refused():
    q, rec = fixture()
    rec[cid(3)[:16]]["exact_match"] = False
    with pytest.raises(T.CampaignRefusal, match="NOT_EXACT"):
        T.build(q, rec)


def test_a_model_digest_mismatch_is_refused():
    q, rec = fixture()
    rec[cid(1)[:16]]["digests"]["model_sha256"] = "other"
    with pytest.raises(T.CampaignRefusal, match="MODEL_DIGEST_MISMATCH"):
        T.build(q, rec)


def test_an_incumbent_that_is_not_the_minimum_is_refused():
    q, rec = fixture()
    q["incumbent"]["config_id"] = "A" * 64
    with pytest.raises(T.CampaignRefusal, match="INCUMBENT_NOT_MINIMUM"):
        T.build(q, rec)


def test_a_verified_candidate_without_a_verify_attempt_is_refused():
    q, rec = fixture()
    q["attempts"] = [a for a in q["attempts"] if not (a["cid"] == cid(4) and a["kind"] == "verify")]
    with pytest.raises(T.CampaignRefusal, match="MISSING_ATTEMPT"):
        T.build(q, rec)


def test_cli_writes_all_outputs(tmp_path):
    q, rec = fixture()
    (tmp_path / "q.json").write_text(json.dumps(q))
    rd = tmp_path / "r"
    rd.mkdir()
    for k, v in rec.items():
        (rd / f"{k}.json").write_text(json.dumps(v))
    out = tmp_path / "out"
    assert T.main(["--queue", str(tmp_path / "q.json"), "--receipts", str(rd), "--out-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"m04_r0_candidates.md", "m04_r0_candidates.csv", "m04_incumbents.md",
                                               "m04_campaign_tables.json"}
