import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import register_run as rr  # noqa: E402

PLAN = {"identity": "phase1-eurusd-final:test", "population_id": "EURUSD",
        "expected_row_counts_per_shard": {"u1": {"feature_pair_gate": 2, "feature_pair_metrics": 10},
                                          "u2": {"feature_pair_gate": 3, "feature_pair_metrics": 11}}}


def test_expected_counts_sum_over_shards():
    assert rr.expected_from_plan(PLAN) == {"feature_pair_gate": 5, "feature_pair_metrics": 21}
    doc = rr.run_document(PLAN)
    assert doc["run_id"] == "phase1-eurusd-final:test" and doc["registration"] == "FIRST_SUBMISSION"


def test_register_is_idempotent_on_a_throwaway_file(tmp_path, monkeypatch):
    wh = pytest.importorskip("tools.fs_phase23_warehouse")
    monkeypatch.chdir(ROOT)
    plan = tmp_path / "PLAN.json"
    plan.write_text(json.dumps(PLAN))
    db = str(tmp_path / "w.duckdb")
    assert rr.main(["--plan", str(plan), "--warehouse", db]) == 0
    assert rr.main(["--plan", str(plan), "--warehouse", db]) == 0
    w = wh.open_warehouse(db)
    r = w.reconcile("phase1-eurusd-final:test")
    assert r["run_id"] == "phase1-eurusd-final:test"
