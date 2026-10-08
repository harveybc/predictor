"""A result page needs a complete, published, internally consistent closure."""

import copy

import pytest

from tools.fs4_candidates import digest
from tools.i6a_result_catalog import record, summarize
from tools.i6a_publish_olap import experiment_set_key


def closure():
    arms = {
        arm: {"mean_weekly_mae": mae, "mean_weekly_naive_mae": 0.2,
              "weeks_better_than_naive": wins, "pooled_mae": mae,
              "pooled_naive_mae": 0.2}
        for arm, mae, wins in (("ARCH_0", 0.4, 0), ("ARCH_A", 0.21, 12),
                               ("ARCH_B", 0.19, 30), ("ARCH_C", 0.22, 10))
    }
    value = {"schema": "i6a.architecture_closure.v2", "state": "COMPLETE",
             "target_id": "Y_l_24h", "split": "validation", "year": 2024,
             "expected_cells": 208, "verified_cells": 208, "expected_weeks": 52,
             "test_read": False, "selected_set_conditioned_on_validation": True,
             "problems": [], "arms": arms}
    value["sha256"] = digest(value)
    return value


def status(value):
    return {"state": "PUBLISHED", "target": "Y_l_24h", "reports": 208,
            "closure_sha256": value["sha256"],
            "experiment_set_key": experiment_set_key("Y_l_24h"), "metric_horizon": 24}


def test_summary_names_naive_and_development_scope():
    value = closure()
    text = summarize(value, status(value))
    assert "0.190000000" in text
    assert "0.200000000" in text
    assert "30/52" in text
    assert "development" in text
    assert "TEST" in text


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(verified_cells=207),
    lambda c: c.update(test_read=True),
    lambda c: c["arms"]["ARCH_C"].update(mean_weekly_naive_mae=0.3),
    lambda c: c.update(sha256="0" * 64),
])
def test_summary_refuses_incomplete_or_contradictory_evidence(mutation):
    value = closure()
    mutation(value)
    with pytest.raises(ValueError):
        summarize(value, status(value))


def test_summary_refuses_unpublished_or_wrong_identity():
    value = closure()
    bad = copy.deepcopy(status(value))
    bad["target"] = "Y_s_1h"
    with pytest.raises(ValueError):
        summarize(value, bad)


def test_catalog_is_idempotent_and_refuses_a_rewritten_result(tmp_path):
    value = closure()
    page = record(tmp_path, value, status(value))
    first = page.read_bytes()
    record(tmp_path, value, status(value))
    assert page.read_bytes() == first
    assert "Y_l_24h" in (tmp_path / "INDEX.md").read_text()
    value["arms"]["ARCH_B"]["mean_weekly_mae"] = 0.18
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    with pytest.raises(ValueError, match="CATALOG_IDENTITY_CONFLICT"):
        record(tmp_path, value, status(value))
    bad = status(value)
    bad["state"] = "CLOSED"
    with pytest.raises(ValueError):
        summarize(value, bad)
