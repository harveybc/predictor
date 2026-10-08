"""Failures must stop a campaign before any incomparable weekly aggregate."""

import pytest

from tools import fs4_weekly_wrapper as W
from tools.i6a_campaign import atomic_json, paired_week, verify_result
from tools.i6a_architectures import ARMS


def _record(arm):
    result = {"disposition": "COMPLETED", "rows_sha256": "rows", "n_scored": 120,
              "fit_population_digest": "fit", "inner_population_digest": "inner", "fit_rows": 1000,
              "inner_rows": 100, "naive": {"rows_sha256": "rows"}}
    result["result_sha256"] = W.digest(result)
    record = {"arm": arm, "task": {"task_id": "same"}, "result": result}
    record["sha256"] = W.digest(record)
    return record


def test_paired_week_rejects_changed_support():
    records = [_record(a) for a in ARMS]
    paired_week(records)
    records[2]["result"]["n_scored"] = 119
    with pytest.raises(ValueError, match="PAIRED_WEEK_MISMATCH"):
        paired_week(records)


def test_result_digest_and_task_identity_reject_mutation(tmp_path):
    rec = _record("ARCH_A")
    path = tmp_path / "cell.json"
    atomic_json(path, rec)
    assert verify_result(path, "ARCH_A", rec["task"]) == rec
    rec["result"]["rows_sha256"] = "other"
    atomic_json(path, rec)
    with pytest.raises(ValueError, match="RESULT_DIGEST_MISMATCH"):
        verify_result(path, "ARCH_A", rec["task"])
