"""The first FS4 wave may close without impersonating the full campaign."""

from copy import deepcopy

import pytest

from tools import fs4_closure as full
from tools import fs4_preliminary_close as partial


def example():
    manifest = {"schema": "fs3.preliminary_gpu_triage_bundle.v1", "final_selection": False,
                "populations": [{"population_id": "EURUSD", "gpu_feature_ids": ["f"],
                                 "deferred_feature_ids": ["deferred"]}]}
    tasks = []
    receipts = {}
    for fold in range(5):
        for arm in partial.ARMS:
            task_id = f"f-{fold}-{arm}"
            task = {"task_id": task_id, "state": "COMPLETE", "payload": {
                "population_id": "EURUSD", "feature_id": "f", "fold_id": f"fold-{fold}",
                "arm": arm, "identity": "dataset-v1"},
                "result": {"metrics": {"mae": 0.2, "naive_mae": 0.3}, "population_n": 10}}
            tasks.append(task)
            embedded = {"plan_sha256": "parent", "task_ids": [task_id]}
            embedded["receipt_sha256"] = full.core.digest(embedded)
            receipts[task_id] = {"schema": full.SCHEMA_RECEIPT_FILE, "task_id": task_id,
                                 "plan_sha256": "parent", "readback_verified": True,
                                 "terminal_sha256": full.local_terminal_sha256(task),
                                 "receipt_sha256": embedded["receipt_sha256"], "receipt": embedded}
    return manifest, tasks, receipts


def test_closes_only_declared_wave(monkeypatch):
    monkeypatch.setattr(partial.full, "validate_complete", lambda task: None)
    manifest, tasks, receipts = example()
    doc = partial.close_wave(tasks, manifest, "parent", "manifest", receipts)
    assert doc["state"] == "EXTRACTIBILITY_PARTIAL_COMPLETE"
    assert doc["denominator"] == {"admitted": 15, "complete": 15,
                                  "not_available_for_train": 0, "sum_equals_admitted": True}
    assert doc["populations"]["EURUSD"]["features"]["f"]["status"] == "MEASURED"
    assert doc["final_selection"] is False


@pytest.mark.parametrize("mutation", ["missing_receipt", "wrong_digest", "missing_arm", "pending"])
def test_refuses_incomplete_or_tampered_wave(monkeypatch, mutation):
    monkeypatch.setattr(partial.full, "validate_complete", lambda task: None)
    manifest, tasks, receipts = example()
    tasks = deepcopy(tasks)
    receipts = deepcopy(receipts)
    task_id = tasks[0]["task_id"]
    if mutation == "missing_receipt":
        del receipts[task_id]
    elif mutation == "wrong_digest":
        receipts[task_id]["terminal_sha256"] = "bad"
    elif mutation == "missing_arm":
        tasks.pop()
    else:
        tasks[0]["state"] = "PENDING"
    with pytest.raises(partial.Refusal):
        partial.close_wave(tasks, manifest, "parent", "manifest", receipts)
