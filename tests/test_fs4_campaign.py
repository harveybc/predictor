import json
from pathlib import Path

import pytest

from tools import fs4_campaign as campaign


def candidate_file(path: Path):
    path.write_text(json.dumps({
        "identity": "train-v1", "population_id": "EURUSD", "final_selection": False,
        "candidates": [
            {"target_id": "short", "method": "MRMR", "k": 2, "members": ["a", "b"], "subset_sha256": "subset-a"},
            {"target_id": "long", "method": "JMI", "k": 2, "members": ["b", "c"], "subset_sha256": "subset-b"},
        ],
    }))


def test_unique_feature_tasks_and_claim_resume(tmp_path):
    source = tmp_path / "candidates.json"
    candidate_file(source)
    db = tmp_path / "queue.sqlite"
    campaign.initialize(db, [source], ["f1", "f2"])
    assert campaign.status(db)["total"] == 18  # 3 features x 2 folds x 3 arms
    first = campaign.claim(db, "worker-a", "extractibility", now=100)
    assert first is not None
    assert campaign.claim(db, "worker-b", "extractibility", now=101)["task_id"] != first["task_id"]
    assert campaign.claim(db, "worker-a", "extractibility", now=102)["task_id"] != first["task_id"]
    campaign.heartbeat(db, "worker-a", first["task_id"], now=103)
    assert campaign.claim(db, "worker-c", "extractibility", now=100 + campaign.LEASE_SECONDS + 1)["task_id"] != first["task_id"]
    recovered = [campaign.claim(db, "worker-c", "extractibility", now=104 + campaign.LEASE_SECONDS)
                 for _ in range(17)]
    assert first["task_id"] in {task["task_id"] for task in recovered if task}


def test_status_can_scope_to_each_task_layer(tmp_path):
    source = tmp_path / "candidates.json"
    candidate_file(source)
    db = tmp_path / "queue.sqlite"
    campaign.initialize(db, [source], ["f1", "f2"])
    scoped = campaign.status(db, population="EURUSD", feature="b", fold="f1", arm="RAW")
    assert scoped["total"] == 1
    assert scoped["pending"] == 1
    assert scoped["eta_seconds"] is None
    assert campaign.status(db, population="ETH")["total"] == 0


def test_terminal_must_match_task_and_be_finite(tmp_path):
    source = tmp_path / "candidates.json"
    candidate_file(source)
    db = tmp_path / "queue.sqlite"
    campaign.initialize(db, [source], ["f1"])
    task = campaign.claim(db, "worker", "extractibility", now=100)
    result = {"task_id": task["task_id"], "input_sha256": "a" * 64,
              "code_sha256": "b" * 64, "model_sha256": "c" * 64,
              "rows_sha256": "d" * 64, "mask_sha256": "e" * 64,
              "metrics": {"mae": 0.1, "naive_mae": 0.2}, "population_n": 20,
              "seed": 0, "status": "COMPLETE"}
    with pytest.raises(campaign.Refusal):
        campaign.complete(db, "worker", {**result, "task_id": "wrong"}, now=103)
    with pytest.raises(campaign.Refusal):
        campaign.complete(db, "worker", {**result, "metrics": {"mae": float("nan"), "naive_mae": 0.2}}, now=103)
    campaign.complete(db, "worker", result, now=104)
    assert campaign.status(db)["complete"] == 1
    assert campaign.claim(db, "worker", "extractibility", now=105)["task_id"] != task["task_id"]


def test_rerun_rejects_changed_plan(tmp_path):
    source = tmp_path / "candidates.json"
    candidate_file(source)
    db = tmp_path / "queue.sqlite"
    campaign.initialize(db, [source], ["f1"])
    campaign.initialize(db, [source], ["f1"])
    with pytest.raises(campaign.Refusal):
        campaign.initialize(db, [source], ["f2"])


def test_calendar_is_context_and_arms_require_identical_rows(tmp_path):
    source = tmp_path / "candidates.json"
    candidate_file(source)
    data = json.loads(source.read_text())
    data["candidates"][0]["members"].append("cal.hour_sin")
    source.write_text(json.dumps(data))
    db = tmp_path / "queue.sqlite"
    out = campaign.initialize(db, [source], ["f1"])
    assert out["tasks"] == 9
    assert out["sources"][0]["seasonal_context"] == ["cal.hour_sin"]
    raw = campaign.claim(db, "worker", now=100)
    result = {"task_id": raw["task_id"], "status": "COMPLETE", "seed": 0,
              "input_sha256": "a" * 64, "code_sha256": "b" * 64,
              "model_sha256": "c" * 64, "rows_sha256": "d" * 64,
              "mask_sha256": "e" * 64, "population_n": 20,
              "metrics": {"mae": 0.1, "naive_mae": 0.2}}
    campaign.complete(db, "worker", result, now=101)
    random = campaign.claim(db, "worker", now=102)
    while random["feature_id"] != raw["feature_id"]:
        random = campaign.claim(db, "worker", now=103)
    with pytest.raises(campaign.Refusal, match="PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH"):
        campaign.complete(db, "worker", {**result, "task_id": random["task_id"],
                                         "rows_sha256": "f" * 64}, now=104)
    with pytest.raises(campaign.Refusal, match="PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH"):
        campaign.complete(db, "worker", {**result, "task_id": random["task_id"],
                                         "metrics": {"mae": 0.1, "naive_mae": 0.3}}, now=104)
