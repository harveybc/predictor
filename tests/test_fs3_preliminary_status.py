import json
import sqlite3

import pytest

from tools.fs3_preliminary_status import summarize


def test_wave_status_has_exact_arm_denominators(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "fs3.preliminary_gpu_triage_bundle.v1",
                                    "populations": [{"population_id": "EURUSD", "gpu_feature_ids": ["a"]}]}))
    queue = tmp_path / "queue.sqlite"
    with sqlite3.connect(queue) as con:
        con.execute("CREATE TABLE tasks(state TEXT,payload TEXT,result TEXT,started_at REAL,finished_at REAL)")
        for arm in ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER"):
            for fold in range(5):
                con.execute("INSERT INTO tasks VALUES(?,?,NULL,NULL,NULL)", ("COMPLETE" if fold == 0 else "PENDING",
                            json.dumps({"population_id": "EURUSD", "feature_id": "a", "arm": arm})))
        con.execute("INSERT INTO tasks VALUES(?,?,NULL,NULL,NULL)", ("PENDING", json.dumps({
            "population_id": "EURUSD", "feature_id": "deferred", "arm": "TRAINED_ENCODER"})))
    result = summarize(queue, manifest)
    assert result["tasks_per_arm"] == 5
    assert result["done_per_arm"] == {"RAW": 1, "RANDOM_ENCODER": 1, "TRAINED_ENCODER": 1}
    assert result["final_selection"] is False
    with sqlite3.connect(queue) as con:
        con.execute("DELETE FROM tasks WHERE rowid=1")
    with pytest.raises(ValueError, match="FIRST_WAVE_TASK_DENOMINATOR_MISMATCH"):
        summarize(queue, manifest)
