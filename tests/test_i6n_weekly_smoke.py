"""Synthetic end-to-end integration; no predictive claim on market data."""

import json
from tools.i6n_weekly_smoke import _hash, _identity, run, status


def test_weekly_masked_training_to_paired_metrics_and_verified_resume(tmp_path):
    result = run(tmp_path, device="cpu", epochs=1)
    assert result["state"] == "COMPLETED"
    assert result["scientific_measurement"] is False
    assert result["coverage_complete"] is False
    assert result["scope"] == "SYNTHETIC_WEEKLY_INTEGRATION"
    assert result["training"]["observed_updates"] > 0
    assert result["training"]["restored_best_weights"]
    assert result["current_origin_in_window"]
    groups = result["metrics"]["groups"]
    scored = [g for g in groups if g["population_kind"] == "S"]
    assert {g["horizon_hours"] for g in scored} == {2, 72}
    assert status(tmp_path)[1] == 0
    assert run(tmp_path, device="cpu", epochs=1) == result
    result_bytes = (tmp_path / "RESULT.json").read_bytes()
    status_bytes = (tmp_path / "STATUS.json").read_bytes()
    altered = json.loads(result_bytes)
    altered["metrics"]["forged_value"] = 999
    (tmp_path / "RESULT.json").write_text(json.dumps(altered))
    receipt = json.loads(status_bytes)
    receipt["result_sha256"] = _hash(tmp_path / "RESULT.json")
    (tmp_path / "STATUS.json").write_text(json.dumps(receipt))
    assert status(tmp_path)[1] == 6  # A re-sealed result still must match input rescore.
    (tmp_path / "RESULT.json").write_bytes(result_bytes)
    (tmp_path / "STATUS.json").write_bytes(status_bytes)
    with (tmp_path / "model.keras").open("ab") as stream:
        stream.write(b"changed")
    assert status(tmp_path)[1] == 6


def test_unstarted_status_does_not_create_files(tmp_path):
    assert status(tmp_path)[1] == 3
    assert list(tmp_path.iterdir()) == []


def test_dead_runner_is_interrupted_not_running(tmp_path):
    (tmp_path / "RUN.lock").touch()
    original = json.dumps({"state": "RUNNING", "code_sha256": _identity(), "eta_seconds": 10})
    (tmp_path / "STATUS.json").write_text(original)
    value, code = status(tmp_path)
    assert code == 4
    assert value["state"] == "INTERRUPTED"
    assert value["eta_seconds"] is None
    assert (tmp_path / "STATUS.json").read_text() == original
