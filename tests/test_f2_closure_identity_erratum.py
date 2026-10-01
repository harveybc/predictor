import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.f2_closure_identity_erratum import build_erratum  # noqa: E402


def evidence(campaign_id="eurusd-v1"):
    return {
        "artifact": {"campaign_id": campaign_id},
        "population": {"asset": "EURUSD 1h", "dataset_id": "lake:eurusd:v1", "sample_hours": 1,
                       "targets": ["log_return_1"]},
        "scale": {"metric_space": "z_train", "scaler_identity": "scaler:train:v1"},
        "per_horizon": {str(h): {"MAE": 0.1} for h in range(1, 5)},
    }


def test_erratum_binds_original_and_derives_identity_without_changing_metrics(tmp_path):
    closure = tmp_path / "CLOSURE.json"
    closure.write_text(json.dumps({"literature": {"reason": "ETHUSDT"}, "configurations": {"x": 1}}))
    paths = []
    for index in range(2):
        path = tmp_path / f"EVIDENCE_{index}.json"
        path.write_text(json.dumps(evidence()))
        paths.append(path)
    result = build_erratum(closure, paths)
    assert result["metrics_changed"] is False
    assert result["retained_evidence"]["count"] == 2
    assert result["corrected"]["campaign_identity"]["campaign_id"] == "eurusd-v1"
    assert "EURUSD 1h" in result["corrected"]["literature"]["reason"]
    assert len(result["original"]["sha256"]) == 64
    assert json.loads(closure.read_text())["configurations"] == {"x": 1}


def test_erratum_rejects_mixed_evidence(tmp_path):
    closure = tmp_path / "CLOSURE.json"
    closure.write_text(json.dumps({"literature": {"reason": "wrong"}}))
    paths = []
    for index, campaign in enumerate(("eurusd-v1", "eth-v1")):
        path = tmp_path / f"EVIDENCE_{index}.json"
        path.write_text(json.dumps(evidence(campaign)))
        paths.append(path)
    with pytest.raises(ValueError, match="mixed campaign identity"):
        build_erratum(closure, paths)


def test_erratum_accepts_published_list_shaped_per_horizon(tmp_path):
    closure = tmp_path / "CLOSURE.json"
    closure.write_text(json.dumps({"literature": {"reason": "wrong"}}))
    item = evidence()
    item["per_horizon"] = [{"horizon": h, "model_MAE": 0.1} for h in range(1, 5)]
    path = tmp_path / "EVIDENCE_0.json"
    path.write_text(json.dumps(item))
    result = build_erratum(closure, [path])
    assert result["corrected"]["campaign_identity"]["horizons"] == [1, 2, 3, 4]
