"""Phase-3 triage must not silently expand a GPU wave to all features."""

import hashlib
import json

import pytest

from tools.fs3_preliminary_screen import K_GRID, METHODS, screen


def _source(tmp_path):
    closure = {"state": "PHASE_3_FILTER_COMPLETE", "closure_sha256": "a" * 64,
               "final_selection": False, "units": [{"target_id": "Y_s_1h"}]}
    candidates = [{"population_id": "EURUSD", "identity": "run-1", "target_id": "Y_s_1h",
                   "method": method, "k": k, "members": ["useful"],
                   "subset_sha256": hashlib.sha256(f"{method}-{k}".encode()).hexdigest()}
                  for method in METHODS for k in K_GRID]
    candidates.append({"population_id": "EURUSD", "identity": "run-1", "target_id": "Y_s_1h",
                       "method": "ALL_ADMISSIBLE", "k": 2, "members": ["useful", "unranked"],
                       "subset_sha256": "b" * 64})
    path = tmp_path / "CANDIDATES_FOR_VALIDATION.json"
    path.write_text(json.dumps({"population_id": "EURUSD", "identity": "run-1",
                                "phase3_closure_sha256": "a" * 64,
                                "final_selection": False,
                                "admissible_features": {"EURUSD": 2},
                                "candidates": candidates}))
    (tmp_path / "PHASE_3_FILTER_COMPLETE.json").write_text(json.dumps(closure))
    return path


def test_all_admissible_does_not_expand_gpu_wave(tmp_path):
    result = screen(_source(tmp_path))
    assert result["gpu_feature_ids"] == ["useful"]
    assert result["deferred_feature_ids"] == ["unranked"]
    assert result["subset_count"] == len(METHODS) * len(K_GRID)
    assert result["final_selection"] is False


def test_missing_method_k_refuses(tmp_path):
    path = _source(tmp_path)
    source = json.loads(path.read_text())
    source["candidates"].pop(0)
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="MISSING_PRELIMINARY_CANDIDATES"):
        screen(path)


def test_closure_mismatch_refuses(tmp_path):
    path = _source(tmp_path)
    source = json.loads(path.read_text())
    source["phase3_closure_sha256"] = "wrong"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="PHASE3_CLOSURE_MISMATCH"):
        screen(path)
