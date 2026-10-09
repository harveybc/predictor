import hashlib
import json

import pytest

from tools import i6b_weekly_campaign as C
from tools import i6b_weekly_merge as M


def _design():
    return {
        "design_sha256": "d" * 64, "members": ["a"], "config": {"seed": 0},
        "weeks": [{"ordinal": i, "fit_start": "2020-01-01T00:00:00Z",
                   "cutoff": "2024-01-01T00:00:00Z", "start": "2024-01-01T00:00:00Z",
                   "end": "2024-01-08T00:00:00Z"} for i in range(2)],
    }


def _week(root, design, ordinal, payload):
    path = root / f"week_{ordinal:03d}"
    path.mkdir(parents=True)
    model = payload.encode()
    (path / "branch_000.keras").write_bytes(model)
    report = {
        "schema": "predictor.i6b.branch_pretraining.v1", "status": "COMPLETE",
        "branch_count": 1, "branches": [{
            "branch": "branch_000", "artifact": "branch_000.keras",
            "manifest": "branch_000.manifest.json", "manifest_sha256": "1" * 64,
            "model_sha256": hashlib.sha256(model).hexdigest(), "weights_sha256": "2" * 64,
            "data_sha256": "3" * 64,
        }],
    }
    (path / "REPORT.json").write_text(json.dumps(report))
    C.write_week_receipt(root, design, ordinal, {"branch_000": "a"}, report)


def test_merge_disjoint_shards_closes_and_builds_index(tmp_path):
    design = _design()
    left, right, out = tmp_path / "left", tmp_path / "right", tmp_path / "out"
    _week(left, design, 0, "zero")
    _week(right, design, 1, "one")
    status = M.merge_campaign(out, [left, right], design)
    assert status["status"] == "COMPLETE"
    assert status["completed_weeks"] == 2
    assert (out / "I7_DONOR_INDEX.json").is_file()


def test_merge_rejects_conflicting_receipts(tmp_path):
    design = _design()
    left, right = tmp_path / "left", tmp_path / "right"
    _week(left, design, 0, "first")
    _week(right, design, 0, "second")
    with pytest.raises(ValueError, match="conflicting"):
        M.merge_campaign(tmp_path / "out", [left, right], design)
