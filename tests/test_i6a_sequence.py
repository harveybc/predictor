"""Sequential horizon dispatch is resumable and fails closed."""

import json

import pytest

from tools.i6a_sequence import target_list, waiting_state


def test_target_list_rejects_duplicate_or_unknown_horizons():
    assert target_list(["Y_s_3h", "Y_s_4h"]) == ["Y_s_3h", "Y_s_4h"]
    for values in (["Y_s_3h", "Y_s_3h"], ["Y_b_s6"], []):
        with pytest.raises(ValueError):
            target_list(values)


def test_waiting_state_requires_complete_predecessor(tmp_path):
    assert waiting_state(tmp_path / "STATUS.json", "COMPLETE") == "WAITING"
    path = tmp_path / "STATUS.json"
    path.write_text(json.dumps({"state": "COMPLETE", "completed": 208, "total": 208}))
    assert waiting_state(path, "COMPLETE") == "READY"
    path.write_text(json.dumps({"state": "COMPLETE", "completed": 207, "total": 208}))
    assert waiting_state(path, "COMPLETE") == "FAILED"
    path.write_text(json.dumps({"state": "PUBLISHED", "reports": 208}))
    assert waiting_state(path, "PUBLISHED") == "READY"
    path.write_text(json.dumps({"state": "INCOMPLETE_EVIDENCE"}))
    assert waiting_state(path, "PUBLISHED") == "FAILED"
