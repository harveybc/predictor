"""Architecture work must use each forecast target's actual support horizon."""

import pytest

from tools import fs4_weekly_wrapper as W
from tools.i6a_weekly_arch_pilot import make_task, observed_gpu, target_horizon_hours
from tools.i6a_close import close


@pytest.mark.parametrize("target,horizon", [
    ("Y_s_1h", 1), ("Y_s_6h", 6), ("Y_l_24h", 24), ("Y_l_144h", 144),
])
def test_target_horizon(target, horizon):
    assert target_horizon_hours(target) == horizon


@pytest.mark.parametrize("target", ["Y_b_s6", "Y_s_7h", "Y_l_25h"])
def test_unsupported_target_refuses(target):
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        target_horizon_hours(target)


def test_task_binds_long_target_and_purge():
    freeze = {"winners": {"EURUSD": {"Y_l_24h": {"RAW": {
        "members": ["a", "b"], "set_id": "selected-set",
    }}}}}
    freeze["freeze_sha256"] = W.digest(freeze)
    task = make_task(freeze, "Y_l_24h", 0, 2024)
    assert task["target_id"] == "Y_l_24h"
    assert task["horizon_hours"] == 24


def test_gpu_pin_rejects_unpinned_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="GPU_UUID_NOT_PINNED"):
        observed_gpu("GPU-expected")


def test_long_horizon_closure_names_target_and_refuses_missing_cells(tmp_path):
    freeze = {"winners": {"EURUSD": {"Y_l_24h": {"RAW": {
        "members": ["a", "b"], "set_id": "selected-set",
    }}}}}
    freeze["freeze_sha256"] = W.digest(freeze)
    result = close(freeze, [tmp_path], "Y_l_24h")
    assert result["state"] == "INCOMPLETE_EVIDENCE"
    assert result["target_id"] == "Y_l_24h"
    assert result["horizon_hours"] == 24
    assert result["expected_cells"] == 208
