import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from finalize_t3_combination import decide
import finalize_t3_combination as finalizer


def result(scores, valid=True):
    return {"meta": {"support_hash": {"TRAIN": "T", "INNER_A": "V"}},
            "decision": {"means": {"FULL": {"all_valid": valid}}},
            "trials": {"FULL": {str(seed): {"trading": {
                "TRAIN": {"sharpe_daily_sqrt365": 3.0},
                "INNER_A": {"sharpe_daily_sqrt365": s, "valid": valid}}}
                for seed,s in zip((20261019, 20261020, 20261021), scores)}}}


def test_frozen_combination_rule_ignores_2024_and_requires_activity():
    control, candidate = result([0., 1., 2.]), result([1., 2., 3.])
    assert decide(control, candidate)["state"] == "COMBINATION_SELECTED"
    assert decide(control, result([-4., -3., -2.]))["state"] == "CONTROL_RETAINED"
    assert decide(control, result([1., 2., 3.], valid=False))["state"] == "CONTROL_RETAINED"
    before = decide(control, candidate)
    for trial in candidate["trials"]["FULL"].values():
        trial["trading"]["OBS_2024"] = {"sharpe_daily_sqrt365": -99999}
    assert decide(control, candidate) == before
    assert not before["scientific_release"]


def test_final_decision_rejects_missing_seeds_and_changed_support():
    a = result([0., 1., 2.]); b = copy.deepcopy(a)
    b["meta"]["support_hash"]["TRAIN"] = "changed"
    with pytest.raises(ValueError, match="SUPPORT"):
        decide(a, b)
    b = copy.deepcopy(a);b["trials"]["FULL"].pop("20261019")
    with pytest.raises(ValueError, match="SEEDS"):
        decide(a, b)


def test_finalizer_waits_without_starting_replay_or_transfers(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer.subprocess, "run", lambda *a, **k: pytest.fail("unexpected command"))
    assert finalizer.tick({"mirror": str(tmp_path)})["state"] == "WAITING_FOR_COMBINATION"


def test_finalizer_delivers_failed_terminal_without_inventing_metrics(tmp_path, monkeypatch):
    finalizer.w.atomic(tmp_path / "ops/CLOSE.json", {"failed": 1})
    monkeypatch.setattr(finalizer, "validate_mirror", lambda root: {"closed": True})
    commands = []
    monkeypatch.setattr(finalizer.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    cfg = {"mirror": str(tmp_path), "output": str(tmp_path / "FAILED.json"),
           "destination": "peer:/delivery/report.json", "timer": "owned.timer"}
    report = finalizer.tick(cfg)
    assert report["state"] == "COMBINATION_EXECUTION_FAILED" and not report["scientific_release"]
    assert len(commands) == 2
    assert commands[-1] == ["systemctl", "--user", "disable", "--now", "owned.timer"]
