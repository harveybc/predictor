"""Finite context bridge integration boundaries, without scientific fits."""
import pytest

from tools.i6dmo import matched_runner as MR, campaign as C


def test_profile_is_distinct_and_tie_retains_reference():
    p = MR.contrast_profile(("mg2l", "mg12c"))
    assert p["arms"] == ("mg12c", "mg2l")
    assert p["horizons"] == (18, 24, 36, 48, 60, 72)
    assert p["tie_keeps"] == "mg12c"
    assert p["context_bridge"] is True


@pytest.mark.parametrize("arms", [("mg12c", "mg12m"), ("mg2l",), ("mg2m", "mg12c")])
def test_no_mixed_or_partial_bridge(arms):
    with pytest.raises(ValueError):
        MR.contrast_profile(arms)


@pytest.mark.parametrize("decision,arm,expected", [("LOWER_SCORE", "mg2l", "SHORT_CONTEXT_SELECTED"),
    ("LOWER_SCORE", "mg12c", "LONG_CONTEXT_RETAINED"),
    ("TIE_RETAIN_CONTROL", "mg12c", "LONG_CONTEXT_RETAINED_WITHIN_TOLERANCE")])
def test_bridge_results_are_automatically_closeable(decision, arm, expected):
    disposition = MR.disposition_of(dict(decision=decision, selected=arm))
    assert disposition == expected
    assert disposition in C.AUTO_CLOSE_DISPOSITIONS


def test_launcher_rejects_output_in_source_and_wrong_config():
    from tools.i6dmo.context_bridge_campaign import validate_config
    with pytest.raises(ValueError):
        validate_config(dict(schema="i6dmo.context_bridge_launch.v1"))
    with pytest.raises(ValueError):
        validate_config(dict(schema="i6dmo.context_bridge_launch.v1", work_root="/tmp/bridge",
            input_dir="/tmp/bridge/input", source_deployment="/tmp/deployment.json"))


def test_timer_shutdown_is_bound_and_does_not_touch_foreign_units(tmp_path, monkeypatch):
    import json
    from tools.i6dmo import context_bridge_campaign as B, deploy
    config = dict(schema="i6dmo.context_bridge_launch.v1", work_root=str(tmp_path),
                  input_dir="/tmp/source", source_deployment="/tmp/deploy.json")
    name = "i6dmo-context-bridge-" + C.sha256_json(config)[:12]
    receipt = tmp_path / "LAUNCHER.json"
    receipt.write_text(json.dumps(dict(name="foreign.timer")))
    calls = []
    monkeypatch.setattr(deploy, "default_systemctl", lambda *args: calls.append(args) or dict(rc=0))
    with pytest.raises(ValueError, match="FOREIGN_LAUNCH"):
        B.stop_launcher(config)
    assert not calls
    receipt.write_text(json.dumps(dict(name=name)))
    B.stop_launcher(config)
    assert calls == [("disable", "--now", name + ".timer")]
