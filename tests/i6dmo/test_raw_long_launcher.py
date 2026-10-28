"""Bounded launcher contract: private configs, durable continuation, no hidden fits."""
import json

import pytest


def test_launch_config_is_explicit_and_rejects_extra_or_nested_roots(tmp_path):
    from tools.i6dmo import raw_long_campaign as L
    root = tmp_path / "private"
    config = dict(schema="i6dmo.raw_long_launch.v1", work_root=str(root),
        source_deployment=str(tmp_path / "deployment.json"), input_dir=str(tmp_path / "input"),
        short_campaign=str(tmp_path / "short"), strategy_inputs=str(tmp_path / "bars"))
    assert L.validate_config(config)["work_root"] == str(root)
    with pytest.raises(ValueError, match="LAUNCH_CONFIG"):
        L.validate_config(dict(config, unreviewed_setting=1))
    with pytest.raises(ValueError, match="LAUNCH_CONFIG"):
        L.validate_config(dict(config, input_dir=str(root / "input")))


def test_journal_status_is_available_before_any_model_or_network(tmp_path):
    from tools.i6dmo import raw_long_campaign as L
    doc = L.status(tmp_path)
    assert doc["state"] == "NOT_PREPARED"
    L.record(tmp_path, "prepare", "RUNNING", completed=0, total=6)
    doc = L.status(tmp_path)
    assert doc["phase"] == "prepare" and doc["completed"] == 0
    assert doc["eta"] == "UNMEASURED"


def test_existing_campaign_is_not_reinitialised(tmp_path, monkeypatch):
    from tools.i6dmo import raw_long_campaign as L
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "PLAN.json").write_text("{}")
    (root / "CLOSE.json").write_text("{}")
    monkeypatch.setattr(L, "_existing_status", lambda r: dict(state="CLOSED", marker="retained"))
    monkeypatch.setattr(L, "_resume", lambda config, r: L._existing_status(r))
    monkeypatch.setattr(L, "prepare", lambda *a, **k: pytest.fail("must not prepare/repeat fits"))
    spec = dict(schema="i6dmo.raw_long_launch.v1", work_root=str(tmp_path),
        source_deployment="/fixture/deployment.json", input_dir="/fixture/input",
        short_campaign="/fixture/short", strategy_inputs="/fixture/bars")
    assert L.run(spec)["marker"] == "retained"


@pytest.mark.parametrize("closed,expected", [(False, "DISPATCHED"), (True, "CLOSED")])
def test_real_status_merge_never_duplicates_state(tmp_path, monkeypatch, closed, expected):
    from tools.i6dmo import raw_long_campaign as L, status as S
    monkeypatch.setattr(S, "compute", lambda *a, **k: dict(state="RUNNING", campaign={"counts": {}}))
    monkeypatch.setattr(L.C, "closed_doc", lambda r: {"state": "CLOSED"} if closed else None)
    assert L._existing_status(tmp_path)["state"] == expected
