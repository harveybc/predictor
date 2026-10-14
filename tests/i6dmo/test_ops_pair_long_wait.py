"""Pair attempts are never burned while waiting; the long design is sealed automatically after short closure."""
import json

import pytest

from tests.i6dmo.ops_fakes import fake_adapter, make_campaign, run_worker, write_long_design


def _rulings(root):
    from tools.i6dmo import campaign
    campaign.record_gate(root, "STRATEGY_OWNER_RULINGS", status="PASS",
                         evidence={r: "ruled" for r in campaign.OWNER_RULINGS}, splits_read=[])


def _to_pair(root, counter, report_mode="ok"):
    from tools.i6dmo import campaign
    for role, mode in (("preferred", "ok"), ("coordinator", report_mode), ("preferred", "ok"),
                       ("coordinator", report_mode)):
        campaign.tick(root)
        assert run_worker(root, role, counter, mode=mode)["result"] == "COMPLETE"
    campaign.tick(root)


def _assert_waits(root, counter, prefix, state="WAIT_ARTIFACT"):
    from tools.i6dmo import status
    out = run_worker(root, "coordinator", counter)
    assert out["result"] == state and out["cause"].startswith(prefix), out
    unit = status.compute(root, level="unit", unit="pair.validation")
    assert unit["state"] == state and unit["attempts"] == []


def test_pair_waits_for_owner_rulings_before_claim(tmp_path):
    root = make_campaign(tmp_path)
    _to_pair(root, tmp_path / "c.txt")
    _assert_waits(root, tmp_path / "c.txt", "WAIT_OWNER_RULING", state="WAIT_OWNER_RULING")


def test_pair_waits_for_missing_strategy_pins(tmp_path):
    root = make_campaign(tmp_path, strategy_pins=False)
    _to_pair(root, tmp_path / "c.txt")
    _rulings(root)
    _assert_waits(root, tmp_path / "c.txt", "STRATEGY_INPUT_PIN_MISSING:strategy_bars")


def test_pair_waits_for_digest_mismatched_strategy_input(tmp_path):
    root = make_campaign(tmp_path)
    _to_pair(root, tmp_path / "c.txt")
    _rulings(root)
    (tmp_path / "strategy_calendar.json").write_text('{"changed": true}')
    _assert_waits(root, tmp_path / "c.txt", "STRATEGY_INPUT_DIGEST_MISMATCH:strategy_calendar")


def test_pair_waits_when_a_report_lacks_validation_predictions(tmp_path):
    root = make_campaign(tmp_path)
    _to_pair(root, tmp_path / "c.txt", report_mode="no_vp")
    _rulings(root)
    _assert_waits(root, tmp_path / "c.txt", "UPSTREAM_ARTIFACT_MISSING:short.report:validation_predictions")


def test_long_design_is_sealed_automatically_after_short_closure(tmp_path, monkeypatch):
    from tools.i6dmo import campaign
    calls = []

    def sealer(short_path, child_terminal, out_path):
        calls.append(json.loads(open(child_terminal).read())["unit_id"])
        out_path.write_text(write_long_design(tmp_path).read_text())
        return dict(status="SEALED", design_sha256="x")
    monkeypatch.setattr(campaign, "LONG_SEALER", sealer)
    root = make_campaign(tmp_path, long=False)
    campaign.tick(root, adapter=fake_adapter())
    assert calls == []  # not before short closure
    for role in ("preferred", "coordinator"):
        campaign.tick(root, adapter=fake_adapter())
        assert run_worker(root, role, tmp_path / "c.txt")["result"] == "COMPLETE"
    campaign.tick(root, adapter=fake_adapter())
    assert calls == ["short.report"]
    units = {u["unit_id"]: u for u in campaign.load_plan(root)["units"]}
    assert units["long.fit"]["identity_sha256"] and units["pair.validation"]["identity_sha256"]
    assert run_worker(root, "preferred", tmp_path / "c.txt")["unit_id"] == "long.fit"


def test_long_design_refusal_is_a_typed_wait(tmp_path, monkeypatch):
    from tools.i6dmo import campaign, status
    monkeypatch.setattr(campaign, "LONG_SEALER",
                        lambda *a: dict(status="FAILED", code="LONG_DESIGN_UNDECIDED", detail="open"))
    root = make_campaign(tmp_path, long=False)
    for role in ("preferred", "coordinator"):
        campaign.tick(root, adapter=fake_adapter())
        run_worker(root, role, tmp_path / "c.txt")
    campaign.tick(root, adapter=fake_adapter())
    unit = status.compute(root, level="unit", unit="long.fit")
    assert unit["state"] == "WAIT_ARTIFACT" and unit["cause"].startswith("LONG_DESIGN_LONG_DESIGN_UNDECIDED")
    assert status.compute(root, level="campaign")["state"] not in ("FAILED", "CLOSED")


def test_real_model_lane_sealer_accepts_a_short_closure_terminal(tmp_path):
    pytest.importorskip("tools.i6dmo.design")
    from tests.i6dmo.ops_fakes import DESIGN
    from tools.i6dmo import campaign
    child = tmp_path / "CHILD_TERMINAL.json"
    child.write_text(json.dumps(dict(schema="i6dmo.terminal.v1", unit_id="short.report", status="COMPLETE",
                                     identity_sha256="a" * 64)))
    result = campaign._default_long_sealer(DESIGN, child, tmp_path / "EXPERIMENT.long.json")
    assert result["status"] == "SEALED", result
    long_design = json.loads((tmp_path / "EXPERIMENT.long.json").read_text())
    assert long_design["horizons_hours"] == [18, 24, 36, 48, 60, 72]
    verdict = campaign.ModelLaneAdapter(__import__("tools.i6dmo.design", fromlist=["x"])).validate_design(long_design)
    assert verdict["ok"], verdict
