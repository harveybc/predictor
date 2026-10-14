"""Pair attempts are never burned while waiting; the long design is sealed automatically after short closure."""
import json

import pytest

from tests.i6dmo.ops_fakes import fake_adapter, make_campaign, run_worker, write_long_design


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


def test_pair_waits_for_an_absent_execution_policy_receipt_before_claim(tmp_path):
    root = make_campaign(tmp_path)
    (root / "gates" / "EXECUTION_POLICY.json").unlink()  # e.g. a root restored without its seal
    _to_pair(root, tmp_path / "c.txt")
    _assert_waits(root, tmp_path / "c.txt", "EXECUTION_POLICY_ABSENT")


def test_pair_waits_for_missing_strategy_pins(tmp_path):
    root = make_campaign(tmp_path, strategy_pins=False)
    _to_pair(root, tmp_path / "c.txt")
    _assert_waits(root, tmp_path / "c.txt", "STRATEGY_INPUT_PIN_MISSING:strategy_bars")


def test_pair_waits_for_digest_mismatched_strategy_input(tmp_path):
    root = make_campaign(tmp_path)
    _to_pair(root, tmp_path / "c.txt")
    (tmp_path / "strategy_calendar.json").write_text('{"changed": true}')
    _assert_waits(root, tmp_path / "c.txt", "STRATEGY_INPUT_DIGEST_MISMATCH:strategy_calendar")


def test_pair_waits_when_a_report_lacks_validation_predictions(tmp_path):
    root = make_campaign(tmp_path)
    _to_pair(root, tmp_path / "c.txt", report_mode="no_vp")
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


def test_long_bundle_pin_uses_its_own_sampling_and_data_contract(tmp_path):
    from tools.i6dmo import campaign
    short = {"relpath": "s", "sha256": "1" * 64, "bundle_sha256": "3e42c3f5" + "0" * 56,
             "input_sampling": "DS2_EXACT"}
    long_ = {"relpath": "l", "sha256": "2" * 64, "bundle_sha256": "e88f72aa" + "0" * 56,
             "input_sampling": "DS12_EXACT", "data_contract_sha256": "6" * 64}
    make_campaign(tmp_path, name="ok", data_pins={"short_bundle": short, "long_bundle": long_})
    with pytest.raises(campaign.Refused, match="BUNDLE_SAMPLING: long_bundle"):
        make_campaign(tmp_path, name="bad", data_pins={"short_bundle": short,
                                                       "long_bundle": dict(long_, input_sampling="DS2_EXACT")})


def test_publish_units_wait_without_a_warehouse_environment_file(tmp_path):
    from tests.i6dmo.ops_fakes import private_spec
    from tools.i6dmo import deploy
    spec = private_spec(tmp_path)
    spec["warehouse"]["environment_file"] = None
    (tmp_path / "private_spec.json").write_text(json.dumps(spec))
    root = make_campaign(tmp_path)
    log = []
    out = deploy.deploy(root, unit_dir=tmp_path / "units", systemctl=lambda *a: log.append(a) or dict(rc=0, stdout="x"))
    assert not [u for u in out["units"] if "publish" in u] and out["waiting"]


def test_final_test_waits_until_the_strategy_lane_exports_run_final_backtest(tmp_path, monkeypatch):
    from tools.i6dmo import campaign, queue, status, strategy_adapter
    root = make_campaign(tmp_path)
    unit = dict(unit_id="final.pair_test", identity_sha256="f" * 64, kind="final_test", family="pair",
                device="cpu", depends_on=[], stage="FINAL_PAIR_TEST", test_open=True, windows={},
                calls_fit=False, cadence="NONE", identity=None)
    q = queue.Queue(root / "state" / "queue.sqlite3")
    q.ensure_unit("final.pair_test", unit, state="QUEUED")
    q.close()
    plan = campaign.load_plan(root)
    plan["units"].append(unit)
    campaign._write_plan(root, plan)
    assert not hasattr(strategy_adapter, "run_final_backtest")  # current strategy lane state
    out = run_worker(root, "coordinator", tmp_path / "c.txt")
    assert out["result"] == "WAIT_ARTIFACT" and out["cause"].startswith("FINAL_BACKTEST_EXPORT_MISSING")
    assert status.compute(root, level="unit", unit="final.pair_test")["attempts"] == []
    monkeypatch.setattr(strategy_adapter, "run_final_backtest", lambda *a, **k: None, raising=False)
    out = run_worker(root, "coordinator", tmp_path / "c.txt")
    assert not str(out.get("cause", "")).startswith("FINAL_BACKTEST_EXPORT_MISSING")
