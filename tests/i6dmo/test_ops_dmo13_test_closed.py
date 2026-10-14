"""DMO13: TEST cannot be opened by init/run/deploy; freeze/provenance/pair receipts are required."""
import json

import pytest

from tests.i6dmo.ops_fakes import FakeWarehouse, make_campaign, run_worker

EXTERNAL = ("DATA_PROVENANCE", "SOFTWARE_TRACEABILITY")


def _rulings(root):
    from tools.i6dmo import campaign
    campaign.record_gate(root, "STRATEGY_OWNER_RULINGS", status="PASS",
                         evidence={r: "ruled" for r in campaign.OWNER_RULINGS}, splits_read=[])


def _validate_everything(root, counter, pair_mode="ok"):
    from tools.i6dmo import campaign
    _rulings(root)
    for role, mode in (("preferred", "ok"), ("coordinator", "ok"), ("preferred", "ok"), ("coordinator", "ok"),
                       ("coordinator", pair_mode)):
        campaign.tick(root)
        out = run_worker(root, role, counter, mode=mode)
        assert out["result"] == "COMPLETE", out
    campaign.publish(root, warehouse=FakeWarehouse())
    return campaign.tick(root)


def test_dmo13_init_refuses_an_open_test_design(tmp_path):
    from tools.i6dmo import campaign
    with pytest.raises(campaign.Refused, match="TEST_OPEN_IN_DESIGN"):
        make_campaign(tmp_path, splits__test__open=True)


def test_dmo13_init_and_deploy_never_open_test(tmp_path):
    from tools.i6dmo import campaign, deploy
    root = make_campaign(tmp_path)
    plan = campaign.load_plan(root)
    assert plan["test_units_enumerated"] is False
    for unit in plan["units"]:
        assert unit["test_open"] is False and "test" not in unit["windows"]
    rendered = deploy.render_units(root, ["coordinator", "preferred"])
    assert all("--open-test" not in text for text in rendered.values())
    effective = root / "EFFECTIVE_CONFIG.json"
    doc = json.loads(effective.read_text())
    doc["design"]["splits"]["test"]["open"] = True  # tampering breaks the seal
    effective.write_text(json.dumps(doc))
    with pytest.raises(campaign.Refused, match="EFFECTIVE_CONFIG_SEAL"):
        deploy.render_units(root, ["coordinator"])


def test_dmo13_readiness_blocks_with_missing_artifacts_and_owners(tmp_path):
    from tools.i6dmo import campaign
    root = make_campaign(tmp_path)
    _validate_everything(root, tmp_path / "calls.txt")
    ready = campaign.readiness(root)
    assert ready["state"] == "BLOCKED"
    missing = {m["gate"]: m for m in ready["missing"]}
    assert set(EXTERNAL) <= set(missing) and all(missing[g]["owner"] and missing[g]["artifact"] for g in EXTERNAL)
    assert not [u for u in campaign.load_plan(root)["units"] if u["kind"] in ("final_refit", "final_test")]
    for gate in EXTERNAL:  # bound to another campaign -> still blocked
        campaign.record_gate(root, gate, status="PASS", evidence={"x": 1}, splits_read=["train", "validation"],
                             effective_config_sha256="0" * 64)
    assert campaign.readiness(root)["state"] == "BLOCKED"


def test_dmo13_all_receipts_open_test_only_through_final_units(tmp_path):
    from tools.i6dmo import campaign
    root = make_campaign(tmp_path)
    _validate_everything(root, tmp_path / "calls.txt")
    for gate in EXTERNAL:
        campaign.record_gate(root, gate, status="PASS", evidence={"receipt": gate},
                             splits_read=["train", "validation"])
    campaign.tick(root)
    assert campaign.readiness(root)["state"] == "READY"
    units = {u["unit_id"]: u for u in campaign.load_plan(root)["units"]}
    assert units["final.short_refit"]["kind"] == "final_refit" and not units["final.short_refit"]["test_open"]
    assert units["final.short_refit"]["windows"]["train"] == ["2021-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    assert units["final.pair_test"]["test_open"] is True
    assert set(units["final.pair_test"]["depends_on"]) == {"final.short_refit", "final.long_refit"}


def test_dmo13_pair_not_eligible_is_terminal_and_never_opens_test(tmp_path):
    from tools.i6dmo import campaign
    root = make_campaign(tmp_path)
    _validate_everything(root, tmp_path / "calls.txt", pair_mode="ineligible")
    for gate in EXTERNAL:
        campaign.record_gate(root, gate, status="PASS", evidence={"receipt": gate},
                             splits_read=["train", "validation"])
    campaign.tick(root)
    ready = campaign.readiness(root)
    assert ready["state"] == "TERMINAL" and ready["disposition"] == "PAIR_NOT_ELIGIBLE"
    assert not [u for u in campaign.load_plan(root)["units"] if u["test_open"]]


def test_dmo13_pair_waits_for_owner_rulings_without_failing(tmp_path):
    from tools.i6dmo import campaign, status
    root = make_campaign(tmp_path)
    counter = tmp_path / "calls.txt"
    for role in ("preferred", "coordinator", "preferred", "coordinator"):
        campaign.tick(root)
        assert run_worker(root, role, counter)["result"] == "COMPLETE"
    campaign.tick(root)
    out = run_worker(root, "coordinator", counter)
    assert out["result"] == "WAIT_OWNER_RULING" and "barrier_equality" in out["cause"]
    unit = status.compute(root, level="unit", unit="pair.validation")
    assert unit["state"] == "WAIT_OWNER_RULING" and not unit["attempts"]
    missing = {m["gate"] for m in campaign.readiness(root)["missing"]}
    assert "STRATEGY_OWNER_RULINGS" in missing
    _rulings(root)
    assert run_worker(root, "coordinator", counter)["result"] == "COMPLETE"
