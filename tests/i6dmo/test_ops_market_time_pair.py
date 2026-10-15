"""Market-time pair campaign (order 50d20d01, plan sections 4 and 6; ops scope).

One finite graph short.fit -> short.report -> long.fit -> long.report -> pair.eligibility ->
strategy.validation (only when the pair qualifies) -> publication/mirror closure. The long family
is measured even when short eligibility fails; a disqualified pair closes PAIR_NOT_ELIGIBLE; no
TEST unit is enumerable or claimable; old elapsed-grid terminals are never adopted.
"""
import json

import pytest

from tests.i6dmo.ops_fakes import fit_count, make_market_time_campaign, run_worker


def _ids(root):
    from tools.i6dmo import campaign
    return [u["unit_id"] for u in campaign.load_plan(root)["units"]]


def _next_claimable(root):
    from tools.i6dmo import campaign
    q = campaign.open_queue(root)
    try:
        return {u["unit_id"] for u in q.units() if u["state"] in ("QUEUED", "WAIT_RESOURCE")}
    finally:
        q.close()


def _drive(root, counter, *, report_mode="ok", pair_mode="ok"):
    """Advance the finite graph with the fake trainers; returns the per-unit end states."""
    from tools.i6dmo import campaign, status
    for _ in range(12):
        campaign.tick(root)
        progressed = False
        for role in ("preferred", "coordinator"):
            claimable = _next_claimable(root)
            mode = pair_mode if "pair.eligibility" in claimable else report_mode
            if run_worker(root, role, counter, mode=mode)["result"] != "IDLE":
                progressed = True
        if not progressed:
            break
    campaign.tick(root)
    return {u: status.compute(root, level="unit", unit=u)["state"] for u in _ids(root)}


def test_market_time_graph_is_finite_and_has_no_test_unit(tmp_path):
    from tools.i6dmo import campaign
    root = make_market_time_campaign(tmp_path)
    plan = campaign.load_plan(root)
    assert _ids(root) == ["short.fit", "short.report", "long.fit", "long.report",
                          "pair.eligibility", "strategy.validation"]
    units = {u["unit_id"]: u for u in plan["units"]}
    assert units["long.fit"]["depends_on"] == ["short.report"]
    assert units["pair.eligibility"]["depends_on"] == ["short.report", "long.report"]
    assert units["strategy.validation"]["depends_on"] == ["pair.eligibility"]
    assert all(u["identity_sha256"] for u in plan["units"])  # both designs sealed up front
    assert plan["mode"] == "MARKET_TIME_PAIR" and plan["test_units_enumerated"] is False
    assert all(u["test_open"] is False and "test" not in u["windows"] for u in plan["units"])
    assert plan["expected"]["final_units"] == "NOT_ENUMERABLE_IN_MARKET_TIME_PAIR"


def test_long_family_runs_even_when_short_is_not_eligible(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    root = make_market_time_campaign(tmp_path)
    states = _drive(root, counter, report_mode="ineligible")
    short_report = campaign.open_queue(root)
    try:
        term = short_report.terminal("short.report")
    finally:
        short_report.close()
    assert not any(h["eligible"] for h in term["terminal"]["eligibility"].values())
    assert states["long.fit"] == "COMPLETE" and states["long.report"] == "COMPLETE"
    assert fit_count(counter) == 2  # the long fit really ran after a failed short eligibility
    assert states["strategy.validation"] == "BLOCKED"


def test_disqualified_pair_closes_with_a_typed_disposition_and_no_strategy_run(tmp_path):
    from tools.i6dmo import campaign, status
    counter = tmp_path / "c.txt"
    root = make_market_time_campaign(tmp_path)
    states = _drive(root, counter, pair_mode="ineligible")
    assert states["pair.eligibility"] == "COMPLETE"
    unit = status.compute(root, level="unit", unit="strategy.validation")
    assert unit["state"] == "BLOCKED" and unit["disposition"] == "PAIR_NOT_ELIGIBLE"
    assert unit["attempts"] == []  # the strategy never ran on a disqualified pair
    assert "strategy.validation" not in counter.read_text()
    ready = campaign.readiness(root)
    assert ready["state"] == "TERMINAL" and ready["disposition"] == "PAIR_NOT_ELIGIBLE"


def test_qualified_pair_runs_the_validation_strategy(tmp_path):
    from tools.i6dmo import status
    counter = tmp_path / "c.txt"
    root = make_market_time_campaign(tmp_path)
    states = _drive(root, counter)
    assert states["pair.eligibility"] == "COMPLETE" and states["strategy.validation"] == "COMPLETE"
    assert "strategy:strategy.validation" in counter.read_text()
    assert status.compute(root, level="campaign")["counts"]["COMPLETE"] == 6


def test_no_test_unit_can_be_enumerated_or_claimed(tmp_path):
    from tools.i6dmo import campaign, queue
    counter = tmp_path / "c.txt"
    root = make_market_time_campaign(tmp_path)
    _drive(root, counter)
    for gate in ("DATA_PROVENANCE", "SOFTWARE_TRACEABILITY"):
        campaign.record_gate(root, gate, status="PASS", evidence={"x": 1},
                             splits_read=["train", "validation"])
    out = campaign.tick(root)
    assert out["finals_enumerated"] is False
    assert not [u for u in campaign.load_plan(root)["units"] if u["kind"] in ("final_refit", "final_test")]
    q = queue.Queue(root / "state" / "queue.sqlite3")
    try:
        with pytest.raises(campaign.Refused, match="TEST_NOT_ENUMERABLE_IN_MARKET_TIME_PAIR"):
            campaign.enumerate_test_units(root)
        q.ensure_unit("final.pair_test", dict(identity_sha256="f" * 64, kind="final_test", family="pair",
                                              device="cpu", depends_on=[], stage="FINAL_PAIR_TEST",
                                              test_open=True), state="QUEUED")
    finally:
        q.close()
    assert run_worker(root, "coordinator", counter)["result"] == "IDLE"  # not in the sealed plan


def test_an_elapsed_grid_terminal_is_never_adopted_by_shape_or_horizon_match(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    donor = make_market_time_campaign(tmp_path, name="ds2", rule="DS2_EXACT")
    assert run_worker(donor, "preferred", counter)["result"] == "COMPLETE"
    donor_identity = next(p.name for p in (donor / "terminals").iterdir())
    root = make_market_time_campaign(tmp_path, name="mg2", prior_roots=[donor])
    target = {u["unit_id"]: u for u in campaign.load_plan(root)["units"]}["short.fit"]
    assert target["identity_sha256"] != donor_identity  # different rule -> different identity
    # Forge the donor terminal under the market-time identity: same shapes, same horizons.
    forged = donor / "terminals" / target["identity_sha256"]
    (donor / "terminals" / donor_identity).rename(forged)
    assert run_worker(root, "preferred", counter)["result"] == "COMPLETE"
    assert fit_count(counter) == 2  # refitted; the elapsed-grid terminal was refused
    assert any(e["event"] == "PRIOR_REJECTED" for e in campaign.events(root))


def test_a_technical_failure_is_a_typed_terminal_without_retries(tmp_path):
    from tools.i6dmo import status
    counter = tmp_path / "c.txt"
    root = make_market_time_campaign(tmp_path)
    assert run_worker(root, "preferred", counter, mode="fail")["result"] == "FAILED"
    unit = status.compute(root, level="unit", unit="short.fit")
    assert unit["state"] == "FAILED" and len(unit["attempts"]) == 1
    assert run_worker(root, "preferred", counter)["result"] in ("IDLE", "REFUSED")
    assert fit_count(counter) == 1  # no retry of a typed failure
    assert status.compute(root, level="campaign")["state"] == "FAILED"
