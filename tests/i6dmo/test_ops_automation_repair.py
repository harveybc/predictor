"""Automation repair (recovery handoff 2026-10-15; plan SC08/SC12) on the REAL worker path.

1. pair.eligibility decides from authenticated report metrics + the sealed minimum-head policy
   only: a NEGATIVE decision is computed with NO strategy OHLC pin, through the actual worker claim,
   governed subprocess and stored-terminal verification (real sealed designs, real market bundles,
   real report writer; only the trainer is replaced by a prediction fixture that trains nothing).
2. strategy.validation is the real adapter's ``strategy`` kind with authenticated forecast
   upstreams: disqualified -> typed abstention, qualified -> the hook's real source-pin check
   refuses typed when the pins are absent (never fabricated, no trade).
3. Terminal disposition + fully delivered outbox (exact readback) + verified mirrors -> explicit
   CLOSE automatically, with immutable closure-input snapshots; a re-tick changes nothing.
4. Status shows code identity, reuse and negative decisions; a dead or failing service is a typed
   failure after a bounded threshold, never indefinite WAITING.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.i6dmo import ops_real_chain as R
from tests.i6dmo.ops_fakes import FakeWarehouse, make_campaign, make_market_time_campaign, run_worker

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def market_data(tmp_path_factory):
    return R.build_market_data(tmp_path_factory.mktemp("market_data"))


def _tree_digest(root):
    out = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _terminal(root, unit_id):
    from tools.i6dmo import campaign
    q = campaign.open_queue(root)
    try:
        return q.terminal(unit_id)
    finally:
        q.close()


# --------------------------------------------------------------------- 1 + 2 (disqualified)
@pytest.fixture(scope="module")
def negative_chain(market_data, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("negative")
    root = R.make_real_chain_campaign(tmp, market_data, name="negative")
    counter = tmp / "fits.txt"
    results = R.drive(root, env={"I6DMO_FAKE_COUNTER": str(counter)})
    return dict(root=root, results=results, counter=counter, tmp=tmp)


def test_negative_pair_decision_needs_no_strategy_pin_on_the_real_worker_path(negative_chain):
    from tools.i6dmo import campaign
    root = negative_chain["root"]
    effective = campaign.load_effective(root)
    assert "strategy_bars" not in effective["data_pins"] and "strategy_calendar" not in effective["data_pins"]
    st = R.states(root)
    assert st["short.report"][0] == st["long.report"][0] == "COMPLETE"
    assert st["pair.eligibility"][:2] == ("COMPLETE", "PAIR_NOT_ELIGIBLE")
    # the child really ran: one governed attempt, stored and verified
    term = _terminal(root, "pair.eligibility")
    assert term["source"] == "LOCAL"
    stored = root / "terminals" / term["identity_sha256"]
    from tools.i6dmo import worker
    unit = campaign.load_plan(root)["units"][4]
    assert unit["unit_id"] == "pair.eligibility"
    worker._verify_stored(stored, unit)
    decision = json.loads((stored / "out" / "pair" / "ELIGIBILITY.json").read_text())
    assert decision["schema"] == "i6dmo.pair_decision.v1"
    assert decision["status"] == "PAIR_NOT_ELIGIBLE"
    assert decision["eligible"]["short"] == [2, 3, 4, 5, 6, 12] and decision["eligible"]["long"] == []
    assert decision["missing"] == ["LONG_ELIGIBLE_HEADS_0_LT_2"]
    assert decision["rule"]["min_valid_long_heads"] == 2 and decision["rule"]["min_valid_short_heads"] == 1
    # the decision names the exact authenticated report terminals and METRICS it was computed from
    for report in ("short.report", "long.report"):
        rt = _terminal(root, report)
        assert decision["inputs"][report]["identity_sha256"] == rt["identity_sha256"]
        metrics = rt["terminal"]["artifacts"]["metrics"]["sha256"]
        assert decision["inputs"][report]["metrics_sha256"] == metrics
    # no bars/backtest artifacts; the decision report is in the terminal (enqueued by the worker)
    assert set(term["terminal"]["artifacts"]) == {"execution_policy", "eligibility"}
    assert [r["experiment_key"].rsplit(":", 1)[-1] for r in term["terminal"]["reports"]] == ["decision"]
    assert term["terminal"]["costs"]["fit_seconds"] is None


def test_disqualified_strategy_abstains_typed_without_running(negative_chain):
    from tools.i6dmo import campaign, status
    root = negative_chain["root"]
    view = status.compute(root, level="unit", unit="strategy.validation")
    assert view["state"] == "BLOCKED" and view["disposition"] == "PAIR_NOT_ELIGIBLE"
    assert view["cause"] == "PAIR_NOT_ELIGIBLE:PAIR_NOT_ELIGIBLE:LONG_ELIGIBLE_HEADS_0_LT_2"
    assert view["attempts"] == []
    assert view["decision"]["decision"] == "PAIR_NOT_ELIGIBLE" and view["decision"]["strategy_status"] == "ABSTAINED"
    ready = campaign.readiness(root)
    assert ready["state"] == "TERMINAL" and ready["disposition"] == "PAIR_NOT_ELIGIBLE"
    # only the two fixture fits; no report/pair/strategy fitted anything
    assert Path(negative_chain["counter"]).read_text().splitlines() == ["fit:short.fit", "fit:long.fit"]


def test_strategy_hook_abstains_on_a_disqualified_pair_through_the_real_runner(negative_chain):
    """Defense in depth: even if claimed, the real strategy hook abstains typed and reads no pin."""
    from tools.i6dmo import campaign, worker
    root = negative_chain["root"]
    out = _run_strategy_hook(root, negative_chain["tmp"] / "abstain")
    assert out["status"] == "PAIR_NOT_ELIGIBLE"
    assert out["disposition"] == "PAIR_NOT_ELIGIBLE:LONG_ELIGIBLE_HEADS_0_LT_2"
    assert out["gates"]["strategy_status"] == "ABSTAINED" and out["reports"] == []
    assert "backtest" not in out["artifacts"]


def _run_strategy_hook(root, folder):
    """Build the UNIT_SPEC exactly as the worker does and run the real unit_runner child."""
    from tools.i6dmo import campaign, worker
    folder = Path(folder)
    (folder / "out").mkdir(parents=True)
    plan, effective, spec = campaign.load_plan(root), campaign.load_effective(root), campaign.load_deployment(root)
    q = campaign.open_queue(root)
    try:
        row = q.unit("strategy.validation")
        unit = dict(row["spec"], identity_sha256=row["identity_sha256"])
        unit_spec = worker.build_unit_spec(root, q, plan, effective, unit, spec["roles"]["coordinator"],
                                           device_class="cpu", gpu=None, cap=1 << 30, bundle=None,
                                           attempt=dict(index=1, kind="initial"))
    finally:
        q.close()
    assert set(unit_spec["forecast_upstream"]) == {"short.report", "long.report"}
    (folder / "UNIT_SPEC.json").write_text(json.dumps(unit_spec))
    argv = worker._child_argv(effective["entrypoints"], "strategy", sys.executable, folder)
    proc = subprocess.run(argv, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads((folder / "out" / "TERMINAL.json").read_text())


# --------------------------------------------------------------------------- 2 (qualified)
@pytest.fixture(scope="module")
def qualified_chain(market_data, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qualified")
    root = R.make_real_chain_campaign(tmp, market_data, name="qualified")
    R.drive(root, env={"I6DMO_FIXTURE_ERROR_LONG": "0.1"})
    return dict(root=root, tmp=tmp)


def test_qualified_pair_queues_the_strategy_and_the_worker_keeps_its_pin_precondition(qualified_chain):
    root = qualified_chain["root"]
    st = R.states(root)
    assert st["pair.eligibility"][:2] == ("COMPLETE", "COMPLETE")
    assert _terminal(root, "pair.eligibility")["terminal"]["gates"]["pair_status"] == "ELIGIBLE"
    # trading stage: the real source-pin check holds the unit (typed, unclaimed, nothing fabricated)
    assert st["strategy.validation"][0] == "WAIT_ARTIFACT"
    assert st["strategy.validation"][2].startswith("STRATEGY_INPUT_PIN_MISSING:strategy_bars")
    from tools.i6dmo import status
    assert status.compute(root, level="unit", unit="strategy.validation")["attempts"] == []


def test_qualified_strategy_hook_reaches_its_real_pin_check_and_refuses_typed(qualified_chain):
    out = _run_strategy_hook(qualified_chain["root"], qualified_chain["tmp"] / "hook")
    assert out["status"] == "FAILED" and out["disposition"] == "STRATEGY_INPUT_PIN_MISSING:strategy_bars"
    assert "backtest" not in out["artifacts"] and out["reports"] == []
    assert out["gates"]["pair_decision_sha256"]  # it authenticated the pair decision first


def test_strategy_hook_refuses_a_forecast_upstream_that_is_not_the_decided_input(qualified_chain, tmp_path):
    """A forecast report swapped after the decision is refused before any pin or bar is read."""
    from tools.i6dmo import strategy_adapter as SA, worker, campaign
    root = qualified_chain["root"]
    plan, effective, spec = campaign.load_plan(root), campaign.load_effective(root), campaign.load_deployment(root)
    q = campaign.open_queue(root)
    try:
        row = q.unit("strategy.validation")
        unit = dict(row["spec"], identity_sha256=row["identity_sha256"])
        unit_spec = worker.build_unit_spec(root, q, plan, effective, unit, spec["roles"]["coordinator"],
                                           device_class="cpu", gpu=None, cap=1 << 30, bundle=None,
                                           attempt=dict(index=1, kind="initial"))
    finally:
        q.close()
    unit_spec["forecast_upstream"]["long.report"] = unit_spec["forecast_upstream"]["short.report"]
    out = SA.run_unit(unit_spec, tmp_path / "out", tmp_path / "progress.json")
    assert out["status"] == "FAILED" and out["disposition"].startswith("UPSTREAM_IDENTITY_MISMATCH:long.report")


def test_pair_refuses_report_eligibility_that_disagrees_with_its_metrics(negative_chain, tmp_path):
    """Eligibility is taken from AUTHENTICATED metrics: a doctored eligibility row is refused."""
    import shutil
    from tools.i6dmo import strategy_adapter as SA, worker, campaign
    root = negative_chain["root"]
    plan, effective, spec = campaign.load_plan(root), campaign.load_effective(root), campaign.load_deployment(root)
    q = campaign.open_queue(root)
    try:
        row = q.unit("pair.eligibility")
        unit = dict(row["spec"], identity_sha256=row["identity_sha256"])
        unit_spec = worker.build_unit_spec(root, q, plan, effective, unit, spec["roles"]["coordinator"],
                                           device_class="cpu", gpu=None, cap=1 << 30, bundle=None,
                                           attempt=dict(index=1, kind="initial"))
    finally:
        q.close()
    # forge a copy of the long report where h18 claims eligibility; re-seal every digest consistently
    src = Path(unit_spec["upstream"]["long.report"]["path"]).parent
    forged = root / "terminals" / ("f" * 64)
    shutil.copytree(src, forged)
    stored = json.loads((forged / "TERMINAL.json").read_text())
    stored["identity_sha256"] = "f" * 64
    stored["terminal"]["eligibility"]["18"].update(mae=0.0, eligible=True)
    (forged / "CHILD_TERMINAL.json").write_text(json.dumps(stored["terminal"]))
    stored["child_terminal_sha256"] = hashlib.sha256((forged / "CHILD_TERMINAL.json").read_bytes()).hexdigest()
    (forged / "TERMINAL.json").write_text(json.dumps(stored))
    unit_spec["upstream"]["long.report"] = dict(
        path=str(forged / "TERMINAL.json"),
        sha256=hashlib.sha256((forged / "TERMINAL.json").read_bytes()).hexdigest())
    try:
        out = SA.run_unit(unit_spec, tmp_path / "out", tmp_path / "progress.json")
    finally:
        shutil.rmtree(forged)
    assert out["status"] == "FAILED" and out["disposition"] == "REPORT_METRICS_DISAGREE:long.report:h18"


# ----------------------------------------------------------------------------------- 3 close
def test_negative_campaign_closes_automatically_with_retained_proofs(negative_chain):
    from tools.i6dmo import campaign, status
    root = negative_chain["root"]
    out = campaign.tick(root)
    assert out["close"] == "NOT_CLOSABLE" and "PUBLICATION_NOT_DELIVERED" in out["close_problems"]
    assert not (root / "CLOSE.json").exists()
    warehouse = FakeWarehouse()
    pub = campaign.publish(root, warehouse=warehouse)
    assert pub["publication"]["state"] == "DELIVERED"
    assert pub["close"] == "CLOSED"
    close = json.loads((root / "CLOSE.json").read_text())
    assert close["state"] == "CLOSED" and close["disposition"] == "PAIR_NOT_ELIGIBLE"
    assert close["trigger"] == "AUTOMATIC"
    inputs = close["closure_inputs"]
    assert set(inputs) == {"PUBLICATION.json", "MIRRORS.json", "REPORTS.json", "OUTBOX_READBACK.json"}
    for doc in inputs.values():
        snap = root / doc["snapshot"]
        assert hashlib.sha256(snap.read_bytes()).hexdigest() == doc["sha256"]
        assert not os.access(snap, os.W_OK) or os.geteuid() == 0
    assert inputs["PUBLICATION.json"]["sha256"] == close["publication_sha256"]
    assert inputs["MIRRORS.json"]["sha256"] == close["mirrors_sha256"]
    readback = json.loads((root / inputs["OUTBOX_READBACK.json"]["snapshot"]).read_text())
    assert readback["summary"]["pending"] == readback["summary"]["mismatch"] == 0
    assert all(i["readback_sha256"] == i["expected_rows_sha256"] for i in readback["items"])
    assert any(i["source_unit"] == "pair.eligibility" for i in readback["items"])
    # the post-close mirror pass rewrote MIRRORS.json, yet the closure's proof is intact
    mirrors = json.loads((root / "MIRRORS.json").read_text())
    assert mirrors["state"] == "VERIFIED" and mirrors["close_sha256"] == hashlib.sha256(
        (root / "CLOSE.json").read_bytes()).hexdigest()
    assert hashlib.sha256((root / "MIRRORS.json").read_bytes()).hexdigest() != close["mirrors_sha256"]
    for target in (Path(campaign.load_deployment(root)["mirrors"]["local"]),
                   Path(campaign.load_deployment(root)["mirrors"]["second_host_path"])):
        assert (target / "CLOSE.json").read_bytes() == (root / "CLOSE.json").read_bytes()
        assert (target / inputs["PUBLICATION.json"]["snapshot"]).is_file()
    assert status.compute(root, level="campaign")["state"] == "CLOSED"
    # idempotent: a re-tick, re-publish and worker change nothing
    before = _tree_digest(root)
    assert campaign.tick(root)["close"] == "ALREADY_CLOSED"
    assert campaign.publish(root, warehouse=warehouse)["result"] == "ALREADY_CLOSED_AND_MIRRORED"
    assert R.worker(root, "coordinator")["result"] == "CLOSED"
    assert _tree_digest(root) == before


def test_close_waits_for_mirrors_bound_to_this_publication_and_never_closes_failures(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    root = make_market_time_campaign(tmp_path)
    # failed unit: terminal for a human, never closed automatically
    assert run_worker(root, "preferred", counter, mode="fail")["result"] == "FAILED"
    campaign.publish(root, warehouse=FakeWarehouse())
    assert not (root / "CLOSE.json").exists()
    problems, ready, _ = campaign.closure_problems(root)
    assert any(p.startswith("DISPOSITION_NOT_AUTO_CLOSABLE:UNIT_FAILED") for p in problems)


def test_close_refuses_a_stale_publication_or_a_mismatched_readback(tmp_path):
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign
    from tools.i6dmo.outbox import Outbox
    root = make_market_time_campaign(tmp_path)
    _drive(root, tmp_path / "c.txt", pair_mode="ineligible")
    campaign.publish(root, warehouse=FakeWarehouse(corrupt=True), mirror=True)
    assert not (root / "CLOSE.json").exists()
    problems, _, _ = campaign.closure_problems(root)
    assert any(p.startswith("READBACK_NOT_EXACT") for p in problems)
    assert "PUBLICATION_NOT_DELIVERED" in problems
    root2 = make_market_time_campaign(tmp_path, name="stale")
    _drive(root2, tmp_path / "c2.txt", pair_mode="ineligible")
    # publication without mirrors: nothing verifies the copies -> no close
    campaign.publish(root2, warehouse=FakeWarehouse(), mirror=False)
    assert not (root2 / "CLOSE.json").exists()
    box = Outbox(root2 / "state" / "outbox.sqlite3")
    try:
        box.enqueue(dict(experiment_key="i6dmo:late", experiment_set_key="i6dmo:x", actor="predictor",
                         lake="LOCAL_RETAINED", lineage="UNVERIFIED", config_sha256="0" * 64, code_commit="c",
                         project="predictor", phase="I6DMO", tags={"test_read": "false"},
                         metrics=[dict(metric="Late", value=1.0, split="validation", horizon=None, unit="count")]),
                    source_unit="late")
    finally:
        box.close()
    problems, _, _ = campaign.closure_problems(root2)
    assert any(p.startswith("OUTBOX_NOT_FULLY_DELIVERED") for p in problems)
    assert campaign.tick(root2)["close"] == "NOT_CLOSABLE"


def test_positive_fixture_disposition_also_closes_automatically(tmp_path):
    """SC08 positive branch (orchestration fakes): a completed strategy closes the same way."""
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign
    root = make_market_time_campaign(tmp_path)
    _drive(root, tmp_path / "c.txt")
    assert campaign.readiness(root)["disposition"] == "MARKET_TIME_PAIR_VALIDATION_COMPLETE"
    out = campaign.publish(root, warehouse=FakeWarehouse())
    assert out["close"] == "CLOSED"
    close = json.loads((root / "CLOSE.json").read_text())
    assert close["disposition"] == "MARKET_TIME_PAIR_VALIDATION_COMPLETE" and close["trigger"] == "AUTOMATIC"


# ---------------------------------------------------------------------------------- 4 status
def test_status_shows_code_identity_reuse_and_negative_decisions(tmp_path, monkeypatch):
    from tests.i6dmo import ops_fakes
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign, recovery, status
    source = ops_fakes.make_market_time_campaign(tmp_path, name="source")
    assert ops_fakes.run_worker(source, "preferred", tmp_path / "fits.txt")["result"] == "COMPLETE"
    original = campaign.load_plan(source)["units"][0]
    monkeypatch.setattr(ops_fakes, "CODE", "a" * 40)
    monkeypatch.setattr(recovery, "reporting_only_changes", lambda *a: ["tools/i6dmo/reporting.py"])
    root = ops_fakes.make_market_time_campaign(tmp_path, name="successor", completed_short_fit_source=source)
    _drive(root, tmp_path / "fits.txt", pair_mode="ineligible")
    fit = status.compute(root, level="unit", unit="short.fit")
    assert fit["code"] == dict(unit_code_commit="c" * 40, campaign_code_commit="a" * 40,
                               relation="ORIGINAL_PIN_IMPORTED")
    assert fit["reuse"]["policy"] == "ADOPT_ONLY" and fit["reuse"]["terminal_source"] == "ADOPTED"
    assert fit["reuse"]["original_identity_sha256"] == original["identity_sha256"]
    assert fit["reuse"]["original_code_commit"] == "c" * 40 and fit["reuse"]["reporting_code_commit"] == "a" * 40
    assert fit["reuse"]["source_terminal_sha256"] and fit["reuse"]["training_repeated"] is False
    report = status.compute(root, level="unit", unit="short.report")
    assert report["code"]["relation"] == "CAMPAIGN_PIN" and report["reuse"] is None
    camp = status.compute(root, level="campaign")
    assert camp["code_identity"]["units_by_code_commit"]["c" * 40] == ["short.fit"]
    assert [r["unit_id"] for r in camp["reuse"]] == ["short.fit"]
    assert {d["unit_id"] for d in camp["decisions"]} >= {"pair.eligibility", "strategy.validation"}


def test_service_failures_are_typed_after_a_bounded_threshold(tmp_path):
    from tools.i6dmo import campaign, services, status
    root = make_market_time_campaign(tmp_path)
    campaign.tick(root)
    now = time.time()
    camp = status.compute(root, level="campaign", now=now)
    assert camp["state"] == "WAITING_FOR_WORKER" and not camp["failures"]
    # dead timer: no service ran for longer than the bounded threshold
    later = status.compute(root, level="campaign", now=now + services.SILENT_AFTER + 1)
    assert later["state"] == "SERVICE_FAILED"
    assert [f["kind"] for f in later["failures"]] == ["SERVICE_SILENT"]
    # a crashing service run is an immediate typed failure; a later good run clears it
    with pytest.raises(RuntimeError):
        with services.run(root, "publish"):
            raise RuntimeError("WAREHOUSE_ENV_UNREADABLE")
    camp = status.compute(root, level="campaign")
    assert camp["state"] == "SERVICE_FAILED"
    failure = next(f for f in camp["failures"] if f["kind"] == "SERVICE_ERROR")
    assert failure["service"] == "publish" and failure["cause"].startswith("RuntimeError:WAREHOUSE_ENV_UNREADABLE")
    with services.run(root, "publish"):
        pass
    assert status.compute(root, level="campaign")["state"] == "WAITING_FOR_WORKER"


def test_deployed_service_that_never_ran_or_whose_unit_failed_is_typed(tmp_path):
    from tools.i6dmo import campaign, deploy, services, status
    root = make_market_time_campaign(tmp_path)
    calls = []
    deploy.deploy(root, unit_dir=tmp_path / "units", systemctl=lambda *a: calls.append(a) or dict(rc=0, stdout="x"),
                  start=False)
    now = time.time()
    camp = status.compute(root, level="campaign", now=now + services.SILENT_AFTER + 5)
    never = {f["service"] for f in camp["failures"] if f["kind"] == "SERVICE_NEVER_RAN"}
    assert {"tick", "worker-preferred", "worker-coordinator"} <= never
    assert camp["state"] == "SERVICE_FAILED"
    campaign.tick(root)
    probe_units = []

    def probe(units):
        probe_units.extend(units)
        return {u: dict(active_state="failed" if "coordinator-worker" in u else "inactive",
                        result="exit-code" if "coordinator-worker" in u else "success") for u in units}
    camp = status.compute(root, level="campaign", systemd_probe=probe)
    failed = [f for f in camp["failures"] if f["kind"] == "SERVICE_UNIT_FAILED"]
    assert [f["service"] for f in failed] == ["worker-coordinator"]
    assert all(u.endswith(".service") for u in probe_units)


def test_stale_running_attempt_is_a_typed_failure_not_running_forever(tmp_path):
    from tools.i6dmo import queue, status
    root = make_campaign(tmp_path)
    now = time.time()
    q = queue.Queue(root / "state" / "queue.sqlite3", clock=lambda: now - 100)
    q.claim("short.fit", owner="w", role="preferred", host_key="host-preferred", boot_id="b",
            device_class="rtx5090", heavy=True, ttl=600)
    q.close()
    camp = status.compute(root, level="campaign", now=now + 3600)
    assert camp["state"] != "RUNNING"
    assert any(f["kind"] == "ATTEMPT_STALE" and f["unit"] == "short.fit" for f in camp["failures"])


def test_cli_status_accepts_the_systemd_probe_flag():
    proc = subprocess.run([sys.executable, "-m", "tools.i6dmo", "status", "--help"], cwd=REPO,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "--probe-systemd" in proc.stdout


# ------------------------------------------------- terminal: stop ONLY this campaign's own timers
class Recorder:
    """Injected ``systemctl --user`` runner: records calls; the real user systemd is never touched."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, *args):
        self.calls.append(args)
        return dict(rc=1 if args[-1] in self.fail else 0, stdout="failed" if args[-1] in self.fail else "")


def _deployed(tmp_path, name, **kw):
    from tools.i6dmo import deploy
    root = make_market_time_campaign(tmp_path, name=name, **kw)
    deploy.deploy(root, unit_dir=tmp_path / "units", systemctl=Recorder(), start=False)
    return root


@pytest.fixture(autouse=True)
def _real_systemctl_forbidden(monkeypatch):
    from tools.i6dmo import deploy
    monkeypatch.setattr(deploy, "default_systemctl", lambda *a: pytest.fail(f"real systemctl called: {a}"))


def test_close_stops_only_this_campaigns_own_units_with_a_durable_receipt(tmp_path):
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign, deploy, status
    root = _deployed(tmp_path, "mine")
    other = _deployed(tmp_path, "other", prior_roots=[tmp_path / "elsewhere"])  # a second live campaign
    cid, other_cid = (campaign.load_plan(r)["campaign_id"] for r in (root, other))
    assert cid != other_cid
    _drive(root, tmp_path / "c.txt", pair_mode="ineligible")
    systemctl = Recorder()
    me = f"{deploy.prefix(cid)}publish.service"
    out = campaign.publish(root, warehouse=FakeWarehouse(), systemctl=systemctl, self_unit=me)
    assert out["close"] == "CLOSED" and out["timers"] == "STOPPED"
    receipt = json.loads((root / "TIMERS_STOPPED.json").read_text())
    listed = json.loads((root / "DEPLOY.json").read_text())["units"]
    assert receipt["state"] == "STOPPED" and receipt["trigger"] == "CLOSED:PAIR_NOT_ELIGIBLE"
    assert receipt["close_sha256"] == hashlib.sha256((root / "CLOSE.json").read_bytes()).hexdigest()
    assert sorted(u["unit"] for u in receipt["units"]) == sorted(listed)
    timers = sorted(n for n in listed if n.endswith(".timer"))
    assert sorted(a[-1] for a in systemctl.calls if a[:2] == ("disable", "--now")) == timers
    assert all(a[-1].startswith(deploy.prefix(cid)) for a in systemctl.calls)
    assert not any(other_cid in a[-1] for a in systemctl.calls)
    assert ("stop", me) not in systemctl.calls  # the invoking oneshot exits by itself
    assert next(u for u in receipt["units"] if u["unit"] == me)["action"] == "SELF_NOT_STOPPED"
    mirrors = json.loads((root / "MIRRORS.json").read_text())
    assert mirrors["timers_sha256"] == hashlib.sha256((root / "TIMERS_STOPPED.json").read_bytes()).hexdigest()
    camp = status.compute(root, level="campaign")
    assert camp["state"] == "CLOSED" and camp["timers"]["state"] == "STOPPED"
    calls = len(systemctl.calls)
    before = _tree_digest(root)
    assert campaign.tick(root, systemctl=systemctl)["close"] == "ALREADY_CLOSED"
    assert campaign.publish(root, warehouse=FakeWarehouse(), systemctl=systemctl)["result"] == \
        "ALREADY_CLOSED_AND_MIRRORED"
    assert len(systemctl.calls) == calls and _tree_digest(root) == before  # idempotent


def test_a_failed_stop_is_typed_visible_and_never_alters_close(tmp_path):
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign, deploy, status
    root = _deployed(tmp_path, "mine")
    cid = campaign.load_plan(root)["campaign_id"]
    _drive(root, tmp_path / "c.txt", pair_mode="ineligible")
    tick_timer = f"{deploy.prefix(cid)}tick.timer"
    out = campaign.publish(root, warehouse=FakeWarehouse(), systemctl=Recorder(fail=[tick_timer]), self_unit="")
    assert out["close"] == "CLOSED" and out["timers"] == "TIMER_STOP_FAILED"
    close_bytes = (root / "CLOSE.json").read_bytes()
    camp = status.compute(root, level="campaign")
    assert camp["state"] == "CLOSED" and camp["close"]["disposition"] == "PAIR_NOT_ELIGIBLE"
    failure = next(f for f in camp["failures"] if f["kind"] == "TIMER_STOP_FAILED")
    assert failure["units"] == [tick_timer]
    assert next(u for u in json.loads((root / "TIMERS_STOPPED.json").read_text())["units"]
                if u["unit"] == tick_timer)["rc"] == 1
    retry = Recorder()
    assert campaign.tick(root, systemctl=retry, self_unit="")["timers"] == "STOPPED"  # retried, then final
    assert (root / "CLOSE.json").read_bytes() == close_bytes
    assert not [f for f in status.compute(root, level="campaign")["failures"] if f["kind"] == "TIMER_STOP_FAILED"]


def test_foreign_units_in_the_deploy_record_are_refused_never_touched(tmp_path):
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign
    root = _deployed(tmp_path, "mine")
    record = json.loads((root / "DEPLOY.json").read_text())
    record["units"] += ["postgres.service", "metabase.timer", "i6dmo-*.timer", "fs4-weekly.timer"]
    (root / "DEPLOY.json").write_text(json.dumps(record))
    _drive(root, tmp_path / "c.txt", pair_mode="ineligible")
    systemctl = Recorder()
    out = campaign.publish(root, warehouse=FakeWarehouse(), systemctl=systemctl, self_unit="")
    assert out["close"] == "CLOSED" and out["timers"] == "TIMER_STOP_FAILED"
    touched = {a[-1] for a in systemctl.calls}
    assert not touched & {"postgres.service", "metabase.timer", "i6dmo-*.timer", "fs4-weekly.timer"}
    receipt = json.loads((root / "TIMERS_STOPPED.json").read_text())
    assert sorted(receipt["failed"]) == sorted(["postgres.service", "metabase.timer", "i6dmo-*.timer",
                                                "fs4-weekly.timer"])


def test_a_technical_terminal_stops_its_timers_without_closing(tmp_path):
    from tools.i6dmo import campaign
    root = _deployed(tmp_path, "failed")
    assert run_worker(root, "preferred", tmp_path / "c.txt", mode="fail")["result"] == "FAILED"
    systemctl = Recorder()
    out = campaign.publish(root, warehouse=FakeWarehouse(), systemctl=systemctl, self_unit="")
    assert out["timers"] == "STOPPED" and not (root / "CLOSE.json").exists()
    receipt = json.loads((root / "TIMERS_STOPPED.json").read_text())
    assert receipt["trigger"] == "TECHNICAL_TERMINAL:UNIT_FAILED:short.fit" and receipt["close_sha256"] is None
    assert any(a[:2] == ("disable", "--now") for a in systemctl.calls)


def test_an_uninjected_systemctl_under_test_is_a_typed_failure_not_a_real_call(tmp_path, monkeypatch):
    from tests.i6dmo.test_ops_market_time_pair import _drive
    from tools.i6dmo import campaign
    root = _deployed(tmp_path, "mine")
    _drive(root, tmp_path / "c.txt", pair_mode="ineligible")
    out = campaign.publish(root, warehouse=FakeWarehouse(), self_unit="")
    assert out["close"] == "CLOSED" and out["timers"] == "TIMER_STOP_FAILED"
    receipt = json.loads((root / "TIMERS_STOPPED.json").read_text())
    assert all(u["stdout"].startswith("RuntimeError:SYSTEMCTL_NOT_INJECTED_UNDER_TEST") for u in receipt["units"]
               if u.get("rc") is None and str(u.get("action", "")).startswith("systemctl"))
