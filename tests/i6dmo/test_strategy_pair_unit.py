"""Pair-stage hook ``tools.i6dmo.strategy_adapter.run_unit`` against the ops contract.

Fake family outputs (synthetic predictions, bars and calendar) only: no fit,
no TEST row, no trading. The owner-rulings gate is recorded with the ops lane's
own ``campaign.record_gate`` so the digest/binding rules are the real ones.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import campaign as C
from tools.i6dmo import strategy_adapter as sa
from tools.i6dmo import unit_runner
from tools.i6dmo import worker as W

UTC = timezone.utc
H = timedelta(hours=1)
P0 = 1.10000
O = datetime(2024, 1, 7, 22, tzinfo=UTC)
CL = datetime(2024, 1, 12, 22, tzinfo=UTC)
K = datetime(2024, 1, 11, 5, tzinfo=UTC)
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
ENTRY_LONG = [P0 + d for d in (0.0010, 0.0012, 0.0015, 0.0011, 0.0013, 0.0014)]

RULINGS = {
    "barrier_equality": "INCLUSIVE_TOUCH",
    "costs_and_pip_value": {"pip_cost": 0.00001, "spread_price": 0.00008, "slippage_price": 0.00002,
                            "commission_per_lot_round_trip": 7.0, "swap_per_lot_per_day": 10.0},
    "fill_model": {"fill": "MARKET_ORDER_AT_OPEN_OF_NEXT_AVAILABLE_BAR",
                   "intrabar_both_barriers": "STOP_LOSS_FIRST", "force_close_lead_hours": 1},
    "sharpe_convention": {"return_clock": "UTC_DAY_LAST_MARK_TO_MARKET", "periods_per_year": 260},
    "min_long_heads": 2,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _origins():
    t, out = O + H, []
    while t <= CL:
        out.append(t)
        t += H
    return out


def _stored_terminal(root, unit_id, family, better):
    ident = hashlib.sha256(unit_id.encode()).hexdigest()
    folder = root / "terminals" / ident
    (folder / "out" / "predictions").mkdir(parents=True)
    origins = _origins()
    hs = SHORT if family == "short" else LONG
    values = [[P0] * 6 if family == "short" else list(ENTRY_LONG) for _ in origins]
    np.savez(folder / "out" / "predictions" / "validation.npz",
             row_id=np.array([int(t.timestamp()) for t in origins], dtype=np.int64),
             horizons_hours=np.array(hs), predicted_price=np.array(values, dtype=float),
             mask=np.ones((len(origins), 6), dtype=bool))
    pred = folder / "out" / "predictions" / "validation.npz"
    child = dict(schema="i6dmo.terminal.v1", unit_id=unit_id, identity_sha256=ident, status="COMPLETE",
                 disposition="COMPLETE", costs={},
                 artifacts={"validation_predictions": {"path": "predictions/validation.npz",
                                                       "sha256": _sha(pred)}},
                 eligibility={str(h): {"mae": 0.9 if h in better else 1.0, "naive": 1.0,
                                       "eligible": h in better, "n": 50} for h in hs},
                 gates={}, reports=[])
    (folder / "TERMINAL.json").write_text(json.dumps(dict(
        schema="i6dmo.stored_terminal.v1", unit_id=unit_id, identity_sha256=ident, terminal=child)))
    path = folder / "TERMINAL.json"
    return dict(path=str(path), sha256=_sha(path), terminal_sha256="t" * 64)


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "campaign"
    (root / "gates").mkdir(parents=True)
    (root / "PLAN.json").write_text(json.dumps(dict(campaign_id="i6dmo-fake", effective_config_sha256="e" * 64)))
    data = tmp_path / "data"
    data.mkdir()
    times = _origins()
    np.savez(data / "bars.npz", close_time=np.array([int(t.timestamp()) for t in times], dtype=np.int64),
             open=np.full(len(times), P0), high=np.full(len(times), P0 + 0.00005),
             low=np.full(len(times), P0 - 0.00005), close=np.full(len(times), P0))
    (data / "calendar.json").write_text(json.dumps({
        "market_calendar": {"calendar_id": "synthetic.assumed_fx.test", "version": "v1",
                            "weeks": [{"week_id": "2024-W02", "market_open": O.isoformat(),
                                       "market_close": CL.isoformat(), "entry_cutoff": K.isoformat()}]},
        "entry_delay_after_open_hours": 12, "force_flat_before_close_hours": 12}))
    effective = dict(
        code_commit="c" * 40,
        design={"splits": {"validation": {"start": "2024-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
                           "test": {"start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z",
                                    "open": False}}},
        data_pins={"strategy_bars": {"relpath": "bars.npz", "sha256": _sha(data / "bars.npz")},
                   "strategy_calendar": {"relpath": "calendar.json", "sha256": _sha(data / "calendar.json")}})
    spec = dict(schema="i6dmo.unit_spec.v1", unit_id="pair.validation", kind="pair", family="pair",
                identity_sha256="i" * 64, campaign_id="i6dmo-fake", test_open=False, data_root=str(data),
                effective_config=effective, effective_config_sha256="e" * 64,
                upstream={"short.report": _stored_terminal(root, "short.report", "short", {2, 3}),
                          "long.report": _stored_terminal(root, "long.report", "long", {18, 24, 72})})
    out = root / "attempts" / "pair.validation" / "a01" / "out"
    out.mkdir(parents=True)
    return dict(root=root, spec=spec, out=out, progress=out.parent / "progress.json")


def _rule(root, status="PASS", rulings=RULINGS):
    return C.record_gate(root, "STRATEGY_OWNER_RULINGS", status=status, evidence={"rulings": rulings},
                         splits_read=[])


def _run(env):
    return sa.run_unit(env["spec"], env["out"], env["progress"])


def test_pair_refused_while_owner_rulings_gate_is_absent(env):
    doc = _run(env)
    assert doc["status"] == "FAILED"
    assert doc["disposition"] == "STRATEGY_OWNER_RULINGS_OPEN:ABSENT"
    assert doc["gates"] == {"pair_status": "BLOCKED_OWNER_RULINGS"}
    assert doc["artifacts"] == {} and doc["reports"] == []
    assert not (env["out"] / "pair").exists()  # nothing computed, no business result


def test_pair_refused_while_owner_rulings_gate_is_not_pass(env):
    _rule(env["root"], status="PENDING")
    doc = _run(env)
    assert (doc["status"], doc["disposition"]) == ("FAILED", "STRATEGY_OWNER_RULINGS_OPEN:STATUS_PENDING")


def test_tampered_owner_gate_is_refused(env):
    _rule(env["root"])
    path = env["root"] / "gates" / "STRATEGY_OWNER_RULINGS.json"
    doc = json.loads(path.read_text())
    doc["evidence"]["rulings"]["min_long_heads"] = 1
    path.write_text(json.dumps(doc))
    assert _run(env)["disposition"] == "STRATEGY_OWNER_RULINGS_OPEN:RECEIPT_DIGEST_MISMATCH"


@pytest.mark.parametrize("change,code", [
    (lambda r: r.pop("min_long_heads"), "OWNER_RULINGS_INCOMPLETE:min_long_heads"),
    (lambda r: r.update(barrier_equality="STRICT_CROSS"), "RULING_NOT_IMPLEMENTED:barrier_equality"),
    (lambda r: r["fill_model"].update(intrabar_both_barriers="TAKE_PROFIT_FIRST"),
     "RULING_NOT_IMPLEMENTED:fill_model"),
    (lambda r: r["sharpe_convention"].update(return_clock="HOURLY"), "RULING_NOT_IMPLEMENTED:sharpe_convention"),
    (lambda r: r["costs_and_pip_value"].pop("spread_price"), "RULING_INVALID:costs_and_pip_value"),
])
def test_rulings_must_be_complete_and_implemented(env, change, code):
    rulings = json.loads(json.dumps(RULINGS))
    change(rulings)
    _rule(env["root"], rulings=rulings)
    doc = _run(env)
    assert (doc["status"], doc["disposition"]) == ("FAILED", code)
    assert not (env["out"] / "pair").exists()


def test_pair_runs_with_owner_ruled_parameters_and_matches_the_ops_terminal_contract(env):
    _rule(env["root"])
    doc = _run(env)
    assert doc["status"] == "COMPLETE", doc["disposition"]
    g = doc["gates"]
    assert g["pair_status"] == "PASS"
    assert g["strategy_version"] == f"{sa.STRATEGY_PIN['sha256']}:heuristic_strategy.business_pair_policy.v1"
    costs = RULINGS["costs_and_pip_value"]
    assert g["costs_sha256"] == hashlib.sha256(
        json.dumps(costs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for name, art in doc["artifacts"].items():
        assert _sha(env["out"] / art["path"]) == art["sha256"], name
    backtest = json.loads((env["out"] / doc["artifacts"]["backtest"]["path"]).read_text())
    assert backtest["execution"]["spread_price"] == costs["spread_price"]  # owner value, not default
    assert backtest["entry_parameters"]["min_valid_long_heads"] == 2
    assert backtest["metrics"]["closed_trades"] >= 1
    assert backtest["eligibility"]["eligible"] == {"short": [2, 3], "long": [18, 24, 72]}
    assert not doc["costs"].get("fit_seconds")
    for report in doc["reports"]:
        assert report["tags"]["test_read"] == "false"
        assert all(isinstance(m["value"], (int, float)) for m in report["metrics"])
    unit = dict(unit_id="pair.validation", kind="pair", identity_sha256="i" * 64)
    W._validate_child(dict(doc, schema="i6dmo.terminal.v1", unit_id="pair.validation",
                           identity_sha256="i" * 64), unit)
    progress = json.loads(env["progress"].read_text())
    assert progress["phase"] == "publish" and progress["done"] == progress["total"]


def test_ineligible_pair_returns_pair_not_eligible_without_trading(env):
    env["spec"]["upstream"]["long.report"] = _stored_terminal(env["root"], "long.report2", "long", {18})
    _rule(env["root"])
    doc = _run(env)
    assert doc["status"] == "PAIR_NOT_ELIGIBLE"
    assert doc["gates"]["pair_status"] == "PAIR_NOT_ELIGIBLE"
    assert doc["disposition"] == "LONG_ELIGIBLE_HEADS_1_LT_2"
    assert "backtest" not in doc["artifacts"]


def test_tampered_upstream_and_missing_input_pins_are_typed_failures(env):
    _rule(env["root"])
    up = env["spec"]["upstream"]["short.report"]
    env["spec"]["upstream"]["short.report"] = dict(up, sha256="0" * 64)
    assert _run(env)["disposition"] == "UPSTREAM_DIGEST_MISMATCH:short.report"
    env["spec"]["upstream"]["short.report"] = up
    pins = env["spec"]["effective_config"]["data_pins"]
    del pins["strategy_bars"]
    assert _run(env)["disposition"] == "STRATEGY_INPUT_PIN_MISSING:strategy_bars"


def test_test_reads_are_refused(env):
    _rule(env["root"])
    env["spec"]["test_open"] = True
    assert _run(env)["disposition"] == "TEST_OPEN_REFUSED"
    env["spec"]["test_open"] = False
    env["spec"]["kind"] = "final_test"
    assert _run(env)["disposition"] == "FINAL_TEST_HOOK_NOT_PREPARED"


def test_unit_runner_cli_writes_the_terminal(env, capsys):
    _rule(env["root"])
    spec_path = env["out"].parent / "UNIT_SPEC.json"
    spec_path.write_text(json.dumps(env["spec"]))
    rc = unit_runner.main(["--kind", "pair", "run-unit", "--spec", str(spec_path), "--out", str(env["out"]),
                           "--progress", str(env["progress"])])
    assert rc == 0
    term = json.loads((env["out"] / "TERMINAL.json").read_text())
    assert term["schema"] == "i6dmo.terminal.v1" and term["status"] == "COMPLETE"
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["terminal_sha256"]
