"""Pair-stage hook ``tools.i6dmo.strategy_adapter.run_unit`` against the ops contract.

Fake family outputs (synthetic predictions, bars and calendar) only: no fit,
no TEST row, no trading. AUD-DMO-06: there is no owner-approval gate; the hook
derives a SEALED execution-policy receipt from the versioned instrument
configuration in the sealed effective config plus the business contract, and
refuses tampered or mismatching receipts. The policy under test is the ACTUAL
pinned heuristic-strategy blob (d40fd3b0), loaded by ``load_pinned_policy``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

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

INSTRUMENT = {
    "instrument_id": "EURUSD",
    "config_version": "i6dmo-eurusd-exec-2026-10-14",
    "price_decimals": 5,
    "point_size": 0.00001,
    "pip_size": 0.0001,
    "pip_cost_parameter": 0.00001,
    "lot_units": 100000.0,
    "costs": {"spread_price": 0.00008, "slippage_price": 0.00002,
              "commission_per_lot_round_trip": 7.0, "swap_per_lot_per_day": 10.0},
    "cost_verification": "NOT_BROKER_VERIFIED",
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
        strategy_execution={"instrument": json.loads(json.dumps(INSTRUMENT))},
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


def _run(env):
    return sa.run_unit(env["spec"], env["out"], env["progress"])


def test_pair_runs_without_any_owner_approval_gate(env):
    # AUD-DMO-06: no STRATEGY_OWNER_RULINGS gate exists and none is required.
    assert not (env["root"] / "gates" / "STRATEGY_OWNER_RULINGS.json").exists()
    doc = _run(env)
    assert doc["status"] == "COMPLETE", doc["disposition"]
    assert "owner" not in doc["disposition"].lower()


def test_without_inline_instrument_the_versioned_sourced_file_governs(env):
    # Order 85ea134e section 3: the typed wait clears exactly when the sealed
    # versioned file exists; the committed file exists, so the unit completes
    # with that file's seal bound into the receipt.
    del env["spec"]["effective_config"]["strategy_execution"]
    doc = _run(env)
    assert doc["status"] == "COMPLETE", doc["disposition"]
    file_doc = json.loads(sa.DEFAULT_INSTRUMENT_FILE.read_text())
    derived = sa.derive_execution_policy_from_file(sa.DEFAULT_INSTRUMENT_FILE)
    assert doc["gates"]["execution_policy_sha256"] == derived["receipt_sha256"]
    assert derived["instrument_config_sha256"] == file_doc["config_sha256"]
    # An explicitly named file that does not exist is the typed wait, not a prompt.
    env["spec"]["effective_config"]["strategy_execution"] = {
        "instrument_file": str(env["root"] / "absent_instrument.json")}
    env["out2"] = env["out"].parent.parent / "a02" / "out"
    env["out2"].mkdir(parents=True)
    doc2 = sa.run_unit(env["spec"], env["out2"], env["out2"].parent / "progress.json")
    assert (doc2["status"], doc2["disposition"]) == ("FAILED", "INSTRUMENT_CONFIG_MISSING")


def test_invalid_instrument_configuration_is_typed(env):
    env["spec"]["effective_config"]["strategy_execution"]["instrument"]["cost_verification"] = "LIVE"
    assert _run(env)["disposition"] == "COST_VERIFICATION_INVALID"


def test_presealed_receipt_must_match_and_tampering_is_refused(env):
    receipt = sa.seal_execution_policy(env["root"], INSTRUMENT)
    doc = _run(env)
    assert doc["status"] == "COMPLETE"
    assert doc["gates"]["execution_policy_sha256"] == receipt["receipt_sha256"]
    # Tampered sealed receipt: refused, nothing recomputed silently.
    path = env["root"] / "gates" / "EXECUTION_POLICY.json"
    body = json.loads(path.read_text())
    body["business_rules"]["min_valid_long_heads"] = 1
    path.write_text(json.dumps(body))
    assert _run(env)["disposition"] == "EXECUTION_POLICY_TAMPERED"
    # A sealed receipt for a DIFFERENT configuration than the effective config: mismatch.
    other = json.loads(json.dumps(INSTRUMENT))
    other["costs"]["spread_price"] = 0.0005
    sa.seal_execution_policy(env["root"], other)
    assert _run(env)["disposition"] == "EXECUTION_POLICY_MISMATCH"


def test_pair_runs_with_the_sealed_policy_and_matches_the_ops_terminal_contract(env):
    doc = _run(env)
    assert doc["status"] == "COMPLETE", doc["disposition"]
    assert doc["status"] == "COMPLETE", doc["disposition"]
    g = doc["gates"]
    assert g["pair_status"] == "PASS"
    assert g["strategy_version"] == f"{sa.STRATEGY_PIN['sha256']}:heuristic_strategy.business_pair_policy.v1"
    costs = INSTRUMENT["costs"]
    assert g["costs_sha256"] == hashlib.sha256(
        json.dumps(costs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert g["execution_policy_sha256"] == sa.derive_execution_policy(INSTRUMENT)["receipt_sha256"]
    for name, art in doc["artifacts"].items():
        assert _sha(env["out"] / art["path"]) == art["sha256"], name
    sealed = json.loads((env["out"] / doc["artifacts"]["execution_policy"]["path"]).read_text())
    assert sealed["receipt_sha256"] == g["execution_policy_sha256"]
    assert sealed["instrument"]["cost_verification"] == "NOT_BROKER_VERIFIED"
    backtest = json.loads((env["out"] / doc["artifacts"]["backtest"]["path"]).read_text())
    assert backtest["execution"]["spread_price"] == costs["spread_price"]  # configured, not default
    assert backtest["entry_parameters"]["min_valid_long_heads"] == 2
    assert backtest["metrics"]["closed_trades"] >= 1
    assert backtest["eligibility"]["eligible"] == {"short": [2, 3], "long": [18, 24, 72]}
    assert backtest["execution_policy_sha256"] == g["execution_policy_sha256"]
    assert not doc["costs"].get("fit_seconds")
    for report in doc["reports"]:
        assert report["tags"]["test_read"] == "false"
        assert report["tags"]["cost_verification"] == "NOT_BROKER_VERIFIED"
        assert all(isinstance(m["value"], (int, float)) for m in report["metrics"])
    unit = dict(unit_id="pair.validation", kind="pair", identity_sha256="i" * 64)
    W._validate_child(dict(doc, schema="i6dmo.terminal.v1", unit_id="pair.validation",
                           identity_sha256="i" * 64), unit)
    progress = json.loads(env["progress"].read_text())
    assert progress["phase"] == "publish" and progress["done"] == progress["total"]


def test_ineligible_pair_returns_pair_not_eligible_without_trading(env):
    env["spec"]["upstream"]["long.report"] = _stored_terminal(env["root"], "long.report2", "long", {18})
    doc = _run(env)
    assert doc["status"] == "PAIR_NOT_ELIGIBLE"
    assert doc["gates"]["pair_status"] == "PAIR_NOT_ELIGIBLE"
    assert doc["disposition"] == "LONG_ELIGIBLE_HEADS_1_LT_2"
    assert "backtest" not in doc["artifacts"]


def test_tampered_upstream_and_missing_input_pins_are_typed_failures(env):
    up = env["spec"]["upstream"]["short.report"]
    env["spec"]["upstream"]["short.report"] = dict(up, sha256="0" * 64)
    assert _run(env)["disposition"] == "UPSTREAM_DIGEST_MISMATCH:short.report"
    env["spec"]["upstream"]["short.report"] = up
    pins = env["spec"]["effective_config"]["data_pins"]
    del pins["strategy_bars"]
    assert _run(env)["disposition"] == "STRATEGY_INPUT_PIN_MISSING:strategy_bars"


def test_test_reads_are_refused(env):
    env["spec"]["test_open"] = True
    assert _run(env)["disposition"] == "TEST_OPEN_REFUSED"
    env["spec"]["test_open"] = False
    env["spec"]["kind"] = "final_test"
    assert _run(env)["disposition"] == "FINAL_TEST_HOOK_NOT_PREPARED"


def test_unit_runner_cli_writes_the_terminal(env, capsys):
    spec_path = env["out"].parent / "UNIT_SPEC.json"
    spec_path.write_text(json.dumps(env["spec"]))
    rc = unit_runner.main(["--kind", "pair", "run-unit", "--spec", str(spec_path), "--out", str(env["out"]),
                           "--progress", str(env["progress"])])
    assert rc == 0
    term = json.loads((env["out"] / "TERMINAL.json").read_text())
    assert term["schema"] == "i6dmo.terminal.v1" and term["status"] == "COMPLETE"
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["terminal_sha256"]
