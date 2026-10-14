"""`run_final_backtest(spec, predictions, execution_params, out_dir)` (lane 2 delegation).

Synthetic predictions and bars only — the TEST window here is a synthetic 2025
week, never real TEST rows. The policy is the actual pinned blob. The export
consumes ONLY the execution parameters passed in (no re-derivation), binds the
final identities into its receipt, refuses typed, and never fits.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import strategy_adapter as sa
from tools.i6n_contracts import eligibility as c1

UTC = timezone.utc
H = timedelta(hours=1)
P0 = 1.10000
O = datetime(2025, 1, 5, 22, tzinfo=UTC)
CL = datetime(2025, 1, 10, 22, tzinfo=UTC)
K = datetime(2025, 1, 9, 5, tzinfo=UTC)
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
ENTRY_LONG = [P0 + d for d in (0.0010, 0.0012, 0.0015, 0.0011, 0.0013, 0.0014)]
READINESS = "r" * 64


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _origins():
    out, t = [], O + H
    while t <= CL:
        out.append(t)
        t += H
    return out


def _write_family(out_dir, family):
    folder = Path(out_dir) / f"predictions_{family}"
    folder.mkdir(parents=True, exist_ok=True)
    origins = _origins()
    hs = SHORT if family == "short" else LONG
    values = [([P0] * 6 if family == "short" else list(ENTRY_LONG)) for _ in origins]
    np.savez(folder / "test.npz",
             row_id=np.array([int(t.timestamp()) for t in origins], dtype=np.int64),
             horizons_hours=np.array(hs, dtype=np.int64),
             origin_price=np.full(len(origins), P0),
             predicted_price=np.array(values, dtype=float),
             target_price=np.array(values, dtype=float),
             mask=np.ones((len(origins), 6), dtype=bool))
    return {"schema": "i6dmo.final_test_predictions.v1", "rows": len(origins),
            "horizons_hours": hs, "weights_sha256": f"{family[0]}w" * 32,
            "scales_sha256": "s" * 64, "design_sha256": f"{family[0]}d" * 32,
            "readiness_sha256": READINESS,
            "predictions_sha256": _sha(folder / "test.npz"), "trained": False}


@pytest.fixture
def env(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    times = _origins()
    np.savez(data / "bars.npz",
             close_time=np.array([int(t.timestamp()) for t in times], dtype=np.int64),
             open=np.full(len(times), P0), high=np.full(len(times), P0 + 0.00005),
             low=np.full(len(times), P0 - 0.00005), close=np.full(len(times), P0))
    (data / "calendar.json").write_text(json.dumps({
        "market_calendar": {"calendar_id": "synthetic.assumed_fx.test", "version": "v1",
                            "weeks": [{"week_id": "2025-W02", "market_open": O.isoformat(),
                                       "market_close": CL.isoformat(),
                                       "entry_cutoff": K.isoformat()}]},
        "entry_delay_after_open_hours": 12, "force_flat_before_close_hours": 12}))
    predictions = {f: _write_family(out, f) for f in ("short", "long")}
    spec = {"schema": "i6dmo.unit_spec.v1", "unit_id": "final.pair_test", "kind": "final_test",
            "family": "pair", "identity_sha256": "i" * 64, "campaign_id": "i6dmo-fake",
            "test_open": True, "data_root": str(data),
            "effective_config": {
                "code_commit": "c" * 40,
                "design": {"splits": {"test": {"start": "2025-01-01T00:00:00Z",
                                               "end": "2026-01-01T00:00:00Z", "open": False}}},
                "data_pins": {
                    "strategy_bars": {"relpath": "bars.npz", "sha256": _sha(data / "bars.npz")},
                    "strategy_calendar": {"relpath": "calendar.json",
                                          "sha256": _sha(data / "calendar.json")}}}}
    instrument = {
        "instrument_id": "EURUSD", "config_version": "i6dmo-eurusd-exec-2026-10-14",
        "price_decimals": 5, "point_size": 0.00001, "pip_size": 0.0001,
        "pip_cost_parameter": 0.00001, "lot_units": 100000.0,
        "costs": {"spread_price": 0.00006, "slippage_price": 0.00003,
                  "commission_per_lot_round_trip": 7.0, "swap_per_lot_per_day": 10.0},
        "cost_verification": "NOT_BROKER_VERIFIED"}
    params = sa.execution_from_policy(sa.derive_execution_policy(instrument))
    return dict(spec=spec, predictions=predictions, params=params, out=out)


def _run(env):
    return sa.run_final_backtest(env["spec"], env["predictions"], env["params"], env["out"])


def test_final_backtest_consumes_only_the_passed_sealed_params(env):
    result = _run(env)
    assert result["pair_status"] == "PASS"
    assert result["disposition"] == "FINAL_TEST_COMPLETE"
    receipt = json.loads((env["out"] / result["artifacts"]["final_backtest"]["path"]).read_text())
    # The distinct configured costs prove no re-derivation of a default policy.
    assert receipt["execution"]["spread_price"] == 0.00006
    assert receipt["execution"]["slippage_price"] == 0.00003
    assert receipt["metrics"]["initial_equity"] == 10000.0
    assert receipt["metrics"]["closed_trades"] >= 1
    assert receipt["cost_verification"] == "NOT_BROKER_VERIFIED"
    assert receipt["costs_sha256"] == env["params"]["costs_sha256"]
    # Identity-bound: weights/design/readiness/prediction digests of BOTH families.
    ident = receipt["identity"]
    for family in ("short", "long"):
        assert ident[family]["weights_sha256"] == env["predictions"][family]["weights_sha256"]
        assert ident[family]["design_sha256"] == env["predictions"][family]["design_sha256"]
        assert ident[family]["predictions_sha256"] == env["predictions"][family]["predictions_sha256"]
    assert ident["readiness_sha256"] == READINESS
    assert ident["strategy_policy_sha256"] == sa.STRATEGY_PIN["sha256"]
    assert result["artifacts"]["final_backtest"]["sha256"] == _sha(
        env["out"] / result["artifacts"]["final_backtest"]["path"])
    (report,) = result["reports"]
    assert report["tags"]["test_read"] == "true"
    assert report["tags"]["cost_verification"] == "NOT_BROKER_VERIFIED"
    assert all(m["split"] == "test" for m in report["metrics"])
    assert result.get("fit_seconds") is None  # nothing fitted


def test_params_must_be_the_sealed_execution_params_not_a_receipt(env):
    receipt = {"schema": "i6dmo.execution_policy.v1"}
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.run_final_backtest(env["spec"], env["predictions"], receipt, env["out"])
    assert e.value.code == "EXECUTION_PARAMS_INVALID"
    bad = dict(env["params"], execution="not-execution-parameters")
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.run_final_backtest(env["spec"], env["predictions"], bad, env["out"])
    assert e.value.code == "EXECUTION_PARAMS_INVALID"


def test_prediction_digest_and_identity_bindings_are_enforced(env):
    env["predictions"]["short"]["predictions_sha256"] = "0" * 64
    with pytest.raises(sa.StrategyAdapterError) as e:
        _run(env)
    assert e.value.code == "PREDICTIONS_DIGEST_MISMATCH"


def test_families_must_bind_the_same_readiness_receipt(env):
    env["predictions"]["long"]["readiness_sha256"] = "x" * 64
    with pytest.raises(sa.StrategyAdapterError) as e:
        _run(env)
    assert e.value.code == "READINESS_MISMATCH"


def test_missing_identity_field_is_typed(env):
    del env["predictions"]["long"]["weights_sha256"]
    with pytest.raises(sa.StrategyAdapterError) as e:
        _run(env)
    assert e.value.code == "IDENTITY_UNBOUND"


def test_kind_and_window_guards(env):
    env["spec"]["kind"] = "pair"
    with pytest.raises(sa.StrategyAdapterError) as e:
        _run(env)
    assert e.value.code == "KIND_INVALID"
    env["spec"]["kind"] = "final_test"
    env["spec"]["test_open"] = False
    with pytest.raises(sa.StrategyAdapterError) as e:
        _run(env)
    assert e.value.code == "TEST_NOT_OPEN"


def test_predictions_outside_the_test_window_are_refused(env):
    env["spec"]["effective_config"]["design"]["splits"]["test"]["start"] = "2025-02-01T00:00:00Z"
    with pytest.raises(sa.StrategyAdapterError) as e:
        _run(env)
    assert e.value.code == "PREDICTION_OUTSIDE_TEST_WINDOW"


def test_final_decisions_still_use_the_pinned_chronological_policy(env):
    # SL-first reversed-order counterexample inside the final path: replace the short
    # family at one management origin and see the early close appear in the receipt.
    folder = env["out"] / "predictions_short"
    with np.load(folder / "test.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    origins = _origins()
    params = env["params"]
    heads = [sa.load_pinned_policy().HeadForecast("long", h, True, v)
             for h, v in zip(LONG, ENTRY_LONG)]
    g = sa.load_pinned_policy().entry_decision(
        current_price=P0, long_heads=heads,
        params=sa.load_pinned_policy().EntryParameters(**params["entry"])).geometry
    idx = origins.index(O + 14 * H)  # first management origin after the entry fill
    arrays["predicted_price"][idx] = [g.stop_loss_price - 1e-5, P0, P0, P0, P0,
                                      g.take_profit_price + 1e-5]
    np.savez(folder / "test.npz", **arrays)
    env["predictions"]["short"]["predictions_sha256"] = _sha(folder / "test.npz")
    result = _run(env)
    receipt = json.loads((env["out"] / result["artifacts"]["final_backtest"]["path"]).read_text())
    trade = receipt["trades"][0]  # a later re-entry in the same week is legitimate
    assert trade["exit_reason"] == "EARLY_CLOSE_SL_FIRST"
    assert trade["exit_decision_time"] == (O + 14 * H).isoformat()
    assert math.isclose(receipt["metrics"]["final_equity"],
                        10000.0 + receipt["metrics"]["net_pnl"])
