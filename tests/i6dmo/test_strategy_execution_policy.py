"""AUD-DMO-06 correction: sealed execution-policy receipt and unit-conversion audit.

The receipt is derived automatically from a versioned instrument configuration
plus the established business contract; it never waits for an approval. The
conversion tests run against the ACTUAL pinned heuristic-strategy policy
(blob 79b6df7a at d40fd3b0), not a fake.
"""

from __future__ import annotations

import json
import math

import pytest

from tools.i6dmo import strategy_adapter as sa

INSTRUMENT = {
    "instrument_id": "EURUSD",
    "config_version": "i6dmo-eurusd-exec-2026-10-14",
    "price_decimals": 5,
    "point_size": 0.00001,
    "pip_size": 0.0001,
    "pip_cost_parameter": 0.00001,
    "lot_units": 100000.0,
    "costs": {"spread_price": 0.00002, "slippage_price": 0.00001,
              "commission_per_lot_round_trip": 7.0, "swap_per_lot_per_day": 10.0},
    "cost_verification": "NOT_BROKER_VERIFIED",
}


@pytest.fixture(scope="module")
def policy():
    return sa.load_pinned_policy()


@pytest.fixture(scope="module")
def receipt(policy):
    return sa.derive_execution_policy(INSTRUMENT, policy=policy)


def test_receipt_is_sealed_and_fixes_the_audited_values(receipt, policy):
    assert receipt["schema"] == "i6dmo.execution_policy.v1"
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == sa._canonical_sha(body)
    rules = receipt["business_rules"]
    # Minimum longs is taken from the existing pinned policy (2), not invented here.
    assert rules["min_valid_long_heads"] == policy.EntryParameters().min_valid_long_heads == 2
    assert rules["barrier_equality"] == "INCLUSIVE_TOUCH"
    assert rules["initial_equity"] == 10000.0
    assert rules["entry_family"] == "LONG_FAMILY_ONLY"
    sim = receipt["simulation_assumptions"]
    assert sim["fill"] == "MARKET_ORDER_AT_OPEN_OF_NEXT_AVAILABLE_BAR"
    assert sim["intrabar_both_barriers"] == "STOP_LOSS_FIRST"
    assert "SIMULATION_ASSUMPTION" in sim["statement"]
    rep = receipt["reporting"]
    assert rep == {"return_clock": "UTC_DAY_LAST_MARK_TO_MARKET", "periods_per_year": 260}
    assert receipt["instrument"]["cost_verification"] == "NOT_BROKER_VERIFIED"
    assert receipt["pip_semantics"]["pip_cost_parameter_unit"] == "POINT"
    assert receipt["policy_pin"]["sha256"] == sa.STRATEGY_PIN["sha256"]
    # Deterministic: the same configuration gives the same seal, no prompt involved.
    assert sa.derive_execution_policy(INSTRUMENT, policy=policy)["receipt_sha256"] == \
        receipt["receipt_sha256"]


@pytest.mark.parametrize("change,code", [
    (lambda i: i.pop("point_size"), "INSTRUMENT_CONFIG_INVALID"),
    (lambda i: i.update(config_version=""), "INSTRUMENT_CONFIG_INVALID"),
    (lambda i: i.update(pip_size=0.001), "PIP_POINT_INCONSISTENT"),
    (lambda i: i.update(pip_cost_parameter=0.0005), "PIP_COST_PARAMETER_UNRECOGNIZED"),
    (lambda i: i["costs"].update(spread_price=-1.0), "INSTRUMENT_CONFIG_INVALID"),
    (lambda i: i["costs"].pop("swap_per_lot_per_day"), "INSTRUMENT_CONFIG_INVALID"),
    (lambda i: i.update(cost_verification="BROKER_VERIFIED"), "COST_VERIFICATION_INVALID"),
    (lambda i: i.update(cost_verification="LIVE"), "COST_VERIFICATION_INVALID"),
    (lambda i: i.update(extra_field=1), "INSTRUMENT_CONFIG_INVALID"),
])
def test_invalid_instrument_configuration_is_refused(policy, change, code):
    instrument = json.loads(json.dumps(INSTRUMENT))
    change(instrument)
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.derive_execution_policy(instrument, policy=policy)
    assert e.value.code == code


def test_broker_verified_costs_need_evidence(policy):
    instrument = json.loads(json.dumps(INSTRUMENT))
    instrument["cost_verification"] = {"status": "BROKER_VERIFIED",
                                       "evidence": "broker statement sha256 deadbeef"}
    r = sa.derive_execution_policy(instrument, policy=policy)
    assert r["instrument"]["cost_verification"]["status"] == "BROKER_VERIFIED"
    instrument["cost_verification"] = {"status": "BROKER_VERIFIED", "evidence": ""}
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.derive_execution_policy(instrument, policy=policy)
    assert e.value.code == "COST_VERIFICATION_INVALID"


def test_seal_load_and_tamper(tmp_path, policy):
    receipt = sa.seal_execution_policy(tmp_path, INSTRUMENT, policy=policy)
    loaded = sa.load_execution_policy(tmp_path)
    assert loaded == receipt
    path = tmp_path / "gates" / "EXECUTION_POLICY.json"
    doc = json.loads(path.read_text())
    doc["business_rules"]["min_valid_long_heads"] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.load_execution_policy(tmp_path)
    assert e.value.code == "EXECUTION_POLICY_TAMPERED"
    assert sa.load_execution_policy(tmp_path / "absent") is None


def test_execution_parameters_from_receipt_use_configured_costs(receipt):
    params = sa.execution_from_policy(receipt)
    assert isinstance(params["execution"], sa.ExecutionParameters)
    ex = params["execution"]
    assert (ex.spread_price, ex.slippage_price) == (0.00002, 0.00001)
    assert (ex.commission_per_lot_round_trip, ex.swap_per_lot_per_day) == (7.0, 10.0)
    assert ex.initial_equity == 10000.0 and ex.lot_units == 100000.0
    assert ex.sharpe_periods_per_year == 260 and ex.force_close_lead_hours == 1
    entry = params["entry"]
    assert entry["pip_cost"] == 0.00001 and entry["min_valid_long_heads"] == 2
    assert params["costs_sha256"] == sa._canonical_sha(INSTRUMENT["costs"])


# ---------------------------------------------------------------------------
# Point/pip/lot conversion and sizing audit at non-unit prices (pinned policy)


@pytest.mark.parametrize("price", [1.10, 0.95])
def test_entry_geometry_distances_are_in_configured_point_units(policy, price):
    params = policy.EntryParameters(**sa.DEFAULT_ENTRY_PARAMETERS)
    highest, lowest = price + 200e-5, price - 30e-5  # +200 / -30 points
    longs = [policy.HeadForecast("long", h, True, v) for h, v in zip(
        sa.LONG_HORIZONS_HOURS, [highest, price, price, price, price, lowest])]
    d = policy.entry_decision(current_price=price, long_heads=longs, params=params)
    assert d.action == "OPEN_LONG"
    g = d.geometry
    # "pips" in the pinned policy are POINTS of 0.00001, independent of price level.
    assert math.isclose(g.ideal_profit_pips, 200.0, rel_tol=1e-9)
    assert math.isclose(g.ideal_drawdown_pips, 30.0, rel_tol=1e-9)
    assert math.isclose(g.take_profit_price - price, 0.9 * 200e-5, rel_tol=1e-9)
    assert math.isclose(price - g.stop_loss_price, 2.0 * 30e-5, rel_tol=1e-9)


def test_profit_threshold_is_in_points_not_pips(policy):
    # 5 "pips" at pip_cost 0.00001 is 5 points = 0.5 real pips: an edge of 6 points enters.
    params = policy.EntryParameters(**sa.DEFAULT_ENTRY_PARAMETERS)
    price = 1.10
    longs = [policy.HeadForecast("long", h, True, price + (6e-5 if h == 18 else 0))
             for h in sa.LONG_HORIZONS_HOURS]
    d = policy.entry_decision(current_price=price, long_heads=longs, params=params)
    assert d.action == "OPEN_LONG" and math.isclose(d.geometry.ideal_profit_pips, 6.0)


@pytest.mark.parametrize("price", [1.10, 0.95])
def test_sizing_is_base_currency_units_without_price_conversion(policy, price):
    params = policy.EntryParameters(**sa.DEFAULT_ENTRY_PARAMETERS)
    units = policy.compute_order_units(reward_risk_ratio=3.0, available_cash=10000.0,
                                       params=params)
    assert units == 20000.0  # 10000 * 0.02 * 100, identical at 1.10 and 0.95
    # The notional therefore differs with price; a SIMULATION_ASSUMPTION, not broker logic.
    assert math.isclose(units * 1.10, 22000.0) and math.isclose(units * 0.95, 19000.0)


@pytest.mark.parametrize("price", [1.10, 0.95])
def test_costs_scale_with_units_and_lots_not_with_price(receipt, price):
    ex = sa.execution_from_policy(receipt)["execution"]
    book = sa._Book(ex)
    units = 20000.0
    assert math.isclose(book.commission_per_fill(units), 7.0 / 2 * units / 100000.0)
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2024, 1, 8, 10, tzinfo=timezone.utc)
    swap = book.swap(units, t0, t0 + timedelta(hours=48))
    assert math.isclose(swap, 10.0 * (units / 100000.0) * 2.0)  # per lot per day, price-free
    buy = book.fill_price(price, True)
    assert math.isclose(buy - price, 0.00002 / 2 + 0.00001, rel_tol=1e-9)
