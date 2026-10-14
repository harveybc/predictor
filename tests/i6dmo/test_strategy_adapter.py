"""I6-DMO strategy lane: DMO14 (masks, chronological TP/SL, no head filling)
and DMO15 (fill-based force-flat, no invented stop() fill, one position).

Synthetic prices and calendar only. No market data, no fit, no TEST rows.
The strategy policy is the pinned heuristic-strategy blob loaded from the git
object store (see tools/i6dmo/strategy_adapter.py STRATEGY_PIN).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from tools.i6dmo import strategy_adapter as sa
from tools.i6n_contracts import eligibility as c1

UTC = timezone.utc
H = timedelta(hours=1)
P0 = 1.10000

CAL = c1.CalendarRef("synthetic.assumed_fx.test", "v1")
# Week of 2024-01-07: O Sunday 22:00Z, C Friday 22:00Z, K Thursday 05:00Z.
WEEK = c1.OperationalWeek.derive(
    week_id="2024-W02", calendar=CAL,
    market_open=datetime(2024, 1, 7, 22, tzinfo=UTC),
    market_close=datetime(2024, 1, 12, 22, tzinfo=UTC),
    entry_cutoff=datetime(2024, 1, 11, 5, tzinfo=UTC),
    entry_delay_after_open_hours=12, force_flat_before_close_hours=12)
A, B, K = WEEK.operational_start, WEEK.force_flat, WEEK.entry_cutoff
WEEK2 = c1.OperationalWeek.derive(
    week_id="2024-W03", calendar=CAL,
    market_open=datetime(2024, 1, 14, 22, tzinfo=UTC),
    market_close=datetime(2024, 1, 19, 22, tzinfo=UTC),
    entry_cutoff=datetime(2024, 1, 18, 5, tzinfo=UTC),
    entry_delay_after_open_hours=12, force_flat_before_close_hours=12)

SHORT = (2, 3, 4, 5, 6, 12)
LONG = (18, 24, 36, 48, 60, 72)
ENTRY_LONG = [P0 + d for d in (0.0010, 0.0012, 0.0015, 0.0011, 0.0013, 0.0014)]


def bars_for(week, *, skip=(), price=P0):
    out = []
    t = week.market_open + H
    while t <= week.market_close:
        if t not in skip:
            out.append(sa.Bar(close_time=t, open=price, high=price + 0.00005,
                              low=price - 0.00005, close=price))
        t += H
    return out


def flat_predictions(week, long_values=ENTRY_LONG, short_value=P0):
    short, long_ = {}, {}
    t = week.market_open + H
    while t <= week.market_close:
        short[t] = {h: short_value for h in SHORT}
        long_[t] = {h: v for h, v in zip(LONG, long_values)}
        t += H
    return short, long_


@pytest.fixture(scope="module")
def policy():
    return sa.load_pinned_policy()


@pytest.fixture(scope="module")
def geometry(policy):
    heads = [policy.HeadForecast("long", h, True, v) for h, v in zip(LONG, ENTRY_LONG)]
    d = policy.entry_decision(current_price=P0, long_heads=heads,
                              params=policy.EntryParameters(**sa.DEFAULT_ENTRY_PARAMETERS))
    assert d.action == "OPEN_LONG"
    return d.geometry


# --------------------------------------------------------------------------
# Plugin mapping and pin


def test_dmo14_pinned_policy_loads_from_commit_and_rejects_a_wrong_digest(policy):
    assert policy.RULES_VERSION == "heuristic_strategy.business_pair_policy.v1"
    assert policy.SHORT_HORIZONS_HOURS == SHORT and policy.LONG_HORIZONS_HOURS == LONG
    assert sa.policy_identity(policy)["sha256"] == sa.STRATEGY_PIN["sha256"]
    bad = dict(sa.STRATEGY_PIN, sha256="0" * 64)
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.load_pinned_policy(pin=bad)
    assert e.value.code == "STRATEGY_PIN_MISMATCH"
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.load_pinned_policy(pin=dict(sa.STRATEGY_PIN, commit="f" * 40))
    assert e.value.code == "STRATEGY_PIN_UNRESOLVED"


def test_dmo14_plugin_map_names_the_executor_source_and_refuses_the_entry_point():
    report = sa.map_strategy_resolution()
    assert report["executor_source"] == "PINNED_GIT_BLOB"
    assert report["pinned_policy"]["sha256"] == sa.STRATEGY_PIN["sha256"]
    assert report["entry_point"]["name"] == "heuristic_strategy.plugins:ls_pred_strategy"
    assert report["entry_point"]["used_by_executor"] is False
    legacy = report["legacy_plugin_at_pin"]
    assert legacy["path"] == "app/plugins/plugin_long_short_predictions.py"
    assert legacy["blob"] == "3a19b3f73917a9714d496d10a719135c9810c3ce"  # audited blob
    assert set(legacy["known_contract_defects"]) >= {
        "POSITIONAL_PREDICTION_COLUMNS", "MISSING_HEAD_FILLED_WITH_CURRENT_PRICE",
        "MIN_MAX_EARLY_CLOSE_ORDER_BLIND_IGNORES_TP", "STOP_CLOSE_WITHOUT_FILL"}
    text = json.dumps(report)
    assert "/home/" not in text and "/var/tmp/" not in text


# --------------------------------------------------------------------------
# DMO14: named outputs and per-head masks


def test_dmo14_named_heads_mask_near_force_flat_and_never_fill(policy):
    t = B - 20 * H  # R = 20 h: only long 18 h remains operational
    recs = sa.build_family_heads(origin=t, family="long",
                                 predicted={h: P0 + 0.001 for h in LONG},
                                 week=WEEK, role=c1.Role.MANAGEMENT)
    by_h = {r["horizon_hours"]: r for r in recs}
    assert [r["horizon_hours"] for r in recs] == list(LONG)
    assert by_h[18]["available"] is True and by_h[18]["value"] == P0 + 0.001
    for h in (24, 36, 48, 60, 72):
        assert by_h[h]["available"] is False and by_h[h]["value"] is None
        assert by_h[h]["reason"].startswith("OPERATIONALLY_INELIGIBLE")
        assert by_h[h]["target_timestamp"] == c1.iso(t + h * H)
        assert by_h[h]["eligible_until"] == c1.iso(B - h * H)
    shorts = sa.build_family_heads(origin=t, family="short",
                                   predicted={h: P0 for h in SHORT},
                                   week=WEEK, role=c1.Role.MANAGEMENT)
    assert all(r["available"] for r in shorts)  # long ineligibility never drops shorts
    heads = sa.policy_heads(recs, policy)
    assert [x.value for x in heads if not x.available] == [None] * 5


def test_dmo14_missing_nonfinite_and_validation_ineligible_heads_carry_reasons():
    pred = {h: P0 for h in SHORT}
    pred[2], pred[3] = None, float("nan")
    recs = sa.build_family_heads(origin=A, family="short", predicted=pred, week=WEEK,
                                 role=c1.Role.MANAGEMENT, validation_eligible={4, 5, 6, 12, 3})
    by_h = {r["horizon_hours"]: r for r in recs}
    assert (by_h[2]["available"], by_h[2]["value"]) == (False, None)
    assert "HEAD_NOT_ELIGIBLE_IN_VALIDATION" in by_h[2]["reason"]
    assert "PREDICTION_MISSING" in by_h[2]["reason"]
    assert (by_h[3]["available"], by_h[3]["reason"]) == (False, "PREDICTION_NON_FINITE")
    assert all(by_h[h]["available"] for h in (4, 5, 6, 12))


@pytest.mark.parametrize("bad,code", [
    ([P0] * 6, "POSITIONAL_INPUT_REJECTED"),
    ({h: P0 for h in SHORT[:-1]}, "HEAD_KEYS_MISMATCH"),
    ({**{h: P0 for h in SHORT}, 1: P0}, "HEAD_KEYS_MISMATCH"),
    ({**{h: P0 for h in SHORT[1:]}, True: P0}, "HEAD_KEYS_MISMATCH"),
])
def test_dmo14_positional_or_partial_head_maps_are_refused(bad, code):
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.build_family_heads(origin=A, family="short", predicted=bad, week=WEEK,
                              role=c1.Role.MANAGEMENT)
    assert e.value.code == code


def test_dmo14_named_columns_parse_and_legacy_positional_columns_are_refused():
    row = {f"short_h{h}": P0 for h in SHORT} | {f"long_h{h}": P0 + 1e-4 for h in LONG}
    parsed = sa.parse_named_columns(row)
    assert parsed["short"] == {h: P0 for h in SHORT}
    assert parsed["long"] == {h: P0 + 1e-4 for h in LONG}
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.parse_named_columns({"Prediction_h_1": P0, **row})
    assert e.value.code == "POSITIONAL_COLUMN_REJECTED"


def test_dmo14_model_arrays_map_by_horizons_hours_not_position():
    origins = [int((A + i * H).timestamp()) for i in range(2)]
    horizons = [12, 2, 3, 4, 5, 6]  # deliberately permuted
    values = [[1.0 + h for h in horizons], [2.0 + h for h in horizons]]
    values[1][0] = float("nan")
    out = sa.family_predictions_from_arrays(row_id=origins, horizons_hours=horizons,
                                            predicted_price=values, family="short")
    first = out[A]
    assert first == {2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: 7.0, 12: 13.0}
    assert out[A + H][12] is None  # NaN becomes an explicit missing head
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.family_predictions_from_arrays(row_id=origins, horizons_hours=[2, 3, 4, 5, 6, 18],
                                          predicted_price=values, family="short")
    assert e.value.code == "HEAD_KEYS_MISMATCH"


# --------------------------------------------------------------------------
# DMO14: chronological consumption, buy and sell, decisions distinguished


def _mgmt(policy, *, direction, tp, sl, short_values, long_values=None, origin=A + 2 * H):
    short = sa.build_family_heads(origin=origin, family="short",
                                  predicted=dict(zip(SHORT, short_values)),
                                  week=WEEK, role=c1.Role.MANAGEMENT)
    long_ = sa.build_family_heads(origin=origin, family="long",
                                  predicted=dict(zip(LONG, long_values or [P0] * 6)),
                                  week=WEEK, role=c1.Role.MANAGEMENT)
    return sa.management_decision(policy, direction=direction, take_profit=tp,
                                  stop_loss=sl, short_records=short, long_records=long_)


@pytest.mark.parametrize("direction", ["long", "short"])
def test_dmo14_reversed_hit_order_distinguishes_decisions(policy, direction):
    if direction == "long":
        tp, sl = 1.10100, 1.09800
        s1 = [1.10120, 1.10060, 1.10040, 1.10020, 1.10000, 1.09750]  # TP@2h, SL@12h
    else:
        tp, sl = 1.09900, 1.10200
        s1 = [1.09880, 1.09940, 1.09960, 1.09980, 1.10000, 1.10250]
    s2 = list(reversed(s1))  # SL@2h, TP@12h; same multiset of values
    d1 = _mgmt(policy, direction=direction, tp=tp, sl=sl, short_values=s1)
    d2 = _mgmt(policy, direction=direction, tp=tp, sl=sl, short_values=s2)
    assert (d1["action"], d1["reason"], d1["first_crossing_horizon_hours"]) == ("HOLD", "TP_FIRST", 2)
    assert (d2["action"], d2["reason"], d2["first_crossing_horizon_hours"]) == (
        "EARLY_CLOSE", "SL_FIRST", 2)
    # Negative control: the legacy min/max rule (variant A) cannot tell them apart.
    def legacy_a(path):
        return min(path) < sl if direction == "long" else max(path) > sl
    assert legacy_a(s1 + [P0] * 6) == legacy_a(s2 + [P0] * 6)
    # Every consulted head is receipted in horizon order, including those after the touch.
    assert [c["horizon_hours"] for c in d1["consulted"]] == list(SHORT + LONG)


def test_dmo14_unavailable_crossing_head_is_not_consumed(policy):
    tp, sl = 1.10100, 1.09800
    short = sa.build_family_heads(
        origin=A + 2 * H, family="short",
        predicted={2: None, 3: 1.10120, 4: P0, 5: P0, 6: P0, 12: 1.09700},
        week=WEEK, role=c1.Role.MANAGEMENT)
    long_ = sa.build_family_heads(origin=A + 2 * H, family="long",
                                  predicted={h: P0 for h in LONG}, week=WEEK,
                                  role=c1.Role.MANAGEMENT)
    d = sa.management_decision(policy, direction="long", take_profit=tp, stop_loss=sl,
                               short_records=short, long_records=long_)
    assert (d["action"], d["reason"], d["first_crossing_horizon_hours"]) == ("HOLD", "TP_FIRST", 3)
    assert d["skipped"] == [{"family": "short", "horizon_hours": 2,
                             "reason": "PREDICTION_MISSING"}]


def test_dmo14_entry_uses_long_family_only_with_minimum_two_heads(policy):
    long_ok = sa.build_family_heads(origin=A, family="long",
                                    predicted=dict(zip(LONG, ENTRY_LONG)),
                                    week=WEEK, role=c1.Role.ENTRY)
    e1 = sa.entry_decision_from_records(policy, current_price=P0, long_records=long_ok)
    assert e1["action"] == "OPEN_LONG"
    one = dict(zip(LONG, ENTRY_LONG))
    for h in (24, 36, 48, 60, 72):
        one[h] = None
    long_one = sa.build_family_heads(origin=A, family="long", predicted=one, week=WEEK,
                                     role=c1.Role.ENTRY)
    e2 = sa.entry_decision_from_records(policy, current_price=P0, long_records=long_one)
    assert (e2["action"], e2["reason"]) == ("ABSTAIN", "INSUFFICIENT_LONG_HEADS")
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.entry_decision_from_records(
            policy, current_price=P0,
            long_records=sa.build_family_heads(origin=A, family="short",
                                               predicted={h: P0 for h in SHORT},
                                               week=WEEK, role=c1.Role.ENTRY))
    assert e.value.code == "SHORT_HEAD_IN_ENTRY"


def test_dmo14_backtest_reversed_order_changes_the_trade_and_short_family_never_changes_entry(
        policy, geometry):
    tp, sl = geometry.take_profit_price, geometry.stop_loss_price
    bars = bars_for(WEEK)
    results = {}
    for name, first, last in (("S1", tp + 1e-5, sl - 1e-5), ("S2", sl - 1e-5, tp + 1e-5)):
        short, long_ = flat_predictions(WEEK)
        mgmt = A + 2 * H  # first management origin after the entry fill (fill bar A+1h)
        short[mgmt] = {2: first, 3: P0, 4: P0, 5: P0, 6: P0, 12: last}
        results[name] = sa.run_backtest(bars=bars, weeks=[WEEK], short_predictions=short,
                                        long_predictions=long_, policy=policy)
    t1, t2 = results["S1"]["trades"], results["S2"]["trades"]
    assert len(t1) == 1 and len(t2) >= 1  # after the early close S2 may re-enter
    for trades in (t1, t2):  # positions never overlap (one position per asset)
        for a, b in zip(trades, trades[1:]):
            assert sa.parse_time(b["entry_fill_time"]) >= sa.parse_time(a["exit_fill_time"])
    assert t1[0]["entry_decision_time"] == t2[0]["entry_decision_time"] == c1.iso(A)
    assert (t1[0]["entry_fill_price"], t1[0]["units"]) == (t2[0]["entry_fill_price"], t2[0]["units"])
    assert t1[0]["exit_reason"] == "FORCE_FLAT"
    assert t2[0]["exit_reason"] == "EARLY_CLOSE_SL_FIRST"
    assert t2[0]["exit_decision_time"] == c1.iso(A + 2 * H)


# --------------------------------------------------------------------------
# DMO15: fills, force flat, no invented stop() fill, one position, metrics


def test_dmo15_force_flat_obtains_an_actual_fill_before_B(policy):
    short, long_ = flat_predictions(WEEK)
    res = sa.run_backtest(bars=bars_for(WEEK), weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    (trade,) = res["trades"]
    assert trade["exit_reason"] == "FORCE_FLAT"
    assert trade["exit_decision_time"] == c1.iso(B - H)
    assert trade["exit_fill_bar_close_time"] == c1.iso(B)
    assert sa.parse_time(trade["exit_fill_time"]) < B
    assert trade["force_flat_violation"] is False
    assert res["open_positions"] == [] and res["failures"] == {}
    assert res["position_at"][c1.iso(B)] == 0.0


def test_dmo15_missing_fill_bar_is_a_retained_failure_not_an_invented_close(policy):
    short, long_ = flat_predictions(WEEK)
    # Every bar from B onward is missing in this week: the close order cannot fill by B.
    skip = {B + i * H for i in range(0, 13)}
    week1 = bars_for(WEEK, skip=skip)
    res = sa.run_backtest(bars=week1, weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    assert res["trades"] == []
    (pos,) = res["open_positions"]
    assert pos["status"] == "OPEN_NO_FILL"
    assert res["failures"] == {"CLOSE_ORDER_UNFILLED_AT_END": 1, "FORCE_FLAT_NOT_FILLED": 1}
    assert res["metrics"]["closed_trades"] == 0 and res["metrics"]["open_trades"] == 1
    # With next-week bars present the order fills there and the violation is retained.
    res2 = sa.run_backtest(bars=week1 + bars_for(WEEK2), weeks=[WEEK, WEEK2],
                           short_predictions=short, long_predictions=long_, policy=policy)
    first = res2["trades"][0]
    assert first["force_flat_violation"] is True
    assert first["exit_fill_bar_close_time"] == c1.iso(WEEK2.market_open + H)
    assert res2["failures"]["FORCE_FLAT_FILL_AFTER_DEADLINE"] == 1


def test_dmo15_end_of_data_finalization_invents_no_fill(policy):
    short, long_ = flat_predictions(WEEK)
    cut = [b for b in bars_for(WEEK) if b.close_time <= A + 10 * H]
    res = sa.run_backtest(bars=cut, weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    assert res["trades"] == []
    (pos,) = res["open_positions"]
    assert pos["status"] == "OPEN_AT_END_OF_DATA"
    assert res["metrics"]["closed_trades"] == 0 and res["metrics"]["open_trades"] == 1
    assert res["failures"] == {"POSITION_OPEN_AT_END_OF_DATA": 1}
    # Marked to market at the last close, entry costs included, no exit fill.
    units, entry = pos["units"], pos["entry_fill_price"]
    assert math.isclose(res["metrics"]["final_equity"],
                        10000.0 + units * (cut[-1].close - entry) - pos["accrued_commission"]
                        - pos["accrued_swap"], rel_tol=0, abs_tol=1e-9)


def test_dmo15_one_position_per_asset_and_no_entry_while_pending(policy):
    short, long_ = flat_predictions(WEEK)
    res = sa.run_backtest(bars=bars_for(WEEK), weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    exposure = [v for v in res["position_at"].values()]
    assert max(abs(v) for v in exposure) == res["trades"][0]["units"]
    entries = [d for d in res["decisions"] if d["role"] == "ENTRY"]
    assert len(entries) == 1  # entry decisions stop once an order or position exists
    assert all(d["role"] == "MANAGEMENT" for d in res["decisions"][1:])


def test_dmo15_entry_order_that_would_fill_after_the_cutoff_is_cancelled(policy):
    short, long_ = flat_predictions(WEEK)
    # Only origins at K-1h carry predictions; bar K is missing, so the next fill would be
    # at or after the cutoff.
    only = K - H
    short = {only: short[only]}
    long_ = {only: long_[only]}
    res = sa.run_backtest(bars=bars_for(WEEK, skip={K}), weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    assert res["trades"] == [] and res["open_positions"] == []
    assert res["cancelled_orders"][0]["reason"] == "ENTRY_FILL_AFTER_DEADLINE"
    assert res["failures"] == {}


def test_dmo15_protective_barriers_intrabar_both_touched_takes_stop_first(policy, geometry):
    short, long_ = flat_predictions(WEEK)
    bars = bars_for(WEEK)
    i = next(n for n, b in enumerate(bars) if b.close_time == A + 3 * H)
    b = bars[i]
    bars[i] = sa.Bar(close_time=b.close_time, open=P0,
                     high=geometry.take_profit_price + 1e-5,
                     low=geometry.stop_loss_price - 1e-5, close=P0)
    res = sa.run_backtest(bars=bars, weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    trade = res["trades"][0]
    assert trade["exit_fill_bar_close_time"] == c1.iso(A + 3 * H)
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_reference_price"] == geometry.stop_loss_price
    assert res["rules"]["intrabar_both_barriers"] == "STOP_LOSS_FIRST"


def test_dmo15_initial_equity_costs_and_metrics_are_recomputed_by_hand(policy):
    short, long_ = flat_predictions(WEEK)
    res = sa.run_backtest(bars=bars_for(WEEK), weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    (t,) = res["trades"]
    m = res["metrics"]
    ex = sa.ExecutionParameters()
    assert m["initial_equity"] == 10000.0 == ex.initial_equity
    units = 20000.0  # rr >= upper threshold -> max volume, capped by 10000*0.02*100
    half, slip = ex.spread_price / 2, ex.slippage_price
    assert t["units"] == units
    assert math.isclose(t["entry_fill_price"], P0 + half + slip, abs_tol=1e-15)
    assert math.isclose(t["exit_fill_price"], P0 - half - slip, abs_tol=1e-15)
    hours = (B - H - A) / H  # entry fill at open of bar A+1h, exit at open of bar B
    costs = {"spread": units * ex.spread_price, "slippage": units * 2 * slip,
             "commission": ex.commission_per_lot_round_trip * units / 100000.0,
             "swap": ex.swap_per_lot_per_day * (units / 100000.0) * hours / 24.0}
    for k, v in costs.items():
        assert math.isclose(m["costs"][k], v, abs_tol=1e-9), k
    net = -sum(costs.values())
    assert math.isclose(t["net_pnl"], net, abs_tol=1e-9)
    assert math.isclose(m["net_pnl"], net, abs_tol=1e-9)
    assert math.isclose(m["final_equity"], 10000.0 + net, abs_tol=1e-9)
    assert math.isclose(m["return"], net / 10000.0, abs_tol=1e-12)
    assert m["closed_trades"] == 1 and m["open_trades"] == 0
    assert (m["profit_factor"], m["profit_factor_reason"]) == (0.0, None)  # 0 / loss
    assert m["max_drawdown"] > 0 and m["max_drawdown_fraction"] > 0
    assert m["return_clock"] == "UTC_DAY_LAST_MARK_TO_MARKET"
    assert m["turnover"] > 0 and 0 < m["exposure"] < 1
    refs = res["references"]
    assert refs["cash"] == {"final_equity": 10000.0, "net_pnl": 0.0, "return": 0.0}
    assert refs["business_passive_long"]["label"] == "BUSINESS_MATCHED_PASSIVE_LONG"
    assert refs["conventional_buy_and_hold"]["label"] == \
        "CONVENTIONAL_BUY_AND_HOLD_VIOLATES_NO_WEEKEND_POSITION"


def test_dmo15_zero_trade_metrics_are_explicitly_undefined(policy):
    short, long_ = flat_predictions(WEEK, long_values=[P0] * 6)  # no entry edge
    res = sa.run_backtest(bars=bars_for(WEEK), weeks=[WEEK], short_predictions=short,
                          long_predictions=long_, policy=policy)
    m = res["metrics"]
    assert m["closed_trades"] == 0 and m["net_pnl"] == 0.0
    assert (m["profit_factor"], m["profit_factor_reason"]) == (None, "NO_CLOSED_TRADES")
    assert (m["sharpe"], m["sharpe_reason"]) == (None, "ZERO_RETURN_VARIANCE")
    assert (m["sortino"], m["sortino_reason"]) == (None, "NO_DOWNSIDE_RETURNS")
    assert {d["reason"] for d in res["decisions"]} == {"NO_ENTRY_EDGE"}


def test_dmo15_test_period_is_refused_by_the_adapter(policy):
    w = c1.OperationalWeek.derive(
        week_id="2025-W02", calendar=CAL, market_open=datetime(2025, 1, 5, 22, tzinfo=UTC),
        market_close=datetime(2025, 1, 10, 22, tzinfo=UTC),
        entry_cutoff=datetime(2025, 1, 9, 5, tzinfo=UTC),
        entry_delay_after_open_hours=12, force_flat_before_close_hours=12)
    short, long_ = flat_predictions(w)
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.run_backtest(bars=bars_for(w), weeks=[w], short_predictions=short,
                        long_predictions=long_, policy=policy)
    assert e.value.code == "TEST_FORBIDDEN"


# --------------------------------------------------------------------------
# Pair eligibility gate (PLAN section 5)


def _report(family, better):
    hs = SHORT if family == "short" else LONG
    return {str(h): {"mae": 0.9 if h in better else 1.0, "naive": 1.0, "n": 100} for h in hs}


def test_pair_gate_requires_strict_improvement_and_enough_heads():
    ok = sa.pair_eligibility(_report("short", {2, 3}), _report("long", {18, 72}))
    assert ok["status"] == "ELIGIBLE"
    assert ok["eligible"] == {"short": [2, 3], "long": [18, 72]}
    few = sa.pair_eligibility(_report("short", {2}), _report("long", {18}))
    assert few["status"] == "PAIR_NOT_ELIGIBLE"
    assert few["missing"] == ["LONG_ELIGIBLE_HEADS_1_LT_2"]
    none_short = sa.pair_eligibility(_report("short", set()), _report("long", {18, 24}))
    assert none_short["missing"] == ["SHORT_ELIGIBLE_HEADS_0_LT_1"]
    tie = _report("long", {18})
    tie["24"]["mae"] = tie["24"]["naive"]  # equality is not an improvement
    assert sa.pair_eligibility(_report("short", {2}), tie)["status"] == "PAIR_NOT_ELIGIBLE"
    bad = _report("long", {18, 24})
    bad["36"]["eligible"] = True  # declared flag disagrees with MAE < naive
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.pair_eligibility(_report("short", {2}), bad)
    assert e.value.code == "ELIGIBILITY_FLAG_MISMATCH"
    missing = _report("long", {18, 24})
    del missing["72"]
    r = sa.pair_eligibility(_report("short", {2}), missing)
    assert r["eligible"]["long"] == [18, 24]
    assert r["head_status"]["long"]["72"] == "HEAD_NOT_REPORTED"
