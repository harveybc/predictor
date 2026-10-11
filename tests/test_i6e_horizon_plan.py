"""Contracts for the approved multihorizon grid, not forecasting-quality tests."""

import pytest

from tools.i6e_horizon_plan import (
    build_plan,
    compare_prices,
    experiment_grid,
    weekly_schedule,
)


def test_grid_covers_each_business_horizon_without_one_hour():
    arms, requests = experiment_grid()
    horizons = {2, 3, 4, 5, 6, 24, 48, 72, 96, 120}
    assert {a["horizon_hours"] for a in arms} == horizons
    assert len(arms) == 38
    assert len(requests) == 41  # 31 coarse requests + 10 hourly controls.
    assert sum(a["stage"] == "DOWNSAMPLING" for a in arms) == 28
    assert sum(a["stage"] == "HOURLY" for a in arms) == 10


def test_downsampled_steps_include_three_and_keep_physical_horizon():
    arms, _ = experiment_grid()
    long72 = [a for a in arms if a["horizon_hours"] == 72]
    assert {(a["spacing_hours"], a["forecast_steps"]) for a in long72} == {
        (72, 1), (36, 2), (24, 3), (18, 4), (1, 72)
    }
    for arm in arms:
        assert arm["spacing_hours"] * arm["forecast_steps"] == arm["horizon_hours"]
        assert (arm["samples"] - 1) * arm["spacing_hours"] == 2 * arm["horizon_hours"]


def test_native_clock_never_fabricates_fractional_hour_observations():
    arms, _ = experiment_grid()
    for arm in arms:
        assert type(arm["spacing_hours"]) is int
        assert arm["spacing_hours"] >= 1
    assert not any(a["horizon_hours"] == 3 and a["forecast_steps"] == 2 for a in arms)


def test_equivalent_hourly_requests_do_not_duplicate_fits():
    arms, requests = experiment_grid()
    assert len({a["arm_id"] for a in arms}) == len(arms)
    for horizon, steps in ((2, 2), (3, 3), (4, 4)):
        aliases = [r for r in requests if r["horizon_hours"] == horizon
                   and (r["mode"] == "HOURLY" or r["requested_steps"] == steps)]
        assert len(aliases) == 2
        assert aliases[0]["arm_id"] == aliases[1]["arm_id"]


def test_whole_downsampling_stage_precedes_ascending_hourly_windows():
    arms, _ = experiment_grid()
    assert all(a["stage"] == "DOWNSAMPLING" for a in arms[:28])
    assert [a["horizon_hours"] for a in arms[28:]] == [2, 3, 4, 5, 6, 24, 48, 72, 96, 120]
    assert [a["samples"] for a in arms[:28]] == sorted(a["samples"] for a in arms[:28])


def test_calendar_includes_the_final_partial_week():
    weeks = weekly_schedule(2024)
    assert len(weeks) == 53
    assert weeks[0]["score_start"] == "2024-01-01T00:00:00Z"
    assert weeks[-1]["score_end"] == "2025-01-01T00:00:00Z"
    assert weeks[-1]["score_start"] == "2024-12-30T00:00:00Z"
    assert weeks[-1]["fit_start"] == "2020-12-30T00:00:00Z"
    assert all(a["score_end"] == b["score_start"] for a, b in zip(weeks, weeks[1:]))


def test_test_calendar_uses_the_preceding_monday_cutoff():
    weeks = weekly_schedule(2025)
    assert weeks[0]["fit_cutoff"] == "2024-12-30T00:00:00Z"
    assert weeks[0]["score_start"] == "2025-01-01T00:00:00Z"
    assert weeks[-1]["score_end"] == "2026-01-01T00:00:00Z"


def test_naive_is_recomputed_for_each_physical_horizon_and_in_price_units():
    short = compare_prices([1, 2], [1.1, 2.2], [1.05, 2.1],
                           horizon_hours=2, naive_horizon_hours=2)
    long = compare_prices([1, 2], [1.4, 2.8], [1.2, 2.4],
                          horizon_hours=24, naive_horizon_hours=24)
    assert short["naive_mae"] == pytest.approx(.15)
    assert long["naive_mae"] == pytest.approx(.6)
    assert short["mae"] == pytest.approx(.075)
    assert short["unit"] == long["unit"] == "USD_per_EUR"
    assert short["naive_label"] == "naive para el horizonte de 2 horas"
    assert long["naive_label"] == "naive para el horizonte de 24 horas"


def test_a_model_cannot_be_compared_to_another_horizons_naive():
    with pytest.raises(ValueError, match="horizon"):
        compare_prices([1], [1.2], [1.1], horizon_hours=24, naive_horizon_hours=1)


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")], [-1], [0]])
def test_invalid_prices_never_turn_into_metrics(bad):
    with pytest.raises(ValueError):
        compare_prices(bad, bad, bad, horizon_hours=2, naive_horizon_hours=2)


def test_grid_is_reproducible_and_claims_no_execution_or_test_read():
    first = build_plan()
    assert first == build_plan()
    assert first["status"] == "PLANNED_NOT_EXECUTED"
    assert first["test_read"] is False
    assert first["execution_ready"] is False
    assert first["validation_cell_dispositions_expected"] == 38 * 53
    assert first["measured_scored_rows"] is None
