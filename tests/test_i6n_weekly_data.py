"""Business-clock partition tests; no market data or framework required."""

from datetime import datetime, timedelta, timezone

import pytest

from tools.i6n_contracts.eligibility import CalendarRef, OperationalWeek, SplitBounds
from tools.i6n_weekly_data import build_weekly_partitions

UTC = timezone.utc


def dt(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def week(name, opening):
    return OperationalWeek.derive(
        week_id=name, calendar=CalendarRef("synthetic", "v1"), market_open=opening,
        market_close=opening + timedelta(days=5), entry_cutoff=opening + timedelta(days=4),
        entry_delay_after_open_hours=12, force_flat_before_close_hours=12,
    )


def setup():
    weeks = [week("train", dt("2023-11-05T22:00")),
             week("inner", dt("2023-12-17T22:00")),
             week("previous", dt("2023-12-24T22:00")),
             week("eval", dt("2023-12-31T22:00"))]
    origins = tuple(w.market_open + timedelta(hours=i) for w in weeks for i in range(120))
    return dict(
        origins=origins, weeks=weeks, observed_targets={t: t for t in origins},
        input_available_at={t: t for t in origins}, horizons=(2, 72),
        evaluation_week_id="eval",
        evaluation_split=SplitBounds("development", dt("2024-01-01"), dt("2025-01-01"), None),
        source_coverage_start=dt("2010-01-01"),
        coverage_mode="synthetic_sparse",
    )


def test_four_calendar_years_and_previous_close_cutoff():
    result = build_weekly_partitions(**setup())
    assert result.fit_cutoff == dt("2023-12-29T22:00")
    assert result.history_start == dt("2019-12-29T22:00")
    assert result.inner_start == dt("2023-12-01T22:00")
    assert result.train.n_eligible > 0
    assert result.inner.n_eligible > 0
    assert result.evaluation.n_eligible > 0
    assert result.coverage_complete is False


def test_short_survives_when_long_crosses_force_flat():
    result = build_weekly_partitions(**setup())
    i = result.evaluation.origins.index(dt("2024-01-03T12:00"))
    assert result.evaluation.mask[i] == (True, False)
    assert result.evaluation.target_instants[i][0] == dt("2024-01-03T14:00")


def test_evaluation_uses_target_membership_and_keeps_all_week_origins():
    args = setup()
    args["evaluation_split"] = SplitBounds("development", dt("2024-01-02"), dt("2024-01-04"), None)
    result = build_weekly_partitions(**args)
    assert result.evaluation.origins == tuple(
        t for t in args["origins"]
        if args["weeks"][-1].market_open <= t < args["weeks"][-1].market_close)
    i = result.evaluation.origins.index(dt("2024-01-01T23:00"))
    assert result.evaluation.mask[i][0] is True
    assert result.evaluation.target_instants[i][0] == dt("2024-01-02T01:00")
    for targets, mask in zip(result.evaluation.target_instants, result.evaluation.mask):
        assert all(not m or dt("2024-01-02") <= t < dt("2024-01-04") for t, m in zip(targets, mask))


def test_training_targets_cannot_cross_inner_boundary():
    args = setup()
    args["inner_days"] = 53  # boundary Monday 2023-11-06 22:00 within TRAIN week
    result = build_weekly_partitions(**args)
    i = result.train.origins.index(dt("2023-11-06T20:00"))
    assert result.train.mask[i] == (False, False)
    assert result.train.exclusions["TARGET_AT_OR_AFTER_SPLIT_END"] > 0
    assert set(result.train.origins).isdisjoint(result.inner.origins)
    assert dt("2023-11-06T20:00") not in result.inner.origins


@pytest.mark.parametrize("kind", ["late_input", "missing_input", "late_label", "missing_label"])
def test_unavailable_data_cannot_enter_fit(kind):
    args = setup()
    origin = dt("2023-11-06T12:00")
    target = origin + timedelta(hours=2)
    if kind == "late_input":
        args["input_available_at"][origin] = origin + timedelta(seconds=1)
    elif kind == "missing_input":
        del args["input_available_at"][origin]
    elif kind == "late_label":
        args["observed_targets"][target] = dt("2024-01-10")
    else:
        del args["observed_targets"][target]
    result = build_weekly_partitions(**args)
    i = result.train.origins.index(origin)
    assert not result.train.mask[i][0]


def test_preoperational_inputs_remain_context_not_labels():
    result = build_weekly_partitions(**setup())
    i = result.train.origins.index(dt("2023-11-06T02:00"))
    assert result.train.mask[i] == (False, False)
    assert result.train.exclusions["ORIGIN_BEFORE_OPERATIONAL_START"] > 0


def test_real_saturday_in_declared_closure_is_masked_and_counted():
    args = setup()
    saturday = dt("2023-12-23T12:00")
    args["origins"] = tuple(sorted((*args["origins"], saturday)))
    args["input_available_at"][saturday] = saturday
    result = build_weekly_partitions(**args)
    i = result.inner.origins.index(saturday)
    assert result.inner.mask[i] == (False, False)
    assert result.inner.exclusions["ORIGIN_IN_MARKET_CLOSURE"] == 2
    assert result.inner.indices[i] == args["origins"].index(saturday)
    assert result.inner.target_instants[i] == tuple(
        saturday + timedelta(hours=h) for h in args["horizons"])


@pytest.mark.parametrize("origin", ["2023-10-30T12:00", "2023-11-11T12:00"])
def test_sparse_unknown_calendar_is_not_assumed_weekend(origin):
    args = setup()
    args["origins"] = tuple(sorted((*args["origins"], dt(origin))))
    with pytest.raises(ValueError, match="CALENDAR_COVERAGE_MISSING") as error:
        build_weekly_partitions(**args)
    assert error.value.code == "CALENDAR_COVERAGE_MISSING"


def complete_setup():
    """One genuinely enumerated year of a supplied synthetic weekly calendar."""
    args = setup()
    weeks = [week(f"w{i}", dt("2022-12-25T22:00") + timedelta(weeks=i))
             for i in range(54)]
    origins = tuple(w.market_open + timedelta(hours=i) for w in weeks for i in range(120))
    args.update(weeks=weeks, origins=origins, observed_targets={t: t for t in origins},
                input_available_at={t: t for t in origins}, history_years=1,
                evaluation_week_id=weeks[-1].week_id)
    del args["coverage_mode"]
    return args


def test_default_complete_accepts_full_calendar_and_hourly_enumeration():
    args = complete_setup()
    result = build_weekly_partitions(**args)
    assert result.coverage_complete is True
    assert result.coverage_mode == "complete"
    assert result.history_start == dt("2022-12-29T22:00")
    assert args["weeks"][0].market_open <= result.history_start
    assert args["weeks"][-2].market_close == result.fit_cutoff
    expected_history = tuple(t for t in args["origins"]
                             if result.history_start <= t < result.fit_cutoff)
    assert result.train.origins + result.inner.origins == expected_history
    assert len(result.evaluation.origins) == 120
    assert all(p.n_eligible > 0 for p in (result.train, result.inner, result.evaluation))
    args["coverage_mode"] = "synthetic_sparse"
    sparse = build_weekly_partitions(**args)
    assert sparse.coverage_complete is False
    assert sparse.support_sha256 != result.support_sha256


def test_default_four_year_coverage_rejects_sparse_fixture():
    args = setup()
    del args["coverage_mode"]
    with pytest.raises(ValueError, match="CALENDAR_COVERAGE_MISSING"):
        build_weekly_partitions(**args)


@pytest.mark.parametrize("kind", ["late_first_week", "missing_week", "missing_previous",
                                  "missing_train_hour", "missing_inner_hour", "missing_eval_hour",
                                  "missing_pre_split_eval_hour", "off_grid_hour"])
def test_strict_coverage_rejects_calendar_and_origin_holes(kind):
    args = complete_setup()
    code = "INCOMPLETE_HOURLY_ORIGINS"
    if kind == "late_first_week":
        args["weeks"] = args["weeks"][1:]
        code = "CALENDAR_COVERAGE_MISSING"
    elif kind in ("missing_week", "missing_previous"):
        del args["weeks"][10 if kind == "missing_week" else -2]
        code = "CALENDAR_GAP_TOO_LARGE"
    elif kind == "off_grid_hour":
        args["origins"] = tuple(sorted((*args["origins"], dt("2023-06-05T12:30"))))
        code = "ORIGIN_GRID_INVALID"
    else:
        missing = {"missing_train_hour": dt("2023-06-05T12:00"),
                   "missing_inner_hour": dt("2023-12-18T12:00"),
                   "missing_eval_hour": dt("2024-01-01T12:00"),
                   "missing_pre_split_eval_hour": dt("2023-12-31T23:00")}[kind]
        args["origins"] = tuple(t for t in args["origins"] if t != missing)
    with pytest.raises(ValueError, match=code) as error:
        build_weekly_partitions(**args)
    assert error.value.code == code


@pytest.mark.parametrize("split", ["TRAIN", "INNER", "EVAL"])
@pytest.mark.parametrize("kind", ["no_origins", "no_eligible_labels"])
def test_entirely_empty_populations_rejected_with_typed_reason(split, kind):
    args = setup()
    bounds = {"TRAIN": (dt("2019-12-29T22:00"), dt("2023-12-01T22:00")),
              "INNER": (dt("2023-12-01T22:00"), dt("2023-12-29T22:00")),
              "EVAL": (dt("2023-12-31T22:00"), dt("2024-01-05T22:00"))}
    start, end = bounds[split]
    if kind == "no_origins":
        args["origins"] = tuple(t for t in args["origins"] if not start <= t < end)
    else:
        args["input_available_at"] = {t: t for t in args["origins"] if not start <= t < end}
    with pytest.raises(ValueError, match=f"NO_ELIGIBLE_{split}") as error:
        build_weekly_partitions(**args)
    assert error.value.code == f"NO_ELIGIBLE_{split}"


@pytest.mark.parametrize("mode", [None, True, "sparse", "COMPLETE", []])
def test_unknown_coverage_modes_rejected(mode):
    args = setup()
    args["coverage_mode"] = mode
    with pytest.raises(ValueError, match="coverage_mode"):
        build_weekly_partitions(**args)


def test_no_prediction_availability_filter_and_deterministic_identity():
    first = build_weekly_partitions(**setup())
    second = build_weekly_partitions(**setup())
    assert first.support_sha256 == second.support_sha256
    assert len(first.support_sha256) == 64


@pytest.mark.parametrize("kind", ["duplicate_origins", "unordered_origins", "naive_origin",
                                  "missing_previous", "calendar_mix", "overlap", "coverage",
                                  "bool_years", "zero_inner", "unknown_eval", "test_split"])
def test_bad_contract_rejected(kind):
    args = setup()
    if kind == "duplicate_origins":
        args["origins"] += (args["origins"][-1],)
    elif kind == "unordered_origins":
        args["origins"] = tuple(reversed(args["origins"]))
    elif kind == "naive_origin":
        args["origins"] = (datetime(2023, 1, 1),)
    elif kind == "missing_previous":
        args["weeks"] = args["weeks"][-1:]
    elif kind == "calendar_mix":
        from dataclasses import replace
        args["weeks"][0] = replace(args["weeks"][0], calendar=CalendarRef("other", "v1"))
    elif kind == "overlap":
        args["weeks"].insert(1, week("overlap", dt("2023-11-06T22:00")))
    elif kind == "coverage":
        args["source_coverage_start"] = dt("2023-01-01")
    elif kind == "bool_years":
        args["history_years"] = True
    elif kind == "zero_inner":
        args["inner_days"] = 0
    elif kind == "unknown_eval":
        args["evaluation_week_id"] = "absent"
    else:
        args["evaluation_split"] = SplitBounds("TEST", dt("2025-01-01"), dt("2026-01-01"), None)
    with pytest.raises(ValueError):
        build_weekly_partitions(**args)
