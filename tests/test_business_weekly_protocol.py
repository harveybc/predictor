"""Acceptance tests for the dependency-free BUSINESS weekly protocol core."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from tools.business_weekly_protocol import (
    AvailableRecord,
    BusinessWeeklyProtocol,
    Disposition,
    DispositionStatus,
    EvaluationMode,
    EvaluationSplit,
    UpdateMode,
    subtract_calendar_years,
)


UTC = timezone.utc


def dt(year, month, day):
    """Build a midnight UTC test timestamp."""

    return datetime(year, month, day, tzinfo=UTC)


def protocol(
    evaluation_mode=EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD,
    update_mode=UpdateMode.FULL_RETRAIN_ROLLING_4Y,
    **overrides,
):
    """Build a compact protocol spanning three validation and two test weeks."""

    values = {
        "evaluation_mode": evaluation_mode,
        "update_mode": update_mode,
        "validation_start": dt(2024, 1, 1),
        "validation_end": dt(2024, 1, 22),
        "test_start": dt(2024, 2, 5),
        "test_end": dt(2024, 2, 19),
        "procedure_digest": "procedure-v1",
    }
    values.update(overrides)
    return BusinessWeeklyProtocol(**values)


def test_bw01_enumerates_every_complete_week_in_both_half_open_intervals():
    """BW01: the denominator is every planned UTC week, in stable order."""

    weeks = protocol().weeks()

    assert len(weeks) == 5
    assert [week.ordinal for week in weeks] == list(range(5))
    assert [week.split for week in weeks] == [
        EvaluationSplit.VALIDATION,
        EvaluationSplit.VALIDATION,
        EvaluationSplit.VALIDATION,
        EvaluationSplit.TEST,
        EvaluationSplit.TEST,
    ]
    assert all(week.end - week.start == timedelta(days=7) for week in weeks)
    assert all(week.cutoff == week.start for week in weeks)
    with pytest.raises(FrozenInstanceError):
        weeks[0].ordinal = 99


def test_bw01_rejects_partial_or_non_monday_intervals():
    with pytest.raises(ValueError, match="Monday"):
        protocol(validation_start=dt(2024, 1, 2))
    with pytest.raises(ValueError, match="Monday"):
        protocol(validation_end=dt(2024, 1, 29) + timedelta(hours=1))
    with pytest.raises(ValueError, match="ordered intervals"):
        protocol(validation_end=dt(2024, 2, 12))
    with pytest.raises(TypeError, match="EvaluationMode"):
        protocol(evaluation_mode="BUSINESS_WEEKLY_WALK_FORWARD")


def test_bw04_uses_exact_four_calendar_years_and_explicit_leap_policy():
    """BW04: four years are calendar years, not 1,460 elapsed days."""

    leap = datetime(2024, 2, 29, 12, tzinfo=UTC)
    assert subtract_calendar_years(leap, 4) == datetime(2020, 2, 29, 12, tzinfo=UTC)
    assert subtract_calendar_years(leap, 3) == datetime(2021, 2, 28, 12, tzinfo=UTC)
    assert subtract_calendar_years(dt(2024, 3, 4), 4) == dt(2020, 3, 4)
    assert protocol().weeks()[0].fit_start == dt(2020, 1, 1)

    with pytest.raises(TypeError, match="bool"):
        subtract_calendar_years(leap, True)
    with pytest.raises(TypeError, match="bool"):
        protocol(rolling_years=True)


def test_bw04_rejects_records_unavailable_at_the_weekly_cutoff():
    p = protocol()
    cutoff = p.weeks()[0].cutoff
    available = AvailableRecord("known", cutoff)
    late = AvailableRecord("future", cutoff + timedelta(microseconds=1))

    assert p.require_available([available], cutoff) == (available,)
    with pytest.raises(ValueError, match="future"):
        p.require_available([available, late], cutoff)


def test_bw05_warm_update_requires_parent_and_refuses_strict_memory_claim():
    """BW05: inherited weights are explicit and cannot impersonate full retraining."""

    p = protocol(update_mode=UpdateMode.WARM_UPDATE_ROLLING_4Y)
    week = p.weeks()[0]

    with pytest.raises(ValueError, match="parent_digest"):
        p.bind_model(week, "model-a", strict_four_year_memory=False)
    with pytest.raises(ValueError, match="cannot claim"):
        p.bind_model(
            week,
            "model-a",
            parent_digest="parent-a",
            strict_four_year_memory=True,
        )
    identity = p.bind_model(
        week,
        "model-a",
        parent_digest="parent-a",
        strict_four_year_memory=False,
    )
    assert identity.parent_digest == "parent-a"
    assert identity.procedure_digest == p.procedure_identity
    assert identity.model_digest != identity.procedure_digest

    with pytest.raises(TypeError, match="bool"):
        p.bind_model(
            week,
            "model-b",
            parent_digest="parent-a",
            strict_four_year_memory=1,
        )


def test_bw05_full_retrain_has_no_parent_and_claims_strict_window():
    p = protocol()
    week = p.weeks()[0]

    with pytest.raises(ValueError, match="must not declare a parent"):
        p.bind_model(
            week,
            "model-a",
            parent_digest="parent-a",
            strict_four_year_memory=True,
        )
    identity = p.bind_model(week, "model-a", strict_four_year_memory=True)
    assert identity.parent_digest is None
    assert identity.strict_four_year_memory is True


def test_bw06_weekly_and_monthly_cadences_have_distinct_behavior_and_identity():
    """BW06: cadence is explicit rather than an undocumented approximation."""

    weekly = protocol()
    monthly = protocol(
        evaluation_mode=EvaluationMode.BUSINESS_MONTHLY_WALK_FORWARD,
        update_mode=UpdateMode.FULL_RETRAIN_ROLLING_4Y,
    )

    assert weekly.business_label() != monthly.business_label()
    assert [week.retrain_due for week in weekly.weeks()] == [True] * 5
    assert [week.retrain_due for week in monthly.weeks()] == [
        True,
        False,
        False,
        True,
        False,
    ]


def test_bw06_monthly_retrains_at_each_split_boundary_even_in_same_month():
    monthly = protocol(
        evaluation_mode=EvaluationMode.BUSINESS_MONTHLY_WALK_FORWARD,
        update_mode=UpdateMode.FULL_RETRAIN_ROLLING_4Y,
        validation_start=dt(2024, 1, 1),
        validation_end=dt(2024, 1, 15),
        test_start=dt(2024, 1, 22),
        test_end=dt(2024, 2, 12),
    )

    assert [week.retrain_due for week in monthly.weeks()] == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_bw14_literature_mode_refuses_business_update_and_labels():
    """BW14: a static literature run cannot be relabeled as BUSINESS."""

    with pytest.raises(ValueError, match="must not declare"):
        protocol(evaluation_mode=EvaluationMode.LITERATURE_STATIC)

    static = protocol(
        evaluation_mode=EvaluationMode.LITERATURE_STATIC,
        update_mode=None,
    )
    assert not any(week.retrain_due for week in static.weeks())
    assert all(week.fit_start is None for week in static.weeks())
    with pytest.raises(ValueError, match="refuses BUSINESS"):
        static.business_label()
    with pytest.raises(ValueError, match="refuses BUSINESS"):
        static.bind_model(
            static.weeks()[0], "model-a", strict_four_year_memory=True
        )


def test_bw15_requires_one_explicit_disposition_per_planned_week():
    """BW15: missing, duplicate, and unexplained weeks cannot disappear."""

    p = protocol()
    weeks = p.weeks()
    dispositions = [
        Disposition(week, DispositionStatus.COMPLETED) for week in weeks
    ]
    assert p.close(dispositions) == tuple(dispositions)

    with pytest.raises(ValueError, match="missing weekly dispositions"):
        p.close(dispositions[:-1])
    with pytest.raises(ValueError, match="duplicate disposition"):
        p.close(dispositions + [dispositions[0]])
    with pytest.raises(ValueError, match="require a reason"):
        Disposition(weeks[0], DispositionStatus.FAILED)
