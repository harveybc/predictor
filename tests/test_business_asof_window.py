"""Acceptance tests for the BUSINESS point-in-time window resolver."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from tools.business_asof_window import AsOfRow, SupportSpec, resolve_asof_window
from tools.business_weekly_protocol import EvaluationSplit, WeekSpec


UTC = timezone.utc


def instant(year, month, day, hour=0):
    """Build a UTC instant used by the compact fixtures."""

    return datetime(year, month, day, hour, tzinfo=UTC)


def week(cutoff=instant(2024, 3, 4), fit_start=instant(2020, 3, 4)):
    """Build one weekly BUSINESS scoring specification."""

    return WeekSpec(
        split=EvaluationSplit.VALIDATION,
        ordinal=0,
        start=cutoff,
        end=cutoff + timedelta(days=7),
        cutoff=cutoff,
        fit_start=fit_start,
        retrain_due=True,
    )


def support(**overrides):
    """Build support whose derived purge is two days."""

    values = {
        "input_lookback": timedelta(days=1),
        "target_horizon": timedelta(days=2),
        "maximum_holding_support": timedelta(days=1),
        "inner_validation_weeks": 1,
    }
    values.update(overrides)
    return SupportSpec(**values)


def row(record_id, event_time, *, digest=None, available_time=None,
        target_available_time=None):
    """Build one fully observed immutable row."""

    return AsOfRow(
        record_id=record_id,
        event_time=event_time,
        available_time=available_time or event_time,
        target_available_time=target_available_time
        or event_time + timedelta(days=2),
        row_digest=digest or f"sha256:{record_id}",
    )


def valid_rows():
    """Return fit, purge-gap and inner-validation rows in shuffled order."""

    return (
        row("validation", instant(2024, 2, 25)),
        row("fit-b", instant(2024, 2, 14)),
        row("purged", instant(2024, 2, 23)),
        row("fit-a", instant(2020, 3, 5)),
    )


def test_resolves_exact_fit_and_inner_validation_populations_with_derived_purge():
    resolved = resolve_asof_window(week(), valid_rows(), support())

    assert resolved.support.purge == timedelta(days=2)
    assert resolved.fit_start == instant(2020, 3, 4)
    assert resolved.fit_end == instant(2024, 2, 22)
    assert resolved.inner_validation_start == instant(2024, 2, 24)
    assert resolved.inner_validation_end == instant(2024, 3, 2)
    assert [item.record_id for item in resolved.fit_rows] == ["fit-a", "fit-b"]
    assert [item.record_id for item in resolved.purged_rows] == ["purged"]
    assert [item.record_id for item in resolved.inner_validation_rows] == [
        "validation"
    ]
    assert resolved.fit_count == 2
    assert resolved.inner_validation_count == 1
    assert resolved.purged_count == 1
    assert resolved.inner_validation_end + resolved.support.purge == week().cutoff
    with pytest.raises(FrozenInstanceError):
        resolved.fit_count = 99


def test_rejects_future_feature_bytes_even_when_they_arrive_before_cutoff():
    changed = list(valid_rows())
    changed[1] = row(
        "fit-b",
        instant(2024, 2, 14),
        available_time=instant(2024, 2, 14, 1),
    )

    with pytest.raises(ValueError, match="feature bytes.*fit-b"):
        resolve_asof_window(week(), changed, support())


def test_rejects_late_or_missing_labels():
    late = list(valid_rows())
    late[1] = row(
        "fit-b",
        instant(2024, 2, 14),
        target_available_time=week().cutoff + timedelta(microseconds=1),
    )
    with pytest.raises(ValueError, match="label.*fit-b"):
        resolve_asof_window(week(), late, support())

    missing = list(valid_rows())
    original = missing[1]
    missing[1] = AsOfRow(
        record_id=original.record_id,
        event_time=original.event_time,
        available_time=original.available_time,
        target_available_time=None,
        row_digest=original.row_digest,
    )
    with pytest.raises(ValueError, match="label.*fit-b"):
        resolve_asof_window(week(), missing, support())


def test_exact_calendar_window_handles_leap_year_boundary_without_day_approximation():
    leap_week = week(
        cutoff=instant(2024, 3, 4),
        fit_start=instant(2020, 3, 4),
    )
    inside = row("leap-day", instant(2020, 3, 5))
    resolved = resolve_asof_window(leap_week, (inside,), support())

    assert resolved.fit_start == instant(2020, 3, 4)
    assert resolved.fit_rows == (inside,)

    outside = row("too-old", instant(2020, 3, 3))
    with pytest.raises(ValueError, match="rolling window.*too-old"):
        resolve_asof_window(leap_week, (outside,), support())


def test_rejects_target_or_holding_support_that_crosses_the_scored_week_cutoff():
    crossing = row(
        "crossing",
        instant(2024, 3, 3),
        target_available_time=week().cutoff,
    )
    with pytest.raises(ValueError, match="support.*crossing"):
        resolve_asof_window(week(), (crossing,), support())

    holding = support(
        target_horizon=timedelta(hours=1),
        maximum_holding_support=timedelta(days=3),
    )
    crossing_holding = row(
        "holding-crossing",
        instant(2024, 3, 2),
        target_available_time=instant(2024, 3, 2, 1),
    )
    with pytest.raises(ValueError, match="support.*holding-crossing"):
        resolve_asof_window(week(), (crossing_holding,), holding)


def test_rejects_duplicate_ids_and_duplicate_content_digests():
    first = row("same", instant(2020, 3, 5), digest="sha256:a")
    duplicate_id = row("same", instant(2020, 3, 6), digest="sha256:b")
    with pytest.raises(ValueError, match="duplicate record_id.*same"):
        resolve_asof_window(week(), (first, duplicate_id), support())

    duplicate_digest = row("other", instant(2020, 3, 6), digest="sha256:a")
    with pytest.raises(ValueError, match="duplicate row_digest.*sha256:a"):
        resolve_asof_window(week(), (first, duplicate_digest), support())


def test_same_timestamp_content_mutation_changes_population_digest():
    original = row("fit-a", instant(2020, 3, 5), digest="sha256:original")
    mutated = row("fit-a", instant(2020, 3, 5), digest="sha256:mutated")

    before = resolve_asof_window(week(), (original,), support())
    after = resolve_asof_window(week(), (mutated,), support())

    assert before.fit_count == after.fit_count == 1
    assert before.fit_population_digest != after.fit_population_digest
    assert before.population_digest != after.population_digest


def test_replay_is_deterministic_and_independent_of_input_order():
    forward = resolve_asof_window(week(), valid_rows(), support())
    reverse = resolve_asof_window(week(), tuple(reversed(valid_rows())), support())

    assert forward == reverse
    assert forward.population_digest == reverse.population_digest
    assert forward.fit_population_digest == reverse.fit_population_digest
    assert (
        forward.inner_validation_population_digest
        == reverse.inner_validation_population_digest
    )


@pytest.mark.parametrize(
    "field",
    ["input_lookback", "target_horizon", "maximum_holding_support"],
)
def test_rejects_booleans_as_support_durations(field):
    with pytest.raises(TypeError, match=f"{field}.*timedelta"):
        support(**{field: True})


def test_rejects_boolean_validation_count_and_static_week_without_fit_start():
    with pytest.raises(TypeError, match="inner_validation_weeks.*int"):
        support(inner_validation_weeks=True)

    static_week = week(fit_start=None)
    with pytest.raises(ValueError, match="fit_start"):
        resolve_asof_window(static_week, valid_rows(), support())
