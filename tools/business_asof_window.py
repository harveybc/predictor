"""Point-in-time population resolver for BUSINESS weekly evaluation.

The resolver is intentionally free of model and storage concerns.  It binds an
immutable set of row identities to one :class:`WeekSpec`, verifies that every
byte and target was knowable at the appropriate time, and creates purged fit
and inner-validation populations with deterministic content identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from typing import Iterable, Optional, Tuple

from tools.business_weekly_protocol import (
    WEEK,
    WeekSpec,
    subtract_calendar_years,
)


def _require_utc(name: str, value: object) -> datetime:
    """Return a timezone-aware UTC instant or reject it."""

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value


def _require_duration(
    name: str,
    value: object,
    *,
    allow_zero: bool,
) -> timedelta:
    """Validate a duration while explicitly excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a timedelta, not {type(value).__name__}")
    if value < timedelta(0) or (not allow_zero and value == timedelta(0)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _iso(value: datetime) -> str:
    """Serialize a validated UTC instant canonically."""

    return value.isoformat(timespec="microseconds")


@dataclass(frozen=True)
class AsOfRow:
    """Immutable identity and availability clocks for one decision-time row.

    ``event_time`` is the prediction origin.  ``available_time`` is the time at
    which all feature bytes represented by ``row_digest`` were available.
    ``target_available_time`` is optional at construction so missing labels can
    be represented and rejected explicitly by the supervised resolver.
    """

    record_id: str
    event_time: datetime
    available_time: datetime
    target_available_time: Optional[datetime]
    row_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(self.row_digest, str) or not self.row_digest.strip():
            raise ValueError("row_digest must be a non-empty string")
        _require_utc("event_time", self.event_time)
        _require_utc("available_time", self.available_time)
        if self.target_available_time is not None:
            _require_utc("target_available_time", self.target_available_time)


@dataclass(frozen=True)
class SupportSpec:
    """Temporal support needed to form inputs, targets, and trade outcomes."""

    input_lookback: timedelta
    target_horizon: timedelta
    maximum_holding_support: timedelta
    inner_validation_weeks: int

    def __post_init__(self) -> None:
        _require_duration("input_lookback", self.input_lookback, allow_zero=True)
        _require_duration("target_horizon", self.target_horizon, allow_zero=False)
        _require_duration(
            "maximum_holding_support",
            self.maximum_holding_support,
            allow_zero=True,
        )
        if isinstance(self.inner_validation_weeks, bool) or not isinstance(
            self.inner_validation_weeks, int
        ):
            raise TypeError("inner_validation_weeks must be an int, not bool")
        if self.inner_validation_weeks <= 0:
            raise ValueError("inner_validation_weeks must be positive")

    @property
    def purge(self) -> timedelta:
        """Return the future support that must not cross a population boundary."""

        return max(self.target_horizon, self.maximum_holding_support)

    @property
    def inner_validation_duration(self) -> timedelta:
        """Return the requested chronological validation duration."""

        return self.inner_validation_weeks * WEEK


@dataclass(frozen=True)
class ResolvedAsOfWindow:
    """Authenticated fit, purge-gap, and inner-validation populations."""

    week: WeekSpec
    support: SupportSpec
    fit_start: datetime
    fit_end: datetime
    inner_validation_start: datetime
    inner_validation_end: datetime
    fit_rows: Tuple[AsOfRow, ...]
    purged_rows: Tuple[AsOfRow, ...]
    inner_validation_rows: Tuple[AsOfRow, ...]
    fit_population_digest: str
    purged_population_digest: str
    inner_validation_population_digest: str
    population_digest: str

    @property
    def fit_count(self) -> int:
        """Number of rows eligible for fitting."""

        return len(self.fit_rows)

    @property
    def purged_count(self) -> int:
        """Number of rows withheld between fit and inner validation."""

        return len(self.purged_rows)

    @property
    def inner_validation_count(self) -> int:
        """Number of rows eligible for inner validation."""

        return len(self.inner_validation_rows)


def _row_payload(row: AsOfRow) -> dict[str, object]:
    """Return the canonical digest payload for one row."""

    return {
        "record_id": row.record_id,
        "event_time": _iso(row.event_time),
        "available_time": _iso(row.available_time),
        "target_available_time": (
            _iso(row.target_available_time)
            if row.target_available_time is not None
            else None
        ),
        "row_digest": row.row_digest,
    }


def _digest(payload: object) -> str:
    """Hash a canonical JSON value with an explicit algorithm prefix."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _population_digest(rows: Tuple[AsOfRow, ...]) -> str:
    """Bind an ordered population to all row identities and availability clocks."""

    return _digest([_row_payload(row) for row in rows])


def _validate_rows(
    rows: Iterable[AsOfRow],
    *,
    week: WeekSpec,
    support: SupportSpec,
) -> Tuple[AsOfRow, ...]:
    """Validate and deterministically order candidate supervised rows."""

    ordered = []
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    assert week.fit_start is not None

    for row in rows:
        if not isinstance(row, AsOfRow):
            raise TypeError("rows must contain only AsOfRow records")
        if row.record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {row.record_id}")
        if row.row_digest in seen_digests:
            raise ValueError(f"duplicate row_digest: {row.row_digest}")
        seen_ids.add(row.record_id)
        seen_digests.add(row.row_digest)

        if not week.fit_start <= row.event_time < week.cutoff:
            raise ValueError(
                f"row outside rolling window: {row.record_id}"
            )
        if row.event_time - support.input_lookback < week.fit_start:
            raise ValueError(
                f"input support crosses rolling window: {row.record_id}"
            )
        if row.available_time > row.event_time:
            raise ValueError(
                f"feature bytes unavailable at event_time: {row.record_id}"
            )
        if row.target_available_time is None or row.target_available_time > week.cutoff:
            raise ValueError(f"label unavailable by cutoff: {row.record_id}")
        target_support_end = row.event_time + support.target_horizon
        if row.target_available_time < target_support_end:
            raise ValueError(f"label predates target support: {row.record_id}")
        support_end = row.event_time + support.purge
        if support_end > week.cutoff:
            raise ValueError(f"future support crosses cutoff: {row.record_id}")
        ordered.append(row)

    return tuple(
        sorted(
            ordered,
            key=lambda item: (item.event_time, item.record_id, item.row_digest),
        )
    )


def resolve_asof_window(
    week: WeekSpec,
    rows: Iterable[AsOfRow],
    support: SupportSpec,
) -> ResolvedAsOfWindow:
    """Resolve one exact rolling BUSINESS training population.

    The last ``inner_validation_weeks`` before the scored week are shifted back
    by the maximum future support.  A second support-sized gap separates that
    inner validation population from fitting.  This prevents labels, holding
    outcomes, or target paths from crossing either boundary.
    """

    if not isinstance(week, WeekSpec):
        raise TypeError("week must be a WeekSpec")
    if not isinstance(support, SupportSpec):
        raise TypeError("support must be a SupportSpec")
    if week.fit_start is None:
        raise ValueError("week.fit_start is required for BUSINESS resolution")
    expected_fit_start = subtract_calendar_years(week.cutoff, 4)
    if week.fit_start != expected_fit_start:
        raise ValueError(
            "week.fit_start must be exactly four calendar years before cutoff"
        )

    inner_validation_end = week.cutoff - support.purge
    inner_validation_start = (
        inner_validation_end - support.inner_validation_duration
    )
    fit_end = inner_validation_start - support.purge
    earliest_origin = week.fit_start + support.input_lookback
    if fit_end <= earliest_origin:
        raise ValueError("temporal support leaves no possible fit population")

    ordered = _validate_rows(rows, week=week, support=support)
    fit_rows = tuple(row for row in ordered if row.event_time < fit_end)
    purged_rows = tuple(
        row
        for row in ordered
        if fit_end <= row.event_time < inner_validation_start
    )
    inner_validation_rows = tuple(
        row
        for row in ordered
        if inner_validation_start <= row.event_time <= inner_validation_end
    )

    fit_digest = _population_digest(fit_rows)
    purged_digest = _population_digest(purged_rows)
    inner_digest = _population_digest(inner_validation_rows)
    complete_digest = _digest(
        {
            "week": {
                "split": week.split.value,
                "ordinal": week.ordinal,
                "start": _iso(week.start),
                "end": _iso(week.end),
                "cutoff": _iso(week.cutoff),
                "fit_start": _iso(week.fit_start),
            },
            "support": {
                "input_lookback_us": support.input_lookback // timedelta(microseconds=1),
                "target_horizon_us": support.target_horizon // timedelta(microseconds=1),
                "maximum_holding_support_us": (
                    support.maximum_holding_support // timedelta(microseconds=1)
                ),
                "purge_us": support.purge // timedelta(microseconds=1),
                "inner_validation_weeks": support.inner_validation_weeks,
            },
            "boundaries": {
                "fit_end": _iso(fit_end),
                "inner_validation_start": _iso(inner_validation_start),
                "inner_validation_end": _iso(inner_validation_end),
            },
            "populations": {
                "fit": {"count": len(fit_rows), "digest": fit_digest},
                "purged": {"count": len(purged_rows), "digest": purged_digest},
                "inner_validation": {
                    "count": len(inner_validation_rows),
                    "digest": inner_digest,
                },
            },
        }
    )

    return ResolvedAsOfWindow(
        week=week,
        support=support,
        fit_start=week.fit_start,
        fit_end=fit_end,
        inner_validation_start=inner_validation_start,
        inner_validation_end=inner_validation_end,
        fit_rows=fit_rows,
        purged_rows=purged_rows,
        inner_validation_rows=inner_validation_rows,
        fit_population_digest=fit_digest,
        purged_population_digest=purged_digest,
        inner_validation_population_digest=inner_digest,
        population_digest=complete_digest,
    )

