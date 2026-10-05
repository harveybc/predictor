"""Pure contracts for business-faithful weekly model evaluation.

This module deliberately contains no training or storage implementation.  It
defines the calendar, point-in-time, identity, and closure rules that an
orchestrator must satisfy before it can call a result a BUSINESS evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple


WEEK = timedelta(days=7)


class EvaluationMode(str, Enum):
    """Supported evaluation calendars."""

    BUSINESS_WEEKLY_WALK_FORWARD = "BUSINESS_WEEKLY_WALK_FORWARD"
    BUSINESS_MONTHLY_WALK_FORWARD = "BUSINESS_MONTHLY_WALK_FORWARD"
    LITERATURE_STATIC = "LITERATURE_STATIC"


class UpdateMode(str, Enum):
    """Supported model-update semantics for BUSINESS evaluation."""

    FULL_RETRAIN_ROLLING_4Y = "FULL_RETRAIN_ROLLING_4Y"
    WARM_UPDATE_ROLLING_4Y = "WARM_UPDATE_ROLLING_4Y"


class EvaluationSplit(str, Enum):
    """Externally scored portions of the evaluation calendar."""

    VALIDATION = "validation"
    TEST = "test"


class DispositionStatus(str, Enum):
    """Terminal states allowed for a planned week."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXCLUDED = "EXCLUDED"


class LeapDayPolicy(str, Enum):
    """Policy used when subtracting years from February 29.

    ``CLAMP_TO_FEBRUARY_28`` maps February 29 to February 28 when the target
    year has no February 29.  All other month/day pairs are preserved.
    """

    CLAMP_TO_FEBRUARY_28 = "CLAMP_TO_FEBRUARY_28"


def _require_plain_int(name: str, value: object) -> int:
    """Return an integer while rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    return value


def _require_utc_boundary(name: str, value: datetime) -> datetime:
    """Validate a Monday 00:00 UTC interval boundary."""

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.weekday() != 0 or any(
        (value.hour, value.minute, value.second, value.microsecond)
    ):
        raise ValueError(f"{name} must be Monday 00:00:00 UTC")
    return value


def _require_utc_instant(name: str, value: datetime) -> datetime:
    """Validate a timezone-aware UTC instant."""

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value


def subtract_calendar_years(
    instant: datetime,
    years: int,
    policy: LeapDayPolicy = LeapDayPolicy.CLAMP_TO_FEBRUARY_28,
) -> datetime:
    """Subtract whole calendar years without approximating them as days.

    Args:
        instant: UTC instant from which calendar years are subtracted.
        years: Positive, non-boolean number of calendar years.
        policy: Explicit handling of February 29 in a non-leap target year.

    Returns:
        The corresponding instant in the target calendar year.
    """

    _require_utc_instant("instant", instant)
    years = _require_plain_int("years", years)
    if years <= 0:
        raise ValueError("years must be positive")
    if policy is not LeapDayPolicy.CLAMP_TO_FEBRUARY_28:
        raise ValueError(f"unsupported leap-day policy: {policy!r}")

    target_year = instant.year - years
    try:
        return instant.replace(year=target_year)
    except ValueError:
        if instant.month == 2 and instant.day == 29:
            return instant.replace(year=target_year, day=28)
        raise


@dataclass(frozen=True)
class WeekSpec:
    """One immutable, scored week in a complete external interval."""

    split: EvaluationSplit
    ordinal: int
    start: datetime
    end: datetime
    cutoff: datetime
    fit_start: Optional[datetime]
    retrain_due: bool

    def __post_init__(self) -> None:
        if not isinstance(self.split, EvaluationSplit):
            raise TypeError("split must be an EvaluationSplit")
        _require_plain_int("ordinal", self.ordinal)
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        _require_utc_boundary("start", self.start)
        _require_utc_boundary("end", self.end)
        _require_utc_instant("cutoff", self.cutoff)
        if self.fit_start is not None:
            _require_utc_instant("fit_start", self.fit_start)
        if self.end - self.start != WEEK:
            raise ValueError("a WeekSpec must span exactly seven days")
        if self.cutoff != self.start:
            raise ValueError("cutoff must equal the scored week's start")
        if type(self.retrain_due) is not bool:
            raise TypeError("retrain_due must be a bool")

    @property
    def key(self) -> Tuple[EvaluationSplit, datetime]:
        """Return the stable calendar key for this week."""

        return self.split, self.start


@dataclass(frozen=True)
class Disposition:
    """Explicit terminal disposition for one planned week."""

    week: WeekSpec
    status: DispositionStatus
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, DispositionStatus):
            raise TypeError("status must be a DispositionStatus")
        if self.status is DispositionStatus.COMPLETED and self.reason is not None:
            raise ValueError("a completed week must not carry a failure reason")
        if self.status is not DispositionStatus.COMPLETED and not self.reason:
            raise ValueError("failed or excluded weeks require a reason")


@dataclass(frozen=True)
class AvailableRecord:
    """Minimal point-in-time identity for a candidate training record."""

    record_id: str
    available_time: datetime

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        _require_utc_instant("available_time", self.available_time)


@dataclass(frozen=True)
class WeeklyModelIdentity:
    """Identity of changing weekly weights under one frozen procedure."""

    procedure_digest: str
    week_start: datetime
    model_digest: str
    update_mode: UpdateMode
    parent_digest: Optional[str]
    strict_four_year_memory: bool


@dataclass(frozen=True)
class BusinessWeeklyProtocol:
    """Immutable protocol definition for validation and test traversal."""

    evaluation_mode: EvaluationMode
    update_mode: Optional[UpdateMode]
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    procedure_digest: str
    rolling_years: int = 4
    leap_day_policy: LeapDayPolicy = LeapDayPolicy.CLAMP_TO_FEBRUARY_28

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_mode, EvaluationMode):
            raise TypeError("evaluation_mode must be an EvaluationMode")
        if self.update_mode is not None and not isinstance(self.update_mode, UpdateMode):
            raise TypeError("update_mode must be an UpdateMode or None")
        if not isinstance(self.leap_day_policy, LeapDayPolicy):
            raise TypeError("leap_day_policy must be a LeapDayPolicy")
        for name in (
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        ):
            _require_utc_boundary(name, getattr(self, name))
        if not self.validation_start < self.validation_end <= self.test_start < self.test_end:
            raise ValueError("validation and test must be non-empty, ordered intervals")
        for start, end in (
            (self.validation_start, self.validation_end),
            (self.test_start, self.test_end),
        ):
            if (end - start) % WEEK:
                raise ValueError("each interval must contain complete UTC weeks")

        years = _require_plain_int("rolling_years", self.rolling_years)
        if years != 4:
            raise ValueError("BUSINESS protocol requires exactly four calendar years")
        if not self.procedure_digest:
            raise ValueError("procedure_digest must be non-empty")

        if self.evaluation_mode is EvaluationMode.LITERATURE_STATIC:
            if self.update_mode is not None:
                raise ValueError("LITERATURE_STATIC must not declare a BUSINESS update mode")
        elif not isinstance(self.update_mode, UpdateMode):
            raise ValueError("BUSINESS modes require an explicit update mode")

    @property
    def procedure_identity(self) -> str:
        """Return the frozen procedure identity, independent of weekly weights."""

        return self.procedure_digest

    def weeks(self) -> Tuple[WeekSpec, ...]:
        """Enumerate every week in validation and test without omission."""

        weeks = []
        ordinal = 0
        previous_month: Optional[Tuple[int, int]] = None
        for split, interval_start, interval_end in (
            (EvaluationSplit.VALIDATION, self.validation_start, self.validation_end),
            (EvaluationSplit.TEST, self.test_start, self.test_end),
        ):
            # Validation and test are independent traversals. The first scored
            # week of either split always needs its own contemporaneous release.
            previous_month = None
            cursor = interval_start
            while cursor < interval_end:
                month = (cursor.year, cursor.month)
                if self.evaluation_mode is EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD:
                    retrain_due = True
                elif self.evaluation_mode is EvaluationMode.BUSINESS_MONTHLY_WALK_FORWARD:
                    retrain_due = month != previous_month
                else:
                    retrain_due = False
                weeks.append(
                    WeekSpec(
                        split=split,
                        ordinal=ordinal,
                        start=cursor,
                        end=cursor + WEEK,
                        cutoff=cursor,
                        fit_start=(
                            None
                            if self.evaluation_mode is EvaluationMode.LITERATURE_STATIC
                            else subtract_calendar_years(
                                cursor, self.rolling_years, self.leap_day_policy
                            )
                        ),
                        retrain_due=retrain_due,
                    )
                )
                ordinal += 1
                previous_month = month
                cursor += WEEK
        return tuple(weeks)

    def require_available(
        self, records: Iterable[AvailableRecord], cutoff: datetime
    ) -> Tuple[AvailableRecord, ...]:
        """Reject records that were unavailable at a weekly cutoff."""

        _require_utc_instant("cutoff", cutoff)
        admitted = tuple(records)
        late = tuple(r.record_id for r in admitted if r.available_time > cutoff)
        if late:
            raise ValueError(f"records unavailable at cutoff: {', '.join(late)}")
        return admitted

    def bind_model(
        self,
        week: WeekSpec,
        model_digest: str,
        *,
        parent_digest: Optional[str] = None,
        strict_four_year_memory: bool,
    ) -> WeeklyModelIdentity:
        """Bind changing model bytes to the frozen procedure and one week."""

        if self.evaluation_mode is EvaluationMode.LITERATURE_STATIC:
            raise ValueError("LITERATURE_STATIC refuses BUSINESS weekly model labels")
        if week not in self.weeks():
            raise ValueError("week is not part of this protocol")
        if not model_digest:
            raise ValueError("model_digest must be non-empty")
        if model_digest == self.procedure_digest:
            raise ValueError("model_digest must differ from procedure_digest")
        if type(strict_four_year_memory) is not bool:
            raise TypeError("strict_four_year_memory must be a bool")

        if self.update_mode is UpdateMode.WARM_UPDATE_ROLLING_4Y:
            if not parent_digest:
                raise ValueError("warm update requires parent_digest")
            if strict_four_year_memory:
                raise ValueError("warm update cannot claim strict four-year memory")
        else:
            if parent_digest is not None:
                raise ValueError("full retraining must not declare a parent checkpoint")
            if not strict_four_year_memory:
                raise ValueError("full retraining must declare strict four-year memory")

        return WeeklyModelIdentity(
            procedure_digest=self.procedure_digest,
            week_start=week.start,
            model_digest=model_digest,
            update_mode=self.update_mode,
            parent_digest=parent_digest,
            strict_four_year_memory=strict_four_year_memory,
        )

    def business_label(self) -> str:
        """Return the BUSINESS cadence label or reject literature mode."""

        if self.evaluation_mode is EvaluationMode.LITERATURE_STATIC:
            raise ValueError("LITERATURE_STATIC refuses BUSINESS labels")
        return self.evaluation_mode.value

    def close(self, dispositions: Sequence[Disposition]) -> Tuple[Disposition, ...]:
        """Validate one and only one explicit disposition per planned week."""

        planned = {week.key: week for week in self.weeks()}
        observed = {}
        for disposition in dispositions:
            key = disposition.week.key
            if key in observed:
                raise ValueError(f"duplicate disposition for {key}")
            if key not in planned or disposition.week != planned[key]:
                raise ValueError(f"disposition references an unplanned week: {key}")
            observed[key] = disposition
        missing = tuple(key for key in planned if key not in observed)
        if missing:
            raise ValueError(f"missing weekly dispositions: {missing}")
        return tuple(observed[week.key] for week in self.weeks())
