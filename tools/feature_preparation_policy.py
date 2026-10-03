"""Dependency-free contracts for leakage-safe feature preparation.

This module defines what a feature group means and which state may cross from
TRAIN into validation, test, or live use. It deliberately does not score or
rank features and does not read datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Iterable, Sequence


CAUSAL_LAG_HOURS = (0, 1, 2, 6, 24, 48, 168)
UNKNOWN_CATEGORY = "UNKNOWN"


class SemanticRole(str, Enum):
    """Semantic role of a complete selectable feature group."""

    CONTINUOUS = "CONTINUOUS"
    BINARY = "BINARY"
    CYCLIC_PAIR = "CYCLIC_PAIR"
    CATEGORICAL_GROUP = "CATEGORICAL_GROUP"
    MASK = "MASK"
    DELTA_TIME = "DELTA_TIME"
    QUALITY_ONLY = "QUALITY_ONLY"
    SELECTOR_EPISODE_ONLY = "SELECTOR_EPISODE_ONLY"


class Representation(str, Enum):
    POINT = "POINT"
    LAGGED = "LAGGED"
    WINDOW_168 = "WINDOW_168"


class ValueVariant(str, Enum):
    RAW = "RAW"
    TRAIN_Q005_CLIP = "TRAIN_Q005_CLIP"


class Partition(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    LIVE = "LIVE"


class CausalStatus(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    NOT_IDENTIFIED = "NOT_IDENTIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class CausalDisposition(str, Enum):
    SUPPORTING = "SUPPORTING"
    NEUTRAL = "NEUTRAL"
    REJECTING = "REJECTING"


def _require_non_bool_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _require_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not a boolean")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class FeatureGroupSpec:
    """Immutable declaration of one atomic semantic selection unit."""

    group_id: str
    role: SemanticRole
    members: tuple[str, ...]
    representation: Representation = Representation.POINT
    elapsed_lag_hours: tuple[int, ...] = CAUSAL_LAG_HOURS
    fixed_calendar_context: bool = False
    economic_calendar: bool = False
    deferred_until_stage: str | None = None
    source_allows_carry: bool = False
    max_carry_age_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if isinstance(self.role, str):
            object.__setattr__(self, "role", SemanticRole(self.role))
        if isinstance(self.representation, str):
            object.__setattr__(self, "representation", Representation(self.representation))
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError("members must be a non-empty tuple")
        if any(not isinstance(member, str) or not member for member in self.members):
            raise ValueError("every member must be a non-empty string")
        if len(set(self.members)) != len(self.members):
            raise ValueError("members must be unique")
        if self.role is SemanticRole.CYCLIC_PAIR and len(self.members) != 2:
            raise ValueError("CYCLIC_PAIR requires exactly two members")
        if self.representation is Representation.LAGGED:
            if self.elapsed_lag_hours != CAUSAL_LAG_HOURS:
                raise ValueError("LAGGED must use the sealed elapsed-hour set")
        elif self.elapsed_lag_hours != CAUSAL_LAG_HOURS:
            raise ValueError("custom lag sets are not part of this contract")
        for lag in self.elapsed_lag_hours:
            _require_non_bool_int("elapsed lag", lag, minimum=0)
        if self.source_allows_carry:
            if self.max_carry_age_seconds is None:
                raise ValueError("source carry requires max_carry_age_seconds")
            _require_non_bool_int(
                "max_carry_age_seconds", self.max_carry_age_seconds, minimum=1
            )
        elif self.max_carry_age_seconds is not None:
            raise ValueError("carry age requires explicit source allowance")
        if self.fixed_calendar_context and self.role not in {
            SemanticRole.CYCLIC_PAIR,
            SemanticRole.BINARY,
            SemanticRole.CATEGORICAL_GROUP,
        }:
            raise ValueError("fixed calendar context must use a calendar-compatible role")
        if self.economic_calendar:
            if self.role is not SemanticRole.SELECTOR_EPISODE_ONLY:
                raise ValueError("economic calendar must be SELECTOR_EPISODE_ONLY")
            if not self.deferred_until_stage:
                raise ValueError("economic calendar requires a deferred stage")

    @property
    def selectable(self) -> bool:
        return self.role not in {
            SemanticRole.QUALITY_ONLY,
            SemanticRole.SELECTOR_EPISODE_ONLY,
        } and not self.fixed_calendar_context


@dataclass(frozen=True)
class TemporalObservation:
    """One temporal value with explicit missingness and elapsed time."""

    value: float | int | None
    observed_mask: int
    delta_time: float | int

    def __post_init__(self) -> None:
        mask = _require_non_bool_int("observed_mask", self.observed_mask, minimum=0)
        if mask not in (0, 1):
            raise ValueError("observed_mask must be 0 or 1")
        delta = _require_number("delta_time", self.delta_time)
        if delta < 0:
            raise ValueError("delta_time must be non-negative")
        if self.value is None and mask != 0:
            raise ValueError("missing value requires observed_mask=0")
        if self.value is not None:
            _require_number("value", self.value)
            if mask != 1:
                raise ValueError("observed value requires observed_mask=1")


@dataclass(frozen=True)
class TrainFitState:
    """State fitted exclusively from one group's TRAIN observations."""

    group_id: str
    role: SemanticRole
    variant: ValueVariant
    fitted_partition: Partition
    mean: float | None
    std: float | None
    median: float | None
    clip_bounds: tuple[float, float] | None
    vocabulary: tuple[str, ...]
    constant: bool
    identity_digest: str

    def __post_init__(self) -> None:
        if self.fitted_partition is not Partition.TRAIN:
            raise ValueError("fit state must be fitted on TRAIN")


@dataclass(frozen=True)
class PreparedOuterData:
    fit_state: TrainFitState
    partition: Partition
    values: tuple[float | str, ...]
    observed_mask: tuple[int, ...] = ()
    delta_time: tuple[float, ...] = ()


@dataclass(frozen=True)
class FeaturePreparationPolicy:
    """Collection-level validation and semantic-group selection boundary."""

    groups: tuple[FeatureGroupSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.groups, tuple):
            raise TypeError("groups must be a tuple")
        ids = [group.group_id for group in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("group_id values must be unique")
        members = [member for group in self.groups for member in group.members]
        if len(members) != len(set(members)):
            raise ValueError("a physical member may belong to only one semantic group")

    @property
    def context_group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.groups if group.fixed_calendar_context)

    @property
    def selectable_group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.groups if group.selectable)

    @property
    def subset_k_group_ids(self) -> tuple[str, ...]:
        return self.selectable_group_ids

    def validate_selection(
        self, selected_group_ids: Sequence[str], *, subset_k: int
    ) -> tuple[str, ...]:
        k = _require_non_bool_int("subset_k", subset_k, minimum=0)
        selected = tuple(selected_group_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("selected semantic groups must be unique")
        group_by_id = {group.group_id: group for group in self.groups}
        member_names = {member for group in self.groups for member in group.members}
        for group_id in selected:
            if group_id not in group_by_id:
                if group_id in member_names:
                    raise ValueError("selection unit must be the complete semantic group")
                raise ValueError(f"unknown semantic group: {group_id}")
            group = group_by_id[group_id]
            if group.fixed_calendar_context:
                raise ValueError("fixed calendar context is excluded from subset K")
            if group.economic_calendar:
                raise ValueError(
                    f"economic calendar is deferred until {group.deferred_until_stage}"
                )
            if not group.selectable:
                raise ValueError(f"semantic group is not selectable: {group_id}")
        if len(selected) != k:
            raise ValueError("selection cardinality must equal subset_k")
        return selected

    @staticmethod
    def causal_disposition(status: CausalStatus) -> CausalDisposition:
        if isinstance(status, str):
            status = CausalStatus(status)
        if status is CausalStatus.IDENTIFIED:
            return CausalDisposition.SUPPORTING
        if status is CausalStatus.REJECTED:
            return CausalDisposition.REJECTING
        return CausalDisposition.NEUTRAL


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _state_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fit_train_state(
    spec: FeatureGroupSpec,
    train_values: Iterable[object],
    variant: ValueVariant = ValueVariant.RAW,
    *,
    partition: Partition = Partition.TRAIN,
) -> TrainFitState:
    """Fit immutable scaling, imputation, clipping, or vocabulary on TRAIN."""

    if isinstance(partition, str):
        partition = Partition(partition)
    if partition is not Partition.TRAIN:
        raise ValueError("fit_train_state accepts TRAIN only")
    if isinstance(variant, str):
        variant = ValueVariant(variant)
    values = tuple(train_values)

    if spec.representation is not Representation.POINT:
        observations: list[TemporalObservation] = []
        for value in values:
            if not isinstance(value, TemporalObservation):
                raise ValueError(
                    "temporal TRAIN inputs require TemporalObservation(value, mask, delta)"
                )
            observations.append(value)
        values = tuple(observation.value for observation in observations)

    if spec.role is SemanticRole.CATEGORICAL_GROUP:
        categories: list[str] = []
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError("categorical TRAIN values must be strings")
            categories.append(value)
        vocabulary = (UNKNOWN_CATEGORY,) + tuple(
            sorted(category for category in set(categories) if category != UNKNOWN_CATEGORY)
        )
        payload = {
            "group_id": spec.group_id,
            "role": spec.role.value,
            "variant": variant.value,
            "vocabulary": vocabulary,
        }
        return TrainFitState(
            group_id=spec.group_id,
            role=spec.role,
            variant=variant,
            fitted_partition=Partition.TRAIN,
            mean=None,
            std=None,
            median=None,
            clip_bounds=None,
            vocabulary=vocabulary,
            constant=False,
            identity_digest=_state_digest(payload),
        )

    numeric = [_require_number("TRAIN value", value) for value in values if value is not None]
    if not numeric:
        raise ValueError("numeric TRAIN values must contain at least one observation")
    clip_bounds = None
    fitted = numeric
    if variant is ValueVariant.TRAIN_Q005_CLIP:
        clip_bounds = (_quantile(numeric, 0.005), _quantile(numeric, 0.995))
        fitted = [min(max(value, clip_bounds[0]), clip_bounds[1]) for value in numeric]
    median = float(statistics.median(fitted))
    passthrough_role = spec.role in {
        SemanticRole.BINARY,
        SemanticRole.CYCLIC_PAIR,
        SemanticRole.MASK,
    }
    mean = None if passthrough_role else float(statistics.fmean(fitted))
    measured_std = float(statistics.pstdev(fitted))
    constant = measured_std < 1e-8
    effective_std = None if passthrough_role else (1.0 if constant else measured_std)
    payload = {
        "group_id": spec.group_id,
        "role": spec.role.value,
        "variant": variant.value,
        "mean": mean,
        "std": effective_std,
        "median": median,
        "clip_bounds": clip_bounds,
        "constant": constant,
    }
    return TrainFitState(
        group_id=spec.group_id,
        role=spec.role,
        variant=variant,
        fitted_partition=Partition.TRAIN,
        mean=mean,
        std=effective_std,
        median=median,
        clip_bounds=clip_bounds,
        vocabulary=(),
        constant=constant,
        identity_digest=_state_digest(payload),
    )


def _transform_number(value: object, state: TrainFitState) -> float:
    numeric = state.median if value is None else _require_number("outer value", value)
    assert numeric is not None
    if state.clip_bounds is not None:
        numeric = min(max(numeric, state.clip_bounds[0]), state.clip_bounds[1])
    if state.mean is None and state.std is None:
        return numeric
    assert state.mean is not None and state.std is not None
    return (numeric - state.mean) / state.std


def prepare_outer(
    spec: FeatureGroupSpec,
    state: TrainFitState,
    outer_values: Iterable[object],
    partition: Partition,
) -> PreparedOuterData:
    """Transform validation, test, or live values without fitting new state."""

    if isinstance(partition, str):
        partition = Partition(partition)
    if partition is Partition.TRAIN:
        raise ValueError("prepare_outer requires an outer partition")
    if state.fitted_partition is not Partition.TRAIN or state.group_id != spec.group_id:
        raise ValueError("outer transform requires matching TRAIN-fit state")
    values = tuple(outer_values)

    if spec.role is SemanticRole.CATEGORICAL_GROUP:
        transformed = tuple(
            value if isinstance(value, str) and value in state.vocabulary else UNKNOWN_CATEGORY
            for value in values
        )
        return PreparedOuterData(state, partition, transformed)

    if spec.representation is Representation.POINT:
        transformed = tuple(_transform_number(value, state) for value in values)
        return PreparedOuterData(state, partition, transformed)

    observations: list[TemporalObservation] = []
    for value in values:
        if not isinstance(value, TemporalObservation):
            raise ValueError(
                "temporal missing inputs require TemporalObservation(value, mask, delta)"
            )
        observations.append(value)
    transformed_values = tuple(
        0.0 if observation.value is None else _transform_number(observation.value, state)
        for observation in observations
    )
    return PreparedOuterData(
        state,
        partition,
        transformed_values,
        tuple(observation.observed_mask for observation in observations),
        tuple(float(observation.delta_time) for observation in observations),
    )


__all__ = [
    "CAUSAL_LAG_HOURS",
    "UNKNOWN_CATEGORY",
    "CausalDisposition",
    "CausalStatus",
    "FeatureGroupSpec",
    "FeaturePreparationPolicy",
    "Partition",
    "PreparedOuterData",
    "Representation",
    "SemanticRole",
    "TemporalObservation",
    "TrainFitState",
    "ValueVariant",
    "fit_train_state",
    "prepare_outer",
]
