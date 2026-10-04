"""Fail-closed scoring of one weekly forecast release against its paired naive.

MAE is the arithmetic mean of absolute errors, the same reduction as
``tools.df_adequacy_models.mae``. MSE is the arithmetic mean of squared
errors. Both stay in this module: importing the adequacy trainer would set
``CUDA_VISIBLE_DEVICES``, and the TensorFlow magnitude helper scores only the
first column of a two-column plugin tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from tools.business_objective_firewall import (
    Actor,
    BusinessObjectiveFirewall,
    FirewallError,
    Phase,
)
from tools.business_weekly_protocol import DispositionStatus, EvaluationSplit, WeekSpec


NOT_CONFIGURED = "NOT_CONFIGURED"
PRIMARY_ERROR_METRIC_FIELD = "primary_error_metric"
FINANCIAL_TASK = "financial_forecast"
DECLARED_TARGET_SCALE = "declared_target_scale"
NOT_ESTABLISHED = "NOT_ESTABLISHED"


class ForecastFamily(str, Enum):
    """Short and long forecast families stay separately identified."""

    SHORT = "short"
    LONG = "long"


class ScoreRefusal(ValueError):
    """Raised before any metric is emitted when a release is not scoreable."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def repository_primary_error_metric() -> str:
    """Return the primary metric named by the weekly business contract.

    Section 3 freezes a metric catalog before the external test. Neither that
    catalog nor ``BusinessWeeklyProtocol`` names ``primary_error_metric``.
    """

    return NOT_CONFIGURED


def financial_task_contract(
    required_families: tuple[ForecastFamily, ...] = (ForecastFamily.SHORT, ForecastFamily.LONG),
    required_horizons: tuple[str, ...] = (),
) -> "ScoringContract":
    """MAE on each target's declared scale is primary. MSE stays secondary.

    The repository default stays ``NOT_CONFIGURED`` until a business contract names it.
    """

    return ScoringContract(
        primary_error_metric="mae",
        required_families=required_families,
        required_horizons=required_horizons,
        task=FINANCIAL_TASK,
        scale_policy=DECLARED_TARGET_SCALE,
    )


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoreRefusal("IDENTITY", f"{name} must be a non-empty string")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ScoreRefusal("EMPTY", "a horizon with no finite origins has no metric")
    return sum(values) / len(values)


def _skill(model: float, naive: float) -> Optional[float]:
    if naive == 0.0:
        return None
    return (naive - model) / naive


@dataclass(frozen=True)
class ScoringContract:
    """Explicit comparison rule. The repository default leaves the gate closed."""

    primary_error_metric: str = NOT_CONFIGURED
    required_families: tuple[ForecastFamily, ...] = (ForecastFamily.SHORT, ForecastFamily.LONG)
    required_horizons: tuple[str, ...] = ()
    task: str = ""
    scale_policy: str = DECLARED_TARGET_SCALE

    def __post_init__(self) -> None:
        allowed = {NOT_CONFIGURED, "mae", "mse"}
        if self.primary_error_metric not in allowed:
            raise ScoreRefusal(
                "PRIMARY_METRIC",
                f"{PRIMARY_ERROR_METRIC_FIELD} must be mae, mse, or {NOT_CONFIGURED}",
            )
        families = tuple(self.required_families)
        if not families or any(not isinstance(item, ForecastFamily) for item in families):
            raise ScoreRefusal("FAMILY", "required families must be explicit ForecastFamily values")
        if not isinstance(self.task, str):
            raise ScoreRefusal("TASK", "task must be a string")
        if self.scale_policy != DECLARED_TARGET_SCALE:
            raise ScoreRefusal("SCALE", "error metrics stay on each target's declared scale")
        object.__setattr__(self, "required_families", families)
        object.__setattr__(self, "required_horizons", tuple(self.required_horizons))


def _contract_payload(contract: ScoringContract) -> dict[str, Any]:
    return {
        "primary_error_metric": contract.primary_error_metric,
        "required_families": [item.value for item in contract.required_families],
        "required_horizons": list(contract.required_horizons),
        "scale_policy": contract.scale_policy,
        "task": contract.task,
    }


def _contract_digest(contract: ScoringContract) -> str:
    return _digest(_contract_payload(contract))


@dataclass(frozen=True)
class HorizonForecast:
    """One horizon bound to ordered origins, the target, the model, and the naive."""

    family: ForecastFamily
    horizon: str
    origins: tuple[str, ...]
    target: tuple[float, ...]
    prediction: tuple[float, ...]
    naive: tuple[float, ...]
    unit: str
    scale: str
    model_digest: str
    dataset_id: str
    week: WeekSpec
    target_origins: tuple[str, ...] | None = None
    prediction_origins: tuple[str, ...] | None = None
    naive_origins: tuple[str, ...] | None = None
    task: str = ""
    population: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.family, ForecastFamily):
            raise ScoreRefusal("FAMILY", "family must be a ForecastFamily")
        if not isinstance(self.week, WeekSpec):
            raise ScoreRefusal("WEEK", "week must be a WeekSpec")
        for name in ("horizon", "unit", "scale", "model_digest", "dataset_id"):
            _require_text(name, getattr(self, name))
        if not isinstance(self.task, str) or not isinstance(self.population, str):
            raise ScoreRefusal("IDENTITY", "task and population must be strings")
        if not isinstance(self.origins, tuple):
            object.__setattr__(self, "origins", tuple(self.origins))
        if not isinstance(self.target, tuple):
            object.__setattr__(self, "target", tuple(self.target))
        if not isinstance(self.prediction, tuple):
            object.__setattr__(self, "prediction", tuple(self.prediction))
        if not isinstance(self.naive, tuple):
            object.__setattr__(self, "naive", tuple(self.naive))


@dataclass(frozen=True)
class WeekRelease:
    """Forecast families presented together for one scored week."""

    week: WeekSpec
    horizons: tuple[HorizonForecast, ...]
    consumes_short_and_long: bool = True
    selection_payload: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.week, WeekSpec):
            raise ScoreRefusal("WEEK", "week must be a WeekSpec")
        if type(self.consumes_short_and_long) is not bool:
            raise ScoreRefusal("FAMILY", "consumes_short_and_long must be a bool")
        object.__setattr__(self, "horizons", tuple(self.horizons))


@dataclass(frozen=True)
class OriginDifference:
    """Paired per-origin difference of absolute errors: model minus naive."""

    origin: str
    absolute_error_difference: float


@dataclass(frozen=True)
class HorizonScore:
    """Metrics for one horizon. Overlapping hours are not extra replicates."""

    family: str
    horizon: str
    sample_count: int
    model_mae: float
    model_mse: float
    naive_mae: float
    naive_mse: float
    unit: str
    scale: str
    week_start: str
    model_digest: str
    dataset_id: str
    release_digest: str
    paired_differences: tuple[OriginDifference, ...]
    skill_mae: Optional[float]
    skill_mse: Optional[float]
    primary_error_metric: str
    beats_primary_naive: Optional[bool]


@dataclass(frozen=True)
class WeekScore:
    """Terminal disposition of one week. Eligibility never fills a missing family."""

    week_start: str
    split: str
    disposition: str
    reason: Optional[str]
    strategy_eligible: bool
    primary_error_metric: str
    contract_digest: str
    week_identity_digest: str
    horizons: tuple[HorizonScore, ...]
    digest: str


@dataclass(frozen=True)
class AnnualClose:
    """Sealed weeks are longitudinal units. Their independence is not established."""

    expected_weeks: int
    denominator: int
    longitudinal_evaluation_units: int
    independence_status: str
    origin_rows: int
    eligible_weeks: int
    dispositions: tuple[WeekScore, ...]


def _iso(value: datetime) -> str:
    return value.astimezone(tz=value.tzinfo).strftime("%Y-%m-%dT%H:%M:%SZ")


def _week_payload(week: WeekSpec) -> dict[str, Any]:
    return {
        "cutoff": _iso(week.cutoff),
        "end": _iso(week.end),
        "fit_start": None if week.fit_start is None else _iso(week.fit_start),
        "ordinal": week.ordinal,
        "retrain_due": week.retrain_due,
        "split": week.split.value,
        "start": _iso(week.start),
    }


def _week_digest(week: WeekSpec) -> str:
    return _digest(_week_payload(week))


def _guard_firewall(firewall: Optional[BusinessObjectiveFirewall], release: WeekRelease) -> None:
    """Reject external-test material during validation before any row is read."""

    if firewall is None:
        return
    if firewall.phase is not Phase.VALIDATION_SELECTION:
        return
    payload: dict[str, Any] = {
        "split": release.week.split.value,
        "week_start": _iso(release.week.start),
    }
    if release.selection_payload is not None:
        payload["selection_payload"] = dict(release.selection_payload)
    if release.week.split is EvaluationSplit.TEST:
        raise FirewallError("external test material is forbidden during selection at release.split")
    firewall.record_selection_payload(Actor.CANDIDATE_WORKER, payload)


def _check_origins(forecast: HorizonForecast) -> None:
    lengths = {
        "origins": len(forecast.origins),
        "target": len(forecast.target),
        "prediction": len(forecast.prediction),
        "naive": len(forecast.naive),
    }
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={count}" for name, count in lengths.items())
        raise ScoreRefusal("LENGTH_MISMATCH", detail)
    if not forecast.origins:
        raise ScoreRefusal("EMPTY", f"horizon {forecast.horizon} has no origins")
    seen: set[str] = set()
    previous: Optional[str] = None
    for origin in forecast.origins:
        if not isinstance(origin, str) or not origin:
            raise ScoreRefusal("ORIGIN", "origin identities must be non-empty strings")
        if origin in seen:
            raise ScoreRefusal("DUPLICATE_ORIGIN", origin)
        if previous is not None and not origin > previous:
            raise ScoreRefusal("REORDERED_ORIGIN", f"{previous} then {origin}")
        seen.add(origin)
        previous = origin
    for field, side in (
        ("target", forecast.target_origins),
        ("prediction", forecast.prediction_origins),
        ("naive", forecast.naive_origins),
    ):
        if side is None:
            continue
        side_origins = tuple(side)
        if len(side_origins) != len(forecast.origins):
            if set(side_origins) < set(forecast.origins):
                raise ScoreRefusal("MISSING_ORIGIN", f"{field} missing {set(forecast.origins) - set(side_origins)}")
            raise ScoreRefusal("LENGTH_MISMATCH", f"{field} origins={len(side_origins)}")
        if side_origins != forecast.origins:
            raise ScoreRefusal("ORIGIN_CHANGED", field)


def _finite_at(field: str, origins: tuple[str, ...], values: tuple[float, ...]) -> tuple[float, ...]:
    checked: list[float] = []
    for origin, value in zip(origins, values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoreRefusal("NON_FINITE", f"{field} origin={origin}")
        number = float(value)
        if not math.isfinite(number):
            raise ScoreRefusal("NON_FINITE", f"{field} origin={origin}")
        checked.append(number)
    return tuple(checked)


def _release_digest(forecast: HorizonForecast) -> str:
    return _digest(
        {
            "dataset_id": forecast.dataset_id,
            "family": forecast.family.value,
            "horizon": forecast.horizon,
            "model_digest": forecast.model_digest,
            "naive": list(forecast.naive),
            "origins": list(forecast.origins),
            "population": forecast.population,
            "prediction": list(forecast.prediction),
            "scale": forecast.scale,
            "target": list(forecast.target),
            "task": forecast.task,
            "unit": forecast.unit,
            "week": _week_payload(forecast.week),
        }
    )


def score_horizon(forecast: HorizonForecast, contract: ScoringContract) -> HorizonScore:
    """Score one aligned horizon. Structural refusals happen before the means."""

    _check_origins(forecast)
    target = _finite_at("target", forecast.origins, forecast.target)
    prediction = _finite_at("prediction", forecast.origins, forecast.prediction)
    naive = _finite_at("naive", forecast.origins, forecast.naive)
    model_abs = tuple(abs(left - right) for left, right in zip(target, prediction))
    naive_abs = tuple(abs(left - right) for left, right in zip(target, naive))
    model_sq = tuple((left - right) ** 2 for left, right in zip(target, prediction))
    naive_sq = tuple((left - right) ** 2 for left, right in zip(target, naive))
    model_mae = _mean(model_abs)
    naive_mae = _mean(naive_abs)
    model_mse = _mean(model_sq)
    naive_mse = _mean(naive_sq)
    beats: Optional[bool]
    if contract.primary_error_metric == "mae":
        beats = model_mae < naive_mae
    elif contract.primary_error_metric == "mse":
        beats = model_mse < naive_mse
    else:
        beats = None
    return HorizonScore(
        family=forecast.family.value,
        horizon=forecast.horizon,
        sample_count=len(forecast.origins),
        model_mae=model_mae,
        model_mse=model_mse,
        naive_mae=naive_mae,
        naive_mse=naive_mse,
        unit=forecast.unit,
        scale=forecast.scale,
        week_start=_iso(forecast.week.start),
        model_digest=forecast.model_digest,
        dataset_id=forecast.dataset_id,
        release_digest=_release_digest(forecast),
        paired_differences=tuple(
            OriginDifference(origin, model - baseline)
            for origin, model, baseline in zip(forecast.origins, model_abs, naive_abs)
        ),
        skill_mae=_skill(model_mae, naive_mae),
        skill_mse=_skill(model_mse, naive_mse),
        primary_error_metric=contract.primary_error_metric,
        beats_primary_naive=beats,
    )


def _identity_key(
    forecast: HorizonForecast,
) -> tuple[str, str, str, str, tuple[str, ...], str, str, str]:
    """Shared week identity. Each forecast keeps its own model digest."""

    return (
        forecast.task,
        _iso(forecast.week.cutoff),
        forecast.week.split.value,
        forecast.population,
        forecast.origins,
        forecast.unit,
        forecast.scale,
        _iso(forecast.week.start),
    )


def _paired_differences(raw: object) -> tuple[OriginDifference, ...]:
    if not isinstance(raw, list):
        raise ScoreRefusal("PAIRED_DIFFERENCE", "serialized horizon omits paired origin differences")
    pairs: list[OriginDifference] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ScoreRefusal("PAIRED_DIFFERENCE", "a paired difference must be an object")
        origin = entry.get("origin")
        value = entry.get("absolute_error_difference")
        if not isinstance(origin, str) or not origin:
            raise ScoreRefusal("PAIRED_DIFFERENCE", "paired difference origin is missing")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ScoreRefusal("PAIRED_DIFFERENCE", f"paired difference at {origin} is not finite")
        pairs.append(OriginDifference(origin, float(value)))
    return tuple(pairs)


def _score_payload(score: WeekScore) -> dict[str, Any]:
    return {
        "disposition": score.disposition,
        "horizons": [
            {
                "beats_primary_naive": item.beats_primary_naive,
                "dataset_id": item.dataset_id,
                "family": item.family,
                "horizon": item.horizon,
                "model_digest": item.model_digest,
                "model_mae": item.model_mae,
                "model_mse": item.model_mse,
                "naive_mae": item.naive_mae,
                "naive_mse": item.naive_mse,
                "paired_differences": [
                    {
                        "absolute_error_difference": pair.absolute_error_difference,
                        "origin": pair.origin,
                    }
                    for pair in item.paired_differences
                ],
                "release_digest": item.release_digest,
                "sample_count": item.sample_count,
                "scale": item.scale,
                "skill_mae": item.skill_mae,
                "skill_mse": item.skill_mse,
                "unit": item.unit,
            }
            for item in score.horizons
        ],
        "primary_error_metric": score.primary_error_metric,
        "contract_digest": score.contract_digest,
        "reason": score.reason,
        "split": score.split,
        "strategy_eligible": score.strategy_eligible,
        "week_identity_digest": score.week_identity_digest,
        "week_start": score.week_start,
    }


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreRefusal("SCORE_SCHEMA", f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ScoreRefusal("SCORE_SCHEMA", f"{name} must be finite")
    return number


def _validate_restored_score(score: WeekScore, contract: ScoringContract) -> None:
    if (
        score.primary_error_metric != contract.primary_error_metric
        or score.contract_digest != _contract_digest(contract)
    ):
        raise ScoreRefusal("CONTRACT_MISMATCH", score.week_start)
    if type(score.strategy_eligible) is not bool:
        raise ScoreRefusal("SCORE_SCHEMA", "strategy_eligible must be a bool")
    if score.disposition not in {item.value for item in DispositionStatus} | {"INCOMPLETE"}:
        raise ScoreRefusal("SCORE_SCHEMA", "unsupported disposition")
    if score.reason is not None and (not isinstance(score.reason, str) or not score.reason):
        raise ScoreRefusal("SCORE_SCHEMA", "reason must be null or non-empty text")
    completed = score.disposition == DispositionStatus.COMPLETED.value
    if completed != bool(score.horizons):
        raise ScoreRefusal("SCORE_SCHEMA", "only completed scores carry horizon metrics")
    if not completed:
        if score.strategy_eligible or score.reason is None:
            raise ScoreRefusal("SCORE_SCHEMA", "terminal/incomplete score requires reason and is ineligible")
        return
    seen: set[tuple[str, str]] = set()
    for item in score.horizons:
        if item.family not in {family.value for family in ForecastFamily}:
            raise ScoreRefusal("SCORE_SCHEMA", f"unknown family {item.family!r}")
        slot = (item.family, item.horizon)
        if slot in seen:
            raise ScoreRefusal("SCORE_SCHEMA", f"duplicate horizon {slot}")
        seen.add(slot)
        if type(item.sample_count) is not int or item.sample_count <= 0:
            raise ScoreRefusal("SCORE_SCHEMA", "sample_count must be a positive integer")
        if item.sample_count != len(item.paired_differences):
            raise ScoreRefusal("SCORE_SCHEMA", "sample_count differs from paired population")
        if len({pair.origin for pair in item.paired_differences}) != item.sample_count:
            raise ScoreRefusal("SCORE_SCHEMA", "paired origins must be unique")
        model_mae = _finite_number("model_mae", item.model_mae)
        model_mse = _finite_number("model_mse", item.model_mse)
        naive_mae = _finite_number("naive_mae", item.naive_mae)
        naive_mse = _finite_number("naive_mse", item.naive_mse)
        if min(model_mae, model_mse, naive_mae, naive_mse) < 0.0:
            raise ScoreRefusal("SCORE_SCHEMA", "error metrics cannot be negative")
        model_floor = model_mae**2
        naive_floor = naive_mae**2
        if (
            model_mse < model_floor
            and not math.isclose(model_mse, model_floor, rel_tol=1e-12, abs_tol=0.0)
        ) or (
            naive_mse < naive_floor
            and not math.isclose(naive_mse, naive_floor, rel_tol=1e-12, abs_tol=0.0)
        ):
            raise ScoreRefusal("SCORE_SCHEMA", "MSE cannot be below squared MAE")
        expected_mae_skill = _skill(model_mae, naive_mae)
        expected_mse_skill = _skill(model_mse, naive_mse)
        actual_mae_skill = (
            None if item.skill_mae is None else _finite_number("skill_mae", item.skill_mae)
        )
        actual_mse_skill = (
            None if item.skill_mse is None else _finite_number("skill_mse", item.skill_mse)
        )
        if actual_mae_skill != expected_mae_skill or actual_mse_skill != expected_mse_skill:
            raise ScoreRefusal("SCORE_SCHEMA", "skill is not derived from retained errors")
        expected_beats = (
            model_mae < naive_mae
            if contract.primary_error_metric == "mae"
            else model_mse < naive_mse
            if contract.primary_error_metric == "mse"
            else None
        )
        if item.beats_primary_naive is not expected_beats:
            raise ScoreRefusal("SCORE_SCHEMA", "naive verdict is not derived from retained errors")
        mean_delta = _mean(
            tuple(pair.absolute_error_difference for pair in item.paired_differences)
        )
        if not math.isclose(mean_delta, model_mae - naive_mae, rel_tol=0.0, abs_tol=1e-12):
            raise ScoreRefusal("SCORE_SCHEMA", "paired differences contradict retained MAE")
        for name in ("horizon", "unit", "scale", "model_digest", "dataset_id", "release_digest"):
            _require_text(name, getattr(item, name))
        if item.week_start != score.week_start:
            raise ScoreRefusal("SCORE_SCHEMA", "horizon week differs from score week")
    required = {
        (family.value, horizon)
        for family in contract.required_families
        for horizon in contract.required_horizons
    }
    if required and not required.issubset(seen):
        raise ScoreRefusal("SCORE_SCHEMA", "retained score omits a required family/horizon")
    expected_eligible = (
        bool(score.horizons)
        and score.disposition == DispositionStatus.COMPLETED.value
        and contract.primary_error_metric != NOT_CONFIGURED
        and all(item.beats_primary_naive is True for item in score.horizons)
    )
    if score.strategy_eligible is not expected_eligible:
        raise ScoreRefusal("SCORE_SCHEMA", "strategy eligibility is not derived from retained metrics")
    if (score.strategy_eligible and score.reason is not None) or (
        not score.strategy_eligible and score.reason is None
    ):
        raise ScoreRefusal("SCORE_SCHEMA", "reason contradicts completed score eligibility")


def _freeze(score: WeekScore) -> WeekScore:
    digest = _digest(_score_payload(score))
    return WeekScore(
        week_start=score.week_start,
        split=score.split,
        disposition=score.disposition,
        reason=score.reason,
        strategy_eligible=score.strategy_eligible,
        primary_error_metric=score.primary_error_metric,
        contract_digest=score.contract_digest,
        week_identity_digest=score.week_identity_digest,
        horizons=score.horizons,
        digest=digest,
    )


def score_week(
    release: WeekRelease,
    contract: ScoringContract,
    *,
    firewall: Optional[BusinessObjectiveFirewall] = None,
    strategy: Optional[Callable[[WeekScore], str]] = None,
) -> WeekScore:
    """Score one week without side effects; delivery requires ``WeeklyScoreLedger``."""

    if strategy is not None:
        raise ScoreRefusal(
            "CALLBACK_REQUIRES_LEDGER",
            "strategy delivery requires WeeklyScoreLedger durable idempotency state",
        )

    _guard_firewall(firewall, release)
    for forecast in release.horizons:
        if contract.task and forecast.task != contract.task:
            raise ScoreRefusal(
                "TASK_MISMATCH",
                f"{forecast.family.value} {forecast.horizon} task {forecast.task!r} != {contract.task!r}",
            )
        if forecast.week != release.week:
            raise ScoreRefusal(
                "WEEK_MISMATCH",
                f"{forecast.horizon} week {_iso(forecast.week.start)} != {_iso(release.week.start)}",
            )
    if release.horizons:
        anchor = _identity_key(release.horizons[0])
        seen_slots: set[tuple[ForecastFamily, str]] = set()
        family_digest: dict[ForecastFamily, str] = {}
        for forecast in release.horizons:
            if _identity_key(forecast) != anchor:
                raise ScoreRefusal(
                    "IDENTITY_MISMATCH",
                    (
                        f"{forecast.family.value} {forecast.horizon} task, cutoff, split, "
                        "population, origins, unit, scale, or week differs"
                    ),
                )
            slot = (forecast.family, forecast.horizon)
            if slot in seen_slots:
                raise ScoreRefusal("DUPLICATE_HORIZON", f"{forecast.family.value} {forecast.horizon}")
            seen_slots.add(slot)
            bound = family_digest.get(forecast.family)
            if bound is None:
                family_digest[forecast.family] = forecast.model_digest
            elif bound != forecast.model_digest:
                raise ScoreRefusal(
                    "DIGEST_CHANGED",
                    f"{forecast.family.value} model digest changed inside the forecast",
                )
    families = {item.family for item in release.horizons}
    required = set(contract.required_families)
    if required - families:
        missing = ",".join(item.value for item in contract.required_families if item not in families)
        scored = _freeze(
            WeekScore(
                week_start=_iso(release.week.start),
                split=release.week.split.value,
                disposition="INCOMPLETE",
                reason=f"missing forecast family: {missing}",
                strategy_eligible=False,
                primary_error_metric=contract.primary_error_metric,
                contract_digest=_contract_digest(contract),
                week_identity_digest=_week_digest(release.week),
                horizons=(),
                digest="",
            )
        )
        return scored
    if contract.required_horizons:
        present = {(item.family, item.horizon) for item in release.horizons}
        missing = [
            f"{family.value}:{horizon}"
            for family in contract.required_families
            for horizon in contract.required_horizons
            if (family, horizon) not in present
        ]
        if missing:
            raise ScoreRefusal("MISSING_HORIZON", ",".join(missing))
    horizons = tuple(score_horizon(item, contract) for item in release.horizons)
    beats = [item.beats_primary_naive for item in horizons]
    eligible = bool(horizons) and contract.primary_error_metric != NOT_CONFIGURED and all(beats)
    reason = None
    if contract.primary_error_metric == NOT_CONFIGURED:
        reason = f"{PRIMARY_ERROR_METRIC_FIELD} is {NOT_CONFIGURED}"
        eligible = False
    elif not eligible:
        reason = "primary error is not strictly below the same-row naive at every horizon"
    scored = _freeze(
        WeekScore(
            week_start=_iso(release.week.start),
            split=release.week.split.value,
            disposition=DispositionStatus.COMPLETED.value,
            reason=reason,
            strategy_eligible=eligible,
            primary_error_metric=contract.primary_error_metric,
            contract_digest=_contract_digest(contract),
            week_identity_digest=_week_digest(release.week),
            horizons=horizons,
            digest="",
        )
    )
    return scored


def terminal_week(
    week: WeekSpec,
    status: DispositionStatus,
    reason: str,
    contract: ScoringContract,
) -> WeekScore:
    """Preserve a failed or excluded week without inventing metrics."""

    if status is DispositionStatus.COMPLETED:
        raise ScoreRefusal("DISPOSITION", "a completed week must come from score_week")
    if not reason:
        raise ScoreRefusal("DISPOSITION", "failed or excluded weeks require a reason")
    return _freeze(
        WeekScore(
            week_start=_iso(week.start),
            split=week.split.value,
            disposition=status.value,
            reason=reason,
            strategy_eligible=False,
            primary_error_metric=contract.primary_error_metric,
            contract_digest=_contract_digest(contract),
            week_identity_digest=_week_digest(week),
            horizons=(),
            digest="",
        )
    )


@dataclass
class WeeklyScoreLedger:
    """Ordered weekly receipts. A restored completed week is not scored again."""

    expected_weeks: tuple[WeekSpec, ...]
    contract: ScoringContract
    _scores: dict[str, WeekScore]
    _strategy_deliveries: dict[str, str]
    score_events: int = 0

    def __init__(self, expected_weeks: Sequence[WeekSpec], contract: ScoringContract) -> None:
        weeks = tuple(expected_weeks)
        keys = [_iso(week.start) + "|" + week.split.value for week in weeks]
        if len(keys) != len(set(keys)):
            raise ScoreRefusal("DUPLICATE_WEEK", "the sealed week list contains a duplicate")
        if not weeks:
            raise ScoreRefusal("EMPTY", "the sealed week list is empty")
        self.expected_weeks = weeks
        self.contract = contract
        self._scores = {}
        self._strategy_deliveries = {}
        self.score_events = 0

    def _key(self, week: WeekSpec) -> str:
        key = _iso(week.start) + "|" + week.split.value
        allowed = {
            _iso(item.start) + "|" + item.split.value: item for item in self.expected_weeks
        }
        if key not in allowed:
            raise ScoreRefusal("UNKNOWN_WEEK", key)
        if allowed[key] != week:
            raise ScoreRefusal("WEEK_IDENTITY_MISMATCH", key)
        return key

    def record(self, score: WeekScore, *, counted: bool) -> WeekScore:
        key = score.week_start + "|" + score.split
        allowed = {_iso(item.start) + "|" + item.split.value for item in self.expected_weeks}
        if key not in allowed:
            raise ScoreRefusal("UNKNOWN_WEEK", key)
        if score.primary_error_metric != self.contract.primary_error_metric or any(
            item.primary_error_metric != self.contract.primary_error_metric
            for item in score.horizons
        ):
            raise ScoreRefusal("CONTRACT_MISMATCH", key)
        if score.contract_digest != _contract_digest(self.contract):
            raise ScoreRefusal("CONTRACT_MISMATCH", key)
        expected_week = next(
            item
            for item in self.expected_weeks
            if _iso(item.start) + "|" + item.split.value == key
        )
        if score.week_identity_digest != _week_digest(expected_week):
            raise ScoreRefusal("WEEK_IDENTITY_MISMATCH", key)
        if score.digest != _digest(_score_payload(score)):
            raise ScoreRefusal("DIGEST", key)
        prior = self._scores.get(key)
        if prior is not None:
            if prior.digest != score.digest:
                raise ScoreRefusal("CONFLICTING_WEEK", key)
            return prior
        self._scores[key] = score
        self._strategy_deliveries[key] = (
            "PENDING" if score.strategy_eligible else "NOT_APPLICABLE"
        )
        if counted:
            self.score_events += 1
        return score

    def submit(
        self,
        release: WeekRelease,
        *,
        firewall: Optional[BusinessObjectiveFirewall] = None,
        strategy: Optional[Callable[[WeekScore], None]] = None,
    ) -> WeekScore:
        key = self._key(release.week)
        prior = self._scores.get(key)
        scored = score_week(release, self.contract, firewall=firewall, strategy=None)
        if prior is not None:
            if prior.digest != scored.digest:
                raise ScoreRefusal("CONFLICTING_WEEK", key)
            stored = prior
        else:
            stored = self.record(scored, counted=True)
        if stored.strategy_eligible and strategy is not None:
            delivery = self._strategy_deliveries.get(key, "UNKNOWN")
            if delivery == "UNKNOWN":
                raise ScoreRefusal(
                    "DELIVERY_STATE_UNKNOWN",
                    f"{key} predates durable strategy-delivery receipts",
                )
            if delivery != "DELIVERED":
                acknowledgement = strategy(stored)
                if acknowledgement != stored.digest:
                    raise ScoreRefusal(
                        "DELIVERY_NOT_ACKNOWLEDGED",
                        f"strategy must acknowledge score digest {stored.digest}",
                    )
                self._strategy_deliveries[key] = "DELIVERED"
        return stored

    def record_terminal(self, week: WeekSpec, status: DispositionStatus, reason: str) -> WeekScore:
        self._key(week)
        scored = terminal_week(week, status, reason, self.contract)
        return self.record(scored, counted=True)

    def close(self, proposed_denominator: Optional[int] = None) -> AnnualClose:
        """Close only the sealed week list. A smaller proposed count is refused."""

        ordered: list[WeekScore] = []
        for week in self.expected_weeks:
            key = _iso(week.start) + "|" + week.split.value
            score = self._scores.get(key)
            if score is None:
                raise ScoreRefusal("MISSING_DISPOSITION", key)
            ordered.append(score)
        denominator = len(self.expected_weeks)
        if proposed_denominator is not None and proposed_denominator != denominator:
            raise ScoreRefusal(
                "REDUCED_DENOMINATOR",
                f"proposed {proposed_denominator} != sealed {denominator}",
            )
        origin_rows = sum(item.sample_count for score in ordered for item in score.horizons)
        return AnnualClose(
            expected_weeks=denominator,
            denominator=denominator,
            longitudinal_evaluation_units=denominator,
            independence_status=NOT_ESTABLISHED,
            origin_rows=origin_rows,
            eligible_weeks=sum(1 for score in ordered if score.strategy_eligible),
            dispositions=tuple(ordered),
        )

    def to_json(self) -> str:
        payload = {
            "schema": "business_weekly_score_ledger.v3",
            "delivery_semantics": "AT_LEAST_ONCE_IDEMPOTENT_SCORE_DIGEST",
            "primary_error_metric": self.contract.primary_error_metric,
            "required_families": [item.value for item in self.contract.required_families],
            "required_horizons": list(self.contract.required_horizons),
            "scale_policy": self.contract.scale_policy,
            "task": self.contract.task,
            "score_events": self.score_events,
            "expected_weeks": [
                {
                    "split": week.split.value,
                    "ordinal": week.ordinal,
                    "start": _iso(week.start),
                    "end": _iso(week.end),
                    "cutoff": _iso(week.cutoff),
                    "fit_start": None if week.fit_start is None else _iso(week.fit_start),
                    "retrain_due": week.retrain_due,
                }
                for week in self.expected_weeks
            ],
            "scores": [_score_payload(score) | {"digest": score.digest} for score in self._scores.values()],
        }
        payload["ledger_digest"] = _digest(payload)
        return _canonical(payload).decode("ascii")

    @classmethod
    def from_json(cls, encoded: str) -> "WeeklyScoreLedger":
        payload = json.loads(encoded)
        schema = payload.get("schema")
        if schema != "business_weekly_score_ledger.v3":
            raise ScoreRefusal("SCHEMA", "unsupported score ledger schema")
        expected_fields = {
            "delivery_semantics",
            "expected_weeks",
            "ledger_digest",
            "primary_error_metric",
            "required_families",
            "required_horizons",
            "scale_policy",
            "schema",
            "score_events",
            "scores",
            "task",
        }
        if set(payload) != expected_fields:
            raise ScoreRefusal("SCHEMA_FIELDS", "score ledger fields differ from schema v3")
        if payload["delivery_semantics"] != "AT_LEAST_ONCE_IDEMPOTENT_SCORE_DIGEST":
            raise ScoreRefusal("DELIVERY_SEMANTICS", "unsupported delivery semantics")
        claimed_ledger_digest = payload.get("ledger_digest")
        unsigned_payload = dict(payload)
        unsigned_payload.pop("ledger_digest", None)
        families = tuple(ForecastFamily(item) for item in payload["required_families"])
        contract = ScoringContract(
            primary_error_metric=payload["primary_error_metric"],
            required_families=families,
            required_horizons=tuple(payload["required_horizons"]),
            task=payload.get("task", ""),
            scale_policy=payload.get("scale_policy", DECLARED_TARGET_SCALE),
        )
        weeks = tuple(
            WeekSpec(
                split=EvaluationSplit(item["split"]),
                ordinal=item["ordinal"],
                start=datetime.strptime(item["start"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=_utc()
                ),
                end=datetime.strptime(item["end"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_utc()),
                cutoff=datetime.strptime(item["cutoff"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_utc()),
                fit_start=None
                if item["fit_start"] is None
                else datetime.strptime(item["fit_start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_utc()),
                retrain_due=item["retrain_due"],
            )
            for item in payload["expected_weeks"]
        )
        ledger = cls(weeks, contract)
        restored_keys: set[str] = set()
        for raw in payload["scores"]:
            horizons = tuple(
                HorizonScore(
                    family=item["family"],
                    horizon=item["horizon"],
                    sample_count=item["sample_count"],
                    model_mae=item["model_mae"],
                    model_mse=item["model_mse"],
                    naive_mae=item["naive_mae"],
                    naive_mse=item["naive_mse"],
                    unit=item["unit"],
                    scale=item["scale"],
                    week_start=raw["week_start"],
                    model_digest=item["model_digest"],
                    dataset_id=item["dataset_id"],
                    release_digest=item["release_digest"],
                    paired_differences=_paired_differences(item.get("paired_differences")),
                    skill_mae=item["skill_mae"],
                    skill_mse=item["skill_mse"],
                    primary_error_metric=raw["primary_error_metric"],
                    beats_primary_naive=item["beats_primary_naive"],
                )
                for item in raw["horizons"]
            )
            score = WeekScore(
                week_start=raw["week_start"],
                split=raw["split"],
                disposition=raw["disposition"],
                reason=raw["reason"],
                strategy_eligible=raw["strategy_eligible"],
                primary_error_metric=raw["primary_error_metric"],
                contract_digest=raw.get("contract_digest", ""),
                week_identity_digest=raw.get("week_identity_digest", ""),
                horizons=horizons,
                digest=raw["digest"],
            )
            if score.digest != _digest(_score_payload(score)):
                raise ScoreRefusal("DIGEST", score.week_start)
            _validate_restored_score(score, contract)
            key = score.week_start + "|" + score.split
            if key in restored_keys:
                raise ScoreRefusal("DUPLICATE_SCORE", key)
            restored_keys.add(key)
            ledger.record(score, counted=False)
        ledger._strategy_deliveries = {
            key: "PENDING" if score.strategy_eligible else "NOT_APPLICABLE"
            for key, score in ledger._scores.items()
        }
        events = payload["score_events"]
        if isinstance(events, bool) or not isinstance(events, int) or events != len(ledger._scores):
            raise ScoreRefusal("SCORE_EVENTS", "score event count does not match retained scores")
        ledger.score_events = events
        if claimed_ledger_digest != _digest(unsigned_payload):
            raise ScoreRefusal("LEDGER_DIGEST", "serialized ledger content changed")
        return ledger


def _utc():
    from datetime import timezone

    return timezone.utc


def score_weekly_forecast_release(
    release: WeekRelease,
    contract: Optional[ScoringContract] = None,
    *,
    firewall: Optional[BusinessObjectiveFirewall] = None,
    strategy: Optional[Callable[[WeekScore], None]] = None,
) -> WeekScore:
    """Integration entry used by the weekly business modules. It does not train."""

    active = contract if contract is not None else financial_task_contract()
    scored = score_week(release, active, firewall=firewall, strategy=None)
    if scored.strategy_eligible and strategy is not None:
        raise ScoreRefusal(
            "CALLBACK_REQUIRES_LEDGER",
            "strategy delivery requires WeeklyScoreLedger durable idempotency state",
        )
    return scored
