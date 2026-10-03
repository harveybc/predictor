"""Callback-based training orchestration for BUSINESS walk-forward evaluation.

The module coordinates calendars, point-in-time populations, model lineage,
and execution identity.  Model frameworks remain behind the supplied trainer
callback so forecasting, heuristic heads, SAC, and DQN share one protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional, Tuple

from tools.business_asof_window import (
    AsOfRow,
    ResolvedAsOfWindow,
    SupportSpec,
    resolve_asof_window,
)
from tools.business_objective_firewall import (
    BusinessObjectiveFirewall,
    FirewallError,
    Phase,
)
from tools.business_weekly_protocol import (
    BusinessWeeklyProtocol,
    EvaluationMode,
    EvaluationSplit,
    UpdateMode,
    WeekSpec,
)


class TrainingContractError(ValueError):
    """Raised when a callback or orchestration request violates the contract."""


class TaskFamily(str, Enum):
    """Model families evaluated under the same weekly BUSINESS identity."""

    FORECAST = "FORECAST"
    HEURISTIC_HEAD = "HEURISTIC_HEAD"
    SAC = "SAC"
    DQN = "DQN"


class TrainingStatus(str, Enum):
    """Terminal status of one family in one scored week."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingContractError(f"{name} must be a non-empty string")
    return value


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingContractError("seed must be an int, not bool")
    return value


@dataclass(frozen=True)
class TrainingRequest:
    """Immutable input delivered to exactly one model-training callback."""

    family: TaskFamily
    week: WeekSpec
    window: ResolvedAsOfWindow
    seed: int
    procedure_digest: str
    update_mode: UpdateMode
    cost_identity: str
    parent_digest: Optional[str]
    required_backend: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, TaskFamily):
            raise TrainingContractError("family must be a TaskFamily")
        if not isinstance(self.week, WeekSpec) or self.window.week != self.week:
            raise TrainingContractError("window must resolve the request week")
        _require_seed(self.seed)
        _require_text("procedure_digest", self.procedure_digest)
        if not isinstance(self.update_mode, UpdateMode):
            raise TrainingContractError("update_mode must be an UpdateMode")
        _require_text("cost_identity", self.cost_identity)
        _require_text("required_backend", self.required_backend)
        if self.update_mode is UpdateMode.FULL_RETRAIN_ROLLING_4Y:
            if self.parent_digest is not None:
                raise TrainingContractError("full retraining must not have a parent")
        elif not self.parent_digest:
            raise TrainingContractError("warm update requires the exact prior parent")

    @property
    def data_identity(self) -> str:
        """Return the complete point-in-time training-population identity."""

        return self.window.population_digest


@dataclass(frozen=True)
class TrainingResult:
    """Digest-only result returned by a trainer and checked field by field."""

    family: TaskFamily
    week_start: datetime
    cutoff: datetime
    seed: int
    data_identity: str
    procedure_digest: str
    update_mode: UpdateMode
    cost_identity: str
    parent_digest: Optional[str]
    model_digest: str
    execution_backend: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, TaskFamily):
            raise TrainingContractError("family must be a TaskFamily")
        _require_seed(self.seed)
        for name in (
            "data_identity",
            "procedure_digest",
            "cost_identity",
            "model_digest",
            "execution_backend",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.update_mode, UpdateMode):
            raise TrainingContractError("update_mode must be an UpdateMode")


@dataclass(frozen=True)
class WeeklyRelease:
    """A model made available to one scored week, trained now or reused."""

    family: TaskFamily
    scored_week: WeekSpec
    asof_data_identity: str
    result: TrainingResult
    reused: bool

    def __post_init__(self) -> None:
        if self.family is not self.result.family:
            raise TrainingContractError("release and result families differ")
        _require_text("asof_data_identity", self.asof_data_identity)
        if type(self.reused) is not bool:
            raise TrainingContractError("reused must be a bool")
        if self.result.week_start > self.scored_week.start:
            raise TrainingContractError("a release cannot come from a future week")
        if not self.reused and self.result.week_start != self.scored_week.start:
            raise TrainingContractError("a new release must be trained for its scored week")
        if self.reused and self.result.week_start >= self.scored_week.start:
            raise TrainingContractError("a reused release must come from a prior week")


@dataclass(frozen=True)
class TrainingDisposition:
    """Explicit success or failure for one family in one planned week."""

    family: TaskFamily
    week: WeekSpec
    status: TrainingStatus
    release: Optional[WeeklyRelease] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is TrainingStatus.COMPLETED:
            if self.release is None or self.reason is not None:
                raise TrainingContractError("completed disposition requires only a release")
        elif self.status is TrainingStatus.FAILED:
            if self.release is not None or not self.reason:
                raise TrainingContractError("failed disposition requires only a reason")
        else:
            raise TrainingContractError("status must be a TrainingStatus")


@dataclass(frozen=True)
class TrainingRun:
    """Complete ordered dispositions and resulting firewall state."""

    split: EvaluationSplit
    dispositions: Tuple[TrainingDisposition, ...]
    firewall: BusinessObjectiveFirewall

    @property
    def releases(self) -> Tuple[WeeklyRelease, ...]:
        """Return successful releases in traversal order."""

        return tuple(
            item.release
            for item in self.dispositions
            if item.release is not None
        )


Trainer = Callable[[TrainingRequest], TrainingResult]
RowProvider = Callable[[WeekSpec], Iterable[AsOfRow]]


def _validate_result(request: TrainingRequest, result: object) -> TrainingResult:
    """Reject partial, mismatched, CPU, or otherwise substituted results."""

    if not isinstance(result, TrainingResult):
        raise TrainingContractError("trainer must return TrainingResult; fallback refused")
    expected = {
        "family": request.family,
        "week_start": request.week.start,
        "cutoff": request.week.cutoff,
        "seed": request.seed,
        "data_identity": request.data_identity,
        "procedure_digest": request.procedure_digest,
        "update_mode": request.update_mode,
        "cost_identity": request.cost_identity,
        "parent_digest": request.parent_digest,
        "execution_backend": request.required_backend,
    }
    for field, value in expected.items():
        if getattr(result, field) != value:
            raise TrainingContractError(
                f"trainer result {field} mismatch; silent fallback refused"
            )
    return result


def require_matched_releases(releases: Iterable[WeeklyRelease]) -> None:
    """Require all requested families to share one comparison identity."""

    releases = tuple(releases)
    if not releases:
        raise TrainingContractError("matched comparison requires releases")
    if len({item.family for item in releases}) != len(releases):
        raise TrainingContractError("matched comparison contains duplicate families")
    reference = releases[0]
    fields = (
        "scored_week",
        "asof_data_identity",
    )
    result_fields = (
        "seed",
        "procedure_digest",
        "update_mode",
        "cost_identity",
        "execution_backend",
    )
    for candidate in releases[1:]:
        for field in fields:
            if getattr(candidate, field) != getattr(reference, field):
                raise TrainingContractError(f"unmatched comparison {field}")
        for field in result_fields:
            if getattr(candidate.result, field) != getattr(reference.result, field):
                raise TrainingContractError(f"unmatched comparison {field}")


def run_weekly_training(
    *,
    protocol: BusinessWeeklyProtocol,
    split: EvaluationSplit,
    families: Iterable[TaskFamily],
    support: SupportSpec,
    rows_for_week: RowProvider,
    trainer: Trainer,
    firewall: BusinessObjectiveFirewall,
    seed: int,
    cost_identity: str,
    required_backend: str,
    initial_parent_digests: Optional[Mapping[TaskFamily, str]] = None,
) -> TrainingRun:
    """Traverse one BUSINESS split and train or reuse releases deterministically.

    The firewall is checked before a test week or its row provider is touched.
    Callback exceptions become explicit failed dispositions; no alternative
    trainer, backend, static path, or future release is substituted.
    """

    if not isinstance(protocol, BusinessWeeklyProtocol):
        raise TrainingContractError("protocol must be a BusinessWeeklyProtocol")
    if protocol.evaluation_mode is EvaluationMode.LITERATURE_STATIC:
        raise TrainingContractError("static protocol cannot enter BUSINESS training")
    if not isinstance(split, EvaluationSplit):
        raise TrainingContractError("split must be an EvaluationSplit")
    if not isinstance(firewall, BusinessObjectiveFirewall):
        raise TrainingContractError("firewall must be a BusinessObjectiveFirewall")
    _require_seed(seed)
    _require_text("cost_identity", cost_identity)
    _require_text("required_backend", required_backend)

    ordered_families = tuple(families)
    if not ordered_families or any(
        not isinstance(family, TaskFamily) for family in ordered_families
    ):
        raise TrainingContractError("families must contain TaskFamily values")
    if len(set(ordered_families)) != len(ordered_families):
        raise TrainingContractError("families must be unique")

    if split is EvaluationSplit.VALIDATION:
        if firewall.phase is not Phase.VALIDATION_SELECTION:
            raise FirewallError("validation training requires an open selection firewall")
        active_firewall = firewall
    else:
        if firewall.phase is Phase.VALIDATION_SELECTION:
            raise FirewallError("test data is sealed until the procedure is sealed")
        if firewall.phase is Phase.PROCEDURE_SEALED:
            active_firewall = firewall.open_test_traversal()
        elif firewall.phase is Phase.TEST_TRAVERSAL:
            active_firewall = firewall
        else:
            raise FirewallError("test training requires a sealed open procedure")

    parents = dict(initial_parent_digests or {})
    latest: dict[TaskFamily, TrainingResult] = {}
    broken_warm_lineage: set[TaskFamily] = set()
    dispositions = []
    weeks = tuple(week for week in protocol.weeks() if week.split is split)

    for week in weeks:
        try:
            window = resolve_asof_window(week, rows_for_week(week), support)
        except Exception as exc:
            reason = f"as-of resolution failed: {type(exc).__name__}: {exc}"
            if protocol.update_mode is UpdateMode.WARM_UPDATE_ROLLING_4Y:
                broken_warm_lineage.update(ordered_families)
            dispositions.extend(
                TrainingDisposition(family, week, TrainingStatus.FAILED, reason=reason)
                for family in ordered_families
            )
            continue

        week_releases = []
        for family in ordered_families:
            if not week.retrain_due:
                prior = latest.get(family)
                if prior is None:
                    dispositions.append(
                        TrainingDisposition(
                            family,
                            week,
                            TrainingStatus.FAILED,
                            reason="monthly reuse has no prior release",
                        )
                    )
                    continue
                try:
                    release = WeeklyRelease(
                        family=family,
                        scored_week=week,
                        asof_data_identity=window.population_digest,
                        result=prior,
                        reused=True,
                    )
                except Exception as exc:
                    dispositions.append(
                        TrainingDisposition(
                            family,
                            week,
                            TrainingStatus.FAILED,
                            reason=f"release reuse failed: {type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                dispositions.append(
                    TrainingDisposition(
                        family, week, TrainingStatus.COMPLETED, release=release
                    )
                )
                week_releases.append(release)
                continue

            parent_digest = None
            if protocol.update_mode is UpdateMode.WARM_UPDATE_ROLLING_4Y:
                if family in broken_warm_lineage:
                    dispositions.append(
                        TrainingDisposition(
                            family,
                            week,
                            TrainingStatus.FAILED,
                            reason="warm lineage lacks the previous due release",
                        )
                    )
                    continue
                parent_digest = (
                    latest[family].model_digest
                    if family in latest
                    else parents.get(family)
                )
                if not parent_digest:
                    dispositions.append(
                        TrainingDisposition(
                            family,
                            week,
                            TrainingStatus.FAILED,
                            reason="warm update has no exact previous parent",
                        )
                    )
                    continue

            try:
                request = TrainingRequest(
                    family=family,
                    week=week,
                    window=window,
                    seed=seed,
                    procedure_digest=protocol.procedure_digest,
                    update_mode=protocol.update_mode,
                    cost_identity=cost_identity,
                    parent_digest=parent_digest,
                    required_backend=required_backend,
                )
                result = _validate_result(request, trainer(request))
                protocol.bind_model(
                    week,
                    result.model_digest,
                    parent_digest=result.parent_digest,
                    strict_four_year_memory=(
                        protocol.update_mode
                        is UpdateMode.FULL_RETRAIN_ROLLING_4Y
                    ),
                )
                release = WeeklyRelease(
                    family=family,
                    scored_week=week,
                    asof_data_identity=window.population_digest,
                    result=result,
                    reused=False,
                )
            except Exception as exc:
                if protocol.update_mode is UpdateMode.WARM_UPDATE_ROLLING_4Y:
                    broken_warm_lineage.add(family)
                dispositions.append(
                    TrainingDisposition(
                        family,
                        week,
                        TrainingStatus.FAILED,
                        reason=f"trainer failed: {type(exc).__name__}: {exc}",
                    )
                )
                continue

            latest[family] = result
            dispositions.append(
                TrainingDisposition(
                    family, week, TrainingStatus.COMPLETED, release=release
                )
            )
            week_releases.append(release)

        if len(week_releases) == len(ordered_families):
            try:
                require_matched_releases(week_releases)
            except TrainingContractError as exc:
                failed_families = {item.family for item in week_releases}
                dispositions = [
                    item
                    for item in dispositions
                    if not (item.week == week and item.family in failed_families)
                ]
                dispositions.extend(
                    TrainingDisposition(
                        family,
                        week,
                        TrainingStatus.FAILED,
                        reason=f"matched comparison failed: {exc}",
                    )
                    for family in ordered_families
                )
                for family in failed_families:
                    latest.pop(family, None)
                    if protocol.update_mode is UpdateMode.WARM_UPDATE_ROLLING_4Y:
                        broken_warm_lineage.add(family)

    return TrainingRun(split, tuple(dispositions), active_firewall)
