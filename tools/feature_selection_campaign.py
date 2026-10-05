"""Authenticated TRAIN-only orchestration for scalar feature selectors.

The campaign is deliberately unable to receive outer validation or test arrays.
It authenticates the exact TRAIN population, ordered semantic groups, and the
complete selector plan before dispatching independent methods.  Method failures
are retained beside successful complete rankings instead of truncating the run.
Every method is truncated at the primary K and at every sealed sensitivity K;
a budget that fails is retained per K without erasing the complete ranking.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from tools.feature_selector_comparators import (
    ComparatorMethod,
    ComparatorResult,
    ComparatorRunPlan,
    RandomTape,
    compute_row_digest as compute_comparator_row_digest,
    run_comparator,
    validate_sealed_ks,
)
from tools.feature_selector_elastic_net import (
    ChronologicalInnerFold,
    ElasticNetRunPlan,
    ElasticNetSelectorResult,
    compute_train_row_digest,
    run_elastic_net_selector,
)
from tools.feature_selector_predictive import (
    ChronoEpilogiPlan,
    ExtraTreesPlan,
    GroupGatePlan,
    PredictiveSelectorResult,
    SequentialMarginalPlan,
    compute_predictive_row_digest,
    run_chronoepilogi_selector,
    run_extratrees_selector,
    run_group_gate_selector,
    run_sequential_marginal_selector,
)


_CAMPAIGN_VERSION = 2
_COMPARATOR_ORDER = (
    ComparatorMethod.ALL_ADMISSIBLE,
    ComparatorMethod.SPEARMAN_K,
    ComparatorMethod.MI8_K,
    ComparatorMethod.JMI_K,
    ComparatorMethod.CMIM_K,
    ComparatorMethod.MRMR_K,
    ComparatorMethod.REDUNDANCY_K,
)
_PREDICTIVE_ORDER = (
    "ELASTIC_NET_K",
    "GROUP_ELASTIC_NET_K",
    "EXTRATREES_TBP_K",
    "SEQ_MARGINAL_K",
    "GROUP_GATE_K",
    "CHRONOEPILOGI_K",
)


class CampaignDisposition(str, Enum):
    """Terminal state of one independently dispatched selector."""

    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FeatureSelectionCampaignPlan:
    """Sealed common K, sensitivities, seed and method-specific TRAIN-only configuration."""

    subset_k: int
    random_tapes: tuple[RandomTape, ...]
    elastic_lambda_path: tuple[float, ...]
    redundancy_inner_fold_gains: tuple[tuple[str, float], ...]
    causal_annotations: tuple[tuple[str, str], ...]
    elastic_max_iter: int = 10_000
    elastic_tolerance: float = 1e-8
    sensitivity_ks: tuple[int, ...] = ()
    seed: int = 0
    group_membership: tuple[tuple[str, str], ...] = ()
    extratrees: dict[str, Any] | None = None
    sequential_budget: int | None = None
    group_gate_null_shifts: tuple[int, ...] = (720, 1440, 2160, 2880, 3600)
    chronoepilogi: dict[str, Any] | None = None
    enabled_methods: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        comparator_plan = ComparatorRunPlan(primary, tuple(self.random_tapes), sensitivities)
        if not comparator_plan.random_tapes:
            raise ValueError("at least one fixed random tape is required")
        elastic_plan = ElasticNetRunPlan(
            subset_k=primary,
            lambda_path=tuple(self.elastic_lambda_path),
            max_iter=self.elastic_max_iter,
            tolerance=self.elastic_tolerance,
            sensitivity_ks=sensitivities,
        )
        gains = _validate_pairs(self.redundancy_inner_fold_gains, value_name="inner-fold gain", numeric=True)
        annotations = _validate_pairs(self.causal_annotations, value_name="causal annotation", numeric=False)
        membership = _validate_pairs(self.group_membership, value_name="group membership", numeric=False)
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise TypeError("seed must be a non-negative integer")
        extratrees = dict(self.extratrees or {})
        ExtraTreesPlan(primary, sensitivities, int(self.seed), **extratrees)
        SequentialMarginalPlan(primary, sensitivities, self.sequential_budget)
        chrono = dict(self.chronoepilogi or {})
        ChronoEpilogiPlan(primary, sensitivities, **chrono)
        enabled = None
        if self.enabled_methods is not None:
            enabled = tuple(str(method) for method in self.enabled_methods)
            known = set(self.all_method_ids_for(comparator_plan.random_tapes))
            unknown = [method for method in enabled if method not in known]
            if unknown:
                raise ValueError(f"unknown campaign methods: {unknown}")
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "random_tapes", comparator_plan.random_tapes)
        object.__setattr__(self, "elastic_lambda_path", elastic_plan.lambda_path)
        object.__setattr__(self, "elastic_max_iter", elastic_plan.max_iter)
        object.__setattr__(self, "elastic_tolerance", elastic_plan.tolerance)
        object.__setattr__(self, "redundancy_inner_fold_gains", gains)
        object.__setattr__(self, "causal_annotations", annotations)
        object.__setattr__(self, "group_membership", membership)
        object.__setattr__(self, "extratrees", extratrees)
        object.__setattr__(self, "group_gate_null_shifts", tuple(int(v) for v in self.group_gate_null_shifts))
        object.__setattr__(self, "chronoepilogi", chrono)
        object.__setattr__(self, "enabled_methods", enabled)

    @staticmethod
    def all_method_ids_for(random_tapes: tuple[RandomTape, ...]) -> tuple[str, ...]:
        return (
            tuple(method.value for method in _COMPARATOR_ORDER)
            + tuple(f"RANDOM_K:{tape.tape_id}" for tape in random_tapes)
            + _PREDICTIVE_ORDER
        )

    @property
    def method_ids(self) -> tuple[str, ...]:
        """Every outcome id the campaign dispatches, in dispatch order."""

        all_ids = self.all_method_ids_for(self.random_tapes)
        if self.enabled_methods is None:
            return all_ids
        return tuple(method for method in all_ids if method in self.enabled_methods)

    @property
    def sealed_ks(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.sensitivity_ks) | {self.subset_k}))


@dataclass(frozen=True)
class CampaignMethodOutcome:
    """A complete selector result or an explicit isolated failure."""

    outcome_id: str
    disposition: CampaignDisposition
    result: ComparatorResult | ElasticNetSelectorResult | PredictiveSelectorResult | None
    failure_type: str | None
    failure_message: str | None
    wall_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True)
class FeatureSelectionCampaignResult:
    """Authenticated outcomes for every method declared by one campaign."""

    row_digest: str
    group_digest: str
    plan_digest: str
    sealed_subset_k: int
    sealed_ks: tuple[int, ...]
    seed: int
    eligible_groups: tuple[str, ...]
    causal_annotations: tuple[tuple[str, str], ...]
    causal_not_identified_policy: str
    outcomes: tuple[CampaignMethodOutcome, ...]

    def outcome(self, outcome_id: str) -> CampaignMethodOutcome:
        """Return one named outcome without silently choosing a substitute."""

        matches = tuple(item for item in self.outcomes if item.outcome_id == outcome_id)
        if len(matches) != 1:
            raise KeyError(outcome_id)
        return matches[0]


def _validate_pairs(values: Any, *, value_name: str, numeric: bool) -> tuple[tuple[str, Any], ...]:
    result: list[tuple[str, Any]] = []
    for item in tuple(values):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{value_name} entries must be (group, value) pairs")
        group, value = item
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"{value_name} group ids must be non-empty strings")
        if numeric:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"boolean {value_name} is forbidden")
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"{value_name} must be numeric")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{value_name} must be finite")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{value_name} must be a non-empty string")
        result.append((group, value))
    return tuple(result)


def _numeric_train_arrays(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    raw_x = np.asarray(x)
    raw_y = np.asarray(y)
    if raw_x.dtype.kind == "b" or raw_y.dtype.kind == "b":
        raise TypeError("boolean TRAIN values are forbidden")
    if raw_x.dtype.kind not in "iuf" or raw_y.dtype.kind not in "iuf":
        raise TypeError("TRAIN matrix and target must be numeric")
    # Own the buffers before freezing them for selector dispatch.
    matrix = np.array(raw_x, dtype="<f8", order="C", copy=True)
    target = np.array(raw_y, dtype="<f8", order="C", copy=True)
    if matrix.ndim != 2 or target.ndim != 1:
        raise ValueError("TRAIN X must be 2D and y must be 1D")
    if matrix.size == 0 or target.size == 0 or matrix.shape[1] == 0:
        raise ValueError("TRAIN matrix and target must not be empty")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("TRAIN X and y row counts differ")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("TRAIN matrix and target must contain only finite values")
    return matrix, target


def _validated_groups(group_ids: Any, columns: int | None = None) -> tuple[str, ...]:
    groups = tuple(group_ids)
    if columns is not None and len(groups) != columns:
        raise ValueError("one semantic group id is required per scalar column")
    if not groups or any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("semantic group ids must be non-empty strings")
    if len(set(groups)) != len(groups):
        raise ValueError("duplicate semantic groups are forbidden")
    return groups


def _validated_folds(inner_folds: Any, row_count: int | None = None) -> tuple[ChronologicalInnerFold, ...]:
    folds = tuple(inner_folds)
    if not folds:
        raise ValueError("at least one chronological inner fold is required")
    if any(not isinstance(fold, ChronologicalInnerFold) for fold in folds):
        raise TypeError("inner_folds must contain ChronologicalInnerFold values")
    ids = tuple(fold.fold_id for fold in folds)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate inner fold ids are forbidden")
    if row_count is not None and any(fold.validation_end > row_count for fold in folds):
        raise ValueError("inner fold boundary exceeds TRAIN rows")
    previous_end = -1
    for fold in folds:
        if fold.validation_start < previous_end:
            raise ValueError("inner validation folds overlap or are nonchronological")
        previous_end = fold.validation_end
    return folds


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compute_campaign_row_digest(x: Any, y: Any) -> str:
    """Return the exact canonical TRAIN population identity."""

    matrix, target = _numeric_train_arrays(x, y)
    digest = hashlib.sha256(b"feature-selection-campaign-train-rows-v1\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.tobytes(order="C"))
    digest.update(np.asarray(target.shape, dtype="<i8").tobytes())
    digest.update(target.tobytes(order="C"))
    return digest.hexdigest()


def compute_campaign_group_digest(group_ids: Any) -> str:
    """Return the identity of ordered scalar semantic groups."""

    groups = _validated_groups(group_ids)
    return _canonical_digest({"ordered_semantic_groups": groups})


def compute_campaign_plan_digest(run_plan: FeatureSelectionCampaignPlan, inner_folds: Any) -> str:
    """Bind K, sensitivities, seed, all method settings, annotations, membership and folds.

    The digest covers the complete method set regardless of ``enabled_methods``
    so a chunked run and a full run of the same plan share one identity.
    """

    if not isinstance(run_plan, FeatureSelectionCampaignPlan):
        raise TypeError("run_plan must be FeatureSelectionCampaignPlan")
    folds = _validated_folds(inner_folds)
    payload = {
        "version": _CAMPAIGN_VERSION,
        "methods": run_plan.all_method_ids_for(run_plan.random_tapes),
        "subset_k": run_plan.subset_k,
        "sealed_ks": run_plan.sealed_ks,
        "seed": run_plan.seed,
        "random_tapes": tuple({"tape_id": tape.tape_id, "seed": int(tape.seed)} for tape in run_plan.random_tapes),
        "elastic": {
            "lambda_path": run_plan.elastic_lambda_path,
            "max_iter": run_plan.elastic_max_iter,
            "tolerance": run_plan.elastic_tolerance,
        },
        "group_membership": run_plan.group_membership,
        "extratrees": run_plan.extratrees,
        "sequential_budget": run_plan.sequential_budget,
        "group_gate_null_shifts": run_plan.group_gate_null_shifts,
        "chronoepilogi": run_plan.chronoepilogi,
        "redundancy_inner_fold_gains": run_plan.redundancy_inner_fold_gains,
        "causal_annotations": run_plan.causal_annotations,
        "causal_not_identified_policy": "NEUTRAL_NOT_A_REJECTION",
        "inner_folds": tuple(
            {
                "fold_id": fold.fold_id,
                "train": (fold.train_start, fold.train_end),
                "validation": (fold.validation_start, fold.validation_end),
            }
            for fold in folds
        ),
    }
    return _canonical_digest(payload)


def _unique_mapping(pairs: tuple[tuple[str, Any], ...], *, name: str) -> dict[str, Any]:
    keys = tuple(key for key, _ in pairs)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate {name} semantic groups are forbidden")
    return dict(pairs)


def _complete_outcome(outcome_id: str, result: Any, wall: float) -> CampaignMethodOutcome:
    return CampaignMethodOutcome(outcome_id, CampaignDisposition.COMPLETE, result, None, None, wall)


def _failed_outcome(outcome_id: str, error: Exception, wall: float) -> CampaignMethodOutcome:
    return CampaignMethodOutcome(
        outcome_id, CampaignDisposition.FAILED, None, type(error).__name__, str(error), wall
    )


def _validate_complete_ranking(result: Any, groups: tuple[str, ...]) -> None:
    ranking = tuple(result.full_ranking)
    if len(ranking) != len(groups) or set(ranking) != set(groups):
        raise RuntimeError("selector did not return one complete semantic-group ranking")


def run_feature_selection_campaign(
    x: Any,
    y: Any,
    group_ids: Any,
    inner_folds: Any,
    run_plan: FeatureSelectionCampaignPlan,
    *,
    expected_row_digest: str,
    expected_group_digest: str,
    expected_plan_digest: str,
    input_label: str = "TRAIN",
    on_outcome: Callable[[CampaignMethodOutcome], None] | None = None,
) -> FeatureSelectionCampaignResult:
    """Run all sealed selectors independently over one authenticated TRAIN set.

    ``inner_folds`` are chronological splits *inside* the TRAIN rows handed in:
    the elastic-net family selects hyperparameters on all of them and the
    predictive family uses the last one as its nested split.  ``on_outcome`` is
    called after every method so a driver can persist progress incrementally.
    """

    if input_label != "TRAIN":
        raise ValueError("feature-selection campaigns are TRAIN-only")
    if not isinstance(run_plan, FeatureSelectionCampaignPlan):
        raise TypeError("run_plan must be FeatureSelectionCampaignPlan")
    matrix, target = _numeric_train_arrays(x, y)
    groups = _validated_groups(group_ids, matrix.shape[1])
    if run_plan.subset_k > len(groups):
        raise ValueError("sealed subset K exceeds the number of semantic groups")
    folds = _validated_folds(inner_folds, matrix.shape[0])

    row_digest = compute_campaign_row_digest(matrix, target)
    group_digest = compute_campaign_group_digest(groups)
    plan_digest = compute_campaign_plan_digest(run_plan, folds)
    for supplied, actual, name in (
        (expected_row_digest, row_digest, "row"),
        (expected_group_digest, group_digest, "group"),
        (expected_plan_digest, plan_digest, "plan"),
    ):
        if not isinstance(supplied, str) or not supplied:
            raise TypeError(f"expected {name} digest must be a non-empty string")
        if supplied != actual:
            raise ValueError(f"{name} digest mismatch")

    annotations = _unique_mapping(run_plan.causal_annotations, name="causal annotation")
    if set(annotations) != set(groups):
        raise ValueError("causal annotations must name every semantic group exactly")
    ordered_annotations = tuple((group, str(annotations[group])) for group in groups)

    matrix.setflags(write=False)
    target.setflags(write=False)
    comparator_plan = ComparatorRunPlan(run_plan.subset_k, run_plan.random_tapes, run_plan.sensitivity_ks)
    comparator_row_digest = compute_comparator_row_digest(matrix, target)
    elastic_row_digest = compute_train_row_digest(matrix, target)
    predictive_row_digest = compute_predictive_row_digest(matrix, target)
    nested_split = folds[-1]
    enabled = set(run_plan.method_ids)
    outcomes: list[CampaignMethodOutcome] = []

    def dispatch(outcome_id: str, runner: Callable[[], Any]) -> None:
        if outcome_id not in enabled:
            return
        started = time.monotonic()
        try:
            result = runner()
            _validate_complete_ranking(result, groups)
            outcome = _complete_outcome(outcome_id, result, time.monotonic() - started)
        except Exception as error:  # retained beside the other methods
            outcome = _failed_outcome(outcome_id, error, time.monotonic() - started)
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    for method in _COMPARATOR_ORDER:
        def comparator_runner(method=method):
            gains = None
            if method is ComparatorMethod.REDUNDANCY_K:
                gains = _unique_mapping(run_plan.redundancy_inner_fold_gains, name="inner-fold gain")
            return run_comparator(
                method, matrix, target, groups, comparator_plan,
                expected_row_digest=comparator_row_digest, inner_fold_gains=gains,
            )
        dispatch(method.value, comparator_runner)

    for tape in run_plan.random_tapes:
        dispatch(
            f"RANDOM_K:{tape.tape_id}",
            lambda tape=tape: run_comparator(
                ComparatorMethod.RANDOM_K, matrix, target, groups, comparator_plan,
                expected_row_digest=comparator_row_digest, random_tape_id=tape.tape_id,
            ),
        )

    elastic_plan = ElasticNetRunPlan(
        subset_k=run_plan.subset_k,
        lambda_path=run_plan.elastic_lambda_path,
        max_iter=run_plan.elastic_max_iter,
        tolerance=run_plan.elastic_tolerance,
        sensitivity_ks=run_plan.sensitivity_ks,
    )
    dispatch(
        "ELASTIC_NET_K",
        lambda: run_elastic_net_selector(
            matrix, target, groups, folds, elastic_plan, expected_row_digest=elastic_row_digest
        ),
    )

    def group_elastic_runner():
        if not run_plan.group_membership:
            raise ValueError("GROUP_ELASTIC_NET_K requires group membership for every semantic group")
        return run_elastic_net_selector(
            matrix, target, groups, folds, elastic_plan,
            expected_row_digest=elastic_row_digest, group_membership=run_plan.group_membership,
        )
    dispatch("GROUP_ELASTIC_NET_K", group_elastic_runner)

    dispatch(
        "EXTRATREES_TBP_K",
        lambda: run_extratrees_selector(
            matrix, target, groups, nested_split,
            ExtraTreesPlan(run_plan.subset_k, run_plan.sensitivity_ks, run_plan.seed, **run_plan.extratrees),
            expected_row_digest=predictive_row_digest,
        ),
    )
    dispatch(
        "SEQ_MARGINAL_K",
        lambda: run_sequential_marginal_selector(
            matrix, target, groups, nested_split,
            SequentialMarginalPlan(run_plan.subset_k, run_plan.sensitivity_ks, run_plan.sequential_budget),
            expected_row_digest=predictive_row_digest,
        ),
    )

    def group_gate_runner():
        if not run_plan.group_membership:
            raise ValueError("GROUP_GATE_K requires group membership for every semantic group")
        return run_group_gate_selector(
            matrix, target, groups, nested_split,
            GroupGatePlan(
                run_plan.subset_k, run_plan.group_membership, run_plan.sensitivity_ks,
                null_shifts=run_plan.group_gate_null_shifts,
            ),
            expected_row_digest=predictive_row_digest,
        )
    dispatch("GROUP_GATE_K", group_gate_runner)

    dispatch(
        "CHRONOEPILOGI_K",
        lambda: run_chronoepilogi_selector(
            matrix, target, groups, nested_split,
            ChronoEpilogiPlan(run_plan.subset_k, run_plan.sensitivity_ks, **run_plan.chronoepilogi),
            expected_row_digest=predictive_row_digest,
        ),
    )

    return FeatureSelectionCampaignResult(
        row_digest=row_digest,
        group_digest=group_digest,
        plan_digest=plan_digest,
        sealed_subset_k=run_plan.subset_k,
        sealed_ks=run_plan.sealed_ks,
        seed=run_plan.seed,
        eligible_groups=groups,
        causal_annotations=ordered_annotations,
        causal_not_identified_policy="NEUTRAL_NOT_A_REJECTION",
        outcomes=tuple(outcomes),
    )


__all__ = [
    "CampaignDisposition",
    "CampaignMethodOutcome",
    "FeatureSelectionCampaignPlan",
    "FeatureSelectionCampaignResult",
    "compute_campaign_group_digest",
    "compute_campaign_plan_digest",
    "compute_campaign_row_digest",
    "run_feature_selection_campaign",
]
