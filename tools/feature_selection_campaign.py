"""Authenticated TRAIN-only orchestration for scalar feature selectors.

The campaign is deliberately unable to receive outer validation or test arrays.
It authenticates the exact TRAIN population, ordered semantic groups, and the
complete selector plan before dispatching independent methods.  Method failures
are retained beside successful complete rankings instead of truncating the run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from tools.feature_selector_comparators import (
    ComparatorMethod,
    ComparatorResult,
    ComparatorRunPlan,
    RandomTape,
    compute_row_digest as compute_comparator_row_digest,
    run_comparator,
)
from tools.feature_selector_elastic_net import (
    ChronologicalInnerFold,
    ElasticNetRunPlan,
    ElasticNetSelectorResult,
    compute_train_row_digest,
    run_elastic_net_selector,
)


_CAMPAIGN_VERSION = 1
_COMPARATOR_ORDER = (
    ComparatorMethod.ALL_ADMISSIBLE,
    ComparatorMethod.SPEARMAN_K,
    ComparatorMethod.MI8_K,
    ComparatorMethod.REDUNDANCY_K,
)


class CampaignDisposition(str, Enum):
    """Terminal state of one independently dispatched selector."""

    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FeatureSelectionCampaignPlan:
    """Sealed common K and method-specific TRAIN-only configuration."""

    subset_k: int
    random_tapes: tuple[RandomTape, ...]
    elastic_lambda_path: tuple[float, ...]
    redundancy_inner_fold_gains: tuple[tuple[str, float], ...]
    causal_annotations: tuple[tuple[str, str], ...]
    elastic_max_iter: int = 10_000
    elastic_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        comparator_plan = ComparatorRunPlan(self.subset_k, tuple(self.random_tapes))
        if not comparator_plan.random_tapes:
            raise ValueError("at least one fixed random tape is required")
        elastic_plan = ElasticNetRunPlan(
            subset_k=comparator_plan.subset_k,
            lambda_path=tuple(self.elastic_lambda_path),
            max_iter=self.elastic_max_iter,
            tolerance=self.elastic_tolerance,
        )
        gains = _validate_pairs(
            self.redundancy_inner_fold_gains,
            value_name="inner-fold gain",
            numeric=True,
        )
        annotations = _validate_pairs(
            self.causal_annotations,
            value_name="causal annotation",
            numeric=False,
        )
        object.__setattr__(self, "subset_k", comparator_plan.subset_k)
        object.__setattr__(self, "random_tapes", comparator_plan.random_tapes)
        object.__setattr__(self, "elastic_lambda_path", elastic_plan.lambda_path)
        object.__setattr__(self, "elastic_max_iter", elastic_plan.max_iter)
        object.__setattr__(self, "elastic_tolerance", elastic_plan.tolerance)
        object.__setattr__(self, "redundancy_inner_fold_gains", gains)
        object.__setattr__(self, "causal_annotations", annotations)


@dataclass(frozen=True)
class CampaignMethodOutcome:
    """A complete selector result or an explicit isolated failure."""

    outcome_id: str
    disposition: CampaignDisposition
    result: ComparatorResult | ElasticNetSelectorResult | None
    failure_type: str | None
    failure_message: str | None


@dataclass(frozen=True)
class FeatureSelectionCampaignResult:
    """Authenticated outcomes for every method declared by one campaign."""

    row_digest: str
    group_digest: str
    plan_digest: str
    sealed_subset_k: int
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


def _validate_pairs(
    values: Any, *, value_name: str, numeric: bool
) -> tuple[tuple[str, Any], ...]:
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
    matrix = np.asarray(raw_x, dtype="<f8", order="C")
    target = np.asarray(raw_y, dtype="<f8", order="C")
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


def _validated_folds(
    inner_folds: Any, row_count: int | None = None
) -> tuple[ChronologicalInnerFold, ...]:
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
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
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


def compute_campaign_plan_digest(
    run_plan: FeatureSelectionCampaignPlan, inner_folds: Any
) -> str:
    """Bind common K, all method settings, annotations, and chronological folds."""

    if not isinstance(run_plan, FeatureSelectionCampaignPlan):
        raise TypeError("run_plan must be FeatureSelectionCampaignPlan")
    folds = _validated_folds(inner_folds)
    payload = {
        "version": _CAMPAIGN_VERSION,
        "methods": tuple(method.value for method in _COMPARATOR_ORDER)
        + tuple(f"RANDOM_K:{tape.tape_id}" for tape in run_plan.random_tapes)
        + ("ELASTIC_NET_K",),
        "subset_k": run_plan.subset_k,
        "random_tapes": tuple(
            {"tape_id": tape.tape_id, "seed": int(tape.seed)}
            for tape in run_plan.random_tapes
        ),
        "elastic": {
            "lambda_path": run_plan.elastic_lambda_path,
            "max_iter": run_plan.elastic_max_iter,
            "tolerance": run_plan.elastic_tolerance,
        },
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


def _unique_mapping(
    pairs: tuple[tuple[str, Any], ...], *, name: str
) -> dict[str, Any]:
    keys = tuple(key for key, _ in pairs)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate {name} semantic groups are forbidden")
    return dict(pairs)


def _complete_outcome(outcome_id: str, result: Any) -> CampaignMethodOutcome:
    return CampaignMethodOutcome(
        outcome_id=outcome_id,
        disposition=CampaignDisposition.COMPLETE,
        result=result,
        failure_type=None,
        failure_message=None,
    )


def _failed_outcome(outcome_id: str, error: Exception) -> CampaignMethodOutcome:
    return CampaignMethodOutcome(
        outcome_id=outcome_id,
        disposition=CampaignDisposition.FAILED,
        result=None,
        failure_type=type(error).__name__,
        failure_message=str(error),
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
) -> FeatureSelectionCampaignResult:
    """Run all sealed selectors independently over one authenticated TRAIN set."""

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
    comparator_plan = ComparatorRunPlan(run_plan.subset_k, run_plan.random_tapes)
    comparator_row_digest = compute_comparator_row_digest(matrix, target)
    outcomes: list[CampaignMethodOutcome] = []

    for method in _COMPARATOR_ORDER:
        outcome_id = method.value
        try:
            gains = None
            if method is ComparatorMethod.REDUNDANCY_K:
                gains = _unique_mapping(
                    run_plan.redundancy_inner_fold_gains,
                    name="inner-fold gain",
                )
            result = run_comparator(
                method,
                matrix,
                target,
                groups,
                comparator_plan,
                expected_row_digest=comparator_row_digest,
                inner_fold_gains=gains,
            )
            _validate_complete_ranking(result, groups)
            outcomes.append(_complete_outcome(outcome_id, result))
        except Exception as error:
            outcomes.append(_failed_outcome(outcome_id, error))

    for tape in run_plan.random_tapes:
        outcome_id = f"RANDOM_K:{tape.tape_id}"
        try:
            result = run_comparator(
                ComparatorMethod.RANDOM_K,
                matrix,
                target,
                groups,
                comparator_plan,
                expected_row_digest=comparator_row_digest,
                random_tape_id=tape.tape_id,
            )
            _validate_complete_ranking(result, groups)
            outcomes.append(_complete_outcome(outcome_id, result))
        except Exception as error:
            outcomes.append(_failed_outcome(outcome_id, error))

    outcome_id = "ELASTIC_NET_K"
    try:
        elastic_plan = ElasticNetRunPlan(
            subset_k=run_plan.subset_k,
            lambda_path=run_plan.elastic_lambda_path,
            max_iter=run_plan.elastic_max_iter,
            tolerance=run_plan.elastic_tolerance,
        )
        result = run_elastic_net_selector(
            matrix,
            target,
            groups,
            folds,
            elastic_plan,
            expected_row_digest=compute_train_row_digest(matrix, target),
        )
        _validate_complete_ranking(result, groups)
        outcomes.append(_complete_outcome(outcome_id, result))
    except Exception as error:
        outcomes.append(_failed_outcome(outcome_id, error))

    return FeatureSelectionCampaignResult(
        row_digest=row_digest,
        group_digest=group_digest,
        plan_digest=plan_digest,
        sealed_subset_k=run_plan.subset_k,
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
