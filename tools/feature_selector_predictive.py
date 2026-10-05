"""TRAIN-only predictive selectors: ExtraTrees, sequential marginal, group gate, ChronoEpilogi.

Every selector here receives the fold's TRAIN fit rows and one chronological
nested split inside them.  None has an interface for outer validation or test
values.  All return one complete ranking plus a selection at every sealed K;
per-K failures are retained beside the ranking instead of erasing it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tools.feature_selector_comparators import (
    ComparatorRunPlan,
    FrozenDetails,
    truncate_sealed,
    validate_sealed_ks,
)
from tools.feature_selector_elastic_net import ChronologicalInnerFold

_VERSION = 1
CHRONOEPILOGI_PIN = {
    "repository": "https://github.com/ev07/minimal_chronoepilogi",
    "commit": "7de90758e1f232503ae7c29fc76f4e351253369a",
    "licence": "Apache-2.0",
    "paper": "Vareille, Linardi, Tsamardinos, Christophides. ChronoEpilogi: Scalable Time Series Selection with Multiple Solutions. NeurIPS 2024",
}
CHRONOEPILOGI_ADAPTER_UNREPRODUCED = (
    "MASS/FFT accelerated association",
    "ARDL univariate autoregressive start",
    "categorical ANOVA association path",
    "verification pass (phase V)",
    "pingouin/tigramite partial-correlation implementations",
)


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _positive_int(value: Any, name: str) -> int:
    if _is_bool(value) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _numeric_array(values: Any, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise TypeError(f"boolean {name} values are forbidden")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain numeric values")
    result = np.asarray(raw, dtype="<f8", order="C")
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if result.size == 0 or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_groups(group_ids: Any, columns: int) -> tuple[str, ...]:
    groups = tuple(group_ids)
    if len(groups) != columns:
        raise ValueError("one semantic group id is required per scalar column")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("semantic group ids must be non-empty strings")
    if len(set(groups)) != len(groups):
        raise ValueError("duplicate semantic groups are forbidden")
    return groups


def compute_predictive_row_digest(x: Any, y: Any) -> str:
    """Hash the exact finite TRAIN matrix and target handed to predictive selectors."""

    matrix = _numeric_array(x, ndim=2, name="X")
    target = _numeric_array(y, ndim=1, name="y")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y row counts differ")
    digest = hashlib.sha256(b"predictive-selector-train-rows-v1\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.tobytes(order="C"))
    digest.update(np.asarray(target.shape, dtype="<i8").tobytes())
    digest.update(target.tobytes(order="C"))
    return digest.hexdigest()


def _validated_split(split: Any, rows: int) -> ChronologicalInnerFold:
    if not isinstance(split, ChronologicalInnerFold):
        raise TypeError("nested split must be a ChronologicalInnerFold")
    if split.validation_end > rows:
        raise ValueError("nested split boundary exceeds TRAIN rows")
    return split


def _prepare(x: Any, y: Any, group_ids: Any, split: Any, expected_row_digest: str, input_label: str):
    if input_label != "TRAIN":
        raise ValueError("predictive feature selectors are TRAIN-only")
    if not isinstance(expected_row_digest, str) or not expected_row_digest:
        raise TypeError("expected_row_digest must be a non-empty string")
    matrix = _numeric_array(x, ndim=2, name="X")
    target = _numeric_array(y, ndim=1, name="y")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y row counts differ")
    groups = _validate_groups(group_ids, matrix.shape[1])
    row_digest = compute_predictive_row_digest(matrix, target)
    if row_digest != expected_row_digest:
        raise ValueError("TRAIN row digest mismatch")
    nested = _validated_split(split, matrix.shape[0])
    group_digest = _canonical_digest({"ordered_semantic_groups": groups})
    return matrix, target, groups, row_digest, group_digest, nested


def _split_payload(split: ChronologicalInnerFold) -> dict[str, Any]:
    return {
        "fold_id": split.fold_id,
        "train": (split.train_start, split.train_end),
        "validation": (split.validation_start, split.validation_end),
    }


@dataclass(frozen=True)
class PredictiveSelectorResult:
    """Authenticated complete ranking and sealed selections of one predictive method."""

    method: str
    implementation: str
    row_digest: str
    group_digest: str
    config_digest: str
    sealed_subset_k: int
    selected_groups: tuple[str, ...]
    full_ranking: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    selected_by_k: tuple[tuple[int, tuple[str, ...]], ...]
    k_failures: tuple[tuple[int, str], ...]
    details: FrozenDetails

    def selected_for(self, subset_k: int) -> tuple[str, ...]:
        if _is_bool(subset_k):
            raise TypeError("boolean subset K is forbidden")
        if subset_k == self.sealed_subset_k:
            return self.selected_groups
        for sealed_k, selection in self.selected_by_k:
            if sealed_k == subset_k:
                return selection
        raise ValueError("post-hoc truncation is forbidden")


def _finish(
    method: str,
    implementation: str,
    ranking: tuple[str, ...],
    scores: dict[str, float],
    groups: tuple[str, ...],
    comparator_plan: ComparatorRunPlan,
    row_digest: str,
    group_digest: str,
    config: dict[str, Any],
    details: dict[str, Any],
) -> PredictiveSelectorResult:
    if len(ranking) != len(groups) or set(ranking) != set(groups):
        raise RuntimeError(f"{method} did not produce one complete ranking")
    selected_by_k, k_failures = truncate_sealed(ranking, comparator_plan)
    config = dict(config, sealed_ks=comparator_plan.sealed_ks, version=_VERSION, input_label="TRAIN")
    return PredictiveSelectorResult(
        method=method,
        implementation=implementation,
        row_digest=row_digest,
        group_digest=group_digest,
        config_digest=_canonical_digest(config),
        sealed_subset_k=comparator_plan.subset_k,
        selected_groups=ranking[: comparator_plan.subset_k],
        full_ranking=ranking,
        scores=tuple((group, float(scores[group])) for group in ranking),
        selected_by_k=selected_by_k,
        k_failures=k_failures,
        details=FrozenDetails(details),
    )


# --------------------------------------------------------------------------- ridge helpers


def _standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0, dtype=np.float64)
    scales = np.std(matrix, axis=0, dtype=np.float64)
    scales = np.where(scales < 1e-12, 1.0, scales)
    return means, scales


class _RidgeProbe:
    """Closed-form ridge on standardized nested-train columns, scored on nested validation."""

    def __init__(self, matrix: np.ndarray, target: np.ndarray, split: ChronologicalInnerFold, ridge_lambda: float) -> None:
        train_x = matrix[split.train_start : split.train_end]
        self.train_y = target[split.train_start : split.train_end]
        self.val_y = target[split.validation_start : split.validation_end]
        self.means, self.scales = _standardizer(train_x)
        self.std_train = (train_x - self.means) / self.scales
        self.std_val = (matrix[split.validation_start : split.validation_end] - self.means) / self.scales
        self.train_mean = float(np.mean(self.train_y))
        centered = self.train_y - self.train_mean
        rows = float(self.std_train.shape[0])
        self.gram = (self.std_train.T @ self.std_train) / rows
        self.cross = (self.std_train.T @ centered) / rows
        self.ridge_lambda = float(ridge_lambda)
        self.naive_mae = float(np.mean(np.abs(self.val_y - self.train_mean)))

    def validation_mae(self, columns: tuple[int, ...]) -> float:
        index = np.asarray(columns, dtype=np.int64)
        system = self.gram[np.ix_(index, index)] + self.ridge_lambda * np.eye(index.size)
        coefficients = np.linalg.solve(system, self.cross[index])
        predictions = self.train_mean + self.std_val[:, index] @ coefficients
        return float(np.mean(np.abs(self.val_y - predictions)))


# --------------------------------------------------------------------------- ExtraTrees


@dataclass(frozen=True)
class ExtraTreesPlan:
    """Sealed ExtraTrees configuration: one seed, one model recipe for every fit and refit."""

    subset_k: int
    sensitivity_ks: tuple[int, ...] = ()
    seed: int = 0
    n_estimators: int = 100
    max_features: float = 0.3
    min_samples_leaf: int = 20
    block_length: int = 168
    n_jobs: int = 1
    refit_scope: str = "within_primary_k_set"

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        if _is_bool(self.seed) or not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise TypeError("seed must be a non-negative integer")
        object.__setattr__(self, "n_estimators", _positive_int(self.n_estimators, "n_estimators"))
        object.__setattr__(self, "min_samples_leaf", _positive_int(self.min_samples_leaf, "min_samples_leaf"))
        object.__setattr__(self, "block_length", _positive_int(self.block_length, "block_length"))
        if not (0.0 < float(self.max_features) <= 1.0):
            raise ValueError("max_features must lie in (0, 1]")
        if self.refit_scope != "within_primary_k_set":
            raise ValueError("only within_primary_k_set refit scope is implemented")

    def model_config(self) -> dict[str, Any]:
        """The one model recipe shared by the baseline fit and every refit-after-removal."""

        return {
            "estimator": "sklearn.ensemble.ExtraTreesRegressor",
            "n_estimators": self.n_estimators,
            "max_features": float(self.max_features),
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": int(self.seed),
            "bootstrap": False,
            "criterion": "squared_error",
        }


def _extratrees(plan: ExtraTreesPlan):
    from sklearn.ensemble import ExtraTreesRegressor

    config = plan.model_config()
    return ExtraTreesRegressor(
        n_estimators=config["n_estimators"],
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=config["random_state"],
        bootstrap=config["bootstrap"],
        criterion=config["criterion"],
        n_jobs=plan.n_jobs,
    )


def temporal_block_permutation(rows: int, block_length: int, seed: int) -> np.ndarray:
    """Return one row permutation that shuffles contiguous blocks, keeping within-block order."""

    starts = np.arange(0, rows, block_length)
    order = np.random.default_rng(seed).permutation(starts.size)
    pieces = [np.arange(starts[index], min(starts[index] + block_length, rows)) for index in order]
    return np.concatenate(pieces)


def fit_and_score_extratrees(
    plan: ExtraTreesPlan,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> tuple[float, np.ndarray, Any]:
    """Fit the sealed recipe once; return validation MAE, impurity importances and the model."""

    model = _extratrees(plan)
    model.fit(train_x, train_y)
    predictions = model.predict(val_x)
    return float(np.mean(np.abs(val_y - predictions))), np.asarray(model.feature_importances_, dtype=np.float64), model


def run_extratrees_selector(
    x: Any,
    y: Any,
    group_ids: Any,
    nested_split: Any,
    plan: ExtraTreesPlan,
    *,
    expected_row_digest: str,
    input_label: str = "TRAIN",
) -> PredictiveSelectorResult:
    """ExtraTrees ranked by temporal block permutation importance, with refit-after-removal at primary K."""

    if not isinstance(plan, ExtraTreesPlan):
        raise TypeError("plan must be ExtraTreesPlan")
    matrix, target, groups, row_digest, group_digest, split = _prepare(
        x, y, group_ids, nested_split, expected_row_digest, input_label
    )
    comparator_plan = ComparatorRunPlan(plan.subset_k, (), plan.sensitivity_ks)
    if plan.subset_k > len(groups):
        raise ValueError("subset K exceeds the number of semantic groups")
    train_x = matrix[split.train_start : split.train_end]
    train_y = target[split.train_start : split.train_end]
    val_x = matrix[split.validation_start : split.validation_end]
    val_y = target[split.validation_start : split.validation_end]
    naive_mae = float(np.mean(np.abs(val_y - float(np.mean(train_y)))))

    base_mae, impurity, model = fit_and_score_extratrees(plan, train_x, train_y, val_x, val_y)
    permutation = temporal_block_permutation(val_x.shape[0], plan.block_length, plan.seed)
    importance: dict[str, float] = {}
    for index, group in enumerate(groups):
        permuted = val_x.copy()
        permuted[:, index] = val_x[permutation, index]
        importance[group] = float(np.mean(np.abs(val_y - model.predict(permuted)))) - base_mae
    ranking = tuple(
        sorted(groups, key=lambda group: (-importance[group], -float(impurity[groups.index(group)]), group))
    )
    selected = ranking[: plan.subset_k]
    selected_index = tuple(groups.index(group) for group in selected)
    subset_mae, _, _ = fit_and_score_extratrees(
        plan, train_x[:, selected_index], train_y, val_x[:, selected_index], val_y
    )
    refit_delta: dict[str, float] = {}
    for group in selected:
        kept = tuple(index for index in selected_index if groups[index] != group)
        removed_mae, _, _ = fit_and_score_extratrees(plan, train_x[:, kept], train_y, val_x[:, kept], val_y)
        refit_delta[group] = removed_mae - subset_mae
    details = {
        "nested_split": _split_payload(split),
        "naive_validation_mae": naive_mae,
        "full_model_validation_mae": base_mae,
        "primary_k_model_validation_mae": subset_mae,
        "block_permutation": {"block_length": plan.block_length, "seed": int(plan.seed), "shared_across_features": True},
        "impurity_importance": tuple((group, float(impurity[index])) for index, group in enumerate(groups)),
        "refit_after_removal": {
            "scope": plan.refit_scope,
            "model_config": plan.model_config(),
            "rows": "same nested-train rows, same nested-validation rows",
            "delta_validation_mae_when_removed": tuple(sorted(refit_delta.items())),
        },
    }
    config = {
        "method": "EXTRATREES_TBP_K",
        "model_config": plan.model_config(),
        "n_jobs_affects_results": False,
        "block_length": plan.block_length,
        "refit_scope": plan.refit_scope,
        "nested_split": _split_payload(split),
    }
    return _finish(
        "EXTRATREES_TBP_K", "sklearn_extratrees", ranking, importance, groups,
        comparator_plan, row_digest, group_digest, config, details,
    )


# --------------------------------------------------------------------------- sequential marginal


@dataclass(frozen=True)
class SequentialMarginalPlan:
    """Forward marginal selection under a declared step budget, ridge probe on the nested split."""

    subset_k: int
    sensitivity_ks: tuple[int, ...] = ()
    budget: int | None = None
    ridge_lambda: float = 1e-3

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        budget = self.budget if self.budget is not None else max((primary,) + sensitivities)
        object.__setattr__(self, "budget", _positive_int(budget, "budget"))
        if self.budget < max((primary,) + sensitivities):
            raise ValueError("declared budget must cover every sealed K")
        if not (float(self.ridge_lambda) > 0.0):
            raise ValueError("ridge_lambda must be positive")


def run_sequential_marginal_selector(
    x: Any,
    y: Any,
    group_ids: Any,
    nested_split: Any,
    plan: SequentialMarginalPlan,
    *,
    expected_row_digest: str,
    input_label: str = "TRAIN",
) -> PredictiveSelectorResult:
    """Greedy forward selection by nested-validation MAE reduction, budget steps, remainder by marginal gain."""

    if not isinstance(plan, SequentialMarginalPlan):
        raise TypeError("plan must be SequentialMarginalPlan")
    matrix, target, groups, row_digest, group_digest, split = _prepare(
        x, y, group_ids, nested_split, expected_row_digest, input_label
    )
    comparator_plan = ComparatorRunPlan(plan.subset_k, (), plan.sensitivity_ks)
    if plan.subset_k > len(groups):
        raise ValueError("subset K exceeds the number of semantic groups")
    probe = _RidgeProbe(matrix, target, split, plan.ridge_lambda)
    univariate = {group: probe.naive_mae - probe.validation_mae((index,)) for index, group in enumerate(groups)}
    remaining = list(range(len(groups)))
    chosen: list[int] = []
    scores: dict[str, float] = {}
    path: list[dict[str, Any]] = []
    current_mae = probe.naive_mae
    steps = min(plan.budget, len(groups))
    for _ in range(steps):
        best_index = None
        best_mae = None
        for index in remaining:
            mae = probe.validation_mae(tuple(chosen) + (index,))
            if best_mae is None or mae < best_mae or (mae == best_mae and groups[index] < groups[best_index]):
                best_mae, best_index = mae, index
        chosen.append(best_index)
        remaining.remove(best_index)
        scores[groups[best_index]] = current_mae - best_mae
        path.append({"step": len(chosen), "group": groups[best_index], "validation_mae": best_mae})
        current_mae = best_mae
    tail = sorted(remaining, key=lambda index: (-univariate[groups[index]], groups[index]))
    offset = min(scores.values()) if scores else 0.0
    for position, index in enumerate(tail):
        # Tail scores stay strictly below every budgeted step's score so the
        # order of the complete ranking is the order of the score tuple.
        scores[groups[index]] = offset - 1.0 - position
    ranking = tuple(groups[index] for index in chosen + tail)
    details = {
        "nested_split": _split_payload(split),
        "naive_validation_mae": probe.naive_mae,
        "budget": plan.budget,
        "steps_taken": steps,
        "path": tuple(path),
        "univariate_gain": tuple(sorted(univariate.items())),
        "tail_rule": "remaining groups ordered by univariate nested-validation gain",
    }
    config = {
        "method": "SEQ_MARGINAL_K",
        "probe": "closed_form_ridge_standardized_on_nested_train",
        "ridge_lambda": plan.ridge_lambda,
        "budget": plan.budget,
        "nested_split": _split_payload(split),
    }
    return _finish(
        "SEQ_MARGINAL_K", "ridge_forward_stepwise", ranking, scores, groups,
        comparator_plan, row_digest, group_digest, config, details,
    )


# --------------------------------------------------------------------------- temporal group gate


@dataclass(frozen=True)
class GroupGatePlan:
    """Cluster gate on the nested split with a circular-shift temporal null."""

    subset_k: int
    group_membership: tuple[tuple[str, str], ...]
    sensitivity_ks: tuple[int, ...] = ()
    ridge_lambda: float = 1e-3
    null_shifts: tuple[int, ...] = (720, 1440, 2160, 2880, 3600)

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        pairs = tuple((str(group), str(cluster)) for group, cluster in tuple(self.group_membership))
        if len({group for group, _ in pairs}) != len(pairs):
            raise ValueError("duplicate group membership entry")
        object.__setattr__(self, "group_membership", pairs)
        shifts = tuple(_positive_int(value, "null shift") for value in self.null_shifts)
        if not shifts:
            raise ValueError("at least one temporal null shift is required")
        object.__setattr__(self, "null_shifts", shifts)
        if not (float(self.ridge_lambda) > 0.0):
            raise ValueError("ridge_lambda must be positive")


def run_group_gate_selector(
    x: Any,
    y: Any,
    group_ids: Any,
    nested_split: Any,
    plan: GroupGatePlan,
    *,
    expected_row_digest: str,
    input_label: str = "TRAIN",
) -> PredictiveSelectorResult:
    """Gate dependence clusters by nested-validation gain against a temporal shift null."""

    if not isinstance(plan, GroupGatePlan):
        raise TypeError("plan must be GroupGatePlan")
    matrix, target, groups, row_digest, group_digest, split = _prepare(
        x, y, group_ids, nested_split, expected_row_digest, input_label
    )
    comparator_plan = ComparatorRunPlan(plan.subset_k, (), plan.sensitivity_ks)
    if plan.subset_k > len(groups):
        raise ValueError("subset K exceeds the number of semantic groups")
    membership = dict(plan.group_membership)
    if set(membership) != set(groups):
        raise ValueError("group membership must name every semantic group exactly")
    clusters: dict[str, tuple[int, ...]] = {}
    for index, group in enumerate(groups):
        clusters.setdefault(membership[group], ())
        clusters[membership[group]] += (index,)
    train_rows = split.train_end - split.train_start
    if any(shift >= train_rows for shift in plan.null_shifts):
        raise ValueError("temporal null shift must be smaller than the nested-train rows")

    probe = _RidgeProbe(matrix, target, split, plan.ridge_lambda)
    null_probes = []
    for shift in plan.null_shifts:
        shifted = target.copy()
        fit_slice = slice(split.train_start, split.validation_end)
        shifted[fit_slice] = np.roll(target[fit_slice], shift)
        null_probes.append(_RidgeProbe(matrix, shifted, split, plan.ridge_lambda))

    gain: dict[str, float] = {}
    null_max: dict[str, float] = {}
    null_threshold: dict[str, float] = {}
    passed: dict[str, bool] = {}
    for cluster, columns in clusters.items():
        gain[cluster] = probe.naive_mae - probe.validation_mae(columns)
        nulls = np.asarray([null.naive_mae - null.validation_mae(columns) for null in null_probes])
        null_max[cluster] = float(np.max(nulls))
        spread = float(np.std(nulls, ddof=1)) if nulls.size > 1 else 0.0
        # Pass only when the gain beats every temporal null AND clears the null
        # scale by three standard deviations: with few shifts "beats the max"
        # alone passes one noise cluster in (shifts + 1).
        null_threshold[cluster] = max(null_max[cluster], float(np.mean(nulls)) + 3.0 * spread)
        passed[cluster] = gain[cluster] > 0.0 and gain[cluster] > null_threshold[cluster]
    univariate = {group: probe.naive_mae - probe.validation_mae((index,)) for index, group in enumerate(groups)}
    ranking = tuple(
        sorted(
            groups,
            key=lambda group: (
                0 if passed[membership[group]] else 1,
                -gain[membership[group]],
                membership[group],
                -univariate[group],
                group,
            ),
        )
    )
    scores = {group: gain[membership[group]] if passed[membership[group]] else gain[membership[group]] - 1.0e6 for group in groups}
    details = {
        "nested_split": _split_payload(split),
        "naive_validation_mae": probe.naive_mae,
        "clusters": tuple((cluster, tuple(groups[index] for index in columns)) for cluster, columns in sorted(clusters.items())),
        "cluster_gain": tuple(sorted(gain.items())),
        "cluster_null_max_gain": tuple(sorted(null_max.items())),
        "cluster_null_threshold": tuple(sorted(null_threshold.items())),
        "cluster_passed": tuple(sorted(passed.items())),
        "gate_rule": "gain > 0 and gain > max(max(null), mean(null) + 3*std(null))",
        "n_passed": int(sum(passed.values())),
        "null": {"kind": "circular_shift_of_target_within_fit_rows", "shifts": plan.null_shifts},
        "univariate_gain": tuple(sorted(univariate.items())),
        "score_rule": "gated clusters carry their gain; failed clusters carry gain - 1e6 so they sort after",
    }
    config = {
        "method": "GROUP_GATE_K",
        "membership": plan.group_membership,
        "ridge_lambda": plan.ridge_lambda,
        "null_shifts": plan.null_shifts,
        "nested_split": _split_payload(split),
    }
    return _finish(
        "GROUP_GATE_K", "ridge_cluster_gate_temporal_null", ranking, scores, groups,
        comparator_plan, row_digest, group_digest, config, details,
    )


# --------------------------------------------------------------------------- ChronoEpilogi


@dataclass(frozen=True)
class ChronoEpilogiPlan:
    """ChronoEpilogi configuration: official pinned package when importable, else the faithful adapter."""

    subset_k: int
    sensitivity_ks: tuple[int, ...] = ()
    implementation: str = "auto"
    max_lag: int = 1
    forward_threshold: float = 0.05
    backward_threshold: float = 0.05
    equivalence_threshold: float = 0.05
    maximal_selected_size: int | None = None

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        if self.implementation not in ("auto", "official", "adapter"):
            raise ValueError("implementation must be auto, official or adapter")
        object.__setattr__(self, "max_lag", _positive_int(self.max_lag, "max_lag"))
        for name in ("forward_threshold", "backward_threshold", "equivalence_threshold"):
            value = float(getattr(self, name))
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.maximal_selected_size is not None:
            object.__setattr__(self, "maximal_selected_size", _positive_int(self.maximal_selected_size, "maximal_selected_size"))


def _ols_rss(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return residual, float(np.dot(residual, residual))


def _lr_pvalue(rss_restricted: float, rss_full: float, rows: int, extra_dof: int) -> float:
    from scipy.stats import chi2

    if rss_full <= 0.0 or rss_restricted <= 0.0:
        return 0.0
    statistic = rows * np.log(rss_restricted / rss_full)
    return float(chi2.sf(max(statistic, 0.0), df=extra_dof))


def _pearson_pvalue(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    from scipy.stats import pearsonr

    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0, 1.0
    result = pearsonr(left, right)
    return float(result[0]), float(result[1])


def _partial_pvalue(left: np.ndarray, right: np.ndarray, given: np.ndarray) -> float:
    design = np.column_stack([np.ones(given.shape[0]), given])
    left_res, _ = _ols_rss(design, left)
    right_res, _ = _ols_rss(design, right)
    return _pearson_pvalue(left_res, right_res)[1]


def _chronoepilogi_adapter(
    matrix: np.ndarray, target: np.ndarray, groups: tuple[str, ...], plan: ChronoEpilogiPlan
) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    """Forward-backward-equivalence selection with OLS LR tests and partial-correlation equivalence."""

    rows = matrix.shape[0]
    intercept = np.ones((rows, 1))
    selected: list[int] = []
    equivalents: dict[str, list[str]] = {}
    residual, rss = _ols_rss(intercept, target)
    limit = plan.maximal_selected_size or len(groups)
    forward_log: list[dict[str, Any]] = []
    while len(selected) < min(limit, len(groups)):
        remaining = [index for index in range(len(groups)) if index not in selected]
        if not remaining or rows <= len(selected) + 3:
            break
        association = {index: abs(_pearson_pvalue(matrix[:, index], residual)[0]) for index in remaining}
        chosen = min(remaining, key=lambda index: (-association[index], groups[index]))
        design = np.column_stack([intercept, matrix[:, selected + [chosen]]])
        new_residual, new_rss = _ols_rss(design, target)
        p_value = _lr_pvalue(rss, new_rss, rows, 1)
        forward_log.append({"candidate": groups[chosen], "association": association[chosen], "lr_pvalue": p_value})
        if p_value >= plan.forward_threshold:
            break
        # equivalence heuristic (parcorr): candidates that are conditionally independent of the
        # residual given the chosen variable, and vice versa, while marginally associated.
        equal: list[str] = []
        for index in sorted(remaining, key=lambda idx: -association[idx]):
            if index == chosen:
                continue
            if _pearson_pvalue(matrix[:, index], residual)[1] >= plan.equivalence_threshold:
                break
            p_c_given_v = _partial_pvalue(matrix[:, index], residual, matrix[:, [chosen]])
            p_v_given_c = _partial_pvalue(matrix[:, chosen], residual, matrix[:, [index]])
            if p_c_given_v > plan.equivalence_threshold and p_v_given_c > plan.equivalence_threshold:
                equal.append(groups[index])
            else:
                break
        selected.append(chosen)
        equivalents[groups[chosen]] = equal
        residual, rss = new_residual, new_rss
    # backward phase
    changed = True
    while changed and selected:
        changed = False
        full_design = np.column_stack([intercept, matrix[:, selected]])
        _, full_rss = _ols_rss(full_design, target)
        worst, worst_p = None, -1.0
        for index in selected:
            kept = [other for other in selected if other != index]
            _, restricted_rss = _ols_rss(np.column_stack([intercept, matrix[:, kept]]) if kept else intercept, target)
            p_value = _lr_pvalue(restricted_rss, full_rss, rows, 1)
            if p_value > worst_p:
                worst, worst_p = index, p_value
        if worst is not None and worst_p >= plan.backward_threshold:
            selected.remove(worst)
            equivalents.pop(groups[worst], None)
            changed = True
    boundary = [groups[index] for index in selected]
    classes = [[name] + equivalents.get(name, []) for name in boundary]
    return boundary, classes, {"forward_log": tuple(forward_log)}


def _chronoepilogi_official(
    matrix: np.ndarray, target: np.ndarray, groups: tuple[str, ...], plan: ChronoEpilogiPlan
) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    import pandas as pd
    from chronoepilogi import ChronoEpilogi  # pinned: CHRONOEPILOGI_PIN

    # Row r holds y[r] and x[r + 1]; with max_lag = 1 the package reads x at lag 1,
    # i.e. x[r], the information available at the decision time of y[r].
    lag = plan.max_lag
    frame = {"__target__": target[:-lag]}
    for index, group in enumerate(groups):
        frame[group] = matrix[lag:, index]
    data = pd.DataFrame(frame)
    kwargs: dict[str, Any] = dict(
        phases="FBEV",
        forward_test_threshold=plan.forward_threshold,
        backward_test_threshold=plan.backward_threshold,
        equivalence_test_threshold=plan.equivalence_threshold,
        start_with_univariate_autoregressive_model=False,
        default_max_lag=lag,
        target_type="continuous",
    )
    if plan.maximal_selected_size is not None:
        kwargs["maximal_selected_size"] = plan.maximal_selected_size
    try:
        from chronoepilogi.associations import TemporalSlowAssociation

        kwargs["association_class"] = TemporalSlowAssociation
        kwargs["association_config"] = {"lags": lag, "n_jobs": 1}
    except Exception:  # pragma: no cover - depends on the pinned package layout
        pass
    selector = ChronoEpilogi(data, "__target__", **kwargs)
    selector.fit()
    boundary = [name for name in selector.get_first_markov_boundary() if name != "__target__"]
    try:
        classes = [[member for member in cls if member != "__target__"] for cls in selector.get_equivalence_classes()]
        classes = [cls for cls in classes if cls]
    except Exception:  # pragma: no cover
        classes = [[name] for name in boundary]
    version = getattr(__import__("chronoepilogi"), "__version__", "unknown")
    return boundary, classes, {"package_version": str(version), "kwargs": {k: str(v) for k, v in kwargs.items()}}


def run_chronoepilogi_selector(
    x: Any,
    y: Any,
    group_ids: Any,
    nested_split: Any,
    plan: ChronoEpilogiPlan,
    *,
    expected_row_digest: str,
    input_label: str = "TRAIN",
) -> PredictiveSelectorResult:
    """ChronoEpilogi minimal predictive subsets with equivalence classes, extended to a complete ranking.

    The method is a predictive multiple-solution selector.  It is never
    reported as causal evidence.  The complete ranking is: first Markov
    boundary in forward-selection order, then the other members of each
    equivalence class in class order, then every remaining group by absolute
    Spearman association with the final OLS residual.
    """

    if not isinstance(plan, ChronoEpilogiPlan):
        raise TypeError("plan must be ChronoEpilogiPlan")
    matrix, target, groups, row_digest, group_digest, split = _prepare(
        x, y, group_ids, nested_split, expected_row_digest, input_label
    )
    comparator_plan = ComparatorRunPlan(plan.subset_k, (), plan.sensitivity_ks)
    if plan.subset_k > len(groups):
        raise ValueError("subset K exceeds the number of semantic groups")
    fit_x = matrix[split.train_start : split.validation_end]
    fit_y = target[split.train_start : split.validation_end]
    means, scales = _standardizer(fit_x)
    std_x = (fit_x - means) / scales

    implementation = plan.implementation
    official_available = False
    if implementation in ("auto", "official"):
        try:
            import chronoepilogi  # noqa: F401

            official_available = True
        except Exception:
            official_available = False
        if implementation == "official" and not official_available:
            raise RuntimeError("official ChronoEpilogi package is not importable")
        implementation = "official" if official_available else "adapter"
    if implementation == "official":
        boundary, classes, extra = _chronoepilogi_official(std_x, fit_y, groups, plan)
        implementation_label = f"official_pinned_{CHRONOEPILOGI_PIN['commit'][:12]}"
        unreproduced: tuple[str, ...] = ()
    else:
        boundary, classes, extra = _chronoepilogi_adapter(std_x, fit_y, groups, plan)
        implementation_label = "adapter_faithful"
        unreproduced = CHRONOEPILOGI_ADAPTER_UNREPRODUCED

    ordered: list[str] = list(boundary)
    for cls in classes:
        for member in cls:
            if member not in ordered:
                ordered.append(member)
    design = np.column_stack([np.ones(fit_y.shape[0])] + [std_x[:, groups.index(name)] for name in boundary])
    residual, _ = _ols_rss(design, fit_y)
    from tools.feature_selector_comparators import _spearman

    association = {group: abs(_spearman(std_x[:, index], residual)) for index, group in enumerate(groups)}
    tail = sorted((group for group in groups if group not in ordered), key=lambda group: (-association[group], group))
    ranking = tuple(ordered + tail)
    scores: dict[str, float] = {}
    for position, group in enumerate(ranking):
        if group in boundary:
            scores[group] = float(len(groups) + len(boundary) - boundary.index(group))
        elif group in ordered:
            scores[group] = float(len(groups) - position)
        else:
            scores[group] = association[group]
    details = {
        "nested_split": _split_payload(split),
        "fit_rows": "nested train + nested validation (the fold's TRAIN fit rows); no outer data",
        "first_markov_boundary": tuple(boundary),
        "equivalence_classes": tuple(tuple(cls) for cls in classes),
        "n_alternative_solutions": int(np.prod([len(cls) for cls in classes])) if classes else 0,
        "residual_association": tuple(sorted(association.items())),
        "pin": CHRONOEPILOGI_PIN,
        "unreproduced_scope": unreproduced,
        "evidence_class": "predictive multiple-solution selection; NOT causal evidence",
        "extra": extra,
    }
    config = {
        "method": "CHRONOEPILOGI_K",
        "implementation": implementation_label,
        "max_lag": plan.max_lag,
        "thresholds": {
            "forward": plan.forward_threshold,
            "backward": plan.backward_threshold,
            "equivalence": plan.equivalence_threshold,
        },
        "maximal_selected_size": plan.maximal_selected_size,
        "nested_split": _split_payload(split),
        "ranking_rule": "boundary order, equivalence members, then |spearman| with final residual",
    }
    return _finish(
        "CHRONOEPILOGI_K", implementation_label, ranking, scores, groups,
        comparator_plan, row_digest, group_digest, config, details,
    )


__all__ = [
    "CHRONOEPILOGI_ADAPTER_UNREPRODUCED",
    "CHRONOEPILOGI_PIN",
    "ChronoEpilogiPlan",
    "ExtraTreesPlan",
    "GroupGatePlan",
    "PredictiveSelectorResult",
    "SequentialMarginalPlan",
    "compute_predictive_row_digest",
    "fit_and_score_extratrees",
    "run_chronoepilogi_selector",
    "run_extratrees_selector",
    "run_group_gate_selector",
    "run_sequential_marginal_selector",
    "temporal_block_permutation",
]
