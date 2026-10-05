"""Deterministic TRAIN-only elastic-net selector for scalar feature groups.

The selector owns hyperparameter selection, the final refit, ranking, and
truncation.  It accepts one current scalar column per semantic group and has no
interface for outer validation or test data.  Every standardizer is fitted on
the corresponding inner-training rows only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.feature_selector_comparators import truncate_sealed, validate_sealed_ks


L1_RATIO_PATH = (0.1, 0.5, 0.9, 1.0)
_VERSION = 1


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _finite_float(value: Any, name: str) -> float:
    if _is_bool(value):
        raise TypeError(f"boolean {name} is forbidden")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if _is_bool(value):
        raise TypeError(f"boolean {name} is forbidden")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ChronologicalInnerFold:
    """Half-open row boundaries for one chronological inner fold."""

    fold_id: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id.strip():
            raise ValueError("fold_id must be a non-empty string")
        for name in (
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
        ):
            value = getattr(self, name)
            if _is_bool(value):
                raise TypeError(f"boolean {name} is forbidden")
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.train_start < 0:
            raise ValueError("fold boundaries must be non-negative")
        if self.train_start >= self.train_end:
            raise ValueError("inner-training fold must not be empty")
        if self.validation_start >= self.validation_end:
            raise ValueError("inner-validation fold must not be empty")
        if self.train_end > self.validation_start:
            raise ValueError("inner training and validation must not overlap")


@dataclass(frozen=True)
class ElasticNetRunPlan:
    """Sealed selection and deterministic coordinate-descent budget."""

    subset_k: int
    lambda_path: tuple[float, ...]
    max_iter: int = 10_000
    tolerance: float = 1e-8
    sensitivity_ks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        object.__setattr__(self, "max_iter", _positive_int(self.max_iter, "max_iter"))
        tolerance = _finite_float(self.tolerance, "tolerance")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        object.__setattr__(self, "tolerance", tolerance)

        path = tuple(_finite_float(value, "lambda") for value in self.lambda_path)
        if len(path) < 2:
            raise ValueError("lambda path must contain at least two values")
        if any(value <= 0.0 for value in path):
            raise ValueError("lambda values must be positive")
        if any(left <= right for left, right in zip(path, path[1:])):
            raise ValueError("lambda path must be strictly descending")
        log_steps = np.diff(np.log(np.asarray(path, dtype=np.float64)))
        if not np.allclose(log_steps, log_steps[0], rtol=1e-10, atol=1e-12):
            raise ValueError("lambda path must be logarithmically spaced")
        object.__setattr__(self, "lambda_path", path)

    @property
    def sealed_ks(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.sensitivity_ks) | {self.subset_k}))

    def comparator_view(self):
        """The sealed budgets expressed as a ComparatorRunPlan for truncation."""

        from tools.feature_selector_comparators import ComparatorRunPlan

        return ComparatorRunPlan(self.subset_k, (), self.sensitivity_ks)


@dataclass(frozen=True)
class FitDiagnostic:
    """Convergence and score evidence for one fit."""

    fold_id: str
    l1_ratio: float
    lambda_value: float
    iterations: int
    converged: bool
    objective: float
    validation_mae: float | None


@dataclass(frozen=True)
class ElasticNetSelectorResult:
    """Authenticated complete ranking under one immutable run plan."""

    selected_groups: tuple[str, ...]
    full_ranking: tuple[str, ...]
    standardized_coefficients: tuple[tuple[str, float], ...]
    coefficient_norms: tuple[tuple[str, float], ...]
    selected_l1_ratio: float
    selected_lambda: float
    selected_mean_inner_validation_mae: float
    sealed_subset_k: int
    row_digest: str
    group_digest: str
    fold_digest: str
    hyperparameter_digest: str
    config_digest: str
    diagnostics: tuple[FitDiagnostic, ...]
    final_iterations: int
    final_objective: float
    target_mean: float
    train_means: tuple[float, ...]
    train_scales: tuple[float, ...]
    selected_by_k: tuple[tuple[int, tuple[str, ...]], ...] = ()
    k_failures: tuple[tuple[int, str], ...] = ()
    penalty: str = "elastic_net"
    penalty_groups: tuple[tuple[str, str], ...] = ()

    def selected_for(self, subset_k: int) -> tuple[str, ...]:
        """Return only a selection made under a sealed K."""

        if _is_bool(subset_k):
            raise TypeError("boolean subset K is forbidden")
        if subset_k == self.sealed_subset_k:
            return self.selected_groups
        for sealed_k, selection in self.selected_by_k:
            if sealed_k == subset_k:
                return selection
        raise ValueError("post-hoc K changes are forbidden")


@dataclass(frozen=True)
class _Fit:
    coefficients: np.ndarray
    iterations: int
    converged: bool
    objective: float


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


def compute_train_row_digest(x: Any, y: Any) -> str:
    """Hash the exact canonical finite TRAIN matrix and target."""

    matrix = _numeric_array(x, ndim=2, name="X")
    target = _numeric_array(y, ndim=1, name="y")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y row counts differ")
    digest = hashlib.sha256(b"elastic-net-selector-train-rows-v1\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.tobytes(order="C"))
    digest.update(np.asarray(target.shape, dtype="<i8").tobytes())
    digest.update(target.tobytes(order="C"))
    return digest.hexdigest()


def _validate_folds(
    folds: Any, row_count: int
) -> tuple[ChronologicalInnerFold, ...]:
    result = tuple(folds)
    if not result:
        raise ValueError("at least one chronological inner fold is required")
    if any(not isinstance(fold, ChronologicalInnerFold) for fold in result):
        raise TypeError("folds must contain ChronologicalInnerFold values")
    ids = [fold.fold_id for fold in result]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fold ids are forbidden")
    previous_validation_end = -1
    for fold in result:
        if fold.validation_end > row_count:
            raise ValueError("fold boundary exceeds TRAIN rows")
        if fold.validation_start < previous_validation_end:
            raise ValueError("inner-validation folds overlap or are nonchronological")
        previous_validation_end = fold.validation_end
    return result


def _standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0, dtype=np.float64)
    scales = np.std(matrix, axis=0, dtype=np.float64)
    scales = np.where(scales < 1e-12, 1.0, scales)
    return means, scales


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _objective(
    matrix: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    lambda_value: float,
    l1_ratio: float,
) -> float:
    residual = target - matrix @ coefficients
    loss = 0.5 * float(np.mean(residual * residual))
    penalty = lambda_value * (
        l1_ratio * float(np.sum(np.abs(coefficients)))
        + 0.5 * (1.0 - l1_ratio) * float(np.dot(coefficients, coefficients))
    )
    return loss + penalty


def _coordinate_descent(
    matrix: np.ndarray,
    centered_target: np.ndarray,
    lambda_value: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
    warm_start: np.ndarray | None = None,
) -> _Fit:
    # A warm start along a descending lambda path changes only the iteration
    # count, never the fixed point: the objective is convex and the stopping
    # rule is the same.  It is recorded in the config digest.
    if warm_start is None:
        coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
        prediction = np.zeros(matrix.shape[0], dtype=np.float64)
    else:
        coefficients = np.array(warm_start, dtype=np.float64, copy=True)
        prediction = matrix @ coefficients
    column_energy = np.mean(matrix * matrix, axis=0, dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        largest_change = 0.0
        for column in range(matrix.shape[1]):
            old = coefficients[column]
            residual_without_column = centered_target - prediction + matrix[:, column] * old
            correlation = float(
                np.mean(matrix[:, column] * residual_without_column, dtype=np.float64)
            )
            denominator = column_energy[column] + lambda_value * (1.0 - l1_ratio)
            new = 0.0 if denominator == 0.0 else (
                _soft_threshold(correlation, lambda_value * l1_ratio) / denominator
            )
            coefficients[column] = new
            prediction += matrix[:, column] * (new - old)
            largest_change = max(largest_change, abs(new - old))
        iterations = iteration
        if largest_change <= tolerance:
            converged = True
            break
    return _Fit(
        coefficients=coefficients,
        iterations=iterations,
        converged=converged,
        objective=_objective(
            matrix, centered_target, coefficients, lambda_value, l1_ratio
        ),
    )


def _group_blocks(
    groups: tuple[str, ...], membership: Any
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Validate one cluster id per group and return ordered column blocks."""

    pairs = tuple(membership)
    mapping: dict[str, str] = {}
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("group membership entries must be (group, cluster) pairs")
        group, cluster = item
        if group in mapping:
            raise ValueError("duplicate group membership entry")
        if not isinstance(cluster, str) or not cluster.strip():
            raise ValueError("cluster ids must be non-empty strings")
        mapping[group] = cluster
    if set(mapping) != set(groups):
        raise ValueError("group membership must name every semantic group exactly")
    cluster_ids = tuple(sorted(set(mapping.values())))
    blocks = tuple(
        tuple(index for index, group in enumerate(groups) if mapping[group] == cluster)
        for cluster in cluster_ids
    )
    return cluster_ids, blocks


def _group_proximal_descent(
    matrix: np.ndarray,
    centered_target: np.ndarray,
    blocks: tuple[tuple[int, ...], ...],
    lambda_value: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
    warm_start: np.ndarray | None = None,
) -> _Fit:
    """Block proximal gradient descent for sparse-group (group lasso + ridge)."""

    rows = matrix.shape[0]
    if warm_start is None:
        coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    else:
        coefficients = np.array(warm_start, dtype=np.float64, copy=True)
    prediction = matrix @ coefficients
    block_arrays = tuple(np.asarray(block, dtype=np.int64) for block in blocks)
    lipschitz = []
    weights = []
    for block in block_arrays:
        sub = matrix[:, block]
        gram = (sub.T @ sub) / float(rows)
        largest = float(np.max(np.linalg.eigvalsh(gram))) if block.size > 1 else float(gram[0, 0])
        lipschitz.append(max(largest + lambda_value * (1.0 - l1_ratio), 1e-12))
        weights.append(float(np.sqrt(block.size)))
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        largest_change = 0.0
        for block, step_l, weight in zip(block_arrays, lipschitz, weights):
            old = coefficients[block].copy()
            residual = centered_target - prediction
            gradient = -(matrix[:, block].T @ residual) / float(rows) + lambda_value * (
                1.0 - l1_ratio
            ) * old
            candidate = old - gradient / step_l
            norm = float(np.sqrt(np.dot(candidate, candidate)))
            threshold = lambda_value * l1_ratio * weight / step_l
            new = np.zeros_like(candidate) if norm <= threshold else candidate * (1.0 - threshold / norm)
            delta = new - old
            if np.any(delta != 0.0):
                prediction += matrix[:, block] @ delta
                coefficients[block] = new
                largest_change = max(largest_change, float(np.max(np.abs(delta))))
        iterations = iteration
        if largest_change <= tolerance:
            converged = True
            break
    penalty = 0.0
    for block, weight in zip(block_arrays, weights):
        penalty += weight * float(np.sqrt(np.dot(coefficients[block], coefficients[block])))
    residual = centered_target - prediction
    objective = 0.5 * float(np.mean(residual * residual)) + lambda_value * (
        l1_ratio * penalty + 0.5 * (1.0 - l1_ratio) * float(np.dot(coefficients, coefficients))
    )
    return _Fit(coefficients=coefficients, iterations=iterations, converged=converged, objective=objective)


def run_elastic_net_selector(
    x: Any,
    y: Any,
    group_ids: Any,
    inner_folds: Any,
    run_plan: ElasticNetRunPlan,
    *,
    expected_row_digest: str,
    input_label: str = "TRAIN",
    group_membership: Any | None = None,
) -> ElasticNetSelectorResult:
    """Select scalar semantic groups using inner-fold MAE only.

    With ``group_membership`` (one ``(group, cluster)`` pair per semantic group)
    the penalty becomes sparse-group: a group-lasso term over clusters plus the
    ridge term, fitted by block proximal gradient descent.  Ranking is then by
    cluster norm first and within-cluster coefficient magnitude second.
    """

    if input_label != "TRAIN":
        raise ValueError("elastic-net feature selection accepts TRAIN only")
    if not isinstance(run_plan, ElasticNetRunPlan):
        raise TypeError("run_plan must be ElasticNetRunPlan")
    if not isinstance(expected_row_digest, str) or not expected_row_digest:
        raise TypeError("expected_row_digest must be a non-empty string")

    matrix = _numeric_array(x, ndim=2, name="X")
    target = _numeric_array(y, ndim=1, name="y")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y row counts differ")
    groups = _validate_groups(group_ids, matrix.shape[1])
    if run_plan.subset_k > len(groups):
        raise ValueError("subset K exceeds the number of semantic groups")
    folds = _validate_folds(inner_folds, matrix.shape[0])
    cluster_ids: tuple[str, ...] = ()
    blocks: tuple[tuple[int, ...], ...] = ()
    penalty = "elastic_net"
    penalty_groups: tuple[tuple[str, str], ...] = ()
    if group_membership is not None:
        cluster_ids, blocks = _group_blocks(groups, group_membership)
        membership_map = dict(tuple(group_membership))
        penalty_groups = tuple((group, membership_map[group]) for group in groups)
        penalty = "sparse_group_elastic_net"

    def descend(design, centered, lambda_value, l1_ratio, warm):
        if blocks:
            return _group_proximal_descent(
                design, centered, blocks, lambda_value, l1_ratio,
                run_plan.max_iter, run_plan.tolerance, warm,
            )
        return _coordinate_descent(
            design, centered, lambda_value, l1_ratio,
            run_plan.max_iter, run_plan.tolerance, warm,
        )

    row_digest = compute_train_row_digest(matrix, target)
    if row_digest != expected_row_digest:
        raise ValueError("TRAIN row digest mismatch")
    group_digest = _canonical_digest({"ordered_semantic_groups": groups})
    fold_payload = tuple(
        {
            "fold_id": fold.fold_id,
            "train": (fold.train_start, fold.train_end),
            "validation": (fold.validation_start, fold.validation_end),
        }
        for fold in folds
    )
    fold_digest = _canonical_digest(fold_payload)
    hyperparameter_payload = {
        "l1_ratio_path": L1_RATIO_PATH,
        "lambda_path": run_plan.lambda_path,
    }
    hyperparameter_digest = _canonical_digest(hyperparameter_payload)
    config_payload = {
        "algorithm": (
            "deterministic_block_proximal_gradient"
            if blocks
            else "deterministic_coordinate_descent"
        ),
        "penalty": penalty,
        "penalty_groups": penalty_groups,
        "warm_start": "descending_lambda_path",
        "version": _VERSION,
        "input_label": "TRAIN",
        "subset_k": run_plan.subset_k,
        "sealed_ks": run_plan.sealed_ks,
        "max_iter": run_plan.max_iter,
        "tolerance": run_plan.tolerance,
        "hyperparameter_digest": hyperparameter_digest,
        "fold_digest": fold_digest,
    }
    config_digest = _canonical_digest(config_payload)

    diagnostics: list[FitDiagnostic] = []
    candidates: list[tuple[float, float, float]] = []
    fold_cache = []
    for fold in folds:
        train_x = matrix[fold.train_start : fold.train_end]
        train_y = target[fold.train_start : fold.train_end]
        means, scales = _standardizer(train_x)
        fold_cache.append(
            (
                fold,
                (train_x - means) / scales,
                train_y - float(np.mean(train_y)),
                float(np.mean(train_y)),
                means,
                scales,
                matrix[fold.validation_start : fold.validation_end],
                target[fold.validation_start : fold.validation_end],
            )
        )
    for l1_ratio in L1_RATIO_PATH:
        # One warm-start chain per fold along the descending lambda path.
        fold_maes_by_lambda: dict[float, list[float]] = {value: [] for value in run_plan.lambda_path}
        converged_by_lambda: dict[float, bool] = {value: True for value in run_plan.lambda_path}
        for fold, std_train, centered_y, train_mean, means, scales, val_x, val_y in fold_cache:
            warm = None
            for lambda_value in run_plan.lambda_path:
                fit = descend(std_train, centered_y, lambda_value, l1_ratio, warm)
                warm = fit.coefficients
                validation_mae: float | None = None
                if fit.converged:
                    predictions = train_mean + ((val_x - means) / scales) @ fit.coefficients
                    validation_mae = float(np.mean(np.abs(val_y - predictions)))
                    fold_maes_by_lambda[lambda_value].append(validation_mae)
                else:
                    converged_by_lambda[lambda_value] = False
                diagnostics.append(
                    FitDiagnostic(
                        fold_id=fold.fold_id,
                        l1_ratio=l1_ratio,
                        lambda_value=lambda_value,
                        iterations=fit.iterations,
                        converged=fit.converged,
                        objective=fit.objective,
                        validation_mae=validation_mae,
                    )
                )
        for lambda_value in run_plan.lambda_path:
            if converged_by_lambda[lambda_value]:
                candidates.append(
                    (float(np.mean(fold_maes_by_lambda[lambda_value])), l1_ratio, lambda_value)
                )

    if not candidates:
        raise RuntimeError("no hyperparameter candidate converged on every inner fold")
    selected_mean_mae, selected_l1_ratio, selected_lambda = min(
        candidates,
        key=lambda item: (item[0], item[1], -item[2]),
    )

    means, scales = _standardizer(matrix)
    standardized_matrix = (matrix - means) / scales
    target_mean = float(np.mean(target))
    final_fit = descend(
        standardized_matrix, target - target_mean, selected_lambda, selected_l1_ratio, None
    )
    if not final_fit.converged:
        raise RuntimeError("unconverged elastic-net finalist is forbidden")

    coefficient_by_group = {
        group: float(final_fit.coefficients[index])
        for index, group in enumerate(groups)
    }
    if blocks:
        cluster_norm: dict[str, float] = {}
        for cluster, block in zip(cluster_ids, blocks):
            cluster_norm[cluster] = float(
                np.sqrt(sum(coefficient_by_group[groups[index]] ** 2 for index in block))
            )
        membership_map = dict(penalty_groups)
        ranking = tuple(
            sorted(
                groups,
                key=lambda group: (
                    -cluster_norm[membership_map[group]],
                    membership_map[group],
                    -abs(coefficient_by_group[group]),
                    group,
                ),
            )
        )
    else:
        ranking = tuple(
            sorted(groups, key=lambda group: (-abs(coefficient_by_group[group]), group))
        )
    selected_by_k, k_failures = truncate_sealed(ranking, run_plan.comparator_view())
    ordered_coefficients = tuple(
        (group, coefficient_by_group[group]) for group in ranking
    )
    coefficient_norms = tuple(
        (group, abs(coefficient_by_group[group])) for group in ranking
    )
    return ElasticNetSelectorResult(
        selected_groups=ranking[: run_plan.subset_k],
        full_ranking=ranking,
        standardized_coefficients=ordered_coefficients,
        coefficient_norms=coefficient_norms,
        selected_l1_ratio=selected_l1_ratio,
        selected_lambda=selected_lambda,
        selected_mean_inner_validation_mae=selected_mean_mae,
        sealed_subset_k=run_plan.subset_k,
        row_digest=row_digest,
        group_digest=group_digest,
        fold_digest=fold_digest,
        hyperparameter_digest=hyperparameter_digest,
        config_digest=config_digest,
        diagnostics=tuple(diagnostics),
        final_iterations=final_fit.iterations,
        final_objective=final_fit.objective,
        target_mean=target_mean,
        train_means=tuple(float(value) for value in means),
        train_scales=tuple(float(value) for value in scales),
        selected_by_k=selected_by_k,
        k_failures=k_failures,
        penalty=penalty,
        penalty_groups=penalty_groups,
    )


__all__ = [
    "ChronologicalInnerFold",
    "ElasticNetRunPlan",
    "ElasticNetSelectorResult",
    "FitDiagnostic",
    "L1_RATIO_PATH",
    "compute_train_row_digest",
    "run_elastic_net_selector",
]
