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

    def __post_init__(self) -> None:
        object.__setattr__(self, "subset_k", _positive_int(self.subset_k, "subset K"))
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

    def selected_for(self, subset_k: int) -> tuple[str, ...]:
        """Return only the selection made under the originally sealed K."""

        if _is_bool(subset_k):
            raise TypeError("boolean subset K is forbidden")
        if subset_k != self.sealed_subset_k:
            raise ValueError("post-hoc K changes are forbidden")
        return self.selected_groups


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
) -> _Fit:
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    prediction = np.zeros(matrix.shape[0], dtype=np.float64)
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


def run_elastic_net_selector(
    x: Any,
    y: Any,
    group_ids: Any,
    inner_folds: Any,
    run_plan: ElasticNetRunPlan,
    *,
    expected_row_digest: str,
    input_label: str = "TRAIN",
) -> ElasticNetSelectorResult:
    """Select scalar semantic groups using inner-fold MAE only."""

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
        "algorithm": "deterministic_coordinate_descent",
        "version": _VERSION,
        "input_label": "TRAIN",
        "subset_k": run_plan.subset_k,
        "max_iter": run_plan.max_iter,
        "tolerance": run_plan.tolerance,
        "hyperparameter_digest": hyperparameter_digest,
        "fold_digest": fold_digest,
    }
    config_digest = _canonical_digest(config_payload)

    diagnostics: list[FitDiagnostic] = []
    candidates: list[tuple[float, float, float]] = []
    for l1_ratio in L1_RATIO_PATH:
        for lambda_value in run_plan.lambda_path:
            fold_maes: list[float] = []
            candidate_converged = True
            for fold in folds:
                train_x = matrix[fold.train_start : fold.train_end]
                train_y = target[fold.train_start : fold.train_end]
                validation_x = matrix[fold.validation_start : fold.validation_end]
                validation_y = target[fold.validation_start : fold.validation_end]
                means, scales = _standardizer(train_x)
                standardized_train = (train_x - means) / scales
                centered_train_y = train_y - float(np.mean(train_y))
                fit = _coordinate_descent(
                    standardized_train,
                    centered_train_y,
                    lambda_value,
                    l1_ratio,
                    run_plan.max_iter,
                    run_plan.tolerance,
                )
                validation_mae: float | None = None
                if fit.converged:
                    predictions = (
                        float(np.mean(train_y))
                        + ((validation_x - means) / scales) @ fit.coefficients
                    )
                    validation_mae = float(np.mean(np.abs(validation_y - predictions)))
                    fold_maes.append(validation_mae)
                else:
                    candidate_converged = False
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
            if candidate_converged:
                candidates.append(
                    (float(np.mean(fold_maes)), l1_ratio, lambda_value)
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
    final_fit = _coordinate_descent(
        standardized_matrix,
        target - target_mean,
        selected_lambda,
        selected_l1_ratio,
        run_plan.max_iter,
        run_plan.tolerance,
    )
    if not final_fit.converged:
        raise RuntimeError("unconverged elastic-net finalist is forbidden")

    coefficient_by_group = {
        group: float(final_fit.coefficients[index])
        for index, group in enumerate(groups)
    }
    ranking = tuple(
        sorted(groups, key=lambda group: (-abs(coefficient_by_group[group]), group))
    )
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
