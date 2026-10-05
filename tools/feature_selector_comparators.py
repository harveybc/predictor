"""Deterministic TRAIN-only controls for semantic-group feature selection.

The module deliberately owns ranking and truncation together.  A caller seals
``K`` in :class:`ComparatorRunPlan`; a result cannot later be truncated under a
different budget.  No method accepts validation, test, or outer-fold values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ComparatorMethod(str, Enum):
    """Supported deterministic comparator families."""

    ALL_ADMISSIBLE = "ALL_ADMISSIBLE"
    SPEARMAN_K = "SPEARMAN_K"
    MI8_K = "MI8_K"
    REDUNDANCY_K = "REDUNDANCY_K"
    RANDOM_K = "RANDOM_K"
    JMI_K = "JMI_K"
    CMIM_K = "CMIM_K"
    MRMR_K = "MRMR_K"


_GREEDY_MI_METHODS = frozenset(
    {"JMI_K", "CMIM_K", "MRMR_K"}
)


@dataclass(frozen=True)
class RandomTape:
    """Named random permutation seed declared before selection."""

    tape_id: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.tape_id, str) or not self.tape_id.strip():
            raise ValueError("random tape id must be a non-empty string")
        if isinstance(self.seed, (bool, np.bool_)):
            raise TypeError("boolean random seed is forbidden")
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError("random seed must be an integer")
        if int(self.seed) < 0:
            raise ValueError("random seed must be non-negative")


def validate_sealed_ks(subset_k: Any, sensitivity_ks: Any) -> tuple[int, tuple[int, ...]]:
    """Validate the primary K and the sealed sensitivity budgets declared with it.

    Sensitivity budgets are part of the plan identity: they are declared before
    any ranking exists, so truncating one complete ranking at each of them is
    not a post-hoc change.  Any K absent from this tuple stays forbidden.
    """

    if isinstance(subset_k, (bool, np.bool_)):
        raise TypeError("boolean subset K is forbidden")
    if not isinstance(subset_k, (int, np.integer)):
        raise TypeError("subset K must be an integer")
    if int(subset_k) <= 0:
        raise ValueError("subset K must be positive")
    ks: list[int] = []
    for value in tuple(sensitivity_ks):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("boolean sensitivity K is forbidden")
        if not isinstance(value, (int, np.integer)):
            raise TypeError("sensitivity K must be an integer")
        if int(value) <= 0:
            raise ValueError("sensitivity K must be positive")
        ks.append(int(value))
    if len(set(ks)) != len(ks):
        raise ValueError("duplicate sensitivity K")
    return int(subset_k), tuple(sorted(ks))


@dataclass(frozen=True)
class ComparatorRunPlan:
    """Selection budget, sealed sensitivity budgets and predeclared random tapes."""

    subset_k: int
    random_tapes: tuple[RandomTape, ...] = ()
    sensitivity_ks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        primary, sensitivities = validate_sealed_ks(self.subset_k, self.sensitivity_ks)
        object.__setattr__(self, "sensitivity_ks", sensitivities)
        tapes = tuple(self.random_tapes)
        if len(tapes) > 3:
            raise ValueError("a run plan permits at most three random tapes")
        ids = [tape.tape_id for tape in tapes]
        seeds = [int(tape.seed) for tape in tapes]
        if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
            raise ValueError("duplicate random tape id or seed")
        object.__setattr__(self, "subset_k", primary)
        object.__setattr__(self, "random_tapes", tapes)

    @property
    def sealed_ks(self) -> tuple[int, ...]:
        """Every budget a complete ranking may be truncated at, primary included."""

        return tuple(sorted(set(self.sensitivity_ks) | {self.subset_k}))


class FrozenDetails(Mapping[str, Any]):
    """Small recursively immutable mapping used in public results."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        entries = values or {}
        self._items = tuple(
            (str(key), _freeze(value)) for key, value in sorted(entries.items())
        )

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return f"FrozenDetails({dict(self.items())!r})"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDetails(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ComparatorResult:
    """Authenticated ranking and selection produced under one sealed budget."""

    method: ComparatorMethod
    row_digest: str
    group_digest: str
    method_config_digest: str
    sealed_subset_k: int
    selected_groups: tuple[str, ...]
    full_ranking: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    size_exempt: bool
    details: FrozenDetails
    selected_by_k: tuple[tuple[int, tuple[str, ...]], ...] = ()
    k_failures: tuple[tuple[int, str], ...] = ()

    def selected_for(self, subset_k: int) -> tuple[str, ...]:
        """Return a selection made under a sealed K; reject any other budget."""

        if isinstance(subset_k, (bool, np.bool_)):
            raise TypeError("boolean subset K is forbidden")
        if subset_k == self.sealed_subset_k:
            return self.selected_groups
        for sealed_k, selection in self.selected_by_k:
            if sealed_k == subset_k:
                return selection
        raise ValueError("post-hoc truncation is forbidden")


def truncate_sealed(
    ranking: tuple[str, ...], run_plan: ComparatorRunPlan, *, size_exempt: bool = False
) -> tuple[tuple[tuple[int, tuple[str, ...]], ...], tuple[tuple[int, str], ...]]:
    """Truncate one complete ranking at every sealed K, retaining per-K failures.

    A budget larger than the ranking is a retained failure for that K alone; it
    never erases the complete ranking or the other sealed budgets.
    """

    selections: list[tuple[int, tuple[str, ...]]] = []
    failures: list[tuple[int, str]] = []
    for sealed_k in run_plan.sealed_ks:
        if size_exempt:
            selections.append((sealed_k, tuple(ranking)))
        elif sealed_k > len(ranking):
            failures.append((sealed_k, "sealed K exceeds the number of semantic groups"))
        else:
            selections.append((sealed_k, tuple(ranking[:sealed_k])))
    return tuple(selections), tuple(failures)


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_matrix(values: Any, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise TypeError(f"boolean {name} values are forbidden")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain numeric values")
    array = np.asarray(raw, dtype="<f8", order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if array.size == 0 or any(dimension == 0 for dimension in array.shape):
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_arrays(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = _numeric_matrix(x, ndim=2, name="X")
    target = _numeric_matrix(y, ndim=1, name="y")
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y row counts differ")
    return matrix, target


def compute_row_digest(x: Any, y: Any) -> str:
    """Hash the exact finite TRAIN population in canonical float64 form."""

    matrix, target = _validated_arrays(x, y)
    digest = hashlib.sha256()
    digest.update(b"feature-selector-rows-v1\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.tobytes(order="C"))
    digest.update(np.asarray(target.shape, dtype="<i8").tobytes())
    digest.update(target.tobytes(order="C"))
    return digest.hexdigest()


def _validate_groups(group_ids: Any, column_count: int) -> tuple[str, ...]:
    groups = tuple(group_ids)
    if len(groups) != column_count:
        raise ValueError("one semantic group id is required per X column")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("semantic group ids must be non-empty strings")
    if len(set(groups)) != len(groups):
        raise ValueError("duplicate semantic groups are forbidden")
    return groups


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _rank_from_scores(
    groups: tuple[str, ...], scores: Mapping[str, float]
) -> tuple[str, ...]:
    return tuple(sorted(groups, key=lambda group: (-float(scores[group]), group)))


def _quantile_edges(values: np.ndarray) -> tuple[float, ...]:
    # Nine learned boundaries define exactly eight labeled bins; tied boundaries
    # are retained rather than silently changing the method's bin count.
    edges = np.quantile(values, np.linspace(0.0, 1.0, 9), method="linear")
    return tuple(float(edge) for edge in edges)


def _digitize_eight(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges[1:-1]), values, side="right")


def _mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    joint = np.zeros((8, 8), dtype=np.float64)
    np.add.at(joint, (left, right), 1.0)
    joint /= float(left.size)
    left_prob = np.sum(joint, axis=1)
    right_prob = np.sum(joint, axis=0)
    score = 0.0
    for left_bin in range(8):
        for right_bin in range(8):
            probability = joint[left_bin, right_bin]
            if probability > 0.0:
                score += probability * np.log(
                    probability / (left_prob[left_bin] * right_prob[right_bin])
                )
    return float(score)


def _redundancy_components(
    matrix: np.ndarray, groups: tuple[str, ...], threshold: float
) -> tuple[tuple[str, ...], ...]:
    parents = list(range(len(groups)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if abs(_spearman(matrix[:, left], matrix[:, right])) >= threshold:
                union(left, right)

    components: dict[int, list[str]] = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group)
    return tuple(
        sorted((tuple(sorted(component)) for component in components.values()))
    )


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def _pair_terms(
    code_i: np.ndarray, code_j: np.ndarray, code_y: np.ndarray, bins: int
) -> tuple[float, float, float]:
    """Return I(Xi,Xj;Y), I(Xi;Y|Xj) and I(Xi;Xj) from one 3-way count table."""

    joint = np.bincount(
        (code_i * bins + code_j) * bins + code_y, minlength=bins * bins * bins
    ).reshape(bins, bins, bins).astype(np.float64)
    h_ijy = _entropy_from_counts(joint)
    h_ij = _entropy_from_counts(joint.sum(axis=2))
    h_jy = _entropy_from_counts(joint.sum(axis=0))
    h_y = _entropy_from_counts(joint.sum(axis=(0, 1)))
    h_i = _entropy_from_counts(joint.sum(axis=(1, 2)))
    h_j = _entropy_from_counts(joint.sum(axis=(0, 2)))
    joint_mi = h_ij + h_y - h_ijy  # I(Xi,Xj;Y)
    conditional_mi = h_ij + h_jy - h_j - h_ijy  # I(Xi;Y|Xj)
    pair_mi = h_i + h_j - h_ij  # I(Xi;Xj)
    return joint_mi, conditional_mi, pair_mi


def _greedy_information_ranking(
    method: ComparatorMethod,
    codes: np.ndarray,
    target_codes: np.ndarray,
    groups: tuple[str, ...],
    bins: int,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Greedy forward ordering of every group under JMI, CMIM or mRMR.

    The greedy criterion is applied until every group is ordered, so the result
    is one complete ranking whose top-K prefix is the method's K-set for any
    sealed K.  Ties are broken by group id so replays are identical.
    """

    count = len(groups)
    relevance = np.asarray(
        [
            _mutual_information(codes[:, index], target_codes)
            for index in range(count)
        ],
        dtype=np.float64,
    )
    remaining = list(range(count))
    order: list[int] = []
    criterion: dict[str, float] = {}
    if method is ComparatorMethod.JMI_K:
        accumulator = np.zeros(count, dtype=np.float64)
    elif method is ComparatorMethod.CMIM_K:
        accumulator = np.full(count, np.inf, dtype=np.float64)
    else:
        accumulator = np.zeros(count, dtype=np.float64)

    while remaining:
        if not order:
            scores = relevance[remaining]
        elif method is ComparatorMethod.JMI_K:
            scores = accumulator[remaining]
        elif method is ComparatorMethod.CMIM_K:
            scores = np.minimum(accumulator[remaining], relevance[remaining])
        else:
            scores = relevance[remaining] - accumulator[remaining] / float(len(order))
        best = min(
            range(len(remaining)),
            key=lambda position: (-float(scores[position]), groups[remaining[position]]),
        )
        chosen = remaining.pop(best)
        order.append(chosen)
        criterion[groups[chosen]] = float(scores[best])
        if not remaining:
            break
        chosen_codes = codes[:, chosen]
        for index in remaining:
            joint_mi, conditional_mi, pair_mi = _pair_terms(
                codes[:, index], chosen_codes, target_codes, bins
            )
            if method is ComparatorMethod.JMI_K:
                accumulator[index] += joint_mi
            elif method is ComparatorMethod.CMIM_K:
                accumulator[index] = min(accumulator[index], conditional_mi)
            else:
                accumulator[index] += pair_mi
    ranking = tuple(groups[index] for index in order)
    return ranking, criterion


def run_comparator(
    method: ComparatorMethod | str,
    x: Any,
    y: Any,
    group_ids: Any,
    run_plan: ComparatorRunPlan,
    *,
    input_label: str = "TRAIN",
    expected_row_digest: str | None = None,
    inner_fold_gains: Mapping[str, float] | None = None,
    random_tape_id: str | None = None,
) -> ComparatorResult:
    """Run one comparator over authenticated TRAIN rows.

    ``inner_fold_gains`` is mandatory only for ``REDUNDANCY_K`` and must name
    every semantic group.  It is caller evidence from inner TRAIN folds; this
    core intentionally has no interface through which outer values can enter.
    """

    if not isinstance(run_plan, ComparatorRunPlan):
        raise TypeError("run_plan must be ComparatorRunPlan")
    try:
        selected_method = ComparatorMethod(method)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported comparator method: {method!r}") from error
    if input_label != "TRAIN":
        raise ValueError("feature comparators are TRAIN-only")

    matrix, target = _validated_arrays(x, y)
    groups = _validate_groups(group_ids, matrix.shape[1])
    if run_plan.subset_k > len(groups):
        raise ValueError("subset K exceeds the number of semantic groups")
    row_digest = compute_row_digest(matrix, target)
    if expected_row_digest is not None:
        if not isinstance(expected_row_digest, str):
            raise TypeError("expected row digest must be a string")
        if expected_row_digest != row_digest:
            raise ValueError("row digest mismatch")

    group_digest = _canonical_digest({"ordered_semantic_groups": groups})
    scores: dict[str, float]
    details: dict[str, Any] = {}
    config: dict[str, Any] = {
        "method": selected_method.value,
        "sealed_subset_k": run_plan.subset_k,
        "input_label": "TRAIN",
        "version": 1,
    }

    if selected_method is ComparatorMethod.ALL_ADMISSIBLE:
        scores = {group: 0.0 for group in groups}
        ranking = tuple(sorted(groups))
        selected = ranking
        size_exempt = True
    elif selected_method is ComparatorMethod.SPEARMAN_K:
        scores = {
            group: abs(_spearman(matrix[:, index], target))
            for index, group in enumerate(groups)
        }
        ranking = _rank_from_scores(groups, scores)
        selected = ranking[: run_plan.subset_k]
        size_exempt = False
        config["rank"] = "average_ties_absolute_spearman"
    elif selected_method is ComparatorMethod.MI8_K:
        target_edges = _quantile_edges(target)
        target_bins = _digitize_eight(target, target_edges)
        feature_edges: dict[str, tuple[float, ...]] = {}
        scores = {}
        for index, group in enumerate(groups):
            edges = _quantile_edges(matrix[:, index])
            feature_edges[group] = edges
            scores[group] = _mutual_information(
                _digitize_eight(matrix[:, index], edges), target_bins
            )
        ranking = _rank_from_scores(groups, scores)
        selected = ranking[: run_plan.subset_k]
        size_exempt = False
        details.update(target_edges=target_edges, feature_edges=feature_edges)
        config.update(bin_count=8, edge_fit_partition="TRAIN", edge_rule="quantile")
    elif selected_method.value in _GREEDY_MI_METHODS:
        if inner_fold_gains is not None:
            raise ValueError("inner-fold gains are only valid for REDUNDANCY_K")
        bins = 8
        target_edges = _quantile_edges(target)
        target_bins = _digitize_eight(target, target_edges)
        codes = np.empty(matrix.shape, dtype=np.int64)
        feature_edges = {}
        for index, group in enumerate(groups):
            edges = _quantile_edges(matrix[:, index])
            feature_edges[group] = edges
            codes[:, index] = _digitize_eight(matrix[:, index], edges)
        ranking, scores = _greedy_information_ranking(
            selected_method, codes, target_bins, groups, bins
        )
        selected = ranking[: run_plan.subset_k]
        size_exempt = False
        details.update(target_edges=target_edges, feature_edges=feature_edges)
        config.update(
            bin_count=bins,
            edge_fit_partition="TRAIN",
            edge_rule="quantile",
            criterion={
                "JMI_K": "sum_j I(Xi,Xj;Y)",
                "CMIM_K": "min(I(Xi;Y), min_j I(Xi;Y|Xj))",
                "MRMR_K": "I(Xi;Y) - mean_j I(Xi;Xj)",
            }[selected_method.value],
            greedy_scope="complete_ordering",
        )
    elif selected_method is ComparatorMethod.REDUNDANCY_K:
        if inner_fold_gains is None:
            raise ValueError("REDUNDANCY_K requires supplied inner-fold gains")
        if set(inner_fold_gains) != set(groups):
            raise ValueError("inner-fold gains must name every semantic group exactly")
        scores = {}
        for group in groups:
            gain = inner_fold_gains[group]
            if isinstance(gain, (bool, np.bool_)):
                raise TypeError("boolean inner-fold gain is forbidden")
            if not isinstance(gain, (int, float, np.integer, np.floating)):
                raise TypeError("inner-fold gains must be numeric")
            if not np.isfinite(float(gain)):
                raise ValueError("inner-fold gains must be finite")
            scores[group] = float(gain)
        threshold = 0.90
        components = _redundancy_components(matrix, groups, threshold)
        representatives = tuple(
            sorted(
                (
                    sorted(component, key=lambda group: (-scores[group], group))[0]
                    for component in components
                ),
                key=lambda group: (-scores[group], group),
            )
        )
        non_representatives = tuple(
            sorted(
                (group for group in groups if group not in representatives),
                key=lambda group: (-scores[group], group),
            )
        )
        ranking = representatives + non_representatives
        selected = ranking[: run_plan.subset_k]
        size_exempt = False
        details.update(components=components, representatives=representatives)
        config.update(
            absolute_spearman_threshold=threshold,
            ranking_evidence="supplied_inner_fold_gains",
            gains=tuple(sorted(scores.items())),
        )
    else:
        if inner_fold_gains is not None:
            raise ValueError("inner-fold gains are only valid for REDUNDANCY_K")
        tapes = {tape.tape_id: tape for tape in run_plan.random_tapes}
        if random_tape_id is None or random_tape_id not in tapes:
            raise ValueError("RANDOM_K requires a declared random tape")
        tape = tapes[random_tape_id]
        order = np.random.default_rng(tape.seed).permutation(len(groups))
        ranking = tuple(groups[int(index)] for index in order)
        scores = {
            group: float(len(groups) - position)
            for position, group in enumerate(ranking)
        }
        selected = ranking[: run_plan.subset_k]
        size_exempt = False
        details.update(random_tape_id=tape.tape_id, random_seed=int(tape.seed))
        config.update(random_tape_id=tape.tape_id, random_seed=int(tape.seed))

    ordered_scores = tuple((group, float(scores[group])) for group in ranking)
    config["sealed_ks"] = run_plan.sealed_ks
    selected_by_k, k_failures = truncate_sealed(
        tuple(ranking), run_plan, size_exempt=size_exempt
    )
    return ComparatorResult(
        method=selected_method,
        row_digest=row_digest,
        group_digest=group_digest,
        method_config_digest=_canonical_digest(config),
        sealed_subset_k=run_plan.subset_k,
        selected_groups=tuple(selected),
        full_ranking=tuple(ranking),
        scores=ordered_scores,
        size_exempt=size_exempt,
        details=FrozenDetails(details),
        selected_by_k=selected_by_k,
        k_failures=k_failures,
    )


__all__ = [
    "ComparatorMethod",
    "ComparatorResult",
    "ComparatorRunPlan",
    "RandomTape",
    "compute_row_digest",
    "run_comparator",
    "truncate_sealed",
    "validate_sealed_ks",
]
