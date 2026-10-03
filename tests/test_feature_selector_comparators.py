from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tools.feature_selector_comparators import (
    ComparatorMethod,
    ComparatorRunPlan,
    RandomTape,
    compute_row_digest,
    run_comparator,
)


def sample() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x = np.asarray(
        [
            [0.0, 5.0, 0.0, 8.0],
            [1.0, 4.0, 0.0, 7.0],
            [2.0, 3.0, 1.0, 6.0],
            [3.0, 2.0, 1.0, 5.0],
            [4.0, 1.0, 2.0, 4.0],
            [5.0, 0.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0.0, 1.0, 1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    return x, y, ("alpha", "beta", "gamma", "zeta")


def test_spearman_is_complete_stable_and_sealed_to_same_rows_and_k() -> None:
    x, y, groups = sample()
    plan = ComparatorRunPlan(subset_k=2)
    expected = compute_row_digest(x, y)

    first = run_comparator(
        ComparatorMethod.SPEARMAN_K,
        x,
        y,
        groups,
        plan,
        expected_row_digest=expected,
    )
    second = run_comparator(
        ComparatorMethod.SPEARMAN_K,
        x.copy(),
        y.copy(),
        groups,
        plan,
        expected_row_digest=expected,
    )

    assert first == second
    assert len(first.full_ranking) == len(groups)
    assert first.selected_groups == first.full_ranking[:2]
    assert first.row_digest == expected
    assert first.sealed_subset_k == 2
    with pytest.raises(ValueError, match="post-hoc"):
        first.selected_for(1)
    with pytest.raises(FrozenInstanceError):
        first.sealed_subset_k = 1  # type: ignore[misc]


def test_average_rank_ties_use_lexical_tie_break() -> None:
    y = np.arange(8, dtype=np.float64)
    x = np.column_stack((y, y, np.ones_like(y)))
    result = run_comparator(
        ComparatorMethod.SPEARMAN_K,
        x,
        y,
        ("beta", "alpha", "constant"),
        ComparatorRunPlan(subset_k=2),
    )

    assert result.full_ranking == ("alpha", "beta", "constant")
    assert result.scores[0] == ("alpha", 1.0)
    assert result.scores[1] == ("beta", 1.0)


def test_spearman_is_invariant_to_strict_monotone_transforms() -> None:
    x, y, groups = sample()
    plan = ComparatorRunPlan(subset_k=3)
    base = run_comparator(ComparatorMethod.SPEARMAN_K, x, y, groups, plan)
    transformed = run_comparator(
        ComparatorMethod.SPEARMAN_K,
        np.exp(x / 10.0),
        np.power(y + 1.0, 3.0),
        groups,
        plan,
    )

    assert base.full_ranking == transformed.full_ranking
    assert np.allclose(
        [score for _, score in base.scores],
        [score for _, score in transformed.scores],
    )


def test_mi8_uses_exactly_train_learned_edges_and_is_repeatable() -> None:
    x, y, groups = sample()
    result = run_comparator(
        ComparatorMethod.MI8_K,
        x,
        y,
        groups,
        ComparatorRunPlan(subset_k=2),
    )

    assert len(result.details["target_edges"]) == 9
    assert set(result.details["feature_edges"]) == set(groups)
    assert all(len(edges) == 9 for edges in result.details["feature_edges"].values())
    assert result == run_comparator(
        ComparatorMethod.MI8_K,
        x,
        y,
        groups,
        ComparatorRunPlan(subset_k=2),
    )

    changed_outer = np.vstack((x, np.full((1, x.shape[1]), 1e12)))
    with pytest.raises(ValueError, match="row digest"):
        run_comparator(
            ComparatorMethod.MI8_K,
            changed_outer,
            np.append(y, 1e12),
            groups,
            ComparatorRunPlan(subset_k=2),
            expected_row_digest=result.row_digest,
        )


def test_redundancy_selects_one_best_gain_representative_per_component() -> None:
    base = np.arange(10, dtype=np.float64)
    x = np.column_stack((base, base * 2.0 + 7.0, base[::-1], [0, 1] * 5))
    groups = ("a", "b", "c", "d")
    gains = {"a": 0.2, "b": 0.7, "c": 0.5, "d": 0.1}
    result = run_comparator(
        ComparatorMethod.REDUNDANCY_K,
        x,
        base,
        groups,
        ComparatorRunPlan(subset_k=2),
        inner_fold_gains=gains,
    )

    # a, b and c are one absolute-Spearman component; b has its best inner gain.
    assert result.details["components"][0] == ("a", "b", "c")
    assert result.details["representatives"][0] == "b"
    assert result.selected_groups == ("b", "d")
    assert set(result.full_ranking) == set(groups)


def test_random_requires_declared_tape_and_enforces_max_three() -> None:
    x, y, groups = sample()
    tapes = (RandomTape("r0", 7), RandomTape("r1", 11), RandomTape("r2", 13))
    plan = ComparatorRunPlan(subset_k=2, random_tapes=tapes)
    first = run_comparator(
        ComparatorMethod.RANDOM_K, x, y, groups, plan, random_tape_id="r1"
    )
    second = run_comparator(
        ComparatorMethod.RANDOM_K, x, y, groups, plan, random_tape_id="r1"
    )

    assert first == second
    assert first.details["random_tape_id"] == "r1"
    assert first.details["random_seed"] == 11
    with pytest.raises(ValueError, match="declared random tape"):
        run_comparator(
            ComparatorMethod.RANDOM_K,
            x,
            y,
            groups,
            plan,
            random_tape_id="missing",
        )
    with pytest.raises(ValueError, match="at most three"):
        ComparatorRunPlan(
            subset_k=1,
            random_tapes=tuple(RandomTape(f"r{i}", i) for i in range(4)),
        )


def test_all_admissible_is_the_only_size_exempt_method() -> None:
    x, y, groups = sample()
    result = run_comparator(
        ComparatorMethod.ALL_ADMISSIBLE,
        x,
        y,
        groups,
        ComparatorRunPlan(subset_k=2),
    )
    assert result.selected_groups == tuple(sorted(groups))
    assert result.size_exempt is True
    assert result.sealed_subset_k == 2


@pytest.mark.parametrize(
    ("x", "y", "groups", "plan", "message"),
    [
        (np.asarray([[True, False]]), np.asarray([1.0]), ("a", "b"), ComparatorRunPlan(1), "boolean"),
        (np.asarray([[1.0, np.nan]]), np.asarray([1.0]), ("a", "b"), ComparatorRunPlan(1), "finite"),
        (np.asarray([[1.0], [2.0]]), np.asarray([1.0, np.inf]), ("a",), ComparatorRunPlan(1), "finite"),
        (np.asarray([[1.0, 2.0]]), np.asarray([1.0]), ("a", "a"), ComparatorRunPlan(1), "duplicate"),
        (np.asarray([[1.0]]), np.asarray([1.0]), ("a",), ComparatorRunPlan(2), "exceeds"),
    ],
)
def test_invalid_inputs_fail_closed(
    x: np.ndarray,
    y: np.ndarray,
    groups: tuple[str, ...],
    plan: ComparatorRunPlan,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        run_comparator(ComparatorMethod.SPEARMAN_K, x, y, groups, plan)


def test_boolean_plan_values_and_duplicate_tapes_are_rejected() -> None:
    with pytest.raises(TypeError, match="boolean"):
        ComparatorRunPlan(subset_k=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        RandomTape("bad", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate random tape"):
        ComparatorRunPlan(
            subset_k=1,
            random_tapes=(RandomTape("same", 1), RandomTape("same", 2)),
        )


@pytest.mark.parametrize("label", ["TEST", "OUTER_TEST", "VALIDATION", "LIVE"])
def test_outer_or_test_labelled_inputs_are_never_accessed(label: str) -> None:
    x, y, groups = sample()
    with pytest.raises(ValueError, match="TRAIN-only"):
        run_comparator(
            ComparatorMethod.SPEARMAN_K,
            x,
            y,
            groups,
            ComparatorRunPlan(2),
            input_label=label,
        )


def test_group_and_method_config_digests_change_with_identity() -> None:
    x, y, groups = sample()
    plan = ComparatorRunPlan(2)
    spearman = run_comparator(ComparatorMethod.SPEARMAN_K, x, y, groups, plan)
    mi = run_comparator(ComparatorMethod.MI8_K, x, y, groups, plan)
    reordered = run_comparator(
        ComparatorMethod.SPEARMAN_K,
        x[:, ::-1],
        y,
        tuple(reversed(groups)),
        plan,
    )

    assert spearman.method_config_digest != mi.method_config_digest
    assert spearman.group_digest != reordered.group_digest

