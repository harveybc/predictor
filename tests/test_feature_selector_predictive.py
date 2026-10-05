from __future__ import annotations

import numpy as np
import pytest

from tools.feature_selector_comparators import ComparatorRunPlan
from tools.feature_selector_elastic_net import ChronologicalInnerFold
from tools.feature_selector_predictive import (
    CHRONOEPILOGI_ADAPTER_UNREPRODUCED,
    ChronoEpilogiPlan,
    ExtraTreesPlan,
    GroupGatePlan,
    SequentialMarginalPlan,
    compute_predictive_row_digest,
    fit_and_score_extratrees,
    run_chronoepilogi_selector,
    run_extratrees_selector,
    run_group_gate_selector,
    run_sequential_marginal_selector,
    temporal_block_permutation,
)


GROUPS = ("signal_a", "noise_a", "dup_a", "signal_b", "noise_b", "noise_c")


def sample(rows: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 6))
    x[:, 2] = x[:, 0] + 0.01 * rng.normal(size=rows)  # near duplicate of signal_a
    y = 3.0 * x[:, 0] - 2.0 * x[:, 3] + 0.3 * rng.normal(size=rows)
    return x, y


def split(rows: int = 400) -> ChronologicalInnerFold:
    return ChronologicalInnerFold("nested", 0, 300, 310, rows)


MEMBERSHIP = tuple(
    (group, {"signal_a": "c_sig", "dup_a": "c_sig"}.get(group, group)) for group in GROUPS
)


def _check_common(result, k: int, ks: tuple[int, ...]) -> None:
    assert len(result.full_ranking) == len(GROUPS)
    assert set(result.full_ranking) == set(GROUPS)
    assert result.selected_groups == result.full_ranking[:k]
    assert result.selected_for(k) == result.selected_groups
    for sealed in ks:
        assert result.selected_for(sealed) == result.full_ranking[:sealed]
    with pytest.raises(ValueError, match="post-hoc"):
        result.selected_for(5)
    assert [group for group, _ in result.scores] == list(result.full_ranking)


def test_extratrees_complete_ranking_refit_uses_same_recipe_and_is_replayable() -> None:
    x, y = sample()
    plan = ExtraTreesPlan(subset_k=2, sensitivity_ks=(1, 2, 3), seed=0, n_estimators=20, block_length=24)
    digest = compute_predictive_row_digest(x, y)
    first = run_extratrees_selector(x, y, GROUPS, split(), plan, expected_row_digest=digest)
    second = run_extratrees_selector(x.copy(), y.copy(), GROUPS, split(), plan, expected_row_digest=digest)

    _check_common(first, 2, (1, 2, 3))
    assert first == second
    assert set(first.selected_groups) <= {"signal_a", "dup_a", "signal_b"}
    refit = first.details["refit_after_removal"]
    assert refit["model_config"] == plan.model_config()
    assert refit["scope"] == "within_primary_k_set"
    deltas = dict(refit["delta_validation_mae_when_removed"])
    assert set(deltas) == set(first.selected_groups)
    assert first.details["block_permutation"]["shared_across_features"] is True


def test_extratrees_refit_with_nothing_removed_reproduces_baseline_exactly() -> None:
    x, y = sample()
    plan = ExtraTreesPlan(subset_k=2, seed=0, n_estimators=10)
    nested = split()
    train_x, train_y = x[: nested.train_end], y[: nested.train_end]
    val_x, val_y = x[nested.validation_start :], y[nested.validation_start :]
    mae_a, imp_a, _ = fit_and_score_extratrees(plan, train_x, train_y, val_x, val_y)
    mae_b, imp_b, _ = fit_and_score_extratrees(plan, train_x, train_y, val_x, val_y)
    assert mae_a == mae_b
    assert np.array_equal(imp_a, imp_b)
    other_seed = ExtraTreesPlan(subset_k=2, seed=1, n_estimators=10)
    mae_c, _, _ = fit_and_score_extratrees(other_seed, train_x, train_y, val_x, val_y)
    assert mae_c != mae_a


def test_temporal_block_permutation_keeps_blocks_intact() -> None:
    permutation = temporal_block_permutation(10, 3, seed=0)
    assert sorted(permutation.tolist()) == list(range(10))
    assert np.array_equal(temporal_block_permutation(10, 3, seed=0), permutation)
    # Every original block (0-2, 3-5, 6-8, 9) stays contiguous and in order.
    for start in (0, 3, 6, 9):
        block = list(range(start, min(start + 3, 10)))
        position = permutation.tolist().index(start)
        assert permutation[position : position + len(block)].tolist() == block
    assert permutation.tolist() != list(range(10))


def test_extratrees_rejects_mismatched_rows_and_outer_labels() -> None:
    x, y = sample()
    plan = ExtraTreesPlan(subset_k=2, n_estimators=5)
    with pytest.raises(ValueError, match="digest mismatch"):
        run_extratrees_selector(x, y, GROUPS, split(), plan, expected_row_digest="0" * 64)
    with pytest.raises(ValueError, match="TRAIN-only"):
        run_extratrees_selector(
            x, y, GROUPS, split(), plan,
            expected_row_digest=compute_predictive_row_digest(x, y), input_label="VALIDATION",
        )


def test_sequential_marginal_respects_budget_and_puts_signal_first() -> None:
    x, y = sample()
    plan = SequentialMarginalPlan(subset_k=2, sensitivity_ks=(1, 3), budget=3)
    digest = compute_predictive_row_digest(x, y)
    result = run_sequential_marginal_selector(x, y, GROUPS, split(), plan, expected_row_digest=digest)

    _check_common(result, 2, (1, 3))
    assert result.details["steps_taken"] == 3
    assert set(result.full_ranking[:2]) <= {"signal_a", "dup_a", "signal_b"}
    assert "signal_b" in result.full_ranking[:2]
    step_scores = [score for _, score in result.scores[:3]]
    tail_scores = [score for _, score in result.scores[3:]]
    assert max(tail_scores) < min(step_scores)  # tail never outranks a budgeted step
    assert tail_scores == sorted(tail_scores, reverse=True)
    with pytest.raises(ValueError, match="budget"):
        SequentialMarginalPlan(subset_k=4, budget=3)


def test_group_gate_passes_signal_cluster_and_rejects_noise() -> None:
    x, y = sample(rows=4000, seed=11)
    nested = ChronologicalInnerFold("nested", 0, 2900, 3000, 4000)
    plan = GroupGatePlan(
        subset_k=3, group_membership=MEMBERSHIP, sensitivity_ks=(2,), null_shifts=(300, 600, 900, 1200, 1500)
    )
    digest = compute_predictive_row_digest(x, y)
    result = run_group_gate_selector(x, y, GROUPS, nested, plan, expected_row_digest=digest)

    _check_common(result, 3, (2,))
    passed = dict(result.details["cluster_passed"])
    assert passed["c_sig"] is True
    assert passed["signal_b"] is True
    assert passed["noise_a"] is False and passed["noise_b"] is False and passed["noise_c"] is False
    assert set(result.full_ranking[:3]) == {"signal_a", "dup_a", "signal_b"}
    assert dict(result.details["clusters"])["c_sig"] == ("signal_a", "dup_a")


def test_group_gate_requires_membership_for_every_group() -> None:
    x, y = sample()
    plan = GroupGatePlan(subset_k=2, group_membership=MEMBERSHIP[:-1], null_shifts=(50,))
    with pytest.raises(ValueError, match="every semantic group"):
        run_group_gate_selector(
            x, y, GROUPS, split(), plan, expected_row_digest=compute_predictive_row_digest(x, y)
        )


def test_chronoepilogi_adapter_finds_signal_and_duplicate_equivalence() -> None:
    x, y = sample()
    plan = ChronoEpilogiPlan(subset_k=2, sensitivity_ks=(1, 3), implementation="adapter")
    digest = compute_predictive_row_digest(x, y)
    result = run_chronoepilogi_selector(x, y, GROUPS, split(), plan, expected_row_digest=digest)

    _check_common(result, 2, (1, 3))
    boundary = result.details["first_markov_boundary"]
    assert set(boundary) <= {"signal_a", "dup_a", "signal_b"}
    assert "signal_b" in boundary
    classes = result.details["equivalence_classes"]
    flattened = {member for cls in classes for member in cls}
    assert {"signal_a", "dup_a"} <= flattened
    assert result.implementation == "adapter_faithful"
    assert result.details["unreproduced_scope"] == CHRONOEPILOGI_ADAPTER_UNREPRODUCED
    assert "NOT causal" in result.details["evidence_class"]
    assert result == run_chronoepilogi_selector(x.copy(), y.copy(), GROUPS, split(), plan, expected_row_digest=digest)


def test_chronoepilogi_official_is_required_when_demanded() -> None:
    pytest.importorskip("chronoepilogi", reason="official package only checked where installed")
    x, y = sample()
    plan = ChronoEpilogiPlan(subset_k=2, implementation="official")
    result = run_chronoepilogi_selector(
        x, y, GROUPS, split(), plan, expected_row_digest=compute_predictive_row_digest(x, y)
    )
    assert result.implementation.startswith("official_pinned_")
    assert set(result.details["first_markov_boundary"]) <= {"signal_a", "dup_a", "signal_b"}
    _check_common(result, 2, ())


def test_plans_reject_booleans_and_unsealed_values() -> None:
    with pytest.raises(TypeError):
        ExtraTreesPlan(subset_k=True)
    with pytest.raises(ValueError):
        ExtraTreesPlan(subset_k=2, max_features=0.0)
    with pytest.raises(ValueError):
        ChronoEpilogiPlan(subset_k=2, implementation="whatever")
    with pytest.raises(ValueError):
        GroupGatePlan(subset_k=2, group_membership=MEMBERSHIP, null_shifts=())
    assert ComparatorRunPlan(3, (), (1, 5)).sealed_ks == (1, 3, 5)
