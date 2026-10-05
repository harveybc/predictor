from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tools.feature_selector_elastic_net import (
    L1_RATIO_PATH,
    ChronologicalInnerFold,
    ElasticNetRunPlan,
    compute_train_row_digest,
    run_elastic_net_selector,
)


def sparse_sample(
    *, seed: int = 17, rows: int = 180
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 6))
    y = 4.0 * x[:, 0] - 2.5 * x[:, 3] + rng.normal(scale=0.08, size=rows)
    return x, y, ("signal_a", "noise_a", "noise_b", "signal_b", "noise_c", "noise_d")


def folds() -> tuple[ChronologicalInnerFold, ...]:
    return (
        ChronologicalInnerFold("f0", 0, 60, 60, 90),
        ChronologicalInnerFold("f1", 0, 90, 90, 120),
        ChronologicalInnerFold("f2", 0, 120, 120, 150),
    )


def plan(*, subset_k: int = 2) -> ElasticNetRunPlan:
    return ElasticNetRunPlan(
        subset_k=subset_k,
        lambda_path=(0.5, 0.15811388300841897, 0.05, 0.015811388300841896, 0.005),
        max_iter=20_000,
        tolerance=1e-10,
    )


def run(
    x: np.ndarray,
    y: np.ndarray,
    groups: tuple[str, ...],
    *,
    run_plan: ElasticNetRunPlan | None = None,
):
    return run_elastic_net_selector(
        x,
        y,
        groups,
        folds(),
        run_plan or plan(),
        expected_row_digest=compute_train_row_digest(x, y),
    )


def test_sparse_truth_is_ranked_first_and_complete() -> None:
    x, y, groups = sparse_sample()
    result = run(x, y, groups)

    assert set(result.selected_groups) == {"signal_a", "signal_b"}
    assert len(result.full_ranking) == len(groups)
    assert set(result.full_ranking) == set(groups)
    assert result.selected_groups == result.full_ranking[:2]
    assert tuple(group for group, _ in result.coefficient_norms) == result.full_ranking
    assert all(value >= 0.0 for _, value in result.coefficient_norms)
    assert np.isfinite(result.selected_mean_inner_validation_mae)
    assert all(diagnostic.converged for diagnostic in result.diagnostics)
    assert result.final_iterations > 0


def test_deterministic_replay_has_identical_result_and_digests() -> None:
    x, y, groups = sparse_sample()
    first = run(x, y, groups)
    second = run(x.copy(), y.copy(), groups)

    assert first == second
    assert first.row_digest == compute_train_row_digest(x, y)
    assert len(first.fold_digest) == 64
    assert len(first.hyperparameter_digest) == 64
    assert len(first.group_digest) == 64
    assert len(first.config_digest) == 64


def test_standardization_handles_feature_scale_without_changing_selection() -> None:
    x, y, groups = sparse_sample()
    base = run(x, y, groups)
    scaled_x = x.copy()
    scaled_x[:, 0] *= 1_000_000.0
    scaled_x[:, 3] *= 0.0001
    scaled = run(scaled_x, y, groups)

    assert scaled.full_ranking == base.full_ranking
    assert scaled.selected_groups == base.selected_groups
    assert scaled.selected_l1_ratio == base.selected_l1_ratio
    assert scaled.selected_lambda == base.selected_lambda


def test_hyperparameters_are_selected_only_from_inner_validation_rows() -> None:
    x, y, groups = sparse_sample(rows=210)
    base = run(x, y, groups)
    changed = x.copy()
    changed_y = y.copy()
    changed[150:] *= np.asarray([7.0, -4.0, 2.0, 0.5, -3.0, 5.0])
    changed_y[150:] = -6.0 * changed[150:, 1] + 3.0 * changed[150:, 4]
    outer_changed = run(changed, changed_y, groups)

    assert outer_changed.selected_l1_ratio == base.selected_l1_ratio
    assert outer_changed.selected_lambda == base.selected_lambda
    assert outer_changed.diagnostics == base.diagnostics
    assert outer_changed.row_digest != base.row_digest


def test_exact_fixed_l1_and_supplied_lambda_path_are_in_identity() -> None:
    x, y, groups = sparse_sample()
    first_plan = plan()
    second_plan = ElasticNetRunPlan(
        subset_k=2,
        lambda_path=(0.4, 0.12649110640673517, 0.04, 0.012649110640673518, 0.004),
        max_iter=20_000,
        tolerance=1e-10,
    )
    first = run(x, y, groups, run_plan=first_plan)
    second = run(x, y, groups, run_plan=second_plan)

    assert L1_RATIO_PATH == (0.1, 0.5, 0.9, 1.0)
    assert first.hyperparameter_digest != second.hyperparameter_digest
    assert first.config_digest != second.config_digest
    seen = {(item.l1_ratio, item.lambda_value) for item in first.diagnostics}
    assert seen == {
        (ratio, lambda_value)
        for ratio in L1_RATIO_PATH
        for lambda_value in first_plan.lambda_path
    }


def test_post_hoc_k_change_is_refused_and_result_is_frozen() -> None:
    x, y, groups = sparse_sample()
    result = run(x, y, groups)

    assert result.selected_for(2) == result.selected_groups
    with pytest.raises(ValueError, match="post-hoc K"):
        result.selected_for(3)
    with pytest.raises(TypeError, match="boolean"):
        result.selected_for(True)  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        result.sealed_subset_k = 1  # type: ignore[misc]


@pytest.mark.parametrize("label", ["VALIDATION", "TEST", "OUTER_TEST", "LIVE"])
def test_outer_labelled_top_level_inputs_are_rejected(label: str) -> None:
    x, y, groups = sparse_sample()
    with pytest.raises(ValueError, match="TRAIN only"):
        run_elastic_net_selector(
            x,
            y,
            groups,
            folds(),
            plan(),
            expected_row_digest=compute_train_row_digest(x, y),
            input_label=label,
        )


def test_outer_arrays_are_not_part_of_the_api() -> None:
    x, y, groups = sparse_sample()
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_elastic_net_selector(
            x,
            y,
            groups,
            folds(),
            plan(),
            expected_row_digest=compute_train_row_digest(x, y),
            test_x=x,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("x_mutator", "y_mutator", "groups", "message"),
    [
        (lambda x: x.astype(bool), lambda y: y, None, "boolean"),
        (lambda x: np.where(np.indices(x.shape)[1] == 0, np.nan, x), lambda y: y, None, "finite"),
        (lambda x: x, lambda y: np.where(np.arange(y.size) == 0, np.inf, y), None, "finite"),
        (lambda x: x, lambda y: y, ("a", "a", "b", "c", "d", "e"), "duplicate"),
    ],
)
def test_invalid_training_inputs_fail_closed(x_mutator, y_mutator, groups, message) -> None:
    x, y, default_groups = sparse_sample()
    bad_x = x_mutator(x)
    bad_y = y_mutator(y)
    with pytest.raises((TypeError, ValueError), match=message):
        run_elastic_net_selector(
            bad_x,
            bad_y,
            groups or default_groups,
            folds(),
            plan(),
            expected_row_digest="not-trusted",
        )


def test_digest_authentication_and_k_overflow_fail_closed() -> None:
    x, y, groups = sparse_sample()
    with pytest.raises(ValueError, match="row digest mismatch"):
        run_elastic_net_selector(
            x, y, groups, folds(), plan(), expected_row_digest="0" * 64
        )
    with pytest.raises(ValueError, match="exceeds"):
        run(x, y, groups, run_plan=plan(subset_k=7))


@pytest.mark.parametrize(
    "fold_factory",
    [
        lambda: (ChronologicalInnerFold("empty-train", 1, 1, 2, 3),),
        lambda: (ChronologicalInnerFold("empty-validation", 0, 2, 3, 3),),
        lambda: (ChronologicalInnerFold("overlap", 0, 5, 4, 8),),
        lambda: (
            ChronologicalInnerFold("later", 0, 40, 40, 70),
            ChronologicalInnerFold("earlier", 0, 20, 20, 30),
        ),
        lambda: (
            ChronologicalInnerFold("a", 0, 40, 40, 70),
            ChronologicalInnerFold("b", 0, 60, 60, 90),
        ),
    ],
)
def test_empty_overlapping_or_nonchronological_folds_are_rejected(fold_factory) -> None:
    x, y, groups = sparse_sample()
    with pytest.raises(ValueError, match="empty|overlap|nonchronological"):
        run_elastic_net_selector(
            x,
            y,
            groups,
            fold_factory(),
            plan(),
            expected_row_digest=compute_train_row_digest(x, y),
        )


def test_boolean_plan_values_and_non_logarithmic_path_are_rejected() -> None:
    with pytest.raises(TypeError, match="boolean"):
        ElasticNetRunPlan(True, (1.0, 0.1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        ElasticNetRunPlan(1, (1.0, 0.1), max_iter=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        ElasticNetRunPlan(1, (1.0, 0.1), tolerance=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="logarithmically"):
        ElasticNetRunPlan(1, (1.0, 0.4, 0.1))


def test_unconverged_finalist_is_rejected() -> None:
    x, y, groups = sparse_sample()
    too_short = ElasticNetRunPlan(
        subset_k=2,
        lambda_path=(1e-4, 1e-5),
        max_iter=1,
        tolerance=1e-30,
    )
    with pytest.raises(RuntimeError, match="no hyperparameter candidate converged|unconverged"):
        run(x, y, groups, run_plan=too_short)
