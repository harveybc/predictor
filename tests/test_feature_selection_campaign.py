from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tools.feature_selection_campaign import (
    CampaignDisposition,
    FeatureSelectionCampaignPlan,
    compute_campaign_group_digest,
    compute_campaign_plan_digest,
    compute_campaign_row_digest,
    run_feature_selection_campaign,
)
from tools.feature_selector_comparators import RandomTape
from tools.feature_selector_elastic_net import ChronologicalInnerFold


def sample(
    *, rows: int = 180, seed: int = 31
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 6))
    y = 3.5 * x[:, 0] - 2.0 * x[:, 3] + rng.normal(scale=0.1, size=rows)
    return x, y, ("signal_a", "noise_a", "noise_b", "signal_b", "noise_c", "noise_d")


def folds() -> tuple[ChronologicalInnerFold, ...]:
    return (
        ChronologicalInnerFold("f0", 0, 60, 60, 90),
        ChronologicalInnerFold("f1", 0, 90, 90, 120),
        ChronologicalInnerFold("f2", 0, 120, 120, 150),
    )


EXPECTED_METHODS = (
    "ALL_ADMISSIBLE",
    "SPEARMAN_K",
    "MI8_K",
    "JMI_K",
    "CMIM_K",
    "MRMR_K",
    "REDUNDANCY_K",
    "RANDOM_K:random-17",
    "RANDOM_K:random-29",
    "ELASTIC_NET_K",
    "GROUP_ELASTIC_NET_K",
    "EXTRATREES_TBP_K",
    "SEQ_MARGINAL_K",
    "GROUP_GATE_K",
    "CHRONOEPILOGI_K",
)


def plan(
    groups: tuple[str, ...],
    *,
    redundancy_gains: tuple[tuple[str, float], ...] | None = None,
    sensitivity_ks: tuple[int, ...] = (1, 3),
    enabled_methods: tuple[str, ...] | None = None,
) -> FeatureSelectionCampaignPlan:
    return FeatureSelectionCampaignPlan(
        subset_k=2,
        random_tapes=(RandomTape("random-17", 17), RandomTape("random-29", 29)),
        elastic_lambda_path=(0.5, 0.15811388300841897, 0.05, 0.015811388300841896, 0.005),
        elastic_max_iter=20_000,
        elastic_tolerance=1e-10,
        redundancy_inner_fold_gains=redundancy_gains
        or tuple((group, float(len(groups) - index)) for index, group in enumerate(groups)),
        causal_annotations=tuple(
            (group, "NOT_IDENTIFIED" if index % 2 else "IDENTIFIED")
            for index, group in enumerate(groups)
        ),
        sensitivity_ks=sensitivity_ks,
        seed=0,
        group_membership=tuple((group, "c_sig" if group.startswith("signal") else group) for group in groups),
        extratrees={"n_estimators": 10, "block_length": 12},
        group_gate_null_shifts=(20, 30),
        chronoepilogi={"implementation": "adapter"},
        enabled_methods=enabled_methods,
    )


def run(x: np.ndarray, y: np.ndarray, groups: tuple[str, ...], campaign_plan=None):
    selected_plan = campaign_plan or plan(groups)
    return run_feature_selection_campaign(
        x,
        y,
        groups,
        folds(),
        selected_plan,
        expected_row_digest=compute_campaign_row_digest(x, y),
        expected_group_digest=compute_campaign_group_digest(groups),
        expected_plan_digest=compute_campaign_plan_digest(selected_plan, folds()),
    )


def test_campaign_is_complete_authenticated_and_retains_full_rankings() -> None:
    x, y, groups = sample()
    result = run(x, y, groups)

    assert result.row_digest == compute_campaign_row_digest(x, y)
    assert result.group_digest == compute_campaign_group_digest(groups)
    assert result.plan_digest == compute_campaign_plan_digest(plan(groups), folds())
    assert result.sealed_subset_k == 2
    assert result.causal_not_identified_policy == "NEUTRAL_NOT_A_REJECTION"
    assert result.eligible_groups == groups
    assert tuple(outcome.outcome_id for outcome in result.outcomes) == EXPECTED_METHODS
    assert result.sealed_ks == (1, 2, 3)
    assert result.seed == 0
    failures = [(o.outcome_id, o.failure_message) for o in result.outcomes if o.disposition is not CampaignDisposition.COMPLETE]
    assert failures == []
    for outcome in result.outcomes:
        assert outcome.failure_type is None
        assert outcome.failure_message is None
        assert outcome.result is not None
        assert len(outcome.result.full_ranking) == len(groups)
        assert set(outcome.result.full_ranking) == set(groups)


def test_deterministic_replay_is_identical() -> None:
    x, y, groups = sample()
    assert run(x, y, groups) == run(x.copy(), y.copy(), groups)


def test_campaign_does_not_change_caller_array_writeability() -> None:
    x, y, groups = sample()

    run(x, y, groups)

    assert x.flags.writeable
    assert y.flags.writeable


def test_unseen_train_tail_does_not_choose_elastic_hyperparameters() -> None:
    x, y, groups = sample(rows=210)
    base = run(x, y, groups)
    changed_x = x.copy()
    changed_y = y.copy()
    changed_x[150:] *= np.asarray([8.0, -3.0, 2.0, 0.5, -4.0, 6.0])
    changed_y[150:] = -7.0 * changed_x[150:, 1] + changed_x[150:, 5]
    changed = run(changed_x, changed_y, groups)

    base_elastic = base.outcome("ELASTIC_NET_K").result
    changed_elastic = changed.outcome("ELASTIC_NET_K").result
    assert base_elastic is not None and changed_elastic is not None
    assert changed_elastic.selected_l1_ratio == base_elastic.selected_l1_ratio
    assert changed_elastic.selected_lambda == base_elastic.selected_lambda
    assert changed_elastic.diagnostics == base_elastic.diagnostics
    assert changed.row_digest != base.row_digest


def test_one_method_failure_is_retained_without_erasing_other_methods() -> None:
    x, y, groups = sample()
    incomplete_gains = ((groups[0], 1.0),)
    result = run(x, y, groups, plan(groups, redundancy_gains=incomplete_gains))

    failed = result.outcome("REDUNDANCY_K")
    assert failed.disposition is CampaignDisposition.FAILED
    assert failed.result is None
    assert failed.failure_type == "ValueError"
    assert "every semantic group" in failed.failure_message
    assert len(result.outcomes) == len(EXPECTED_METHODS)
    assert all(
        outcome.disposition is CampaignDisposition.COMPLETE
        for outcome in result.outcomes
        if outcome.outcome_id != "REDUNDANCY_K"
    )


def test_every_method_carries_a_selection_at_every_sealed_k() -> None:
    x, y, groups = sample()
    result = run(x, y, groups)
    for outcome in result.outcomes:
        selector = outcome.result
        assert selector is not None, outcome.outcome_id
        for sealed_k in result.sealed_ks:
            selection = selector.selected_for(sealed_k)
            if outcome.outcome_id == "ALL_ADMISSIBLE":
                assert set(selection) == set(groups)
            else:
                assert selection == selector.full_ranking[:sealed_k], outcome.outcome_id
        with pytest.raises(ValueError):
            selector.selected_for(4)


def test_sensitivity_k_beyond_groups_is_retained_per_k_without_losing_the_ranking() -> None:
    x, y, groups = sample()
    wide = plan(groups, sensitivity_ks=(1, 9))
    result = run(x, y, groups, wide)
    for outcome in result.outcomes:
        assert outcome.disposition is CampaignDisposition.COMPLETE, (outcome.outcome_id, outcome.failure_message)
        selector = outcome.result
        assert len(selector.full_ranking) == len(groups)
        if outcome.outcome_id != "ALL_ADMISSIBLE":
            assert dict(selector.k_failures)[9].startswith("sealed K exceeds")
            assert selector.selected_for(1) == selector.full_ranking[:1]


def test_enabled_methods_chunk_shares_the_plan_identity_and_reports_progress() -> None:
    x, y, groups = sample()
    full = plan(groups)
    chunk = plan(groups, enabled_methods=("SPEARMAN_K", "SEQ_MARGINAL_K"))
    assert compute_campaign_plan_digest(full, folds()) == compute_campaign_plan_digest(chunk, folds())
    seen: list[str] = []
    result = run_feature_selection_campaign(
        x, y, groups, folds(), chunk,
        expected_row_digest=compute_campaign_row_digest(x, y),
        expected_group_digest=compute_campaign_group_digest(groups),
        expected_plan_digest=compute_campaign_plan_digest(chunk, folds()),
        on_outcome=lambda outcome: seen.append(outcome.outcome_id),
    )
    assert seen == ["SPEARMAN_K", "SEQ_MARGINAL_K"]
    assert tuple(o.outcome_id for o in result.outcomes) == ("SPEARMAN_K", "SEQ_MARGINAL_K")
    assert all(o.wall_seconds >= 0.0 for o in result.outcomes)
    with pytest.raises(ValueError, match="unknown campaign methods"):
        plan(groups, enabled_methods=("NOT_A_METHOD",))


def test_group_methods_fail_in_isolation_without_membership() -> None:
    x, y, groups = sample()
    base = plan(groups)
    no_groups = replace(base, group_membership=())
    result = run(x, y, groups, no_groups)
    for method in ("GROUP_ELASTIC_NET_K", "GROUP_GATE_K"):
        outcome = result.outcome(method)
        assert outcome.disposition is CampaignDisposition.FAILED
        assert "group membership" in outcome.failure_message
    assert result.outcome("SPEARMAN_K").disposition is CampaignDisposition.COMPLETE


def test_not_identified_is_neutral_and_never_removes_a_group() -> None:
    x, y, groups = sample()
    all_not_identified = replace(
        plan(groups),
        causal_annotations=tuple((group, "NOT_IDENTIFIED") for group in groups),
    )
    result = run(x, y, groups, all_not_identified)

    assert result.eligible_groups == groups
    assert dict(result.causal_annotations) == {group: "NOT_IDENTIFIED" for group in groups}
    assert all(
        set(outcome.result.full_ranking) == set(groups)
        for outcome in result.outcomes
        if outcome.result is not None
    )


@pytest.mark.parametrize(
    ("bad_groups", "message"),
    [
        (("a", "b", "c", "d", "e"), "one semantic group"),
        (("a", "a", "b", "c", "d", "e"), "duplicate semantic groups"),
    ],
)
def test_missing_or_duplicate_groups_fail_before_any_method(
    bad_groups: tuple[str, ...], message: str
) -> None:
    x, y, _ = sample()
    campaign_plan = FeatureSelectionCampaignPlan(
        subset_k=2,
        random_tapes=(RandomTape("r", 1),),
        elastic_lambda_path=(0.1, 0.01),
        redundancy_inner_fold_gains=tuple((group, 1.0) for group in bad_groups),
        causal_annotations=tuple((group, "NOT_IDENTIFIED") for group in bad_groups),
    )
    with pytest.raises(ValueError, match=message):
        run_feature_selection_campaign(
            x,
            y,
            bad_groups,
            folds(),
            campaign_plan,
            expected_row_digest=compute_campaign_row_digest(x, y),
            expected_group_digest="untrusted",
            expected_plan_digest=compute_campaign_plan_digest(campaign_plan, folds()),
        )


def test_row_group_and_plan_identity_mismatches_fail_closed() -> None:
    x, y, groups = sample()
    campaign_plan = plan(groups)
    common = dict(
        x=x,
        y=y,
        group_ids=groups,
        inner_folds=folds(),
        run_plan=campaign_plan,
        expected_row_digest=compute_campaign_row_digest(x, y),
        expected_group_digest=compute_campaign_group_digest(groups),
        expected_plan_digest=compute_campaign_plan_digest(campaign_plan, folds()),
    )
    for key, message in (
        ("expected_row_digest", "row digest mismatch"),
        ("expected_group_digest", "group digest mismatch"),
        ("expected_plan_digest", "plan digest mismatch"),
    ):
        altered = dict(common)
        altered[key] = "0" * 64
        with pytest.raises(ValueError, match=message):
            run_feature_selection_campaign(**altered)


def test_campaign_rejects_outer_labels_arrays_and_more_than_three_tapes() -> None:
    x, y, groups = sample()
    campaign_plan = plan(groups)
    with pytest.raises(ValueError, match="TRAIN-only"):
        run_feature_selection_campaign(
            x,
            y,
            groups,
            folds(),
            campaign_plan,
            expected_row_digest=compute_campaign_row_digest(x, y),
            expected_group_digest=compute_campaign_group_digest(groups),
            expected_plan_digest=compute_campaign_plan_digest(campaign_plan, folds()),
            input_label="TEST",
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_feature_selection_campaign(
            x,
            y,
            groups,
            folds(),
            campaign_plan,
            expected_row_digest=compute_campaign_row_digest(x, y),
            expected_group_digest=compute_campaign_group_digest(groups),
            expected_plan_digest=compute_campaign_plan_digest(campaign_plan, folds()),
            validation_x=x,
        )
    with pytest.raises(ValueError, match="at most three"):
        FeatureSelectionCampaignPlan(
            subset_k=2,
            random_tapes=tuple(RandomTape(f"r{i}", i) for i in range(4)),
            elastic_lambda_path=(0.1, 0.01),
            redundancy_inner_fold_gains=tuple((group, 1.0) for group in groups),
            causal_annotations=tuple((group, "NOT_IDENTIFIED") for group in groups),
        )


def test_campaign_requires_at_least_one_fixed_random_tape() -> None:
    x, y, groups = sample()
    with pytest.raises(ValueError, match="at least one fixed random tape"):
        FeatureSelectionCampaignPlan(
            subset_k=2,
            random_tapes=(),
            elastic_lambda_path=(0.1, 0.01),
            redundancy_inner_fold_gains=tuple((group, 1.0) for group in groups),
            causal_annotations=tuple((group, "NOT_IDENTIFIED") for group in groups),
        )
