from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tools.feature_preparation_policy import (
    CAUSAL_LAG_HOURS,
    UNKNOWN_CATEGORY,
    CausalDisposition,
    CausalStatus,
    FeatureGroupSpec,
    FeaturePreparationPolicy,
    Partition,
    Representation,
    SemanticRole,
    TemporalObservation,
    ValueVariant,
    fit_train_state,
    prepare_outer,
)


def continuous_spec(**overrides: object) -> FeatureGroupSpec:
    values: dict[str, object] = {
        "group_id": "eurusd_close",
        "role": SemanticRole.CONTINUOUS,
        "members": ("eurusd_close",),
    }
    values.update(overrides)
    return FeatureGroupSpec(**values)


def test_contract_enumerates_all_roles_and_representations() -> None:
    assert {role.value for role in SemanticRole} == {
        "CONTINUOUS",
        "BINARY",
        "CYCLIC_PAIR",
        "CATEGORICAL_GROUP",
        "MASK",
        "DELTA_TIME",
        "QUALITY_ONLY",
        "SELECTOR_EPISODE_ONLY",
    }
    assert {representation.value for representation in Representation} == {
        "POINT",
        "LAGGED",
        "WINDOW_168",
    }
    assert CAUSAL_LAG_HOURS == (0, 1, 2, 6, 24, 48, 168)


def test_specs_and_fit_state_are_immutable() -> None:
    spec = continuous_spec()
    state = fit_train_state(spec, [1.0, 2.0, 3.0], ValueVariant.RAW)

    with pytest.raises(FrozenInstanceError):
        spec.group_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.mean = 0.0  # type: ignore[misc]


def test_selection_accepts_only_complete_semantic_groups() -> None:
    cyclic = FeatureGroupSpec(
        group_id="hour_of_day",
        role=SemanticRole.CYCLIC_PAIR,
        members=("hour_sin", "hour_cos"),
        fixed_calendar_context=True,
    )
    category = FeatureGroupSpec(
        group_id="market_session",
        role=SemanticRole.CATEGORICAL_GROUP,
        members=("session_asia", "session_europe", "session_us"),
    )
    policy = FeaturePreparationPolicy((cyclic, category))

    assert policy.validate_selection(("market_session",), subset_k=1) == (
        "market_session",
    )
    for internal_column in ("hour_sin", "hour_cos", "session_europe"):
        with pytest.raises(ValueError, match="semantic group"):
            policy.validate_selection((internal_column,), subset_k=1)
    with pytest.raises(ValueError, match="fixed calendar"):
        policy.validate_selection(("hour_of_day",), subset_k=1)


def test_cyclic_pairs_are_atomic_and_have_exactly_two_members() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        FeatureGroupSpec(
            group_id="hour",
            role=SemanticRole.CYCLIC_PAIR,
            members=("hour_sin",),
            fixed_calendar_context=True,
        )


def test_raw_and_train_q005_clip_are_distinct_train_fitted_states() -> None:
    spec = continuous_spec()
    values = [0.0, 1.0, 2.0, 3.0, 1_000.0]

    raw = fit_train_state(spec, values, ValueVariant.RAW)
    clipped = fit_train_state(spec, values, ValueVariant.TRAIN_Q005_CLIP)

    assert raw.variant is ValueVariant.RAW
    assert raw.clip_bounds is None
    assert clipped.variant is ValueVariant.TRAIN_Q005_CLIP
    assert clipped.clip_bounds is not None
    assert raw.identity_digest != clipped.identity_digest


def test_outer_bytes_never_change_train_fitted_state() -> None:
    spec = continuous_spec()
    train = [1.0, 2.0, None, 4.0]
    state = fit_train_state(spec, train, ValueVariant.TRAIN_Q005_CLIP)

    first = prepare_outer(spec, state, [5.0, None], Partition.VALIDATION)
    second = prepare_outer(spec, state, [-1e12, 1e12], Partition.TEST)

    assert first.fit_state is state
    assert second.fit_state is state
    assert first.fit_state.identity_digest == second.fit_state.identity_digest
    assert state == fit_train_state(spec, train, ValueVariant.TRAIN_Q005_CLIP)


def test_outer_partitions_can_transform_but_cannot_fit() -> None:
    spec = continuous_spec()
    for outer in (Partition.VALIDATION, Partition.TEST, Partition.LIVE):
        with pytest.raises(ValueError, match="TRAIN"):
            fit_train_state(spec, [1.0], ValueVariant.RAW, partition=outer)

    state = fit_train_state(spec, [1.0, 2.0], ValueVariant.RAW)
    with pytest.raises(ValueError, match="outer partition"):
        prepare_outer(spec, state, [3.0], Partition.TRAIN)


def test_unseen_categories_map_to_unknown_without_changing_vocabulary() -> None:
    spec = FeatureGroupSpec(
        group_id="venue",
        role=SemanticRole.CATEGORICAL_GROUP,
        members=("venue",),
    )
    state = fit_train_state(spec, ["LONDON", "NEW_YORK", "LONDON"])
    prepared = prepare_outer(
        spec,
        state,
        ["LONDON", "TOKYO", None],
        Partition.VALIDATION,
    )

    assert state.vocabulary == (UNKNOWN_CATEGORY, "LONDON", "NEW_YORK")
    assert prepared.values == ("LONDON", UNKNOWN_CATEGORY, UNKNOWN_CATEGORY)
    assert "TOKYO" not in state.vocabulary


def test_binary_values_are_train_imputed_but_not_scaled() -> None:
    spec = FeatureGroupSpec(
        group_id="is_market_open",
        role=SemanticRole.BINARY,
        members=("is_market_open",),
    )
    state = fit_train_state(spec, [0, 1, 1, None])
    prepared = prepare_outer(spec, state, [0, 1, None], Partition.TEST)

    assert state.mean is None
    assert state.std is None
    assert state.median == 1.0
    assert prepared.values == (0.0, 1.0, 1.0)


def test_no_generic_forward_fill_and_carry_needs_source_rule_and_max_age() -> None:
    with pytest.raises(ValueError, match="max_carry_age"):
        continuous_spec(source_allows_carry=True)
    with pytest.raises(ValueError, match="source allowance"):
        continuous_spec(max_carry_age_seconds=60)
    with pytest.raises(TypeError, match="boolean"):
        continuous_spec(source_allows_carry=True, max_carry_age_seconds=True)

    spec = continuous_spec(source_allows_carry=True, max_carry_age_seconds=300)
    assert spec.source_allows_carry is True
    assert spec.max_carry_age_seconds == 300


def test_temporal_missing_values_require_value_mask_and_delta() -> None:
    spec = continuous_spec(representation=Representation.WINDOW_168)
    state = fit_train_state(
        spec,
        [
            TemporalObservation(value=1.0, observed_mask=1, delta_time=0.0),
            TemporalObservation(value=2.0, observed_mask=1, delta_time=0.0),
            TemporalObservation(value=None, observed_mask=0, delta_time=1.0),
        ],
    )

    prepared = prepare_outer(
        spec,
        state,
        [TemporalObservation(value=None, observed_mask=0, delta_time=3.0)],
        Partition.TEST,
    )
    assert prepared.values == (0.0,)
    assert prepared.observed_mask == (0,)
    assert prepared.delta_time == (3.0,)

    with pytest.raises(ValueError, match="TemporalObservation"):
        prepare_outer(spec, state, [None], Partition.TEST)
    with pytest.raises(ValueError, match="TemporalObservation"):
        fit_train_state(spec, [1.0, 2.0, None])
    with pytest.raises(ValueError, match="missing value"):
        TemporalObservation(value=None, observed_mask=1, delta_time=1.0)


def test_lagged_representation_uses_only_sealed_elapsed_hour_lags() -> None:
    spec = continuous_spec(representation=Representation.LAGGED)
    assert spec.elapsed_lag_hours == CAUSAL_LAG_HOURS

    with pytest.raises(ValueError, match="sealed elapsed-hour set"):
        continuous_spec(
            representation=Representation.LAGGED,
            elapsed_lag_hours=(0, 1, 24),
        )


def test_boolean_values_are_rejected_where_numbers_or_counts_are_required() -> None:
    spec = continuous_spec()
    with pytest.raises(TypeError, match="boolean"):
        fit_train_state(spec, [1.0, True])

    policy = FeaturePreparationPolicy((spec,))
    with pytest.raises(TypeError, match="boolean"):
        policy.validate_selection(("eurusd_close",), subset_k=True)

    with pytest.raises(TypeError, match="boolean"):
        TemporalObservation(value=1.0, observed_mask=True, delta_time=0.0)


def test_fixed_calendar_is_context_and_economic_calendar_is_deferred() -> None:
    calendar = FeatureGroupSpec(
        group_id="weekday",
        role=SemanticRole.CYCLIC_PAIR,
        members=("weekday_sin", "weekday_cos"),
        fixed_calendar_context=True,
    )
    economic = FeatureGroupSpec(
        group_id="economic_calendar",
        role=SemanticRole.SELECTOR_EPISODE_ONLY,
        members=("event_surprise",),
        economic_calendar=True,
        deferred_until_stage="I11",
    )
    policy = FeaturePreparationPolicy((calendar, economic))

    assert policy.context_group_ids == ("weekday",)
    assert policy.selectable_group_ids == ()
    assert policy.subset_k_group_ids == ()
    with pytest.raises(ValueError, match="deferred"):
        policy.validate_selection(("economic_calendar",), subset_k=1)


def test_not_identified_is_neutral_and_never_automatic_rejection() -> None:
    policy = FeaturePreparationPolicy((continuous_spec(),))

    assert policy.causal_disposition(CausalStatus.NOT_IDENTIFIED) is CausalDisposition.NEUTRAL
    assert policy.causal_disposition(CausalStatus.IDENTIFIED) is CausalDisposition.SUPPORTING
    assert policy.causal_disposition(CausalStatus.REJECTED) is CausalDisposition.REJECTING
