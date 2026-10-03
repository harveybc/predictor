"""Acceptance tests for callback-based BUSINESS weekly training."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from tools.business_asof_window import AsOfRow, SupportSpec
from tools.business_objective_firewall import BusinessObjectiveFirewall, FirewallError
from tools.business_weekly_protocol import (
    BusinessWeeklyProtocol,
    EvaluationMode,
    EvaluationSplit,
    UpdateMode,
)
from tools.business_weekly_training import (
    TaskFamily,
    TrainingContractError,
    TrainingResult,
    TrainingStatus,
    require_matched_releases,
    run_weekly_training,
)


UTC = timezone.utc
FAMILIES = tuple(TaskFamily)


def dt(year, month, day):
    return datetime(year, month, day, tzinfo=UTC)


def protocol(*, monthly=False, warm=False):
    return BusinessWeeklyProtocol(
        evaluation_mode=(
            EvaluationMode.BUSINESS_MONTHLY_WALK_FORWARD
            if monthly
            else EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD
        ),
        update_mode=(
            UpdateMode.WARM_UPDATE_ROLLING_4Y
            if warm
            else UpdateMode.FULL_RETRAIN_ROLLING_4Y
        ),
        validation_start=dt(2024, 1, 1),
        validation_end=dt(2024, 2, 12),
        test_start=dt(2025, 1, 6),
        test_end=dt(2025, 1, 20),
        procedure_digest="procedure-v1",
    )


def support():
    return SupportSpec(
        input_lookback=timedelta(0),
        target_horizon=timedelta(days=1),
        maximum_holding_support=timedelta(0),
        inner_validation_weeks=1,
    )


class Rows:
    def __init__(self):
        self.calls = []

    def __call__(self, week):
        self.calls.append(week)
        event_time = week.fit_start + timedelta(days=1)
        return (
            AsOfRow(
                record_id=f"row-{week.start.date()}",
                event_time=event_time,
                available_time=event_time,
                target_available_time=event_time + timedelta(days=1),
                row_digest=f"sha256:{week.start.date()}",
            ),
        )


class FakeTrainer:
    """Deterministic in-memory trainer used only by this test module."""

    def __init__(self, mutation=None):
        self.requests = []
        self.mutation = mutation

    def __call__(self, request):
        self.requests.append(request)
        digest = hashlib.sha256(
            f"{request.family.value}:{request.week.start.isoformat()}".encode()
        ).hexdigest()
        result = TrainingResult(
            family=request.family,
            week_start=request.week.start,
            cutoff=request.week.cutoff,
            seed=request.seed,
            data_identity=request.data_identity,
            procedure_digest=request.procedure_digest,
            update_mode=request.update_mode,
            cost_identity=request.cost_identity,
            parent_digest=request.parent_digest,
            model_digest=digest,
            execution_backend=request.required_backend,
        )
        return self.mutation(request, result) if self.mutation else result


def firewall(p):
    test_ids = tuple(
        week.start.isoformat()
        for week in p.weeks()
        if week.split is EvaluationSplit.TEST
    )
    return BusinessObjectiveFirewall.create(test_ids)


def run(p, trainer, rows=None, **overrides):
    values = {
        "protocol": p,
        "split": EvaluationSplit.VALIDATION,
        "families": FAMILIES,
        "support": support(),
        "rows_for_week": rows or Rows(),
        "trainer": trainer,
        "firewall": firewall(p),
        "seed": 17,
        "cost_identity": "cost-v1",
        "required_backend": "GPU:physical-device-1",
    }
    values.update(overrides)
    return run_weekly_training(**values)


def test_weekly_invokes_each_family_exactly_once_per_due_release():
    p = protocol()
    trainer = FakeTrainer()
    result = run(p, trainer)
    validation_weeks = [
        week for week in p.weeks() if week.split is EvaluationSplit.VALIDATION
    ]

    assert len(trainer.requests) == len(validation_weeks) * len(FAMILIES)
    assert len(result.releases) == len(trainer.requests)
    assert all(not release.reused for release in result.releases)
    for week in validation_weeks:
        releases = [item for item in result.releases if item.scored_week == week]
        require_matched_releases(releases)


def test_monthly_non_due_weeks_reuse_only_latest_prior_release():
    p = protocol(monthly=True)
    trainer = FakeTrainer()
    result = run(p, trainer)
    due_weeks = [
        week
        for week in p.weeks()
        if week.split is EvaluationSplit.VALIDATION and week.retrain_due
    ]

    assert len(trainer.requests) == len(due_weeks) * len(FAMILIES)
    for family in FAMILIES:
        releases = [item for item in result.releases if item.family is family]
        assert releases[0].reused is False
        for current, previous in zip(releases[1:], releases):
            if current.reused:
                assert current.result is previous.result
                assert current.result.week_start < current.scored_week.start
            else:
                assert current.result.week_start == current.scored_week.start


def test_full_retrain_has_no_parent_and_warm_update_uses_exact_previous_model():
    full_trainer = FakeTrainer()
    run(protocol(), full_trainer)
    assert all(item.parent_digest is None for item in full_trainer.requests)

    warm = protocol(warm=True)
    warm_trainer = FakeTrainer()
    initial = {family: f"initial-{family.value}" for family in FAMILIES}
    result = run(
        warm,
        warm_trainer,
        initial_parent_digests=initial,
    )
    for family in FAMILIES:
        requests = [item for item in warm_trainer.requests if item.family is family]
        releases = [item for item in result.releases if item.family is family]
        assert requests[0].parent_digest == initial[family]
        for request, prior in zip(requests[1:], releases):
            assert request.parent_digest == prior.result.model_digest


def test_future_rows_are_rejected_before_trainer_is_called():
    p = protocol()
    trainer = FakeTrainer()

    def future_rows(week):
        event = week.fit_start + timedelta(days=1)
        return (
            AsOfRow(
                "future",
                event,
                week.cutoff + timedelta(microseconds=1),
                event + timedelta(days=1),
                "sha256:x",
            ),
        )

    result = run(p, trainer, rows=future_rows)

    assert trainer.requests == []
    assert all(item.status is TrainingStatus.FAILED for item in result.dispositions)
    assert all("unavailable" in item.reason for item in result.dispositions)


def test_family_comparison_rejects_different_cost_or_data_identity():
    trainer = FakeTrainer()
    result = run(protocol(), trainer)
    first_week = result.releases[: len(FAMILIES)]
    altered_cost = replace(
        first_week[-1],
        result=replace(first_week[-1].result, cost_identity="other-cost"),
    )
    with pytest.raises(TrainingContractError, match="cost_identity"):
        require_matched_releases(first_week[:-1] + (altered_cost,))

    altered_data = replace(first_week[-1], asof_data_identity="other-data")
    with pytest.raises(TrainingContractError, match="asof_data_identity"):
        require_matched_releases(first_week[:-1] + (altered_data,))


def test_callback_failure_has_explicit_disposition_and_no_substitute():
    failed_once = False

    def fail_one(request, result):
        nonlocal failed_once
        if request.family is TaskFamily.SAC and not failed_once:
            failed_once = True
            raise RuntimeError("synthetic trainer failure")
        return result

    trainer = FakeTrainer(fail_one)
    warm = protocol(warm=True)
    result = run(
        warm,
        trainer,
        initial_parent_digests={
            family: f"initial-{family.value}" for family in FAMILIES
        },
    )
    failures = [
        item for item in result.dispositions if item.status is TrainingStatus.FAILED
    ]

    assert failures
    assert all(item.family is TaskFamily.SAC for item in failures)
    assert "synthetic trainer failure" in failures[0].reason
    assert failures[1:]
    assert all("warm lineage" in item.reason for item in failures[1:])
    assert not any(item.family is TaskFamily.SAC for item in result.releases)


def test_backend_mismatch_and_static_mode_are_not_silent_fallbacks():
    def cpu_fallback(request, result):
        return replace(result, execution_backend="CPU")

    result = run(protocol(), FakeTrainer(cpu_fallback))
    assert result.releases == ()
    assert all("silent fallback refused" in item.reason for item in result.dispositions)

    static = replace(
        protocol(),
        evaluation_mode=EvaluationMode.LITERATURE_STATIC,
        update_mode=None,
    )
    with pytest.raises(TrainingContractError, match="static protocol"):
        run(static, FakeTrainer())


def test_test_rows_are_not_consumed_before_firewall_seals_procedure():
    p = protocol()
    rows = Rows()
    with pytest.raises(FirewallError, match="sealed"):
        run(
            p,
            FakeTrainer(),
            rows=rows,
            split=EvaluationSplit.TEST,
        )
    assert rows.calls == []

    sealed = firewall(p).seal_procedure(
        selected_candidate_id="candidate-1",
        procedure={"digest": p.procedure_digest},
        validation_summary={"weeks": 6},
    )
    result = run(
        p,
        FakeTrainer(),
        rows=rows,
        split=EvaluationSplit.TEST,
        firewall=sealed,
    )
    assert rows.calls
    assert result.firewall.phase.value == "TEST_TRAVERSAL"
