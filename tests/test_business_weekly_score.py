"""Focused tests for the weekly paired-forecast scoring boundary."""

from __future__ import annotations

import json
import hashlib
import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from tools.business_objective_firewall import BusinessObjectiveFirewall, FirewallError, Phase
from tools.business_weekly_protocol import DispositionStatus, EvaluationSplit, WeekSpec
from tools.business_weekly_score import (
    FINANCIAL_TASK,
    NOT_CONFIGURED,
    PRIMARY_ERROR_METRIC_FIELD,
    ForecastFamily,
    HorizonForecast,
    ScoreRefusal,
    ScoringContract,
    WeekRelease,
    WeeklyScoreLedger,
    financial_task_contract,
    repository_primary_error_metric,
    score_week,
    score_weekly_forecast_release,
)
from tools.business_weekly_training import score_forecast_release


UTC = timezone.utc


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def reseal(payload: dict) -> None:
    for score in payload["scores"]:
        score["digest"] = digest({key: value for key, value in score.items() if key != "digest"})
    payload["ledger_digest"] = digest(
        {key: value for key, value in payload.items() if key != "ledger_digest"}
    )


class Bomb(tuple):
    """Row payload that fails if scoring reads it."""

    def __iter__(self):
        raise AssertionError("rows accessed")


def week(ordinal: int, split: EvaluationSplit = EvaluationSplit.VALIDATION) -> WeekSpec:
    start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=7 * ordinal)
    return WeekSpec(split, ordinal, start, start + timedelta(days=7), start, None, False)


def horizon(
    family: ForecastFamily,
    name: str,
    prediction: tuple[float, ...],
    naive: tuple[float, ...],
    *,
    target: tuple[float, ...] = (0.0, 0.0),
    origins: tuple[str, ...] = ("o1", "o2"),
    unit: str = "log_return",
    scale: str = "raw",
    model_digest: str = "model-a",
    scored_week: WeekSpec | None = None,
) -> HorizonForecast:
    return HorizonForecast(
        family=family,
        horizon=name,
        origins=origins,
        target=target,
        prediction=prediction,
        naive=naive,
        unit=unit,
        scale=scale,
        model_digest=model_digest,
        dataset_id="synthetic-week",
        week=scored_week or week(0),
    )


def release_for(*forecasts: HorizonForecast, scored: WeekSpec | None = None, payload=None) -> WeekRelease:
    return WeekRelease(scored or week(0), forecasts, True, payload)


def identified(
    family: ForecastFamily,
    name: str,
    prediction: tuple[float, ...],
    naive: tuple[float, ...],
    *,
    model_digest: str,
    task: str = "eurusd_hourly",
    population: str = "validation-week-0",
    origins: tuple[str, ...] = ("o1", "o2"),
    scale: str = "raw",
    unit: str = "log_return",
) -> HorizonForecast:
    return HorizonForecast(
        family=family,
        horizon=name,
        origins=origins,
        target=(0.0, 0.0),
        prediction=prediction,
        naive=naive,
        unit=unit,
        scale=scale,
        model_digest=model_digest,
        dataset_id="synthetic-week",
        week=week(0),
        task=task,
        population=population,
    )


class WeeklyScoreTests(unittest.TestCase):
    def test_repository_primary_metric_stays_unconfigured(self) -> None:
        self.assertEqual(repository_primary_error_metric(), NOT_CONFIGURED)
        self.assertEqual(PRIMARY_ERROR_METRIC_FIELD, "primary_error_metric")

    def test_better_than_naive_is_eligible_with_exact_metrics(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT, ForecastFamily.LONG), ("h1",))
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),
            horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (3.0, 3.0)),
        )
        scored = score_week(release, contract)
        short, long = scored.horizons
        self.assertEqual(short.sample_count, 2)
        self.assertEqual(short.model_mae, 1.0)
        self.assertEqual(short.naive_mae, 2.0)
        self.assertEqual(short.model_mse, 1.0)
        self.assertEqual(short.naive_mse, 4.0)
        self.assertEqual(short.paired_differences[0].absolute_error_difference, -1.0)
        self.assertEqual(short.skill_mae, 0.5)
        self.assertTrue(scored.strategy_eligible)
        self.assertEqual(long.naive_mae, 3.0)

    def test_unconfigured_primary_beats_naive_and_stays_closed(self) -> None:
        contract = ScoringContract(repository_primary_error_metric())
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),
            horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0)),
        )
        scored = score_forecast_release(release, contract)
        self.assertFalse(scored.strategy_eligible)
        self.assertIn(NOT_CONFIGURED, scored.reason or "")

    def test_tie_or_loss_blocks_the_strategy_callback(self) -> None:
        contract = ScoringContract("mse")
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),
            horizon(ForecastFamily.LONG, "h1", (2.0, 2.0), (2.0, 2.0)),
        )
        scored = score_week(release, contract)
        self.assertFalse(scored.strategy_eligible)

    def test_duplicate_reordered_or_changed_origin_rejects_before_metrics(self) -> None:
        contract = ScoringContract("mae")
        duplicate = horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), origins=("o1", "o1"))
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release_for(duplicate, horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), origins=("o1", "o1"))), contract)
        self.assertEqual(caught.exception.code, "DUPLICATE_ORIGIN")
        reordered = horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), origins=("o2", "o1"))
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release_for(reordered, horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), origins=("o2", "o1"))), contract)
        self.assertEqual(caught.exception.code, "REORDERED_ORIGIN")
        changed = horizon(
            ForecastFamily.SHORT,
            "h1",
            (1.0, 1.0),
            (2.0, 2.0),
        )
        changed = HorizonForecast(
            changed.family,
            changed.horizon,
            changed.origins,
            changed.target,
            changed.prediction,
            changed.naive,
            changed.unit,
            changed.scale,
            changed.model_digest,
            changed.dataset_id,
            changed.week,
            prediction_origins=("o1", "o9"),
        )
        solo = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(WeekRelease(week(0), (changed,), False), solo)
        self.assertEqual(caught.exception.code, "ORIGIN_CHANGED")
        missing = HorizonForecast(
            changed.family,
            changed.horizon,
            changed.origins,
            changed.target,
            changed.prediction,
            changed.naive,
            changed.unit,
            changed.scale,
            changed.model_digest,
            changed.dataset_id,
            changed.week,
            target_origins=("o1",),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(WeekRelease(week(0), (missing,), False), solo)
        self.assertEqual(caught.exception.code, "MISSING_ORIGIN")

    def test_length_mismatch_rejects(self) -> None:
        broken = horizon(ForecastFamily.SHORT, "h1", (1.0,), (2.0, 2.0))
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(
                WeekRelease(week(0), (broken,), consumes_short_and_long=False),
                ScoringContract("mae", (ForecastFamily.SHORT,), ()),
            )
        self.assertEqual(caught.exception.code, "LENGTH_MISMATCH")
        self.assertIn("prediction=1", caught.exception.detail)

    def test_non_finite_names_field_and_origin(self) -> None:
        broken = horizon(ForecastFamily.SHORT, "h1", (1.0, math.inf), (2.0, 2.0))
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(
                WeekRelease(week(0), (broken,), consumes_short_and_long=False),
                ScoringContract("mae", (ForecastFamily.SHORT,), ()),
            )
        self.assertEqual(caught.exception.code, "NON_FINITE")
        self.assertIn("prediction", caught.exception.detail)
        self.assertIn("o2", caught.exception.detail)

    def test_identity_mismatch_rejects_before_aggregation(self) -> None:
        contract = ScoringContract("mae")
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), unit="price"),
            horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), unit="log_return"),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release, contract)
        self.assertEqual(caught.exception.code, "IDENTITY_MISMATCH")

    def test_missing_family_is_an_incomplete_ineligible_week(self) -> None:
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            True,
        )
        scored = score_week(release, ScoringContract("mae"))
        self.assertEqual(scored.disposition, "INCOMPLETE")
        self.assertFalse(scored.strategy_eligible)
        self.assertEqual(scored.horizons, ())

    def test_failed_week_is_kept_and_a_reduced_denominator_is_refused(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        ledger = WeeklyScoreLedger((week(0), week(1)), contract)
        ledger.record_terminal(week(0), DispositionStatus.EXCLUDED, "source gap")
        ledger.record_terminal(week(1), DispositionStatus.FAILED, "fit failed")
        with self.assertRaises(ScoreRefusal) as caught:
            ledger.close(proposed_denominator=1)
        self.assertEqual(caught.exception.code, "REDUCED_DENOMINATOR")
        closed = ledger.close(proposed_denominator=2)
        self.assertEqual(closed.denominator, 2)
        self.assertEqual(closed.longitudinal_evaluation_units, 2)
        self.assertEqual(closed.independence_status, "NOT_ESTABLISHED")
        self.assertEqual(closed.dispositions[0].disposition, "EXCLUDED")
        self.assertEqual(closed.dispositions[1].disposition, "FAILED")
        self.assertEqual(closed.eligible_weeks, 0)

    def test_validation_firewall_rejects_a_test_payload_before_rows(self) -> None:
        firewall = BusinessObjectiveFirewall.create(("2025-W01",))
        self.assertIs(firewall.phase, Phase.VALIDATION_SELECTION)
        forecast = HorizonForecast(
            ForecastFamily.SHORT,
            "h1",
            ("o1", "o2"),
            Bomb((0.0, 0.0)),
            Bomb((1.0, 1.0)),
            Bomb((2.0, 2.0)),
            "log_return",
            "raw",
            "model-a",
            "synthetic-week",
            week(0),
        )
        release = WeekRelease(week(0), (forecast,), False, {"test_labels": [1.0, 2.0]})
        with self.assertRaises(FirewallError):
            score_week(release, ScoringContract("mae", (ForecastFamily.SHORT,), ()), firewall=firewall)

    def test_restart_does_not_rescore_but_retries_idempotent_delivery(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        calls: list[object] = []

        def deliver(score):
            calls.append(score)
            return score.digest

        first = ledger.submit(release, strategy=deliver)
        self.assertEqual(ledger.score_events, 1)
        self.assertEqual(len(calls), 1)
        restored = WeeklyScoreLedger.from_json(ledger.to_json())
        again = restored.submit(release, strategy=deliver)
        self.assertEqual(again.digest, first.digest)
        self.assertEqual(restored.score_events, 1)
        self.assertEqual(len(calls), 2)

    def test_overlapping_hours_do_not_increase_the_week_denominator(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        ledger = WeeklyScoreLedger((week(0), week(1)), contract)
        for ordinal in (0, 1):
            scored_week = week(ordinal)
            release = WeekRelease(
                scored_week,
                (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), scored_week=scored_week),),
                False,
            )
            ledger.submit(release)
        closed = ledger.close()
        self.assertEqual(closed.denominator, 2)
        self.assertEqual(closed.origin_rows, 4)
        self.assertEqual(closed.longitudinal_evaluation_units, closed.denominator)
        self.assertEqual(closed.independence_status, "NOT_ESTABLISHED")

    def test_distinct_short_and_long_model_digests_are_accepted_when_the_week_matches(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT, ForecastFamily.LONG), ("h1",))
        release = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="digest-a"),
            identified(ForecastFamily.LONG, "h1", (1.0, 1.0), (3.0, 3.0), model_digest="digest-b"),
        )
        scored = score_week(release, contract)
        short, long = scored.horizons
        self.assertEqual(short.model_digest, "digest-a")
        self.assertEqual(long.model_digest, "digest-b")
        self.assertNotEqual(short.release_digest, long.release_digest)
        self.assertEqual(scored.split, EvaluationSplit.VALIDATION.value)
        self.assertEqual(scored.week_start, "2024-01-01T00:00:00Z")
        self.assertEqual(release.week.cutoff, release.week.start)
        shifted = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="digest-a"),
            identified(
                ForecastFamily.LONG,
                "h1",
                (1.0, 1.0),
                (3.0, 3.0),
                model_digest="digest-b",
                population="other-population",
            ),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(shifted, contract)
        self.assertEqual(caught.exception.code, "IDENTITY_MISMATCH")
        retasked = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="digest-a"),
            identified(
                ForecastFamily.LONG,
                "h1",
                (1.0, 1.0),
                (3.0, 3.0),
                model_digest="digest-b",
                task="other-task",
            ),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(retasked, contract)
        self.assertEqual(caught.exception.code, "IDENTITY_MISMATCH")

    def test_model_digest_change_inside_the_same_forecast_is_rejected(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        changed = WeekRelease(
            week(0),
            (
                identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="digest-a"),
                identified(ForecastFamily.SHORT, "h2", (1.0, 1.0), (2.0, 2.0), model_digest="digest-b"),
            ),
            False,
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(changed, contract)
        self.assertEqual(caught.exception.code, "DIGEST_CHANGED")

    def test_contradictory_reuse_of_the_same_family_and_horizon_is_rejected(self) -> None:
        contract = ScoringContract("mae")
        release = release_for(
            identified(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="digest-b"),
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="digest-a"),
            identified(ForecastFamily.SHORT, "h1", (0.5, 0.5), (2.0, 2.0), model_digest="digest-a"),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release, contract)
        self.assertEqual(caught.exception.code, "DUPLICATE_HORIZON")

    def test_serialized_paired_differences_remain_under_the_week_digest(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        scored = ledger.submit(release)
        original = scored.horizons[0].paired_differences
        self.assertEqual(len(original), 2)
        restored = WeeklyScoreLedger.from_json(ledger.to_json())
        restored_pairs = restored.close().dispositions[0].horizons[0].paired_differences
        self.assertEqual(restored_pairs, original)
        self.assertEqual(
            [(item.origin, item.absolute_error_difference) for item in restored_pairs],
            [("o1", -1.0), ("o2", -1.0)],
        )
        payload = json.loads(ledger.to_json())
        payload["scores"][0]["horizons"][0]["paired_differences"][0]["absolute_error_difference"] = 4.0
        with self.assertRaises(ScoreRefusal) as caught:
            WeeklyScoreLedger.from_json(json.dumps(payload))
        self.assertEqual(caught.exception.code, "DIGEST")

    def test_financial_task_contract_primary_error_is_mae_on_the_declared_scale(self) -> None:
        self.assertEqual(repository_primary_error_metric(), NOT_CONFIGURED)
        contract = financial_task_contract(required_horizons=("h1",))
        self.assertEqual(contract.primary_error_metric, "mae")
        self.assertEqual(contract.task, "financial_forecast")
        self.assertEqual(contract.scale_policy, "declared_target_scale")
        self.assertNotEqual(contract.primary_error_metric, repository_primary_error_metric())

        def pair(prediction: tuple[float, float], naive: tuple[float, float]) -> WeekRelease:
            return release_for(
                identified(ForecastFamily.SHORT, "h1", prediction, naive, scale="price_level", model_digest="short", task=FINANCIAL_TASK),
                identified(ForecastFamily.LONG, "h1", prediction, naive, scale="price_level", model_digest="long", task=FINANCIAL_TASK),
            )

        better = score_week(pair((0.0, 5.0), (3.0, 3.0)), contract)
        self.assertTrue(better.strategy_eligible)
        self.assertEqual(better.primary_error_metric, "mae")
        short = better.horizons[0]
        self.assertEqual(short.scale, "price_level")
        self.assertEqual(short.model_mae, 2.5)
        self.assertEqual(short.naive_mae, 3.0)
        self.assertGreater(short.model_mse, short.naive_mse)
        worse = score_week(pair((5.0, 5.0), (0.0, 9.0)), contract)
        self.assertFalse(worse.strategy_eligible)
        self.assertGreater(worse.horizons[0].model_mae, worse.horizons[0].naive_mae)
        self.assertLess(worse.horizons[0].model_mse, worse.horizons[0].naive_mse)

    def test_required_horizons_are_required_for_each_family(self) -> None:
        contract = ScoringContract(
            "mae",
            (ForecastFamily.SHORT, ForecastFamily.LONG),
            ("h1", "h2"),
        )
        release = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="short"),
            identified(ForecastFamily.LONG, "h2", (1.0, 1.0), (2.0, 2.0), model_digest="long"),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release, contract)
        self.assertEqual(caught.exception.code, "MISSING_HORIZON")
        self.assertIn("short:h2", caught.exception.detail)
        self.assertIn("long:h1", caught.exception.detail)

    def test_completed_week_rejects_a_contradictory_resubmission(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        ledger = WeeklyScoreLedger((week(0),), contract)
        first = WeekRelease(
            week(0),
            (identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="model-a"),),
            False,
        )
        ledger.submit(first)
        contradictory = WeekRelease(
            week(0),
            (identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="model-b"),),
            False,
        )
        with self.assertRaises(ScoreRefusal) as caught:
            ledger.submit(contradictory)
        self.assertEqual(caught.exception.code, "CONFLICTING_WEEK")
        self.assertEqual(ledger.score_events, 1)

    def test_financial_contract_rejects_another_task_before_scoring(self) -> None:
        contract = financial_task_contract(required_horizons=("h1",))
        release = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="short", task="other"),
            identified(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="long", task="other"),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release, contract)
        self.assertEqual(caught.exception.code, "TASK_MISMATCH")

    def test_public_weekly_entry_uses_the_financial_mae_contract_by_default(self) -> None:
        release = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="short", task=FINANCIAL_TASK),
            identified(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="long", task=FINANCIAL_TASK),
        )
        scored = score_weekly_forecast_release(release)
        self.assertEqual(scored.primary_error_metric, "mae")
        self.assertTrue(scored.strategy_eligible)

    def test_strategy_delivery_retries_failure_and_is_not_repeated_after_success(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        calls: list[str] = []

        def flaky(score):
            calls.append(score.digest)
            if len(calls) == 1:
                raise RuntimeError("delivery failed")
            return score.digest

        with self.assertRaises(RuntimeError):
            ledger.submit(release, strategy=flaky)
        restored = WeeklyScoreLedger.from_json(ledger.to_json())
        restored.submit(release, strategy=flaky)
        restored.submit(release, strategy=flaky)
        self.assertEqual(calls, [calls[0], calls[0]])

    def test_stateless_public_entry_rejects_eligible_strategy_delivery(self) -> None:
        release = release_for(
            identified(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="short", task=FINANCIAL_TASK),
            identified(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0), model_digest="long", task=FINANCIAL_TASK),
        )
        with self.assertRaises(ScoreRefusal) as caught:
            score_weekly_forecast_release(release, strategy=lambda score: None)
        self.assertEqual(caught.exception.code, "CALLBACK_REQUIRES_LEDGER")
        with self.assertRaises(ScoreRefusal) as caught:
            score_week(release, financial_task_contract(), strategy=lambda score: score.digest)
        self.assertEqual(caught.exception.code, "CALLBACK_REQUIRES_LEDGER")

    def test_delivery_state_is_covered_by_the_ledger_digest(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        ledger.submit(release)
        payload = json.loads(ledger.to_json())
        payload["strategy_deliveries"] = {"2024-01-01T00:00:00Z|validation": "DELIVERED"}
        reseal(payload)
        with self.assertRaises(ScoreRefusal) as caught:
            WeeklyScoreLedger.from_json(json.dumps(payload))
        self.assertEqual(caught.exception.code, "SCHEMA_FIELDS")

    def test_restore_rejects_non_boolean_eligibility_and_non_finite_metrics(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        ledger.submit(release)
        for mutate in (
            lambda payload: payload["scores"][0].__setitem__("strategy_eligible", "false"),
            lambda payload: payload["scores"][0]["horizons"][0].__setitem__("model_mae", math.nan),
        ):
            payload = json.loads(ledger.to_json())
            mutate(payload)
            reseal(payload)
            with self.assertRaises(ScoreRefusal) as caught:
                WeeklyScoreLedger.from_json(json.dumps(payload))
            self.assertEqual(caught.exception.code, "SCORE_SCHEMA")

    def test_terminal_dispositions_round_trip_without_horizons(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        for status in (DispositionStatus.FAILED, DispositionStatus.EXCLUDED):
            ledger = WeeklyScoreLedger((week(0),), contract)
            ledger.record_terminal(week(0), status, "retained terminal reason")
            restored = WeeklyScoreLedger.from_json(ledger.to_json())
            score = restored.close().dispositions[0]
            self.assertEqual(score.disposition, status.value)
            self.assertEqual(score.horizons, ())

    def test_restore_rejects_disposition_reason_skill_and_impossible_error_mutations(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        ledger.submit(release)

        def failed_with_metrics(payload):
            score = payload["scores"][0]
            score["disposition"] = "FAILED"
            score["strategy_eligible"] = False
            score["reason"] = "invented failure"

        mutations = (
            failed_with_metrics,
            lambda payload: payload["scores"][0].__setitem__("reason", "invented failure"),
            lambda payload: payload["scores"][0]["horizons"][0].__setitem__("skill_mae", True),
            lambda payload: payload["scores"][0]["horizons"][0].__setitem__("skill_mse", True),
            lambda payload: payload["scores"][0]["horizons"][0].__setitem__("model_mse", 0.1),
        )
        for mutate in mutations:
            payload = json.loads(ledger.to_json())
            mutate(payload)
            reseal(payload)
            with self.assertRaises(ScoreRefusal) as caught:
                WeeklyScoreLedger.from_json(json.dumps(payload))
            self.assertEqual(caught.exception.code, "SCORE_SCHEMA")

    def test_restore_rejects_duplicate_scores_and_metric_from_another_contract(self) -> None:
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        mae = WeeklyScoreLedger((week(0),), ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",)))
        mae.submit(release)
        duplicate = json.loads(mae.to_json())
        other = WeeklyScoreLedger((week(0),), ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",)))
        changed = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (0.5, 0.5), (2.0, 2.0)),),
            False,
        )
        other.submit(changed)
        duplicate["scores"].append(json.loads(other.to_json())["scores"][0])
        with self.assertRaises(ScoreRefusal) as caught:
            WeeklyScoreLedger.from_json(json.dumps(duplicate))
        self.assertEqual(caught.exception.code, "DUPLICATE_SCORE")

        mse = WeeklyScoreLedger((week(0),), ScoringContract("mse", (ForecastFamily.SHORT,), ("h1",)))
        mse.submit(release)
        mixed = json.loads(mae.to_json())
        mixed["scores"] = json.loads(mse.to_json())["scores"]
        with self.assertRaises(ScoreRefusal) as caught:
            WeeklyScoreLedger.from_json(json.dumps(mixed))
        self.assertEqual(caught.exception.code, "CONTRACT_MISMATCH")

        wrong_scope = json.loads(mae.to_json())
        wrong_scope["required_families"] = ["short", "long"]
        with self.assertRaises(ScoreRefusal) as caught:
            WeeklyScoreLedger.from_json(json.dumps(wrong_scope))
        self.assertEqual(caught.exception.code, "CONTRACT_MISMATCH")

    def test_completed_week_identity_includes_training_window_and_retrain_policy(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ("h1",))
        original_week = week(0)
        ledger = WeeklyScoreLedger((original_week,), contract)
        first = WeekRelease(
            original_week,
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), scored_week=original_week),),
            False,
        )
        ledger.submit(first)
        altered_week = replace(original_week, ordinal=99, fit_start=original_week.start - timedelta(days=365), retrain_due=True)
        altered = WeekRelease(
            altered_week,
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0), scored_week=altered_week),),
            False,
        )
        with self.assertRaises(ScoreRefusal) as caught:
            ledger.submit(altered)
        self.assertEqual(caught.exception.code, "WEEK_IDENTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()
