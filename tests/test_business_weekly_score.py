"""Focused tests for the weekly paired-forecast scoring boundary."""

from __future__ import annotations

import json
import math
import unittest
from datetime import datetime, timedelta, timezone

from tools.business_objective_firewall import BusinessObjectiveFirewall, FirewallError, Phase
from tools.business_weekly_protocol import DispositionStatus, EvaluationSplit, WeekSpec
from tools.business_weekly_score import (
    NOT_CONFIGURED,
    PRIMARY_ERROR_METRIC_FIELD,
    ForecastFamily,
    HorizonForecast,
    ScoreRefusal,
    ScoringContract,
    WeekRelease,
    WeeklyScoreLedger,
    repository_primary_error_metric,
    score_week,
)
from tools.business_weekly_training import score_forecast_release


UTC = timezone.utc


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
        calls: list[str] = []
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),
            horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (3.0, 3.0)),
        )
        scored = score_week(release, contract, strategy=lambda item: calls.append(item.digest))
        self.assertEqual(calls and len(calls), 1)
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
        calls: list[object] = []
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),
            horizon(ForecastFamily.LONG, "h1", (1.0, 1.0), (2.0, 2.0)),
        )
        scored = score_forecast_release(release, contract, strategy=calls.append)
        self.assertEqual(calls, [])
        self.assertFalse(scored.strategy_eligible)
        self.assertIn(NOT_CONFIGURED, scored.reason or "")

    def test_tie_or_loss_blocks_the_strategy_callback(self) -> None:
        contract = ScoringContract("mse")
        calls: list[object] = []
        release = release_for(
            horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),
            horizon(ForecastFamily.LONG, "h1", (2.0, 2.0), (2.0, 2.0)),
        )
        scored = score_week(release, contract, strategy=calls.append)
        self.assertEqual(calls, [])
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
        calls: list[object] = []
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            True,
        )
        scored = score_week(release, ScoringContract("mae"), strategy=calls.append)
        self.assertEqual(scored.disposition, "INCOMPLETE")
        self.assertFalse(scored.strategy_eligible)
        self.assertEqual(calls, [])
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

    def test_restart_does_not_rescore_a_completed_week(self) -> None:
        contract = ScoringContract("mae", (ForecastFamily.SHORT,), ())
        ledger = WeeklyScoreLedger((week(0),), contract)
        release = WeekRelease(
            week(0),
            (horizon(ForecastFamily.SHORT, "h1", (1.0, 1.0), (2.0, 2.0)),),
            False,
        )
        calls: list[object] = []
        first = ledger.submit(release, strategy=calls.append)
        self.assertEqual(ledger.score_events, 1)
        self.assertEqual(len(calls), 1)
        restored = WeeklyScoreLedger.from_json(ledger.to_json())
        again = restored.submit(release, strategy=calls.append)
        self.assertEqual(again.digest, first.digest)
        self.assertEqual(restored.score_events, 1)
        self.assertEqual(len(calls), 1)

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
        from tools.business_weekly_score import financial_task_contract

        self.assertEqual(repository_primary_error_metric(), NOT_CONFIGURED)
        contract = financial_task_contract(required_horizons=("h1",))
        self.assertEqual(contract.primary_error_metric, "mae")
        self.assertEqual(contract.task, "financial_forecast")
        self.assertEqual(contract.scale_policy, "declared_target_scale")
        self.assertNotEqual(contract.primary_error_metric, repository_primary_error_metric())

        def pair(prediction: tuple[float, float], naive: tuple[float, float]) -> WeekRelease:
            return release_for(
                horizon(ForecastFamily.SHORT, "h1", prediction, naive, scale="price_level"),
                horizon(ForecastFamily.LONG, "h1", prediction, naive, scale="price_level"),
            )

        calls: list[object] = []
        better = score_week(pair((0.0, 5.0), (3.0, 3.0)), contract, strategy=calls.append)
        self.assertEqual(len(calls), 1)
        self.assertTrue(better.strategy_eligible)
        self.assertEqual(better.primary_error_metric, "mae")
        short = better.horizons[0]
        self.assertEqual(short.scale, "price_level")
        self.assertEqual(short.model_mae, 2.5)
        self.assertEqual(short.naive_mae, 3.0)
        self.assertGreater(short.model_mse, short.naive_mse)
        calls.clear()
        worse = score_week(pair((5.0, 5.0), (0.0, 9.0)), contract, strategy=calls.append)
        self.assertEqual(calls, [])
        self.assertFalse(worse.strategy_eligible)
        self.assertGreater(worse.horizons[0].model_mae, worse.horizons[0].naive_mae)
        self.assertLess(worse.horizons[0].model_mse, worse.horizons[0].naive_mse)


if __name__ == "__main__":
    unittest.main()
