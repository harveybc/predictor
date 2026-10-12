"""Executable contract for the I6N local-island optimizer.

These tests intentionally describe a successor to the shared-only I6N path.
They use stub fold receipts: no data, TensorFlow, GPU, TEST rows, or DOIN
services are touched.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import pytest

from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_search as search


FOLDS = ("2023Q1_DCA", "2023Q2_DCA", "2023Q3_DCA", "2023Q4")
SEED = 6102026


def _support(index: int) -> dict:
    """Return distinct train/inner support for one quarterly fold."""
    return {
        "counts": {"train": 100 + index, "inner": 20 + index},
        "naive_sums": {"train": 10.0, "inner": 4.0},
        "support_digest": {
            "train": format(index + 1, "x") * 64,
            "inner": format(index + 6, "x") * 64,
        },
    }


def _receipt(request: dict, ratio: float) -> dict:
    """Build a candidate-bound receipt whose train and inner ratios agree."""
    support = copy.deepcopy(request["support"])
    genome = {key: request["config"][key] for key in search._GENOME_KEYS}
    support.update(
        status="ok",
        error_sums={"train": 10.0 * ratio, "inner": 4.0 * ratio},
        metadata={
            "evidence_kind": "stub",
            "candidate_receipt": {
                "genotype_canonical_sha256": search.effective_identity(genome),
                "fold": request["context"]["fold"],
                "horizon_hours": request["config"]["horizon"],
            },
        },
    )
    return support


class RecordingEvaluator:
    """Deterministic four-fold evaluator with generation-dependent ratios."""

    def __init__(self, ratio_by_generation=None):
        self.calls = []
        self._ratio_by_generation = ratio_by_generation or (lambda generation: 0.8)

    def __call__(self, request):
        self.calls.append(copy.deepcopy(request))
        return _receipt(request, self._ratio_by_generation(request["generation"]))


class EventRecorder:
    """Capture the existing DOIN optimizer callback surface."""

    def __init__(self):
        self.candidates = []
        self.generations = []

    def on_candidate_evaluated(self, event):
        self.candidates.append(copy.deepcopy(event))

    def on_generation_end(self, population, hall_of_fame, hyper_keys, generation,
                          stage_info, stats):
        self.generations.append(
            {
                "generation": generation,
                "stage_info": copy.deepcopy(stage_info),
                "stats": copy.deepcopy(stats),
            }
        )


def _config(evaluator, events=None, *, horizon=4, seed=SEED, **changes):
    cfg = {
        "i6n_island_mode": True,
        "i6n_horizon": horizon,
        "i6n_folds": {
            fold: {
                "i6n_support": _support(index),
                "i6n_evaluation_context": {"fold": fold, "fixture": "island-contract"},
            }
            for index, fold in enumerate(FOLDS)
        },
        "i6n_evaluator": evaluator,
        "i6n_admitted_classes": ["S1", "S2", "S3"],
        "population_size": 20,
        "n_generations": 100,
        "optimization_patience": 10,
        "min_relative_improvement": 1e-8,
        "seed": seed,
        "higher_is_better": False,
    }
    if events is not None:
        cfg["optimization_callbacks"] = {
            "on_candidate_evaluated": events.on_candidate_evaluated,
            "on_generation_end": events.on_generation_end,
        }
    cfg.update(changes)
    return cfg


def _optimize(config):
    """The island path evaluates receipts; model/preprocessor are not fixtures."""
    plugin = Plugin()
    result = plugin.optimize(None, None, config)
    return plugin, result


@pytest.mark.parametrize("flag", [None, False])
def test_island_mode_requires_explicit_true_flag(flag):
    cfg = _config(RecordingEvaluator())
    if flag is None:
        cfg.pop("i6n_island_mode")
    else:
        cfg["i6n_island_mode"] = flag
    with pytest.raises(ValueError, match="i6n_island_mode"):
        _optimize(cfg)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"population_size": 19}, "population_size"),
        ({"n_generations": 99}, "n_generations"),
        ({"optimization_patience": 9}, "optimization_patience"),
        ({"min_relative_improvement": 1e-4}, "min_relative_improvement"),
        ({"min_relative_improvement": 1e-9}, "min_relative_improvement"),
    ],
)
def test_island_admission_rejects_weaker_or_different_search_settings(change, message):
    def must_not_run(_request):
        raise AssertionError("configuration must be rejected before evaluation")

    with pytest.raises(ValueError, match=message):
        _optimize(_config(must_not_run, **change))


def test_runtime_checkpoint_path_is_unique_per_island_without_config_divergence():
    template = "/state/h{horizon}/{island_id}/checkpoint.json"
    common = _config(RecordingEvaluator(), optimization_resume_file=template)
    first = Plugin()._island_settings(dict(common, i6n_island_id="gamma-5090"))
    second = Plugin()._island_settings(dict(common, i6n_island_id="gamma-5070ti"))

    assert Plugin._runtime_path(template, first) == "/state/h4/gamma-5090/checkpoint.json"
    assert Plugin._runtime_path(template, second) == "/state/h4/gamma-5070ti/checkpoint.json"
    assert common["optimization_resume_file"] == template


@pytest.mark.parametrize("island_id", ["../escape", "bad/island", ""])
def test_runtime_checkpoint_identity_rejects_unsafe_island_ids(island_id):
    with pytest.raises(ValueError, match="i6n_island_id"):
        Plugin()._island_settings(_config(RecordingEvaluator(), i6n_island_id=island_id))


@pytest.mark.parametrize("horizon", [1, 2, 3, 5, 6, 12, 24, 48, 96, 120])
def test_island_sequence_rejects_every_horizon_except_4_and_72(horizon):
    with pytest.raises(ValueError, match="i6n_horizon"):
        _optimize(_config(RecordingEvaluator(), horizon=horizon))


@pytest.mark.parametrize("horizon", [4, 72])
def test_island_sequence_accepts_only_governing_horizons(horizon):
    evaluator = RecordingEvaluator()
    events = EventRecorder()
    plugin, result = _optimize(_config(evaluator, events, horizon=horizon))
    assert result
    assert plugin.best_fitness_so_far == pytest.approx(-0.2, abs=1e-15)


def test_each_candidate_is_evaluated_on_all_four_distinct_folds():
    evaluator = RecordingEvaluator()
    events = EventRecorder()
    _optimize(_config(evaluator, events))

    calls_by_candidate = defaultdict(list)
    for request in evaluator.calls:
        genome = {key: request["config"][key] for key in search._GENOME_KEYS}
        identity = search.effective_identity(genome)
        calls_by_candidate[(request["generation"], identity)].append(
            request["context"]["fold"]
        )

    assert calls_by_candidate
    assert all(folds == list(FOLDS) for folds in calls_by_candidate.values())
    assert len(events.candidates) == len(calls_by_candidate)


def test_reported_fitness_is_ratio_minus_one_and_negative_is_better():
    evaluator = RecordingEvaluator(lambda _generation: 0.8)
    events = EventRecorder()
    plugin, _result = _optimize(_config(evaluator, events))

    assert plugin.best_fitness_so_far == pytest.approx(0.8 - 1.0, abs=1e-15)
    assert events.candidates
    assert all(event["fitness"] == pytest.approx(-0.2, abs=1e-15)
               for event in events.candidates)
    assert all(event["higher_is_better"] is False for event in events.candidates)


def test_same_seed_replays_the_same_local_island():
    first_evaluator = RecordingEvaluator()
    second_evaluator = RecordingEvaluator()
    first_plugin, first_result = _optimize(_config(first_evaluator, seed=SEED))
    second_plugin, second_result = _optimize(_config(second_evaluator, seed=SEED))

    def evaluated_sequence(evaluator):
        sequence = []
        for request in evaluator.calls:
            if request["context"]["fold"] != FOLDS[0]:
                continue
            genome = {key: request["config"][key] for key in search._GENOME_KEYS}
            sequence.append((request["generation"], search.effective_identity(genome)))
        return sequence

    assert evaluated_sequence(first_evaluator) == evaluated_sequence(second_evaluator)
    assert first_result == second_result
    assert first_plugin.best_fitness_so_far == second_plugin.best_fitness_so_far


def test_patience_counts_generations_without_more_than_the_exact_relative_increment():
    # Generations 1..4 do not improve. Generation 5 improves by 2e-8 and
    # resets patience. The search still runs through generation 29 because all
    # three progressive capacity classes must have opened before patience may
    # terminate the island.
    evaluator = RecordingEvaluator(
        lambda generation: 1.0 if generation < 5 else 1.0 - 2e-8
    )
    events = EventRecorder()
    _optimize(_config(evaluator, events))

    assert [event["generation"] for event in events.generations] == list(range(30))
    final = events.generations[-1]["stage_info"]
    assert final["converged"] is True
    assert final["stop_reason"] == "patience"
    assert final["no_improve_count"] == 24


def test_improvement_equal_to_threshold_does_not_reset_patience():
    # The contract says improvement must be strictly greater than 1e-8.
    evaluator = RecordingEvaluator(
        lambda generation: 1.0 if generation == 0 else 1.0 - 1e-8
    )
    events = EventRecorder()
    _optimize(_config(evaluator, events))

    assert [event["generation"] for event in events.generations] == list(range(30))
    assert events.generations[-1]["stage_info"]["stop_reason"] == "patience"

