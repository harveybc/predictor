"""Executable contract for the I6N local-island optimizer.

These tests intentionally describe a successor to the shared-only I6N path.
They use stub fold receipts: no data, TensorFlow, GPU, TEST rows, or DOIN
services are touched.
"""

from __future__ import annotations

import copy
from collections import defaultdict
import json
import math
import random
import threading

import pytest

from optimizer_plugins.i6n_optimizer import Plugin
from tools.i6n_doin_sequence_config import _domain
from tools import i6n_search as search
from tools import i6n_search_space as space


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
                "rows": [
                    {
                        "split": split.upper(),
                        "head_offset_hours": request["config"]["horizon"],
                        "head_role": "ENDPOINT",
                        "comparison": True,
                        "support_sha256": support["support_digest"][split],
                    }
                    for split in ("train", "inner")
                ],
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


class RejectFirstCandidateEvaluator(RecordingEvaluator):
    """Reject every fold request for the first genotype and score the rest."""

    def __init__(self):
        super().__init__()
        self.rejected_identity = None

    def __call__(self, request):
        self.calls.append(copy.deepcopy(request))
        genome = {key: request["config"][key] for key in search._GENOME_KEYS}
        identity = search.effective_identity(genome)
        if self.rejected_identity is None:
            self.rejected_identity = identity
        if identity == self.rejected_identity:
            raise RuntimeError("physical attempt budget exhausted after interruption")
        return _receipt(request, 0.8)


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


def test_runtime_callbacks_with_thread_locks_are_preserved_without_deepcopy():
    class StatefulCallbacks:
        def __init__(self):
            self.lock = threading.Lock()

        def on_candidate_evaluated(self, _event):
            return None

    callback_owner = StatefulCallbacks()
    evaluator = RecordingEvaluator()
    callbacks = {"on_candidate_evaluated": callback_owner.on_candidate_evaluated}
    cfg = Plugin()._config({
        "higher_is_better": False,
        "nested": {"mutable": []},
        "i6n_evaluator": evaluator,
        "optimization_callbacks": callbacks,
        "_non_serializable_keys": {"optimization_callbacks"},
    })

    assert cfg["i6n_evaluator"] is evaluator
    assert cfg["optimization_callbacks"] is callbacks
    assert cfg["nested"] == {"mutable": []}
    assert "_non_serializable_keys" not in cfg


def test_runtime_marker_cannot_hide_scientific_configuration():
    with pytest.raises(ValueError, match="unsupported runtime keys"):
        Plugin()._config({
            "higher_is_better": False,
            "population_size": 20,
            "_non_serializable_keys": {"population_size"},
        })


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


def test_only_designated_island_seeds_fixed_controls():
    plugin = Plugin()
    common = _config(
        RecordingEvaluator(),
        i6n_control_island_id="omega-4070",
    )
    leader = plugin._island_settings(dict(common, i6n_island_id="omega-4070"))
    worker = plugin._island_settings(dict(common, i6n_island_id="dragon-4090"))
    admitted = {"S1", "S2", "S3"}

    leader_population = plugin._initial_island_population(
        leader, admitted, random.Random(11)
    )
    worker_population = plugin._initial_island_population(
        worker, admitted, random.Random(12)
    )
    fixed = {
        space.genotype_sha256(space.seed_genotype(4)),
        space.genotype_sha256(space.alternative_genotype(4)),
    }

    assert fixed <= {space.genotype_sha256(g) for g in leader_population}
    assert not fixed & {space.genotype_sha256(g) for g in worker_population}
    assert len(worker_population) == 20


def test_control_island_must_be_a_safe_nonempty_id():
    with pytest.raises(ValueError, match="i6n_control_island_id"):
        Plugin()._island_settings(_config(
            RecordingEvaluator(),
            i6n_island_id="omega-4070",
            i6n_control_island_id="bad/island",
        ))


def test_deployment_preserves_only_cost_admitted_capacity_classes():
    source = {"domains": [{
        "optimization_config": {
            "i6n_horizon": 4,
            "i6n_folds": {},
            "i6n_admitted_classes": ["S1", "S2"],
        }
    }]}

    generated = _domain(source, 4, "/predictor", "/python")

    assert generated["optimization_config"]["i6n_admitted_classes"] == ["S1", "S2"]
    assert generated["optimization_config"]["i6n_control_island_id"] == "omega-4070"


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


def test_candidate_event_preserves_archive_receipts_and_effective_identity():
    evaluator = RecordingEvaluator()
    events = EventRecorder()
    cfg = _config(evaluator, events, feature_manifest_sha256="a" * 64,
                  i6n_evaluator_identity="tests:recording-evaluator-v1")
    for binding in cfg["i6n_folds"].values():
        binding["i6n_origins_sha256"] = copy.deepcopy(
            binding["i6n_support"]["support_digest"]
        )

    _optimize(cfg)

    event = events.candidates[0]
    metrics = event["metrics"]
    assert set(metrics["receipts"]) == set(FOLDS)
    assert metrics["origins_binding"] == "VERIFIED"
    assert metrics["origins_sha256"] == {
        fold: cfg["i6n_folds"][fold]["i6n_origins_sha256"] for fold in FOLDS
    }
    assert metrics["genotype_canonical_sha256"] == event["genotype_canonical_sha256"]
    assert metrics["effective_config_sha256"] == event["effective_config_sha256"]
    assert metrics["contract_sha256"] == event["contract_sha256"]
    for fold, receipt in metrics["receipts"].items():
        binding = receipt["metadata"]["candidate_receipt"]
        assert binding["fold"] == fold
        assert binding["genotype_canonical_sha256"] == event["genotype_canonical_sha256"]


def test_checkpoint_identity_rejects_config_contract_feature_evaluator_and_class_mutations(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint.json"
    base = _config(
        RecordingEvaluator(),
        i6n_island_id="island-a",
        optimization_resume_file=str(checkpoint),
        feature_manifest_sha256="a" * 64,
        i6n_evaluator_identity="tests:recording-evaluator-v1",
    )
    _plugin, expected = _optimize(base)
    retained = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert retained["schema"] == "i6n.island.checkpoint.v2"
    assert retained["generation_complete"] is True
    assert retained["checkpoint_identity"]["plan_sha256"]
    assert retained["checkpoint_identity"]["config_sha256"]
    assert retained["checkpoint_identity"]["contract_sha256"]
    assert retained["checkpoint_identity"]["folds_sha256"]
    assert retained["checkpoint_identity"]["support_sha256"]
    assert retained["checkpoint_identity"]["features_sha256"]
    assert retained["checkpoint_identity"]["evaluator_sha256"]
    assert retained["checkpoint_identity"]["admitted_classes"] == ["S1", "S2", "S3"]

    same = copy.deepcopy(base)
    same_evaluator = RecordingEvaluator()
    same["i6n_evaluator"] = same_evaluator
    _plugin, replayed = _optimize(same)
    assert replayed == expected
    assert same_evaluator.calls == []

    mutations = []
    changed = copy.deepcopy(base)
    changed["feature_manifest_sha256"] = "b" * 64
    mutations.append(changed)
    changed = copy.deepcopy(base)
    changed["i6n_evaluator_identity"] = "tests:recording-evaluator-v2"
    mutations.append(changed)
    changed = copy.deepcopy(base)
    changed["i6n_admitted_classes"] = ["S1", "S2"]
    mutations.append(changed)
    changed = copy.deepcopy(base)
    changed["i6n_folds"][FOLDS[0]]["i6n_support"]["support_digest"]["train"] = "f" * 64
    mutations.append(changed)
    changed = copy.deepcopy(base)
    changed["i6n_evaluator_timeout"] = 999
    mutations.append(changed)

    for changed in mutations:
        changed["i6n_evaluator"] = RecordingEvaluator()
        with pytest.raises(ValueError, match="checkpoint identity mismatch"):
            _optimize(changed)


def test_checkpoint_never_claims_patience_before_minimum_generation(monkeypatch):
    checkpoints = []

    def capture(_path, payload):
        checkpoints.append(copy.deepcopy(payload))

    monkeypatch.setattr(Plugin, "_atomic_json", staticmethod(capture))
    _optimize(_config(
        RecordingEvaluator(),
        i6n_island_id="island-a",
        optimization_resume_file="/unused/{island_id}/checkpoint.json",
    ))

    assert checkpoints[-1]["generation"] == 30
    assert checkpoints[-1]["stop_reason"] == "patience"
    assert all(
        item["stop_reason"] is None
        for item in checkpoints
        if item["generation"] < 30
    )


def test_resume_from_preminimum_checkpoint_continues_instead_of_terminating(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "checkpoint.json"
    original = Plugin._atomic_json

    class SimulatedPowerLoss(RuntimeError):
        pass

    def interrupt_after_generation_eleven(path, payload):
        original(path, payload)
        if payload["generation"] == 11 and payload["generation_complete"]:
            raise SimulatedPowerLoss

    config = _config(
        RecordingEvaluator(),
        i6n_island_id="island-a",
        optimization_resume_file=str(checkpoint),
    )
    monkeypatch.setattr(
        Plugin, "_atomic_json", staticmethod(interrupt_after_generation_eleven)
    )
    with pytest.raises(SimulatedPowerLoss):
        _optimize(config)

    interrupted = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert interrupted["generation"] == 11
    assert interrupted["generation_complete"] is True
    assert interrupted["no_improve"] >= 10
    assert interrupted["stop_reason"] is None

    # Regression/mutation: the old writer persisted this premature reason.
    # A new reader must continue from the retained state, not return success.
    interrupted["stop_reason"] = "patience"
    checkpoint.write_text(json.dumps(interrupted), encoding="utf-8")

    monkeypatch.setattr(Plugin, "_atomic_json", staticmethod(original))
    events = EventRecorder()
    resumed = copy.deepcopy(config)
    resumed["i6n_evaluator"] = RecordingEvaluator()
    resumed["optimization_callbacks"] = {
        "on_candidate_evaluated": events.on_candidate_evaluated,
        "on_generation_end": events.on_generation_end,
    }
    _optimize(resumed)

    assert events.generations[0]["generation"] == 11
    assert events.generations[-1]["generation"] == 29
    assert events.generations[-1]["stage_info"]["stop_reason"] == "patience"


def test_candidate_checkpoint_prevents_refit_after_mid_generation_restart(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "checkpoint.json"
    original = Plugin._atomic_json

    class SimulatedPowerLoss(RuntimeError):
        pass

    def interrupt_after_first_candidate(path, payload):
        original(path, payload)
        if not payload["generation_complete"] and len(payload["cache"]) == 1:
            raise SimulatedPowerLoss

    config = _config(
        RecordingEvaluator(),
        i6n_island_id="island-a",
        optimization_resume_file=str(checkpoint),
    )
    monkeypatch.setattr(
        Plugin, "_atomic_json", staticmethod(interrupt_after_first_candidate)
    )
    with pytest.raises(SimulatedPowerLoss):
        _optimize(config)

    interrupted = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert interrupted["generation"] == 0
    assert interrupted["generation_complete"] is False
    retained_identity = next(iter(interrupted["cache"]))

    monkeypatch.setattr(Plugin, "_atomic_json", staticmethod(original))
    resumed_evaluator = RecordingEvaluator()
    resumed = copy.deepcopy(config)
    resumed["i6n_evaluator"] = resumed_evaluator
    _optimize(resumed)

    evaluated_identities = []
    for request in resumed_evaluator.calls:
        if request["context"]["fold"] != FOLDS[0]:
            continue
        genome = {key: request["config"][key] for key in search._GENOME_KEYS}
        evaluated_identities.append(search.effective_identity(genome))
    assert retained_identity not in evaluated_identities


def test_rejected_candidate_is_checkpointed_and_does_not_abort_island(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    evaluator = RejectFirstCandidateEvaluator()

    plugin, result = _optimize(_config(
        evaluator,
        i6n_island_id="island-a",
        optimization_resume_file=str(checkpoint),
    ))

    retained = json.loads(checkpoint.read_text(encoding="utf-8"))
    rejection = retained["rejections"][evaluator.rejected_identity]
    assert rejection["disposition"] == "FAILED"
    assert "physical attempt budget exhausted" in rejection["reason"]
    assert evaluator.rejected_identity not in retained["cache"]
    assert result is not None
    assert plugin.island_terminal()["status"] == "COMPLETE"


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


def test_any_representable_improvement_above_threshold_resets_patience():
    improvement = math.nextafter(1e-8, math.inf)
    evaluator = RecordingEvaluator(
        lambda generation: 1.0 if generation < 5 else 1.0 - improvement
    )
    events = EventRecorder()
    _optimize(_config(evaluator, events))

    generation_five = events.generations[5]["stage_info"]
    assert generation_five["relative_improvement"] > 1e-8
    assert generation_five["no_improve_count"] == 0


def test_completed_island_exposes_an_explicit_checkpoint_bound_terminal():
    plugin, _result = _optimize(_config(
        RecordingEvaluator(),
        i6n_island_id="gamma-5090",
    ))

    terminal = plugin.island_terminal()

    assert terminal["status"] == "COMPLETE"
    assert terminal["island_id"] == "gamma-5090"
    assert terminal["horizon"] == 4
    assert terminal["seed"] == SEED
    assert terminal["stop_reason"] == "patience"
    assert terminal["terminal_generation"] == 30
    assert len(terminal["champion_sha256"]) == 64
    assert len(terminal["plan_sha256"]) == 64
    assert len(terminal["contract_sha256"]) == 64
    assert len(terminal["checkpoint_identity_sha256"]) == 64
    assert math.isfinite(terminal["champion_fitness"])
    assert len(terminal["final_population_sha256"]) == 64
    assert len(terminal["candidate_ledger_sha256"]) == 64
    assert terminal["evaluated_candidate_count"] > 0

    terminal["status"] = "TAMPERED"
    assert plugin.island_terminal()["status"] == "COMPLETE"


def test_candidate_is_durably_reported_before_its_champion_is_broadcast():
    order = []
    config = _config(RecordingEvaluator())
    config["optimization_callbacks"] = {
        "on_candidate_evaluated": lambda event: order.append(
            ("candidate", event["genotype_canonical_sha256"])
        ),
        "on_new_champion": lambda params, *_args: order.append(
            ("champion", params["_i6n_genotype_sha256"])
        ),
    }

    _optimize(config)

    first_candidate = next(index for index, item in enumerate(order) if item[0] == "candidate")
    first_champion = next(index for index, item in enumerate(order) if item[0] == "champion")
    assert first_candidate < first_champion
    assert order[first_candidate][1] == order[first_champion][1]


def test_runtime_telemetry_uses_doin_canonical_candidate_fields():
    events = EventRecorder()
    plugin, _result = _optimize(_config(RecordingEvaluator(), events))

    terminal = plugin.island_terminal()
    runtime = plugin.get_runtime_statistics()
    event = events.candidates[-1]

    assert runtime["candidate_evaluations_total"] == terminal["evaluated_candidate_count"]
    assert runtime["candidate_history_source"].endswith(":local-test")
    assert runtime["candidates_per_hour"] > 0
    assert runtime["candidate_seconds_median"] > 0
    assert event["candidate_num"] == event["candidate_in_gen"]
    assert event["total_candidates"] == event["population_size"] == 20
    assert event["total_candidates_evaluated"] == event["total_eval"]

