"""Synthetic/stub component evidence only: no training or real measurements."""

import copy
import json
import random
import subprocess
import sys

import pytest
from deap import base

from tools import i6n_model
from tools.i6n_search import (
    HORIZONS, MUTATION_GROUPS, Search, ScoreRejected, default_genome,
    effective_config, effective_identity, mutate, normalize_genome,
    per_horizon_configs, sample_genome,
)


def support():
    return {
        "counts": {"train": 10, "inner": 5},
        "naive_sums": {"train": 20.0, "inner": 15.0},
        "support_digest": {"train": "a" * 64, "inner": "b" * 64},
    }


def stub_score(config):
    """Fabricated score for control-flow tests; NOT measured predictive quality."""
    return dict(support(), status="ok",
                error_sums={"train": 10.0, "inner": 6.0},
                metadata={"evidence_kind": "stub", "horizon": config["horizon"]})


def search(**kwargs):
    return Search(12, seed=17, budget=8, support=support(), **kwargs)


def test_config_import_and_sampling_are_tensorflow_free_and_rng_local():
    code = """
import random, sys
state = random.getstate()
from tools.i6n_search import per_horizon_configs
per_horizon_configs(seed=71, count=5)
assert state == random.getstate()
assert not any(k == 'tensorflow' or k.startswith('tensorflow.') for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("h", HORIZONS)
def test_window_and_output_contracts(h):
    rng = random.Random(120 + h)
    for _ in range(40):
        genome = sample_genome(h, rng)
        cfg = effective_config(genome)
        assert cfg["history_hours"] == max(24, cfg["multiplier"] * h)
        assert h % cfg["delta_x"] == cfg["history_hours"] % cfg["delta_x"] == 0
        assert cfg["samples"] == 1 + cfg["history_hours"] // cfg["delta_x"]
        assert cfg["input_offsets"][0] == -cfg["history_hours"]
        assert cfg["input_offsets"][-1] == 0
        assert cfg["output_offsets"][-1] == h
        assert cfg["core"] == i6n_model.normalize_config(cfg["core"])
        widths = [cfg["core"]["width"], *cfg["core"]["reducer_filters"]]
        assert widths == sorted(widths, reverse=True)
        steps = cfg["samples"]
        for stride in cfg["core"]["reducer_strides"]:
            steps = (steps + stride - 1) // stride
        assert cfg["final_steps"] == steps >= 2
        if cfg["output_mode"] == "endpoint":
            assert cfg["delta_y"] is None
            assert cfg["output_offsets"] == [h]
            assert cfg["core"]["n_horizons"] == 1
        else:
            assert h % cfg["delta_y"] == 0
            assert cfg["output_offsets"] == list(range(cfg["delta_y"], h + 1, cfg["delta_y"]))
            assert cfg["core"]["n_horizons"] == len(cfg["output_offsets"]) > 1


def test_canonical_inactive_genes_endpoint_alias_and_history_alias():
    genome = default_genome(2)
    alias = copy.deepcopy(genome)
    alias.update(multiplier=6, delta_y=2)
    alias["core"].update(transformer_blocks=2, adapter_width=64, n_horizons=99)
    assert normalize_genome(alias) == normalize_genome(genome)
    assert effective_identity(alias) == effective_identity(genome)
    alias.update(output_mode="trajectory", delta_y=2)
    assert effective_identity(alias) == effective_identity(genome)
    alias["delta_y"] = 1
    assert effective_identity(alias) != effective_identity(genome)


def test_input_and_output_resolution_are_independent():
    genome = default_genome(12)
    genome.update(delta_x=6, output_mode="trajectory", delta_y=3)
    assert effective_config(genome)["output_offsets"] == [3, 6, 9, 12]
    assert effective_config(genome)["samples"] == 5


@pytest.mark.parametrize("updates", [
    {"horizon": 7}, {"horizon": True}, {"horizon": 12.0},
    {"multiplier": 3}, {"multiplier": True}, {"delta_x": 5},
    {"delta_x": 0}, {"delta_x": 2.0}, {"delta_x": True},
    {"input_transform": "mystery"}, {"output_mode": "mystery"},
    {"output_mode": "trajectory", "delta_y": None},
    {"output_mode": "trajectory", "delta_y": 5},
    {"output_mode": "trajectory", "delta_y": True}, {"unknown": 1},
])
def test_reject_invalid_genes(updates):
    genome = default_genome(12)
    genome.update(updates)
    with pytest.raises(ValueError):
        normalize_genome(genome)


def test_reject_nondivisible_history_increasing_filters_and_collapsed_time():
    genome = default_genome(5)
    genome["delta_x"] = 5  # h divides, but 24-hour minimum history does not.
    with pytest.raises(ValueError):
        normalize_genome(genome)
    genome = default_genome(12)
    genome["core"]["reducer_filters"] = [4, 8]
    with pytest.raises(ValueError):
        normalize_genome(genome)
    genome = default_genome(12)
    genome["delta_x"] = 12
    genome["core"].update(reducer_filters=[8, 4, 4], reducer_strides=[2, 2, 2])
    with pytest.raises(ValueError):
        normalize_genome(genome)


def test_normalization_uses_real_core_validator(monkeypatch):
    real = i6n_model.normalize_config
    seen = []

    def spy(config):
        seen.append(copy.deepcopy(config))
        return real(config)

    monkeypatch.setattr(i6n_model, "normalize_config", spy)
    normalize_genome(default_genome(12))
    assert seen
    genome = default_genome(12)
    genome["core"]["unknown"] = 1
    with pytest.raises(ValueError):
        normalize_genome(genome)


@pytest.mark.parametrize("group", MUTATION_GROUPS)
def test_every_joint_mutation_group_changes_effective_identity(group):
    genome = default_genome(12)
    before = copy.deepcopy(genome)
    child = mutate(genome, random.Random(47), group=group)
    assert genome == before
    assert effective_identity(child) != effective_identity(genome)
    assert effective_config(child)["final_steps"] >= 2


def test_seeded_configs_are_detached_unique_and_per_horizon():
    assert per_horizon_configs() == per_horizon_configs(seed=6102026)
    first = per_horizon_configs(seed=91, count=6)
    assert first == per_horizon_configs(seed=91, count=6)
    assert first != per_horizon_configs(seed=92, count=6)
    assert tuple(first) == HORIZONS == (2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72)
    for h, configs in first.items():
        assert all(c["horizon"] == h for c in configs)
        assert len({json.dumps(c, sort_keys=True) for c in configs}) == 6


def test_deap_search_stub_evaluation_reproducible_bounded_and_no_global_rng():
    state = random.getstate()
    first, second = search(), search()
    results = first.run(stub_score)
    assert results == second.run(stub_score)
    assert random.getstate() == state
    assert len(results) == first.evaluations == first.proposed == 8
    assert len({r["identity"] for r in results}) == 8
    assert first.propose() == []
    assert isinstance(first.toolbox, base.Toolbox)
    assert all(isinstance(i.fitness, base.Fitness) and i.fitness.valid
               for i in first.population)
    assert all(r["fitness"] == pytest.approx(.45) for r in results)
    assert all(r["metrics"]["metadata"]["evidence_kind"] == "stub" for r in results)


def test_pending_proposals_reserve_budget_and_return_detached_data():
    engine = search()
    proposal = engine.propose()[0]
    pristine = copy.deepcopy(proposal)
    proposal["config"]["core"]["width"] = 123
    with pytest.raises(ValueError):
        engine.evaluate(proposal, stub_score)
    result = engine.evaluate(pristine, stub_score)
    result["metrics"]["counts"]["train"] = 999
    assert engine.results[0]["metrics"]["counts"]["train"] == 10
    with pytest.raises(ValueError):
        engine.evaluate(pristine, stub_score)
    assert len(engine.propose(100)) == 7
    assert engine.propose() == []


def test_duplicate_proposals_and_failures_have_finite_attempt_budget():
    engine = search(max_proposals=5)
    fixed = default_genome(12)
    engine.toolbox.register("candidate", lambda: copy.deepcopy(fixed))
    assert len(engine.propose(8)) == 1
    assert engine.attempts == 5
    assert engine.propose() == []
    other = search()

    def fail(config):
        raise RuntimeError("explicit stub failure")

    assert other.run(fail) == []
    assert other.evaluations == len(other.rejections) == 8
    assert not other.population


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("status", "timeout"), ("fitness", float("nan")),
    ("fitness", .01), ("counts", {"train": 9, "inner": 5}),
    ("counts", {"train": True, "inner": 5}),
    ("support_digest", {"train": "c" * 64, "inner": "b" * 64}),
    ("naive_sums", {"train": 21.0, "inner": 15.0}),
    ("error_sums", {"train": float("inf"), "inner": 6.0}),
    ("error_sums", {"train": -1.0, "inner": 6.0}),
    ("error_sums", {"train": True, "inner": 6.0}),
    ("error_sums", {"train": 10.0, "inner": 6.0, "test": 1.0}),
])
def test_mismatched_support_and_failed_scores_cannot_enter_population(field, value):
    engine = search()
    proposal = engine.propose()[0]

    def invalid(config):
        return dict(stub_score(config), **{field: value})

    with pytest.raises(ScoreRejected):
        engine.evaluate(proposal, invalid)
    assert engine.evaluations == 1
    assert not engine.population and not engine.results
    assert len(engine.rejections) == 1


def test_callback_cannot_mutate_reserved_candidate_or_frozen_support():
    frozen = support()
    engine = Search(12, seed=1, budget=1, support=frozen)
    frozen["counts"]["train"] = 100
    proposal = engine.propose()[0]
    before = copy.deepcopy(proposal)

    def callback(config):
        config["core"]["width"] = 128
        return stub_score(config)

    result = engine.evaluate(proposal, callback)
    assert result["config"] == before["config"]
    assert proposal == before


@pytest.mark.parametrize("kwargs", [
    {"budget": 0}, {"budget": True}, {"seed": True},
    {"population_size": 0}, {"max_proposals": 0},
    {"support": {}},
])
def test_invalid_search_contract_rejected(kwargs):
    params = dict(seed=1, budget=1, support=support())
    params.update(kwargs)
    with pytest.raises(ValueError):
        Search(12, **params)


@pytest.mark.parametrize("h", HORIZONS)
def test_mutation_chain_preserves_contract_and_does_not_change_global_rng(h):
    import numpy as np

    py_state, np_state = random.getstate(), np.random.get_state()
    rng = random.Random(312 + h)
    genome = sample_genome(h, rng)
    for group in MUTATION_GROUPS * 3:
        genome = mutate(genome, rng, group)
        assert genome == normalize_genome(genome)
        assert effective_config(genome)["final_steps"] >= 2
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0]
    np.testing.assert_array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]


def test_deap_selection_minimizes_and_respects_population_limit():
    engine = search(population_size=2)
    factors = iter((.9, .2, .7, .3, .8, .1, .4, .6))

    def callback(config):
        result = stub_score(config)
        factor = next(factors)
        result["error_sums"] = {s: factor * n for s, n in result["naive_sums"].items()}
        return result

    results = engine.run(callback)
    assert len(results) == 8
    assert len(engine.population) == 2
    assert [i.fitness.values[0] for i in engine.population] == pytest.approx([.1, .2])
    assert engine.run(callback) == results  # no evaluations on a second run


def test_pending_proposals_run_once_and_canonical_duplicates_are_not_refit():
    engine = Search(2, seed=1, budget=4, support=support(), max_proposals=6)
    original = default_genome(2)
    alias = copy.deepcopy(original)
    alias.update(multiplier=6, output_mode="trajectory", delta_y=2)
    candidates = iter([original, alias, original, alias, original, alias])
    engine.toolbox.register("candidate", lambda: next(candidates))
    assert len(engine.propose(4)) == 1
    assert len(engine.run(stub_score)) == engine.evaluations == 1
    assert engine.attempts == 6


@pytest.mark.parametrize("receipt", [None, 1.0, (), {}, {"fitness": .1}])
def test_non_receipts_rejected(receipt):
    engine = search()
    with pytest.raises(ScoreRejected):
        engine.evaluate(engine.propose()[0], lambda config: receipt)
    assert not engine.population


@pytest.mark.parametrize("field", ["status", "counts", "error_sums", "naive_sums",
                                   "support_digest", "metadata"])
def test_missing_receipt_fields_rejected(field):
    engine = search()

    def callback(config):
        receipt = stub_score(config)
        del receipt[field]
        return receipt

    with pytest.raises(ScoreRejected):
        engine.evaluate(engine.propose()[0], callback)


@pytest.mark.parametrize("field,split,value", [
    ("counts", "train", 0), ("counts", "inner", 5.0),
    ("naive_sums", "train", 0.0), ("naive_sums", "inner", float("nan")),
    ("support_digest", "train", "not-a-digest"),
])
def test_invalid_frozen_support_rejected_before_any_proposals(field, split, value):
    frozen = support()
    frozen[field][split] = value
    with pytest.raises(ValueError):
        Search(12, seed=1, budget=1, support=frozen)


def test_json_roundtrip_and_matching_reported_fitness():
    engine = search()
    proposal = json.loads(json.dumps(engine.propose()[0]))

    def callback(config):
        return dict(stub_score(config), fitness=.45)

    assert engine.evaluate(proposal, callback)["fitness"] == pytest.approx(.45)


def test_metadata_must_be_finite_json_and_not_a_scalar():
    for metadata in (.5, {"bad": float("nan")}, {"bad": object()}):
        engine = search()
        with pytest.raises(ScoreRejected):
            engine.evaluate(engine.propose()[0], lambda config: dict(
                stub_score(config), metadata=metadata))


def test_score_overflow_and_cross_horizon_proposal_fail_closed():
    engine = search()
    tiny = support()
    tiny["naive_sums"] = {"train": 1e-300, "inner": 1e-300}
    small = Search(12, seed=1, budget=1, support=tiny)

    def overflow(config):
        return dict(stub_score(config), **tiny,
                    error_sums={"train": 1e300, "inner": 1e300})

    with pytest.raises(ScoreRejected):
        small.evaluate(small.propose()[0], overflow)
    engine.toolbox.register("candidate", lambda: default_genome(24))
    with pytest.raises(ValueError):
        engine.propose()
    assert not engine.population and engine.evaluations == 0
