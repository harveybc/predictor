"""CPU contract tests; all scores are fabricated, never historical evidence."""

import ast
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_search as search


def receipt(request):
    result = copy.deepcopy(request["support"])
    result.update(status="ok", error_sums={"train": 10., "inner": 3.},
                  metadata={"evidence_kind": "stub"})
    return result


def config(**changes):
    cfg = dict(i6n_horizon=12, i6n_support={
        "counts": {"train": 10, "inner": 2},
        "naive_sums": {"train": 20., "inner": 6.},
        "support_digest": {"train": "a" * 64, "inner": "b" * 64}},
        i6n_evaluation_context={"version": "fabricated-unit-fixture-v1"},
        i6n_evaluator=receipt, n_generations=3, optimization_patience=3)
    cfg.update(changes)
    return cfg


def source_tree(repo, relative):
    root = Path(os.environ.get("I6N_SIBLING_ROOT", Path.home() / "Documents/GitHub"))
    path = root / repo / relative
    if not path.exists():
        pytest.skip(f"read-only sibling source unavailable: {path}")
    return ast.parse(path.read_text()), str(path)


@pytest.fixture
def real_wrapper():
    tree, filename = source_tree("doin-plugins", "src/doin_plugins/predictor/optimizer.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "PredictorOptimizer")
    names = {"setup_shared_mode", "get_shared_opt_config", "create_shared_population",
             "evaluate_candidate", "reproduce_shared", "get_domain_metadata"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    # Execute the actual forwarding bodies, not a reimplementation or wire mock.
    shell = ast.ClassDef(name="PredictorOptimizer", bases=[], keywords=[],
                         body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), shell], type_ignores=[])
    scope = {"copy": copy, "os": os}
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), scope)
    wrapper = scope["PredictorOptimizer"]()
    wrapper._predictor_root = None
    wrapper._predictor_config = config()
    wrapper._deap_optimizer = Plugin()
    wrapper._deap_optimizer.set_params(**wrapper._predictor_config)
    return wrapper


def evaluated(plugin, state, cfg):
    pop = copy.deepcopy(state["population"])
    for item in pop:
        item["fitness"] = plugin.evaluate_single_genome(item, state.get("generation", 0), cfg)["fitness"]
    return pop


def reproduce(plugin, state, cfg, pop=None, **overrides):
    args = dict(evaluated_pop=pop if pop is not None else evaluated(plugin, state, cfg),
                generation=state.get("generation", 0), seed=123, opt_config=cfg,
                innovation_tracker_data=state["innovation_tracker"],
                stage_schedule=state["stage_schedule"], param_defaults=state["param_defaults"])
    args.update(overrides)
    return plugin.reproduce_shared(**args)


def test_real_wrapper_roundtrip_and_raw_minimization(real_wrapper):
    assert real_wrapper.get_domain_metadata()["higher_is_better"] is False
    assert isinstance(Plugin.plugin_params, dict)
    state = real_wrapper.create_shared_population(4, seed=6102026)
    assert json.loads(json.dumps(state, allow_nan=False)) == state
    for item in state["population"]:
        result = real_wrapper.evaluate_candidate(item, 0)
        assert result["fitness"] == .5  # positive, lower is better
        assert result["metrics"]["counts"] == {"train": 10, "inner": 2}
        assert result["train_mae"] == 1.
        assert result["val_mae"] == 1.5
        assert result["hyper_dict"]["horizon"] == 12
        item["fitness"] = result["fitness"]  # exactly the node's handoff
    next_state = real_wrapper.reproduce_shared(
        state["population"], 0, 123, state["innovation_tracker"],
        state["stage_schedule"], state["param_defaults"], 0, 0)
    assert next_state["generation"] == 1
    assert len(next_state["population"]) == 4
    assert all("fitness" not in g for g in next_state["population"])


def test_node_consumer_contract_and_rejection_source():
    tree, filename = source_tree("doin-node", "src/doin_node/unified.py")
    rejection = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_candidate_rejection_reason")
    scope = {"math": math}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), rejection], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), scope)
    reject = scope["_candidate_rejection_reason"]
    assert reject({"fitness": .5}) is None
    assert reject({"fitness": float("inf"), "_eval_error": "missing evaluator"})
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_run_shared_optimization")
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "reproduce_shared"]
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == [
        "population", "generation", "repro_seed", "innovation_tracker_data",
        "stage_schedule", "param_defaults", "stage_idx", "no_improve_count"]
    assert 'max(gen_fitnesses) if _hib else min(gen_fitnesses)' in ast.unparse(loop)


def test_determinism_resume_and_global_rng_untouched():
    p, cfg = Plugin(), config()
    before = random.getstate()
    tensorflow_was_loaded = "tensorflow" in sys.modules
    a = p.create_shared_population(4, cfg)
    assert a == Plugin().create_shared_population(4, cfg)
    b = reproduce(p, a, cfg)
    assert b == reproduce(Plugin(), json.loads(json.dumps(a)), cfg)
    identities = [g["identity"] for g in a["population"] + b["population"]]
    assert len(set(identities)) == 8
    assert before == random.getstate()
    assert ("tensorflow" in sys.modules) == tensorflow_was_loaded


@pytest.mark.parametrize("horizon", search.HORIZONS)
def test_all_horizons_valid(horizon):
    state = Plugin().create_shared_population(3, config(i6n_horizon=horizon))
    for item in state["population"]:
        assert item["genome"] == search.normalize_genome(item["genome"])
        assert item["identity"] == search.effective_identity(item["genome"])


def test_callback_is_detached_and_exact_receipt_preserved():
    seen = []
    def callback(request):
        seen.append(copy.deepcopy(request))
        result = receipt(request)
        request["support"]["counts"]["train"] = 900
        return result
    p, cfg = Plugin(), config(i6n_evaluator=callback)
    p.set_params(**cfg)
    state = p.create_shared_population(1, {})
    result = p.evaluate_single_genome(state["population"][0], 0, {})
    assert result["metrics"] == receipt(seen[0])
    assert cfg["i6n_support"]["counts"]["train"] == 10


@pytest.mark.parametrize("change", [
    lambda r: r["counts"].update(inner=3),
    lambda r: r["naive_sums"].update(train=21.),
    lambda r: r["support_digest"].update(inner="c" * 64),
    lambda r: r["error_sums"].update(inner=float("nan")),
    lambda r: r.update(fitness=-.5),
    lambda r: r.update(status="failed"),
    lambda r: r["counts"].update(test=1),
])
def test_incomparable_receipts_fail_closed(change):
    def callback(request):
        r = receipt(request)
        change(r)
        return r
    p, cfg = Plugin(), config(i6n_evaluator=callback)
    state = p.create_shared_population(1, cfg)
    with pytest.raises(search.ScoreRejected):
        p.evaluate_single_genome(state["population"][0], 0, cfg)


@pytest.mark.parametrize("evaluator", [None, "missing_i6n_historical_cell:evaluate"])
def test_missing_evaluator_is_not_stubbed(evaluator):
    p, cfg = Plugin(), config(i6n_evaluator=evaluator)
    state = p.create_shared_population(1, cfg)
    with pytest.raises(search.ScoreRejected, match="evaluator"):
        p.evaluate_single_genome(state["population"][0], 0, cfg)


def test_json_command_and_failures():
    code = ("import json,sys; r=json.load(sys.stdin); s=r['support']; "
            "s.update(status='ok',error_sums={'train':10.,'inner':3.},"
            "metadata={'evidence_kind':'stub'}); print(json.dumps(s))")
    p, cfg = Plugin(), config(i6n_evaluator=None, i6n_evaluator_command=[sys.executable, "-B", "-c", code])
    state = p.create_shared_population(1, cfg)
    assert p.evaluate_single_genome(state["population"][0], 0, cfg)["fitness"] == .5
    for code in ("raise SystemExit(2)", "print('not json')", "import time; time.sleep(3)"):
        cfg["i6n_evaluator_command"] = [sys.executable, "-B", "-c", code]
        cfg["i6n_evaluator_timeout"] = .05
        state = p.create_shared_population(1, cfg)
        with pytest.raises(search.ScoreRejected):
            p.evaluate_single_genome(state["population"][0], 0, cfg)


def test_json_command_failure_reports_bounded_stderr_only(monkeypatch):
    marker = "diagnostic-tail-marker"
    stderr = "x" * 2500 + "\x00" + marker

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            17, args[0], output="secret-stdout", stderr=stderr)

    monkeypatch.setattr("optimizer_plugins.i6n_optimizer.subprocess.run", fail)
    cfg = config(i6n_evaluator=None, i6n_evaluator_command=["evaluator"])
    with pytest.raises(RuntimeError) as raised:
        Plugin()._evaluate(cfg, {"secret-request": "must-not-appear"})
    message = str(raised.value)
    assert "exit 17" in message and marker in message
    assert "secret-stdout" not in message and "secret-request" not in message
    assert "\x00" not in message and len(message) < 2100


def test_canonical_dedup_and_bounded_exhaustion(monkeypatch):
    base = search.default_genome(12)
    alias = copy.deepcopy(base)
    alias["multiplier"] = 2
    alias["core"]["transformer_blocks"] = 2
    candidates = iter([base, alias] * 3)
    monkeypatch.setattr(search, "sample_genome", lambda h, rng: next(candidates))
    state = Plugin().create_shared_population(4, config(i6n_max_proposals=6))
    assert len(state["population"]) == 1
    assert state["innovation_tracker"]["attempts"] == 6
    result = reproduce(Plugin(), state, config(i6n_max_proposals=6))
    assert result["converged"]
    assert result["stop_reason"] == "attempt_budget"


@pytest.mark.parametrize("changes,overrides,reason", [
    ({"n_generations": 1}, {}, "generation_budget"),
    ({"i6n_budget": 2}, {}, "candidate_budget"),
    ({}, {"no_improve_count": 3}, "patience"),
])
def test_finite_stopping(changes, overrides, reason):
    p, cfg = Plugin(), config(**changes)
    state = p.create_shared_population(2, cfg)
    result = reproduce(p, state, cfg, **overrides)
    assert result["converged"] and result["stop_reason"] == reason


def test_context_and_state_tampering_rejected():
    p, cfg = Plugin(), config()
    state = p.create_shared_population(2, cfg)
    altered = copy.deepcopy(state["population"][0])
    altered["genome"]["core"]["width"] = 32
    with pytest.raises(ValueError, match="identity"):
        p.evaluate_single_genome(altered, 0, cfg)
    cfg2 = config(i6n_evaluation_context={"version": "other"})
    with pytest.raises(ValueError, match="context"):
        reproduce(p, state, cfg2, pop=evaluated(p, state, cfg))
    with pytest.raises(ValueError):
        reproduce(p, state, cfg, generation=1)
    with pytest.raises(ValueError):
        reproduce(p, state, cfg, current_stage_idx=1)
    with pytest.raises(ValueError):
        p.create_shared_population(2, config(higher_is_better=True))


def test_deap_selects_only_low_finite_parents(monkeypatch):
    p, cfg = Plugin(), config()
    state = p.create_shared_population(3, cfg)
    pop = copy.deepcopy(state["population"])
    for item, fitness in zip(pop, [.1, float("inf"), -1.e9]):
        item["fitness"] = fitness
    selected = []
    original = search.mutate
    def record(genome, rng):
        selected.append(search.effective_identity(genome))
        return original(genome, rng)
    monkeypatch.setattr(search, "mutate", record)
    reproduce(p, state, cfg, pop=pop)
    assert selected and set(selected) == {pop[0]["identity"]}
    for item in pop:
        item["fitness"] = float("inf")
    result = reproduce(p, state, cfg, pop=pop)
    assert result["converged"] and result["stop_reason"] == "no_eligible_parents"


def test_unequal_split_ratios_are_not_pooled_or_batch_means():
    def callback(request):
        r = receipt(request)
        r["error_sums"] = {"train": 5., "inner": 6.}
        return r
    p, cfg = Plugin(), config(i6n_evaluator=callback)
    state = p.create_shared_population(1, cfg)
    result = p.evaluate_single_genome(state["population"][0], 0, cfg)
    assert result["fitness"] == .625  # mean(5/20, 6/6), not 11/26


def test_valid_import_callable(monkeypatch):
    import types
    module = types.ModuleType("i6n_fixture_callback")
    module.evaluate = receipt
    monkeypatch.setitem(sys.modules, module.__name__, module)
    p, cfg = Plugin(), config(i6n_evaluator="i6n_fixture_callback:evaluate")
    state = p.create_shared_population(1, cfg)
    assert p.evaluate_single_genome(state["population"][0], 0, cfg)["fitness"] == .5


def test_config_precedence_detachment_and_wrong_generation():
    p, cfg = Plugin(), config()
    p.set_params(**cfg)
    cfg["i6n_support"]["counts"]["train"] = 99
    state = p.create_shared_population(2, {"i6n_horizon": 24})
    assert state["config_snapshot"]["support"]["counts"]["train"] == 10
    assert all(g["genome"]["horizon"] == 24 for g in state["population"])
    with pytest.raises(ValueError, match="generation"):
        p.evaluate_single_genome(state["population"][0], 1, {"i6n_horizon": 24})


def test_zero_offspring_exhaustion_converges_without_runtime_retry(monkeypatch):
    p, cfg = Plugin(), config(i6n_max_proposals=5)
    state = p.create_shared_population(1, cfg)
    genome = state["population"][0]["genome"]
    monkeypatch.setattr(search, "sample_genome", lambda h, rng: genome)
    monkeypatch.setattr(search, "mutate", lambda g, rng: genome)
    result = reproduce(p, state, cfg)
    assert result["population"] == []
    assert result["converged"] and result["stop_reason"] == "attempt_budget"
    assert result["innovation_tracker"]["attempts"] == 5


@pytest.mark.parametrize("field,value", [
    ("i6n_horizon", True), ("i6n_horizon", 7), ("n_generations", 0),
    ("i6n_budget", 0), ("i6n_max_proposals", 0), ("i6n_evaluation_context", {}),
    ("i6n_support", {}), ("optimization_patience", False),
])
def test_invalid_admission(field, value):
    with pytest.raises(ValueError):
        Plugin().create_shared_population(1, config(**{field: value}))


def test_reproduction_inputs_are_detached_and_pending_is_checked():
    p, cfg = Plugin(), config()
    state = p.create_shared_population(3, cfg)
    original = copy.deepcopy(state)
    pop = evaluated(p, state, cfg)
    before = copy.deepcopy(pop)
    result = reproduce(p, state, cfg, pop=pop)
    assert state == original and pop == before
    result["innovation_tracker"]["seen"].clear()
    assert state == original
    with pytest.raises(ValueError, match="reserved"):
        reproduce(p, state, cfg, pop=list(reversed(pop)))
    with pytest.raises(ValueError, match="unevaluated"):
        reproduce(p, state, cfg, pop=state["population"])


def test_real_deap_tournaments_minimize(monkeypatch):
    from optimizer_plugins import i6n_optimizer as adapter
    p, cfg = Plugin(), config()
    state = p.create_shared_population(6, cfg)
    pop = copy.deepcopy(state["population"])
    for i, item in enumerate(pop):
        item["fitness"] = i / 10
    original = adapter.tools.selBest
    compared = []
    def select(contenders, k):
        selected = original(contenders, k)
        scores = [c.fitness.values[0] for c in contenders]
        assert selected[0].fitness.weights == (-1.,)
        assert selected[0].fitness.values[0] == min(scores)
        compared.append(scores)
        return selected
    monkeypatch.setattr(adapter.tools, "selBest", select)
    reproduce(p, state, cfg, pop=pop)
    assert any(len(set(scores)) > 1 for scores in compared)
