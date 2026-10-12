"""Shared DEAP plugin, plan-v2 path (four folds, staged classes). Stub receipts only."""

import copy

import pytest

from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_search as v1
from tools import i6n_search_plan as sp
from tools import i6n_search_space as space


FOLDS = ("2023Q1", "2023Q2", "2023Q3", "2023Q4")


def fold_support(i):
    return {"counts": {"train": 100 + i, "inner": 20 + i},
            "naive_sums": {"train": 10.0, "inner": 4.0},
            "support_digest": {"train": format(i + 1, "x") * 64, "inner": format(i + 6, "x") * 64}}


CALLS = []


def evaluator(request):
    CALLS.append(copy.deepcopy(request))
    assert set(request) == {"config", "generation", "context", "support"}
    r = copy.deepcopy(request["support"])
    width = request["config"]["core"]["width"]
    genome = {k: request["config"][k] for k in v1._GENOME_KEYS}  # A's receipt formula
    r.update(status="ok", error_sums={"train": 9.0 - width / 100, "inner": 3.9},
             metadata={"evidence_kind": "stub", "candidate_receipt": {
                 "genotype_canonical_sha256": v1.effective_identity(genome),
                 "fold": request["context"]["fold"], "horizon_hours": request["config"]["horizon"]}})
    return r


def opt(admitted=("S1",), **changes):
    cfg = dict(
        i6n_plan=sp.build_plan(), i6n_horizon=12,
        i6n_folds={f: {"i6n_support": fold_support(i),
                       "i6n_evaluation_context": {"fold": f, "fixture": "unit-v2"}}
                   for i, f in enumerate(FOLDS)},
        i6n_admitted_classes=list(admitted), i6n_evaluator=evaluator)
    cfg.update(changes)
    return cfg


def test_create_population_contains_seed_and_alternative():
    p = Plugin()
    pop = p.create_shared_population(8, opt(), seed=6102026)
    genomes = [item["genotype"] for item in pop["population"]]
    assert genomes[0] == space.seed_genotype(12)
    assert genomes[1] == space.alternative_genotype(12)
    assert len(genomes) == 8
    assert pop["population"][0]["schema"] == "i6n-shared-deap-v2"
    assert pop["config_snapshot"]["plan_sha256"] == sp.build_plan()["seal"]["sha256"]


def test_population_size_must_match_plan():
    with pytest.raises(ValueError):
        Plugin().create_shared_population(16, opt(), seed=6102026)
    with pytest.raises(ValueError):
        Plugin().create_shared_population(8, opt(), seed=42)


def test_evaluate_runs_four_folds_and_averages():
    CALLS.clear()
    p = Plugin()
    cfg = opt()
    pop = p.create_shared_population(8, cfg, seed=6102026)
    res = p.evaluate_single_genome(pop["population"][0], 0, cfg)
    assert len(CALLS) == 4
    assert [c["context"]["fold"] for c in CALLS] == list(FOLDS)
    assert CALLS[0]["config"] == v1.effective_config(space.to_evaluator_genome(space.seed_genotype(12)))
    expected = sum(0.5 * ((9.0 - 0.08) / 10.0 + 3.9 / 4.0) for _ in FOLDS) / 4
    assert res["fitness"] == pytest.approx(expected)
    assert set(res["metrics"]["per_fold"]) == set(FOLDS)


def test_full_search_three_generations_budget_and_classes():
    p = Plugin()
    cfg = opt(admitted=("S1", "S2", "S3"))
    state = p.create_shared_population(8, cfg, seed=6102026)
    gen, total, classes = 0, 0, []
    while state["population"]:
        evaluated = []
        for item in state["population"]:
            r = p.evaluate_single_genome(item, gen, cfg)
            evaluated.append(dict(item, fitness=r["fitness"]))
            classes.append((gen, space.class_of(item["genotype"])))
        total += len(evaluated)
        state = p.reproduce_shared(evaluated, gen, 6102026, cfg, state["innovation_tracker"],
                                   state["stage_schedule"], state["param_defaults"], 0,
                                   state.get("no_improve_count", 0))
        gen += 1
        if state.get("converged"):
            break
    assert total == 24
    assert state["stop_reason"] == "generation_budget"
    assert all(g >= 1 for g, c in classes if c == "S2")
    assert all(g >= 2 for g, c in classes if c == "S3")


def test_v1_path_unchanged_without_plan():
    # Legacy configs (no i6n_plan) keep the v1 schema used by the 48 completed fits.
    from tests.test_i6n_optimizer import config as v1_config
    pop = Plugin().create_shared_population(8, v1_config(), seed=6102026)
    assert pop["population"][0]["schema"] == "i6n-shared-deap-v1"


def test_wait_resource_then_resume_without_closing_twice():
    p = Plugin()
    cfg = opt(admitted=("S1",))
    st = p.create_shared_population(8, cfg, seed=6102026)
    evaluated = [dict(i, fitness=p.evaluate_single_genome(i, 0, cfg)["fitness"]) for i in st["population"]]
    # all pilots revoked: nothing may be proposed, generation 0 is closed once
    st2 = p.reproduce_shared(evaluated, 0, 6102026, opt(admitted=()), st["innovation_tracker"],
                             st["stage_schedule"], st["param_defaults"], 0, 0)
    assert st2["population"] == [] and st2["status"] == "WAIT_RESOURCE" and not st2["converged"]
    assert st2["generation"] == 1
    st3 = p.reproduce_shared([], 1, 6102026, cfg, st2["innovation_tracker"],
                             st2["stage_schedule"], st2["param_defaults"], 0, 0)
    assert len(st3["population"]) == 8 and st3["generation"] == 1
    assert all(i["generation"] == 1 for i in st3["population"])



def test_v2_plugin_accepts_c2_fold_ids():
    c2 = ("2023Q1_DCA", "2023Q2_DCA", "2023Q3_DCA", "2023Q4")
    cfg = opt(i6n_folds={f: {"i6n_support": fold_support(i),
                             "i6n_evaluation_context": {"fold": f, "fixture": "unit-v2"}}
                         for i, f in enumerate(c2)})
    p = Plugin()
    pop = p.create_shared_population(8, cfg, seed=6102026)
    res = p.evaluate_single_genome(pop["population"][0], 0, cfg)
    assert list(res["metrics"]["per_fold"]) == list(c2)
    bad = opt(i6n_folds={f: cfg["i6n_folds"][f] for f in c2[:3]})
    with pytest.raises(ValueError):
        p.create_shared_population(8, bad, seed=6102026)



def test_v2_plugin_refuses_receipt_of_another_candidate():
    def wrong(request):
        r = evaluator(request)
        r["metadata"]["candidate_receipt"]["genotype_canonical_sha256"] = "b" * 64
        return r
    p = Plugin()
    cfg = opt(i6n_evaluator=wrong)
    pop = p.create_shared_population(8, cfg, seed=6102026)
    with pytest.raises(v1.ScoreRejected, match="RECEIPT_CANDIDATE_MISMATCH"):
        p.evaluate_single_genome(pop["population"][0], 0, cfg)
