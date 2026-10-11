"""Sealed plan JSON, budgets, promotion, fold fitness, finalists, config generator.

Plan af0c3570 section 4. Fabricated receipts only (evidence_kind=stub); no
fits, no data, no TensorFlow, no TEST rows.
"""

import copy
import json
import random
import subprocess
import sys

import pytest

from tools import i6n_search_plan as sp
from tools import i6n_search_space as space


FOLDS = ("2023Q1", "2023Q2", "2023Q3", "2023Q4")


@pytest.fixture(scope="module")
def plan():
    return sp.build_plan()


def support(n_train=100, n_inner=20, naive_train=10.0, naive_inner=4.0, tag="a"):
    return {"counts": {"train": n_train, "inner": n_inner},
            "naive_sums": {"train": naive_train, "inner": naive_inner},
            "support_digest": {"train": tag * 64, "inner": ("b" if tag == "a" else "c") * 64}}


def receipt(sup, err_train, err_inner):
    r = copy.deepcopy(sup)
    r.update(status="ok", error_sums={"train": err_train, "inner": err_inner},
             metadata={"evidence_kind": "stub"})
    return r


# --- plan JSON and seal --------------------------------------------------------------

def test_plan_is_sealed_and_verifiable(plan):
    assert plan["schema"] == "i6n.search_plan.v1"
    sp.verify_plan(plan)
    tampered = copy.deepcopy(plan)
    tampered["budgets"]["total_new_fits_max"] = 2000
    with pytest.raises(ValueError, match="seal"):
        sp.verify_plan(tampered)
    assert sp.build_plan() == plan  # deterministic
    json.dumps(plan, allow_nan=False)


def test_plan_binds_governing_documents(plan):
    g = plan["governing"]
    assert g["orders_sha256"].startswith("ac77366a")
    assert g["plan_sha256"].startswith("da51723a")
    assert g["plan_commit"] == "af0c3570"


def test_plan_carries_initial_genotypes_for_b(plan):
    for h in space.HORIZONS:
        init = plan["initial_genotypes"][str(h)]
        assert init["seed"] == space.seed_genotype(h)
        assert init["seed_evaluator_genome"] == space.to_evaluator_genome(space.seed_genotype(h))
        assert init["seed_genotype_canonical_sha256"] == space.genotype_sha256(space.seed_genotype(h))
        assert init["alternative"] == space.alternative_genotype(h)


def test_plan_domain_matches_space(plan):
    d = plan["domain"]
    assert plan["horizons_hours"] == list(space.HORIZONS)
    assert d["history"]["multipliers"] == [1, 2, 4, 6] and d["history"]["min_hours"] == 24
    assert d["history"]["per_horizon"] == {str(h): list(space.history_domain(h)) for h in space.HORIZONS}
    assert d["supervision"]["trajectory_k"] == [2, 3, 4]
    assert d["supervision"]["endpoint_alias"] == "k=1 canonicalizes to endpoint with trajectory_k=null"
    assert d["entrance_widths"] == [8, 16, 32, 64, 128]
    assert d["reducer_filters"] == [4, 8, 16, 32, 64, 128]
    assert d["adapter_widths"] == [8, 16, 32, 64]
    assert d["stride_patterns"] == {"none": "all 1", "progressive": "all 2", "final": "last reducer 2, others 1"}
    assert plan["seed"] == 6102026 and plan["replicas"] == 1


def test_budgets(plan):
    b = plan["budgets"]
    assert b["new_candidates_per_horizon_max"] == 24
    assert b["folds_per_candidate"] == 4
    assert b["total_new_fits_max"] == 12 * 24 * 4 == 1152
    assert b["max_physical_attempts_per_identity_fold"] == 3
    assert b["per_class_new_candidates_per_horizon_max"] == {"S1": 24, "S2": 16, "S3": 8}
    assert b["per_class_new_candidates_total_max"] == {"S1": 288, "S2": 192, "S3": 96}
    assert b["cache_hits_count_as_new_fits"] is False
    assert b["oom_retry_with_lower_cap"] is False
    s = plan["search"]
    assert (s["population_size"], s["generations"], s["new_candidates_per_horizon"]) == (8, 3, 24)
    assert (s["mutation"], s["immigration"], s["crossover"]) == (0.8, 0.2, False)
    assert s["tolerance"] == 1e-4 and s["patience"] == 3
    assert s["direction"] == "minimize"
    assert plan["plan_sha256"] == plan["seal"]["sha256"]
    assert [c["class_id"] for c in plan["classes"]] == ["S1", "S2", "S3"]
    for c in plan["classes"]:
        assert c["child_memory_bytes"] is None and c["admission"] == "FROM_ADMITTED_COST_PILOT_RECEIPT"


def test_folds_are_distinct_quarter_periods_not_replicas(plan):
    f = plan["folds"]
    assert f["fold_ids"] == list(FOLDS)
    assert f["distinct_periods"] is True and f["replicas_of_one_fit"] is False
    assert f["inner_days"] == 28 and f["train_calendar_years"] == 4 and f["purged"] is True
    assert f["all_folds_required_before_ranking"] is True


def test_promotion_rules(plan):
    rules = {r["class_id"]: r for r in plan["promotion"]}
    assert rules["S1"]["opens_at_generation"] == 0
    assert rules["S2"]["opens_at_generation"] == 1
    assert rules["S3"]["opens_at_generation"] == 2
    for c in ("S1", "S2", "S3"):
        assert "admitted_cost_pilot" in rules[c]["requires"]
    assert plan["cost_admission"]["production_cap_rule"] == "1.25 * measured_peak, only if it fits host and slice reserve"


def test_open_classes(plan):
    assert sp.open_classes(plan, 0, {"S1", "S2", "S3"}) == ["S1"]
    assert sp.open_classes(plan, 1, {"S1", "S2"}) == ["S1", "S2"]
    assert sp.open_classes(plan, 2, {"S1"}) == ["S1"]
    assert sp.open_classes(plan, 2, {"S1", "S2", "S3"}) == ["S1", "S2", "S3"]
    assert sp.open_classes(plan, 0, set()) == []
    # S3 admitted without S2 still opens S3 at generation 2 (genes remain free).
    assert sp.open_classes(plan, 2, {"S1", "S3"}) == ["S1", "S3"]


# --- initial population ------------------------------------------------------------

@pytest.mark.parametrize("h", space.HORIZONS)
def test_initial_population_really_contains_minimal_seed(plan, h):
    pop = sp.initial_population(plan, h, open_class_ids=["S1"])
    assert len(pop) == 8
    assert pop[0] == space.seed_genotype(h)
    assert pop[1] == space.alternative_genotype(h)
    ids = [space.genotype_sha256(g) for g in pop]
    assert len(set(ids)) == 8
    assert all(space.class_of(g) == "S1" for g in pop)
    assert pop == sp.initial_population(plan, h, open_class_ids=["S1"])


def test_initial_population_refuses_without_admitted_class(plan):
    with pytest.raises(sp.WaitResource):
        sp.initial_population(plan, 12, open_class_ids=[])


# --- fold fitness ---------------------------------------------------------------------

def test_fold_fitness_is_unweighted_mean_of_sum_ratios(plan):
    sups = {f: support(n_train=100 * (i + 1), naive_train=10.0 * (i + 1), tag="a") for i, f in enumerate(FOLDS)}
    recs = {f: receipt(sups[f], 9.0 * (i + 1), 3.0 + i) for i, f in enumerate(FOLDS)}
    fit = sp.score_folds(plan, recs, sups)
    expected = sum(0.5 * (9.0 / 10.0 + (3.0 + i) / 4.0) for i in range(4)) / 4
    assert fit["fitness"] == pytest.approx(expected, abs=1e-15)
    assert fit["per_fold"]["2023Q2"] == pytest.approx(0.5 * (0.9 + 1.0))
    assert fit["status"] == "COMPLETE"


def test_fold_fitness_refuses_before_four_folds(plan):
    sups = {f: support() for f in FOLDS}
    recs = {f: receipt(sups[f], 9.0, 3.0) for f in FOLDS[:3]}
    with pytest.raises(sp.Incomplete):
        sp.score_folds(plan, recs, sups)


def test_fold_fitness_refuses_support_mismatch_and_failed(plan):
    sups = {f: support() for f in FOLDS}
    recs = {f: receipt(sups[f], 9.0, 3.0) for f in FOLDS}
    bad = copy.deepcopy(recs)
    bad["2023Q3"]["counts"]["inner"] = 19
    with pytest.raises(ValueError):
        sp.score_folds(plan, bad, sups)
    bad = copy.deepcopy(recs)
    bad["2023Q1"]["status"] = "failed"
    with pytest.raises(ValueError):
        sp.score_folds(plan, bad, sups)
    with pytest.raises(ValueError):
        sp.score_folds(plan, dict(recs, **{"2024Q1": recs["2023Q1"]}), sups)


def test_folds_with_identical_support_digests_are_rejected_as_replicas(plan):
    sups = {f: support(tag="a") for f in FOLDS}
    recs = {f: receipt(sups[f], 9.0, 3.0) for f in FOLDS}
    with pytest.raises(ValueError, match="distinct"):
        sp.score_folds(plan, recs, sups, require_distinct_support=True)


# --- horizon search state machine -------------------------------------------------------

def fake_fitness(genotype):
    # deterministic fabricated score favouring larger widths a bit
    return 1.0 - space.genotype_sha256(genotype).count("a") * 1e-3 - genotype["width"] * 1e-5


def run_horizon(plan, h, admitted, fitness=fake_fitness, fail=lambda g: False):
    state = sp.new_horizon_state(plan, h)
    while True:
        state, batch = sp.propose_generation(plan, state, admitted)
        if not batch:
            break
        for cand in batch:
            if fail(cand["genotype"]):
                state = sp.record_disposition(plan, state, cand["genotype_sha256"], "FAILED",
                                              reason="stub failure after 3 attempts")
            else:
                state = sp.record_fitness(plan, state, cand["genotype_sha256"], fitness(cand["genotype"]))
        state = sp.close_generation(plan, state)
        if state["stop_reason"]:
            break
    return state


def test_search_respects_budgets_and_class_caps(plan):
    state = run_horizon(plan, 72, {"S1", "S2", "S3"})
    recs = state["candidates"]
    assert len(recs) == 24
    assert len({r["genotype_sha256"] for r in recs}) == 24
    by_class = {}
    for r in recs:
        by_class[r["size_class"]] = by_class.get(r["size_class"], 0) + 1
    assert by_class.get("S2", 0) <= 16 and by_class.get("S3", 0) <= 8
    assert all(r["generation"] >= 1 for r in recs if r["size_class"] == "S2")
    assert all(r["generation"] >= 2 for r in recs if r["size_class"] == "S3")
    assert state["stop_reason"] == "generation_budget"
    assert state["new_fits_reserved"] == 96


def test_search_only_s1_admitted_never_proposes_bigger(plan):
    state = run_horizon(plan, 12, {"S1"})
    assert {r["size_class"] for r in state["candidates"]} == {"S1"}
    assert state["candidates"][0]["genotype"] == space.seed_genotype(12)


def test_failed_candidates_have_disposition_not_fitness(plan):
    state = run_horizon(plan, 6, {"S1"}, fail=lambda g: g["width"] == 32)
    failed = [r for r in state["candidates"] if r["disposition"] == "FAILED"]
    assert all(r["fitness"] is None and r["reason"] for r in failed)


def test_records_keep_active_inactive_and_effective_config(plan):
    state = run_horizon(plan, 24, {"S1"})
    for r in state["candidates"]:
        cfg = r["effective_config"]
        assert cfg["genotype"] == r["genotype"]
        assert set(cfg["active_genes"]) and "inactive_genes" in cfg
        assert r["origin"] in {"seed", "alternative", "immigrant", "mutation"}
        if r["origin"] == "mutation":
            assert len(r["parent_genotype_sha256"]) == 64


def test_search_is_reproducible(plan):
    a = run_horizon(plan, 36, {"S1", "S2"})
    b = run_horizon(plan, 36, {"S1", "S2"})
    assert [r["genotype_sha256"] for r in a["candidates"]] == [r["genotype_sha256"] for r in b["candidates"]]


def test_cache_hit_by_full_identity_is_not_a_new_fit(plan):
    state = sp.new_horizon_state(plan, 12)
    seed_id = space.genotype_sha256(space.seed_genotype(12))
    state = sp.register_cache(plan, state, {seed_id: 0.97})
    state, batch = sp.propose_generation(plan, state, {"S1"})
    seed_rec = next(r for r in state["candidates"] if r["genotype_sha256"] == seed_id)
    assert seed_rec["disposition"] == "CACHED_VERIFIED" and seed_rec["fitness"] == 0.97
    assert seed_id not in {c["genotype_sha256"] for c in batch}
    assert state["new_fits_reserved"] == 4 * len(batch)


def test_patience_and_tolerance(plan):
    state = sp.new_horizon_state(plan, 12)
    state["best_history"] = [0.9, 0.89995, 0.89991]
    assert sp.no_improve_count(plan, state) == 2
    state["best_history"] = [0.9, 0.8998]
    assert sp.no_improve_count(plan, state) == 0


def test_wait_resource_when_no_class_admitted(plan):
    state = sp.new_horizon_state(plan, 12)
    state, batch = sp.propose_generation(plan, state, set())
    assert batch == [] and state["status"] == "WAIT_RESOURCE"
    assert state["stop_reason"] is None


# --- finalists ---------------------------------------------------------------------------

def test_finalists_best_plus_minimal_control(plan):
    state = run_horizon(plan, 12, {"S1"})
    fin = sp.select_finalists(plan, state)
    complete = [r for r in state["candidates"] if r["fitness"] is not None]
    best = min(complete, key=lambda r: (r["fitness"], r["genotype_sha256"]))
    assert fin[0]["role"] == "BEST" and fin[0]["genotype_sha256"] == best["genotype_sha256"]
    seed_id = space.genotype_sha256(space.seed_genotype(12))
    assert fin[1]["role"] in {"MINIMAL_CONTROL", "SMALL_ALTERNATIVE"}
    if best["genotype_sha256"] != seed_id:
        assert fin[1]["genotype_sha256"] == seed_id
    assert fin[0]["claim"] == "SELECTED_BY_INNER_FOLD_SEARCH_NOT_PROVEN_OPTIMAL"


def test_finalists_when_seed_is_best_use_alternative(plan):
    seed = space.seed_genotype(12)
    state = run_horizon(plan, 12, {"S1"}, fitness=lambda g: 0.5 if g == seed else 0.9)
    fin = sp.select_finalists(plan, state)
    assert fin[0]["genotype"] == seed
    assert fin[1]["role"] == "SMALL_ALTERNATIVE"
    assert fin[1]["genotype"] == space.alternative_genotype(12)


# --- config generator ---------------------------------------------------------------------

def bindings(h):
    return {f: {"experiment": "i6n-swarm-weekly-20261011", "fold_id": f,
                "support_sha256": format(i, "x") * 64, "context_sha256": "e" * 64}
            for i, f in enumerate(FOLDS, start=1)}


def test_fit_specs_per_candidate_fold(plan):
    pop = sp.initial_population(plan, 12, open_class_ids=["S1"])
    specs = sp.fit_specs(plan, 12, pop, bindings(12))
    assert len(specs) == 8 * 4
    assert len({s["fit_id"] for s in specs}) == 32
    s = specs[0]
    assert set(s) >= {"fit_id", "genotype_canonical_sha256", "genotype", "effective_config",
                      "effective_config_sha256", "search_candidate_sha256",
                      "evaluator_request_config", "fold_id", "size_class", "horizon_hours",
                      "plan_sha256", "max_physical_attempts"}
    assert s["max_physical_attempts"] == 3
    assert s["plan_sha256"] == plan["seal"]["sha256"]


def test_fit_specs_reject_wrong_folds(plan):
    pop = sp.initial_population(plan, 12, open_class_ids=["S1"])
    b = bindings(12)
    del b["2023Q4"]
    with pytest.raises(ValueError):
        sp.fit_specs(plan, 12, pop, b)


def test_cli_plan_and_seed_configs(tmp_path):
    out = tmp_path / "plan.json"
    subprocess.run([sys.executable, "-m", "tools.i6n_search_plan", "plan", "--output", str(out)], check=True)
    plan = json.loads(out.read_text())
    sp.verify_plan(plan)
    cfg_dir = tmp_path / "configs"
    subprocess.run([sys.executable, "-m", "tools.i6n_search_plan", "initial-configs", "--plan", str(out),
                    "--output", str(cfg_dir)], check=True)
    files = sorted(p.relative_to(cfg_dir).as_posix() for p in cfg_dir.rglob("*.json"))
    assert "h12/S1/generation0.json" in files and len(files) == 12 + 1  # + manifest
    manifest = json.loads((cfg_dir / "MANIFEST.json").read_text())
    assert manifest["plan_sha256"] == plan["seal"]["sha256"]
    assert manifest["fits_executed"] == 0


# --- legacy mapping of the 48 completed candidates -------------------------------------------

def test_legacy_state_mapping_cli(tmp_path):
    state = {"identity": {"cell": {"config": {"evaluation_week_id": "2024-01-07", "horizon_hours": 12}}},
             "records": {}}
    from tools import i6n_search as v1
    genome = v1.default_genome(12)
    genome.update(delta_x=2)
    genome["core"].update(reducer_filters=[8], reducer_strides=[2])
    ident = v1.effective_identity(genome)
    state["records"][ident] = {"wire": {"genome": genome, "identity": ident}, "state": "COMPLETED",
                               "fitness": 0.99}
    path = tmp_path / "STATE.json"
    path.write_text(json.dumps({"payload": state, "sha256": "0" * 64}))
    out = tmp_path / "map.json"
    subprocess.run([sys.executable, "-m", "tools.i6n_search_plan", "legacy-map", "--state", str(path),
                    "--output", str(out)], check=True)
    result = json.loads(out.read_text())
    assert result["candidates"][0]["v1_identity"] == ident
    assert result["candidates"][0]["v1_identity_recomputed"] is True
    assert result["candidates"][0]["genotype_status"] == "MAPPED"
    assert result["candidates"][0]["fold_score_reuse"] == "NONE"
    assert result["summary"]["refits_scheduled"] == 0
