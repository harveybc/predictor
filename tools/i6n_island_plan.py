"""Sealed successor plan for the cooperative I6N 4 h -> 72 h search."""

from copy import deepcopy

from tools import i6n_search_plan as legacy
from tools import i6n_search_space as space


SCHEMA = "i6n.doin_island_sequence.v1"
SEED = 6102026
HORIZONS = (4, 72)
POPULATION_PER_NODE = 20
MAX_GENERATIONS = 100
GENERATION_PATIENCE = 10
MIN_RELATIVE_IMPROVEMENT = 1e-8
MIN_GENERATIONS_BEFORE_STOP = 30


def build_plan():
    """Return a deterministic, sealed plan without rewriting historical v1.2."""
    plan = deepcopy(legacy.build_plan())
    previous = plan["plan_sha256"]
    plan.update(
        plan_version="2.0.0-island-sequence",
        supersedes=plan["supersedes"] + [{
            "plan_version": "1.2.0",
            "plan_sha256": previous,
            "reason": "replace the 8x3 shared-population pilot with four local DOIN islands; run h4 then h72",
            "fits_under_superseded_plan": 0,
        }],
        horizons_hours=list(HORIZONS),
        short_horizons_hours=[4],
        long_horizons_hours=[72],
        seed=SEED,
        replicas=1,
        initial_genotypes={
            str(h): {
                "seed": space.seed_genotype(h),
                "seed_evaluator_genome": space.to_evaluator_genome(space.seed_genotype(h)),
                "seed_genotype_canonical_sha256": space.genotype_sha256(space.seed_genotype(h)),
                "alternative": space.alternative_genotype(h),
                "alternative_genotype_canonical_sha256": space.genotype_sha256(space.alternative_genotype(h)),
            }
            for h in HORIZONS
        },
    )
    plan["promotion"] = [
        {"class_id": "S1", "opens_at_generation": 0, "requires": ["admitted_cost_pilot"]},
        {"class_id": "S2", "opens_at_generation": 10, "requires": ["admitted_cost_pilot"]},
        {"class_id": "S3", "opens_at_generation": 20, "requires": ["admitted_cost_pilot"]},
    ]
    plan["search"] = {
        "algorithm": "four cooperative DOIN islands; local DEAP tournament, conditional mutation and immigrants",
        "population_size": POPULATION_PER_NODE,
        "population_scope": "per_node",
        "node_count": 4,
        "generations": MAX_GENERATIONS,
        "mutation": 0.8,
        "immigration": 0.2,
        "crossover": False,
        "selection": "three-way tournament, one elite, minimization",
        "tolerance": MIN_RELATIVE_IMPROVEMENT,
        "patience": GENERATION_PATIENCE,
        "min_generations_before_stop": MIN_GENERATIONS_BEFORE_STOP,
        "direction": "minimize",
        "rng": ("training seed 6102026 on every node; search stream = first 64 bits of "
                "sha256(plan_sha256,horizon,training_seed,island_id); network champions are migration"),
        "initial_population": "minimal Conv8->Conv4 seed, small alternative, then unique S1 samples",
        "exhaustion": "stop with a typed reason; never fill with duplicates",
    }
    plan["fitness"]["optimization_objective"] = (
        "mean across four folds of ((endpoint MAE_h - paired naive MAE_h) / paired naive MAE_h); negative is better"
    )
    plan["budgets"] = {
        "population_per_node": POPULATION_PER_NODE,
        "node_count": 4,
        "max_generations": MAX_GENERATIONS,
        "max_candidate_evaluations_per_node": POPULATION_PER_NODE * MAX_GENERATIONS,
        "folds_per_new_candidate": 4,
        "cache_hits_count_as_new_fits": False,
        "oom_retry_with_lower_cap": False,
    }
    digest = legacy._sha(legacy._body(plan))
    plan["plan_sha256"] = digest
    plan["seal"] = {
        "algorithm": "sha256",
        "canonical": "json sort_keys, separators (',',':'), allow_nan=False, UTF-8; keys plan_sha256 and seal excluded",
        "sha256": digest,
    }
    legacy.verify_plan(plan)
    return plan


def sequence_manifest():
    """Public, secret-free identity of the automatic two-campaign sequence."""
    plan = build_plan()
    return {
        "schema": SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "horizons_hours": list(HORIZONS),
        "order": ["H4", "H72"],
        "seed": SEED,
        "population_per_node": POPULATION_PER_NODE,
        "node_count": 4,
        "max_generations": MAX_GENERATIONS,
        "generation_patience": GENERATION_PATIENCE,
        "min_generations_before_stop": MIN_GENERATIONS_BEFORE_STOP,
        "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
    }
