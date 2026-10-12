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


# ---------------------------------------------------------------------------
# Owner order 2026-10-12: H72 successor identity without STLFSI4.
# ``build_plan`` above stays byte-identical: it is the sealed identity of the
# running H4 islands, now labelled DEVELOPMENT_TUNING_POINT_IN_TIME_CONTAMINATED.

PLAN_VERSION_V3 = "2.1.0-island-sequence-no-stlfsi4"
EXPERIMENT_V3 = "i6n-swarm-weekly-2024-v2-no-stlfsi4"
DOMAIN_ID_V3 = "i6n-island-sequence-h72-v3-no-stlfsi4"
RUN_TAG_V3 = "h72-v3-no-stlfsi4"


def build_plan_v3():
    """Sealed successor plan: the 19 availability-gated features, H72 only."""
    from tools import i6n_contamination as contamination
    from tools.i6n_contracts import feature_availability as availability

    contract = availability.load_contract()
    label = contamination.load_label()
    features = list(availability.MANIFEST_NO_STLFSI4)
    availability.gate(features, contract)
    plan = deepcopy(build_plan())
    previous = plan["plan_sha256"]
    plan.pop("seal")
    plan.update(
        plan_version=PLAN_VERSION_V3,
        supersedes=plan["supersedes"] + [{
            "plan_version": "2.0.0-island-sequence",
            "plan_sha256": previous,
            "reason": ("owner ruling 2026-10-12: STLFSI4 consumed about five days before publication; "
                       "exclude it (no assumed-lag shift) and refuse every result of the superseded identity"),
            "fits_under_superseded_plan": None,
            "results_label": label["label"],
        }],
        features=("the 19 availability-gated RAW features: the 20 retained RAW features minus "
                  "fred.stress.stlfsi4.level (STLFSI4, excluded by owner ruling 2026-10-12); identity "
                  "bound in feature_manifest, in every v2 cell and in the evaluator context"),
        experiment=EXPERIMENT_V3,
        sequence_scope="H72 only; H4 is not re-run under this plan",
        feature_manifest={
            "features": features,
            "sha256": availability.manifest_sha256(features),
            "excluded": list(availability.EXCLUDED_FEATURES),
            "legacy_manifest_sha256": availability.manifest_sha256(availability.LEGACY_MANIFEST_20),
            "availability_contract_sha256": contract["contract_sha256"],
        },
        point_in_time={
            "gate": "tools.i6n_contracts.feature_availability.gate: every feature declares event_time, "
                    "available_time, timezone, availability source and revision policy; a window consumes a "
                    "value only when available_time <= t; undeclared, excluded or early-consumed features "
                    "are refused at cell authentication and at config generation",
            "contamination_label": label["label"],
            "contamination_label_sha256": label["label_sha256"],
            "refusals": ["checkpoint", "migrant", "fit cache", "start receipt", "ranking", "comparison"],
        },
    )
    digest = legacy._sha(legacy._body(plan))
    plan["plan_sha256"] = digest
    plan["seal"] = {
        "algorithm": "sha256",
        "canonical": "json sort_keys, separators (',',':'), allow_nan=False, UTF-8; keys plan_sha256 and seal excluded",
        "sha256": digest,
    }
    legacy.verify_plan(plan)
    return plan
