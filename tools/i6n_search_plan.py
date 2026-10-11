"""I6-N v2 search plan: sealed JSON, budgets, promotion, fold fitness, finalists.

Lane E, plan af0c3570 section 4. Pure functions over JSON state so that the
swarm ledger (lane B) can persist and replay them. No data, TensorFlow,
devices, services or fits. CLI subcommands only write files the caller names.

    python -m tools.i6n_search_plan plan --output <plan.json>   # committed: PLAN_PATH
    python -m tools.i6n_search_plan verify --plan <plan.json>
    python -m tools.i6n_search_plan initial-configs --plan <plan.json> --output <dir>
    python -m tools.i6n_search_plan fit-specs --plan <plan.json> --horizon H \
        --bindings <folds.json> --output <dir>
    python -m tools.i6n_search_plan legacy-map --state <STATE.json> [...] --output <map.json>
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from deap import base, tools as deap_tools

from tools import i6n_search as _v1
from tools import i6n_search_space as space


PLAN_SCHEMA = "i6n.search_plan.v1"
STATE_SCHEMA = "i6n.search_state.v1"
PLAN_VERSION = "1.1.0"
PLAN_PATH = "docs/i6n/swarm/I6N_SEARCH_PLAN_v1_1.json"
SUPERSEDED_PLAN_SHA256 = "8d8865f1343c6b0efa80edf711393425c3445b953cbb76a4f841a5a1994a77d3"
FOLD_IDS = ("2023Q1", "2023Q2", "2023Q3", "2023Q4")
LEGACY_MAPPING_PATH = "docs/i6n/swarm/E_LEGACY_MAPPING_v1.json"
# sha256 of the committed legacy mapping artifact (48 endpoint candidates).
LEGACY_MAPPING_SHA256 = "e3254dd63fa4654dadf5c72b1bcdb2a8a0708dd9540202363e303d3e85c77b3f"

STOP_REASONS = ("generation_budget", "candidate_budget", "patience",
                "admissible_generation_exhausted", "attempt_budget")
DISPOSITIONS = ("PENDING", "COMPLETE", "CACHED_VERIFIED", "FAILED", "INTERRUPTED", "NOT_ADMITTED")


class Incomplete(ValueError):
    """A candidate lacks a COMPLETE receipt on at least one plan fold."""


class WaitResource(RuntimeError):
    """No size class is admitted/open; nothing may be proposed yet."""


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


def _body(plan):
    return {k: v for k, v in plan.items() if k not in ("plan_sha256", "seal")}


# --- plan ------------------------------------------------------------------------------

def build_plan():
    """Deterministic sealed plan v1. Values are the subplan's, not measurements."""
    horizons = list(space.HORIZONS)
    per_h_class = {"S1": 24, "S2": 16, "S3": 8}
    plan = {
        "schema": PLAN_SCHEMA, "plan_version": PLAN_VERSION,
        "supersedes": {"plan_version": "1.0.0", "plan_sha256": SUPERSEDED_PLAN_SHA256,
                       "reason": "INT-STRIDE ruling: free per-layer strides (superset, identities unchanged)",
                       "fits_under_superseded_plan": 0},
        "governing": {
            "orders_path": "docs/handoffs/SATOSHI_I6N_SWARM_WEEKLY_2026_10_11.md",
            "orders_sha256": "ac77366af24bcaf4fb9454a51c94ab1f6d5230bb1de4ff14521f76cc0a3f8b9c",
            "plan_commit": "af0c3570",
            "plan_path": "docs/tres_temas_entrevista/program_v3/I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md",
            "plan_sha256": "da51723a08a8ee3f0a94ffaded8b35fd480b72d80f314a812a176cb43c6355df",
            "sections": ["2", "3", "4"],
        },
        "experiment_scope": "BUSINESS_WEEKLY_WALK_FORWARD development search; TEST (2025+) closed",
        "asset": "EURUSD", "features": "the 20 retained RAW features (identity bound in the evaluator context)",
        "horizons_hours": horizons,
        "short_horizons_hours": list(space.SHORT_HORIZONS),
        "long_horizons_hours": list(space.LONG_HORIZONS),
        "seed": 6102026, "replicas": 1,
        "genotype_schema": space.GENOTYPE_SCHEMA,
        "identity": {
            "genotype_canonical_sha256": "tools.i6n_search_space.genotype_sha256 == tools.i6n_search.effective_identity(to_evaluator_genome(g))",
            "effective_config_sha256": "sha256 canonical JSON of tools.i6n_search_space.evaluator_config(g)",
            "search_candidate_sha256": "sha256 canonical {plan_sha256, horizon_hours, genotype_canonical_sha256}",
            "fit_id": "sha256 canonical {experiment, fold, effective_config_sha256, context_sha256, support_sha256}",
            "cache": "a fold result is reused only when its fit_id matches exactly and its terminal receipt verifies",
        },
        "domain": {
            "history": {"rule": "L = max(24, m*h), deduplicated", "multipliers": list(space.HISTORY_MULTIPLIERS),
                        "min_hours": space.MIN_HISTORY_HOURS,
                        "per_horizon": {str(h): list(space.history_domain(h)) for h in horizons}},
            "delta_x": {"rule": "integer-hour divisors common to h and L, including 1; no smoothing or interpolation"},
            "samples": {"rule": "n = 1 + L/dx, including t", "max": 433, "truncation": "none (no 241 cap)"},
            "input_transform": list(space.INPUT_TRANSFORMS),
            "supervision": {"modes": list(space.SUPERVISIONS), "trajectory_k": list(space.TRAJECTORY_K),
                            "dy": "h/k when integer, independent of dx",
                            "per_horizon_k": {str(h): list(space.trajectory_k_domain(h)) for h in horizons},
                            "endpoint_alias": "k=1 canonicalizes to endpoint with trajectory_k=null",
                            "comparison": "endpoint h on frozen fold support; intermediates are diagnostics"},
            "entrances": list(space.ENTRANCES),
            "entrance_widths": list(space.ENTRANCE_WIDTHS),
            "transformer_blocks": {"conv": [0], "transformer": [1, 2], "conv_transformer": [1, 2]},
            "reducer_counts": list(space.REDUCER_COUNTS),
            "reducer_filters": list(space.REDUCER_FILTERS),
            "reducer_order": "entrance width then reducers non-increasing",
            "reducer_strides": {"per_layer": [1, 2],
                                "named_patterns": {"none": "all 1", "progressive": "all 2",
                                                   "final": "last reducer 2, others 1",
                                                   "mixed": "any other per-layer 1|2 list"},
                                "patterns_role": "derived labels and seed choices, not the admissible domain"},
            "stride_anchor": "right-anchored (keeps t); steps = ceil(steps/stride) per reducer",
            "min_final_steps": space.MIN_FINAL_STEPS,
            "adapters": list(space.ADAPTERS),
            "adapter_widths": list(space.ADAPTER_WIDTHS),
            "lstm": "return_sequences=True (sequence-valued)",
            "fixed_branches": "per-feature causal Conv1D filters=16 kernel=3 stride=1, fusion then position",
            "infeasibility": "typed reasons (INTERFACE_E); never repaired into another experiment",
        },
        "interpretations": [
            {"id": "INT-STRIDE", "text": "RULED 2026-10-11 (Satoshi): '1 o 2 por capa' is the domain, a free per-reducer 1|2 list; none/progressive/final are named patterns used as labels and seeds. Combinations that lose t or leave fewer than min_final_steps are still rejected (FINAL_STEPS_BELOW_MINIMUM). Plan v1.1 is a strict superset of v1.0.0; every v1.0.0 in-domain genotype_canonical_sha256 is unchanged.", "status": "RULED"},
            {"id": "INT-REDUCER-ORDER", "text": "non-increasing includes the entrance width (inherited search contract; legacy candidates obey it)."},
            {"id": "INT-S1-DOWNSAMPLED", "text": "S1 requires dx>=2 ('primero downsampling, despues horario'); hourly grids start at S2."},
            {"id": "INT-CLASS-WIDTH", "text": "class width limit applies to entrance, reducer and adapter widths."},
            {"id": "INT-MIN-STEPS", "text": "a reduction may not leave fewer than 2 temporal steps (inherited model rule)."},
            {"id": "INT-SEED-WINDOW", "text": "seed window = smallest history with a downsampled S1 grid, then its finest dx."},
            {"id": "INT-ALTERNATIVE", "text": "small alternative = Conv16 -> Conv8 -> Conv4, same window, no Transformer."},
            {"id": "INT-CACHE-SLOT", "text": "a verified cache hit occupies a population slot and acts as parent but is not a new candidate or fit."},
            {"id": "INT-LEGACY", "text": "the 48 legacy endpoint candidates are not injected into the population: their scores come from one inner interval before week 2024-01-07, which overlaps the 2023Q4 inner period."},
        ],
        "classes": [
            dict(c, child_memory_bytes=None, child_wall_seconds=None,
                 admission="FROM_ADMITTED_COST_PILOT_RECEIPT")
            for c in space.SIZE_CLASSES
        ],
        "promotion": [
            {"class_id": "S1", "opens_at_generation": 0, "requires": ["admitted_cost_pilot"]},
            {"class_id": "S2", "opens_at_generation": 1,
             "requires": ["admitted_cost_pilot", "previous generation of the horizon terminal on all folds"]},
            {"class_id": "S3", "opens_at_generation": 2,
             "requires": ["admitted_cost_pilot", "previous generation of the horizon terminal on all folds",
                          "pilot exercised the 433-sample contract"]},
        ],
        "promotion_notes": "opening a class keeps every earlier gene free (no frozen window); a class not admitted stays closed and the search continues inside the admitted ones",
        "cost_admission": {
            "pilot_data": "real TRAIN rows of the target campaign, no TEST",
            "production_cap_rule": "1.25 * measured_peak, only if it fits host and slice reserve",
            "never": ["learn the bound by provoking OOM", "retry an OOM with a lower cap", "fall back from GPU to CPU"],
        },
        "folds": {
            "fold_ids": list(FOLD_IDS),
            "rule": "last complete market cutoff of each 2023 calendar quarter; 28 inner days before the cutoff, purged; TRAIN = 4 calendar years before the fold cutoff; refit per fold",
            "inner_days": 28, "train_calendar_years": 4, "purged": True,
            "distinct_periods": True, "replicas_of_one_fit": False,
            "all_folds_required_before_ranking": True,
            "definition_owner": "lane C (exact cutoffs, support digests); A freezes support per (h, fold)",
            "missing_history": "materialize/contrast before the fold; never shorten years, substitute a 2024 week or skip",
            "unit_states": "INTERFACE_C C.1 per (fold unit, h): READY -> fit; WAIT_DEPENDENCY -> the horizon waits (no ranking on fewer folds); NO_ELIGIBLE_ROWS -> the horizon closes typed NO_FOLD_SUPPORT, never ranked on 3 folds",
            "cutoff_source": "INTERFACE_C C.1 INNER_FOLD unit: close of the last calendar week entirely inside the quarter",
        },
        "search": {
            "algorithm": "DEAP generational, conditional mutation + immigrants",
            "population_size": 8, "generations": 3, "new_candidates_per_horizon": 24,
            "mutation": 0.8, "immigration": 0.2, "crossover": False,
            "selection": "tools.selBest over a 3-way tournament for the parent; fitness weights (-1,)",
            "tolerance": 1e-4, "patience": 3, "direction": "minimize",
            "rng": "random.Random('i6n-search-plan-v1:<seed>:<h>:gen<g>') per horizon and generation",
            "initial_population": "seed (Conv8->Conv4) + small alternative + immigrants from open classes",
            "exhaustion": "close with a reason; never fill with duplicates",
        },
        "fitness": {
            "per_fold": "0.5 * (sum_abs_error_TRAIN / naive_sum_abs_error_TRAIN + sum_abs_error_INNER / naive_sum_abs_error_INNER), endpoint h, original price, sums/counts within each split (tools.i6n_search._score)",
            "aggregation": "unweighted mean over the 4 folds",
            "requires": "all 4 folds COMPLETE with support equal to the frozen fold support",
            "naive": "same origins and masks per horizon, original price",
            "outer_week_or_test": "never used for stopping or fitness",
            "failed": "disposition with reason, no fitness",
        },
        "training": {"seed": 6102026, "max_epochs": 50, "patience": 5, "min_delta": 1e-6,
                     "monitor": "mean TRAIN/inner loss", "restore_best": True},
        "budgets": {
            "new_candidates_per_horizon_max": 24, "folds_per_candidate": 4,
            "new_fits_per_horizon_max": 96, "total_new_fits_max": 12 * 24 * 4,
            "per_class_new_candidates_per_horizon_max": per_h_class,
            "per_class_new_candidates_total_max": {k: v * len(horizons) for k, v in per_h_class.items()},
            "max_physical_attempts_per_identity_fold": 3,
            "max_recovery_attempts_without_checkpoint": 1,
            "max_proposal_attempts_per_horizon": 32 * 24,
            "cache_hits_count_as_new_fits": False,
            "oom_retry_with_lower_cap": False,
        },
        "initial_genotypes": {
            str(h): {"seed": space.seed_genotype(h),
                     "seed_evaluator_genome": space.to_evaluator_genome(space.seed_genotype(h)),
                     "seed_genotype_canonical_sha256": space.genotype_sha256(space.seed_genotype(h)),
                     "alternative": space.alternative_genotype(h),
                     "alternative_genotype_canonical_sha256": space.genotype_sha256(space.alternative_genotype(h))}
            for h in horizons
        },
        "finalists": {
            "per_horizon": ["BEST: lowest complete fitness (tie: smaller class, then identity)",
                            "MINIMAL_CONTROL: the seed when it is not the best, else SMALL_ALTERNATIVE (the alternative, or the smallest complete S1 non-equivalent candidate)"],
            "claim": "SELECTED_BY_INNER_FOLD_SEARCH_NOT_PROVEN_OPTIMAL",
        },
        "weekly_rollout": {
            "arms": ["BEST", "MINIMAL_CONTROL|SMALL_ALTERNATIVE"],
            "refit": "weekly, 4 calendar years TRAIN per cutoff, random initial weights by default",
            "weeks": "all eligible 2024 weeks generated from the calendar (lane C); max 2*12*53 = 1272 dispositions",
            "annual_mae": "sum(error)/sum(n) with identical naive; weekly mean also recorded",
            "transfer_claim": "the search recipe is NOT transferred as an optimum; each week refits",
        },
        "prior_mapping": {"artifact": LEGACY_MAPPING_PATH, "sha256": LEGACY_MAPPING_SHA256,
                          "verdict": "48/48 genotypes MAPPED with their own v1 identity; fold_score_reuse NONE for all 48; no refit of the legacy context"},
        "stop_reasons": list(STOP_REASONS),
        "dispositions": list(DISPOSITIONS),
    }
    digest = _sha(_body(plan))
    plan["plan_sha256"] = digest
    plan["seal"] = {"algorithm": "sha256",
                    "canonical": "json sort_keys, separators (',',':'), allow_nan=False, UTF-8; keys plan_sha256 and seal excluded",
                    "sha256": digest}
    return plan


def verify_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("not an i6n search plan v1")
    digest = _sha(_body(plan))
    if plan.get("seal", {}).get("sha256") != digest or plan.get("plan_sha256") != digest:
        raise ValueError("plan seal mismatch")
    return plan


def _class_rules(plan):
    return {r["class_id"]: r for r in plan["promotion"]}


def open_classes(plan, generation, admitted):
    """Classes usable at this generation: admitted by pilot AND promoted."""
    rules = _class_rules(plan)
    return [c["class_id"] for c in plan["classes"]
            if c["class_id"] in set(admitted) and generation >= rules[c["class_id"]]["opens_at_generation"]]


def _rng(plan, h, generation):
    return random.Random(f"i6n-search-plan-v1:{plan['seed']}:{h}:gen{generation}")


def initial_population(plan, h, open_class_ids):
    """Seed + alternative + immigrants; requires S1 open (the seed is S1)."""
    if "S1" not in set(open_class_ids):
        raise WaitResource("S1 not admitted: the minimal seed cannot be proposed")
    size = plan["search"]["population_size"]
    pop = [space.seed_genotype(h), space.alternative_genotype(h)]
    seen = {space.genotype_sha256(g) for g in pop}
    rng = _rng(plan, h, 0)
    for _ in range(64 * size):
        if len(pop) >= size:
            break
        g = space.sample_genotype(h, rng, classes=open_class_ids)
        ident = space.genotype_sha256(g)
        if ident not in seen:
            seen.add(ident)
            pop.append(g)
    return pop


# --- fold fitness ----------------------------------------------------------------------------

def score_folds(plan, receipts, supports, require_distinct_support=True):
    """Unweighted mean over the plan folds of the per-fold TRAIN/inner ratio fitness."""
    folds = set(plan["folds"]["fold_ids"])
    if set(supports) != folds:
        raise ValueError(f"frozen support must cover exactly {sorted(folds)}")
    if set(receipts) - folds:
        raise ValueError(f"receipt for a fold outside the plan: {sorted(set(receipts) - folds)}")
    if set(receipts) != folds:
        raise Incomplete(f"missing folds {sorted(folds - set(receipts))}: no fitness before all folds")
    per_fold = {}
    digests = set()
    for fold in plan["folds"]["fold_ids"]:
        frozen = _v1._support(supports[fold])
        per_fold[fold] = _v1._score(receipts[fold], frozen)[0]
        digests.add((frozen["support_digest"]["train"], frozen["support_digest"]["inner"]))
    if require_distinct_support and len(digests) != len(folds):
        raise ValueError("fold supports are not distinct periods (identical support digests)")
    fitness = math.fsum(per_fold.values()) / len(per_fold)
    return {"status": "COMPLETE", "fitness": fitness, "per_fold": per_fold,
            "aggregation": plan["fitness"]["aggregation"]}


# --- per-horizon search state ---------------------------------------------------------------

def new_horizon_state(plan, h):
    verify_plan(plan)
    if h not in plan["horizons_hours"]:
        raise ValueError("horizon outside plan")
    return {"schema": STATE_SCHEMA, "plan_sha256": plan["plan_sha256"], "horizon_hours": h,
            "generation": 0, "phase": "PROPOSE", "status": "QUEUED", "stop_reason": None,
            "candidates": [], "cache": {}, "best_history": [], "new_fits_reserved": 0,
            "new_candidates": 0, "class_counts": {c["class_id"]: 0 for c in plan["classes"]},
            "attempts": 0, "filtered": []}


def register_cache(plan, state, verified):
    """Verified 4-fold fitness by genotype identity (all four fit_ids verified by B)."""
    state = deepcopy(state)
    for ident, fitness in verified.items():
        space._hex64(ident, "genotype identity")
        if type(fitness) not in (int, float) or not math.isfinite(fitness) or fitness < 0:
            raise ValueError("cached fitness must be finite and nonnegative")
        state["cache"][ident] = float(fitness)
    return state


class _Fit(base.Fitness):
    weights = (-1.0,)


class _Ind:
    def __init__(self, rec):
        self.rec = rec
        self.fitness = _Fit((rec["fitness"],))


def _record(plan, state, g, origin, parent=None):
    ident = space.genotype_sha256(g)
    return {"genotype": g, "genotype_canonical_sha256": ident,
            "search_candidate_sha256": space.search_candidate_sha256(g, plan["plan_sha256"]),
            "size_class": space.class_of(g), "generation": state["generation"], "origin": origin,
            "parent_genotype_sha256": parent, "effective_config": space.effective_config(g),
            "disposition": "PENDING", "fitness": None, "reason": None}


def propose_generation(plan, state, admitted):
    """Return (state, new candidates to fit on all folds). Never duplicates."""
    state = deepcopy(state)
    if state["stop_reason"] or state["phase"] != "PROPOSE":
        if state["phase"] == "EVALUATE":
            raise ValueError("current generation is still being evaluated")
        return state, []
    b = plan["budgets"]
    gen, h = state["generation"], state["horizon_hours"]
    opened = open_classes(plan, gen, admitted)
    if not opened or (gen == 0 and "S1" not in opened):
        state["status"] = "WAIT_RESOURCE"
        return state, []
    seen = {r["genotype_canonical_sha256"] for r in state["candidates"]}
    caps = b["per_class_new_candidates_per_horizon_max"]
    size = plan["search"]["population_size"]
    slots = []

    def admit(g, origin, parent=None):
        ident = space.genotype_sha256(g)
        if ident in seen:
            return False
        rec = _record(plan, state, g, origin, parent)
        if ident in state["cache"]:
            rec.update(disposition="CACHED_VERIFIED", fitness=state["cache"][ident])
        else:
            if state["new_candidates"] >= b["new_candidates_per_horizon_max"]:
                return False
            if state["class_counts"][rec["size_class"]] >= caps[rec["size_class"]]:
                state["filtered"].append({"generation": gen, "genotype_canonical_sha256": ident,
                                          "reasons": [{"code": "CLASS_BUDGET_EXHAUSTED", "gene": "*",
                                                       "detail": rec["size_class"]}]})
                return False
            state["new_candidates"] += 1
            state["class_counts"][rec["size_class"]] += 1
            state["new_fits_reserved"] += b["folds_per_candidate"]
        seen.add(ident)
        slots.append(rec)
        return True

    rng = _rng(plan, h, gen)
    if gen == 0:
        for i, g in enumerate(initial_population(plan, h, opened)):
            admit(g, "seed" if i == 0 else "alternative" if i == 1 else "immigrant")
    else:
        parents = [_Ind(r) for r in state["candidates"] if r["fitness"] is not None]
        while (len(slots) < size and state["new_candidates"] < b["new_candidates_per_horizon_max"]
               and state["attempts"] < b["max_proposal_attempts_per_horizon"]):
            state["attempts"] += 1
            if parents and rng.random() < plan["search"]["mutation"]:
                parent = deap_tools.selBest(rng.choices(parents, k=3), 1)[0].rec
                child = space.mutate(parent["genotype"], rng, classes=opened, exclude=seen,
                                     filtered=None)
                if child is None:
                    continue
                admit(child, "mutation", parent["genotype_canonical_sha256"])
            else:
                try:
                    g = space.sample_genotype(h, rng, classes=opened)
                except ValueError:
                    continue
                admit(g, "immigrant")
    state["candidates"].extend(slots)
    batch = [deepcopy(r) for r in slots if r["disposition"] == "PENDING"]
    if not slots:
        state["stop_reason"] = ("attempt_budget" if state["attempts"] >= b["max_proposal_attempts_per_horizon"]
                                else "admissible_generation_exhausted")
        state["status"] = "COMPLETE"
        return state, []
    state["phase"], state["status"] = "EVALUATE", "RUNNING"
    return state, batch


def _find(state, ident):
    for r in state["candidates"]:
        if r["genotype_canonical_sha256"] == ident and r["generation"] == state["generation"]:
            return r
    raise ValueError("unknown candidate for the current generation")


def record_fitness(plan, state, ident, fitness):
    state = deepcopy(state)
    r = _find(state, ident)
    if r["disposition"] != "PENDING":
        raise ValueError("candidate already terminal")
    if type(fitness) not in (int, float) or not math.isfinite(fitness) or fitness < 0:
        raise ValueError("fitness must be finite and nonnegative")
    r.update(disposition="COMPLETE", fitness=float(fitness))
    return state


def record_disposition(plan, state, ident, disposition, reason):
    if disposition not in ("FAILED", "INTERRUPTED", "NOT_ADMITTED") or not reason:
        raise ValueError("terminal non-score disposition requires a reason")
    state = deepcopy(state)
    r = _find(state, ident)
    if r["disposition"] != "PENDING":
        raise ValueError("candidate already terminal")
    r.update(disposition=disposition, fitness=None, reason=str(reason))
    return state


def no_improve_count(plan, state):
    hist, tol = state["best_history"], plan["search"]["tolerance"]
    if not hist:
        return 0
    ref, count = hist[0], 0
    for best in hist[1:]:
        if ref - best > tol:
            ref, count = best, 0
        else:
            count += 1
    return count


def close_generation(plan, state):
    """Barrier: every candidate of the generation is terminal on all folds."""
    state = deepcopy(state)
    current = [r for r in state["candidates"] if r["generation"] == state["generation"]]
    if any(r["disposition"] == "PENDING" for r in current):
        raise ValueError("generation not terminal on all folds")
    scores = [r["fitness"] for r in state["candidates"] if r["fitness"] is not None]
    if scores:
        state["best_history"].append(min(scores))
    state["generation"] += 1
    state["phase"] = "PROPOSE"
    if state["generation"] >= plan["search"]["generations"]:
        state["stop_reason"] = "generation_budget"
    elif state["new_candidates"] >= plan["budgets"]["new_candidates_per_horizon_max"]:
        state["stop_reason"] = "candidate_budget"
    elif no_improve_count(plan, state) >= plan["search"]["patience"]:
        state["stop_reason"] = "patience"
    state["status"] = "COMPLETE" if state["stop_reason"] else "QUEUED"
    return state


def select_finalists(plan, state):
    order = {c: i for i, c in enumerate(space.CLASS_IDS)}
    complete = sorted((r for r in state["candidates"] if r["fitness"] is not None),
                      key=lambda r: (r["fitness"], order[r["size_class"]], r["genotype_canonical_sha256"]))
    if not complete:
        return []
    h = state["horizon_hours"]
    best = complete[0]
    seed_id = space.genotype_sha256(space.seed_genotype(h))
    alt_id = space.genotype_sha256(space.alternative_genotype(h))
    by_id = {r["genotype_canonical_sha256"]: r for r in complete}

    def fin(r, role, claim):
        return {"role": role, "genotype": r["genotype"], "genotype_canonical_sha256": r["genotype_canonical_sha256"],
                "fitness": r["fitness"], "size_class": r["size_class"], "claim": claim}

    out = [fin(best, "BEST", plan["finalists"]["claim"])]
    best_id = best["genotype_canonical_sha256"]
    if seed_id != best_id and seed_id in by_id:
        out.append(fin(by_id[seed_id], "MINIMAL_CONTROL", "CONTROL_NOT_SELECTED_BY_FITNESS"))
    elif alt_id != best_id and alt_id in by_id:
        out.append(fin(by_id[alt_id], "SMALL_ALTERNATIVE", "CONTROL_NOT_SELECTED_BY_FITNESS"))
    else:
        small = [r for r in complete if r["size_class"] == "S1" and r["genotype_canonical_sha256"] != best_id]
        if small:
            out.append(fin(small[0], "SMALL_ALTERNATIVE", "CONTROL_NOT_SELECTED_BY_FITNESS"))
        else:
            out[0]["control_status"] = "CONTROL_UNAVAILABLE"
    return out


# --- config generator ---------------------------------------------------------------------------

def fit_specs(plan, h, genotypes, bindings):
    """One spec per (candidate, fold), carrying active/inactive genes and effective config."""
    verify_plan(plan)
    folds = plan["folds"]["fold_ids"]
    if set(bindings) != set(folds):
        raise ValueError(f"bindings must cover exactly {folds}")
    specs = []
    for g in genotypes:
        g = space.canonicalize(g)
        if g["horizon_hours"] != h:
            raise ValueError("genotype horizon mismatch")
        eff = space.effective_config(g)
        for fold in folds:
            binding = dict(bindings[fold])
            if binding.get("fold_id") != fold:
                raise ValueError("binding fold_id mismatch")
            specs.append({
                "schema": "i6n.fit_spec.v1", "plan_sha256": plan["plan_sha256"], "horizon_hours": h,
                "fold_id": fold, "fit_id": space.fit_id(g, binding),
                "genotype": g, "genotype_canonical_sha256": eff["genotype_canonical_sha256"],
                "search_candidate_sha256": space.search_candidate_sha256(g, plan["plan_sha256"]),
                "effective_config": eff, "effective_config_sha256": eff["effective_config_sha256"],
                "evaluator_request_config": eff["evaluator_config"], "size_class": eff["size_class"],
                "seed": plan["seed"],
                "max_physical_attempts": plan["budgets"]["max_physical_attempts_per_identity_fold"],
                "binding": binding,
            })
    return specs


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initial_configs(plan, output):
    output = Path(output)
    files = {}
    for h in plan["horizons_hours"]:
        pop = initial_population(plan, h, ["S1"])
        rel = f"h{h}/S1/generation0.json"
        files[rel] = _write(output / rel, {
            "schema": "i6n.generation_configs.v1", "plan_sha256": plan["plan_sha256"],
            "horizon_hours": h, "generation": 0, "open_classes": ["S1"],
            "candidates": [space.effective_config(g) for g in pop]})
    _write(output / "MANIFEST.json", {"schema": "i6n.config_manifest.v1", "plan_sha256": plan["plan_sha256"],
                                      "files": files, "fits_executed": 0,
                                      "note": "generation 0 assumes S1 admitted; later generations come from search state"})
    return files


# --- legacy mapping -----------------------------------------------------------------------------------

def legacy_mapping(states):
    """Map completed v1 controller records onto v2 identity. Private fields are dropped."""
    candidates = []
    for state in states:
        cell = state.get("identity", {}).get("cell", {}).get("config", {})
        for key, rec in sorted(state.get("records", {}).items()):
            wire = rec["wire"]
            mapped = space.legacy_map(wire["genome"], stored_identity=key)
            mapped.update(horizon_hours=wire["genome"]["horizon"],
                          legacy_wire_identity_matches=wire.get("identity") == key,
                          legacy_state=rec.get("state"), legacy_fitness=rec.get("fitness"),
                          legacy_evaluation_week_id=cell.get("evaluation_week_id"),
                          legacy_genome=wire["genome"])
            candidates.append(mapped)
    reasons = {}
    for c in candidates:
        for r in c["reasons"]:
            reasons[r["code"]] = reasons.get(r["code"], 0) + 1
    summary = {"n": len(candidates),
               "completed": sum(c["legacy_state"] == "COMPLETED" for c in candidates),
               "v1_identity_recomputed_all": all(c["v1_identity_recomputed"] for c in candidates),
               "mapped": sum(c["genotype_status"] == "MAPPED" for c in candidates),
               "out_of_v2_domain": sum(c["genotype_status"] == "OUT_OF_V2_DOMAIN" for c in candidates),
               "out_of_domain_reasons": reasons,
               "fold_score_reuse_none": sum(c["fold_score_reuse"] == "NONE" for c in candidates),
               "refits_scheduled": 0}
    return {"schema": "i6n.legacy_mapping.v1", "candidates": candidates, "summary": summary}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m tools.i6n_search_plan")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--output", required=True)
    a = sub.add_parser("verify")
    a.add_argument("--plan", required=True)
    a = sub.add_parser("initial-configs")
    a.add_argument("--plan", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("fit-specs")
    a.add_argument("--plan", required=True)
    a.add_argument("--horizon", type=int, required=True)
    a.add_argument("--bindings", required=True, help="JSON {fold_id: {experiment, fold_id, context_sha256, support_sha256}}")
    a.add_argument("--genotypes", help="JSON list of genotypes (default: generation-0 population with S1)")
    a.add_argument("--output", required=True)
    a = sub.add_parser("legacy-map")
    a.add_argument("--state", action="append", required=True)
    a.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.cmd == "plan":
        _write(Path(args.output), build_plan())
    elif args.cmd == "verify":
        plan = verify_plan(json.loads(Path(args.plan).read_text()))
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "status": "VERIFIED"}))
    elif args.cmd == "initial-configs":
        initial_configs(verify_plan(json.loads(Path(args.plan).read_text())), args.output)
    elif args.cmd == "fit-specs":
        plan = verify_plan(json.loads(Path(args.plan).read_text()))
        genotypes = (json.loads(Path(args.genotypes).read_text()) if args.genotypes
                     else initial_population(plan, args.horizon, ["S1"]))
        specs = fit_specs(plan, args.horizon, genotypes, json.loads(Path(args.bindings).read_text()))
        out = Path(args.output)
        files = {}
        for spec in specs:
            rel = f"h{args.horizon}/{spec['size_class']}/{spec['fold_id']}/{spec['fit_id']}.json"
            files[rel] = _write(out / rel, spec)
        _write(out / f"h{args.horizon}/MANIFEST.json", {"plan_sha256": plan["plan_sha256"], "files": files,
                                                       "fits_executed": 0})
    elif args.cmd == "legacy-map":
        states = [json.loads(Path(s).read_text())["payload"] for s in args.state]
        _write(Path(args.output), legacy_mapping(states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
