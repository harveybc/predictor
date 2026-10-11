"""Shared-only I6N DEAP adapter for doin_plugins.predictor.PredictorOptimizer.

No data or training imports. Evaluation requires an explicitly configured
receipt producer; see docs/i6n/DOIN_ADAPTER.md for the runtime boundaries.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib
import json
import math
import random
import subprocess

from deap import base, tools

from tools import i6n_search as search
from tools import i6n_search_plan as plan_v2
from tools import i6n_search_space as space


SCHEMA = "i6n-shared-deap-v1"
SCHEMA_V2 = "i6n-shared-deap-v2"


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


class _Fitness(base.Fitness):
    weights = (-1.,)


class _Individual:
    def __init__(self, genome, fitness):
        self.genome = genome
        self.fitness = _Fitness((fitness,))


class Plugin:
    """Deterministic generational DEAP selection, conditional mutation, immigrants.

    The node owns claims and result persistence. This class owns canonical
    proposal deduplication across generations, carried in innovation_tracker.
    There is no island-mode optimize() implementation or implicit evaluator.
    """

    # Required by app.plugin_loader; admission settings must be explicit.
    plugin_params = {}

    def __init__(self):
        self.params = {}

    def set_params(self, **kwargs):
        self.params.update(deepcopy(kwargs))

    def _config(self, opt_config):
        if not isinstance(opt_config, Mapping):
            raise ValueError("opt_config must be a mapping")
        cfg = deepcopy(self.params)
        cfg.update(deepcopy(dict(opt_config)))
        if cfg.get("higher_is_better", False) is not False:
            raise ValueError("I6N requires higher_is_better=False (raw minimization)")
        return cfg

    def _contract(self, cfg):
        horizon = cfg.get("i6n_horizon")
        search.default_genome(horizon)
        support = search._support(cfg.get("i6n_support"))
        context = cfg.get("i6n_evaluation_context")
        if not isinstance(context, Mapping) or not context:
            raise ValueError("i6n_evaluation_context must be a nonempty JSON mapping")
        contract = dict(schema=SCHEMA, horizon=horizon, support=support,
                        evaluation_context=deepcopy(dict(context)),
                        objective="mean_train_inner_endpoint_mae_ratio", higher_is_better=False)
        _json(contract)
        return contract

    def _settings(self, cfg, population_size):
        size = _int(population_size, "population_size", 1)
        generations = _int(cfg.get("n_generations", 10), "n_generations", 1)
        budget = _int(cfg.get("i6n_budget", size * generations), "i6n_budget", 1)
        return dict(population_size=size, n_generations=generations, budget=budget,
                    max_proposals=_int(cfg.get("i6n_max_proposals", 32 * budget),
                                       "i6n_max_proposals", 1),
                    patience=_int(cfg.get("optimization_patience", 5),
                                  "optimization_patience", 1))

    def _schedule(self, contract, settings):
        return [dict(name=f"i6n-h{contract['horizon']}", start_gen=0,
                     end_gen=settings["n_generations"], patience=settings["patience"])]

    def _wire(self, genome, context, generation):
        genome = search.normalize_genome(genome)
        return dict(schema=SCHEMA, genome=genome, identity=search.effective_identity(genome),
                    context=context, generation=generation)

    def _decode(self, item, context, generation, horizon):
        required = {"schema", "genome", "identity", "context", "generation"}
        if (not isinstance(item, Mapping) or not required <= set(item)
                or set(item) - required - {"fitness"} or item["schema"] != SCHEMA):
            raise ValueError("invalid I6N shared genome schema")
        if item["context"] != context:
            raise ValueError("genome evaluation context mismatch")
        if _int(item["generation"], "genome generation") != generation:
            raise ValueError("genome generation mismatch")
        genome = search.normalize_genome(item["genome"])
        if genome["horizon"] != horizon:
            raise ValueError("genome horizon mismatch")
        if search.effective_identity(genome) != item["identity"]:
            raise ValueError("genome identity mismatch")
        return genome

    def _propose(self, tracker, contract, settings, rng, parents):
        toolbox = base.Toolbox()
        toolbox.register("select", tools.selBest)
        toolbox.register("mutate", search.mutate, rng=rng)
        seen = set(tracker["seen"])
        population = []
        while (len(population) < settings["population_size"]
               and len(seen) < settings["budget"]
               and tracker["attempts"] < settings["max_proposals"]):
            tracker["attempts"] += 1
            if parents and rng.random() < .8:
                parent = toolbox.select(rng.choices(parents, k=3), 1)[0]
                genome = toolbox.mutate(parent.genome)
            else:
                genome = search.sample_genome(contract["horizon"], rng)
            wire = self._wire(genome, tracker["context"], tracker["generation"])
            if wire["identity"] in seen:
                continue
            seen.add(wire["identity"])
            population.append(wire)
        tracker["seen"] = sorted(seen)
        tracker["pending"] = [item["identity"] for item in population]
        return population

    def create_shared_population(self, population_size, opt_config, seed=6102026):
        cfg = self._config(opt_config)
        if "i6n_plan" in cfg:
            return _V2(self).create(population_size, cfg, seed)
        contract = self._contract(cfg)
        settings = self._settings(cfg, population_size)
        _int(seed, "seed")
        context = _digest(contract)
        tracker = dict(schema=SCHEMA, context=context, generation=0,
                       seen=[], pending=[], attempts=0, settings=settings)
        population = self._propose(tracker, contract, settings, random.Random(seed), [])
        if not population:
            raise RuntimeError("no unique initial candidates within proposal bounds")
        return deepcopy(dict(population=population, innovation_tracker=tracker,
                             stage_schedule=self._schedule(contract, settings),
                             param_defaults=settings, config_snapshot=contract,
                             generation=0, stage_idx=0, no_improve_count=0))

    def _evaluate(self, cfg, request):
        callback = cfg.get("i6n_evaluator")
        command = cfg.get("i6n_evaluator_command")
        if (callback is None) == (command is None):
            raise ValueError("configure exactly one I6N evaluator callable or JSON command")
        if command is not None:
            if (not isinstance(command, list) or not command
                    or any(not isinstance(arg, str) or not arg for arg in command)):
                raise ValueError("i6n_evaluator_command must be a nonempty argv list")
            timeout = cfg.get("i6n_evaluator_timeout", 120.)
            if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("i6n_evaluator_timeout must be finite and positive")
            result = subprocess.run(command, input=_json(request), text=True,
                                    capture_output=True, check=True, timeout=timeout,
                                    shell=False)
            return json.loads(result.stdout)
        if isinstance(callback, str):
            module, separator, name = callback.partition(":")
            if not separator or not module or not name:
                raise ValueError("i6n_evaluator import path must be module:callable")
            try:
                callback = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f"I6N evaluator unavailable: {callback}") from exc
        if not callable(callback):
            raise ValueError("I6N evaluator is not callable")
        return callback(deepcopy(request))

    def evaluate_single_genome(self, genome_serialized, generation, opt_config):
        cfg = self._config(opt_config)
        if "i6n_plan" in cfg:
            return _V2(self).evaluate(genome_serialized, generation, cfg)
        contract = self._contract(cfg)
        _int(generation, "generation")
        genome = self._decode(genome_serialized, _digest(contract), generation,
                              contract["horizon"])
        request = dict(config=search.effective_config(genome), generation=generation,
                       context=contract["evaluation_context"], support=contract["support"])
        try:
            receipt = self._evaluate(cfg, request)
            fitness, metrics = search._score(receipt, contract["support"])
        except Exception as exc:
            raise search.ScoreRejected(f"I6N evaluator rejected candidate: {exc}") from exc
        result = dict(fitness=fitness, metrics=metrics, hyper_dict=deepcopy(request["config"]))
        # The node's val_* dashboard fields mean INNER here, never outer/TEST.
        for split, prefix in (("train", "train"), ("inner", "val")):
            count = metrics["counts"][split]
            result[f"{prefix}_mae"] = metrics["error_sums"][split] / count
            result[f"{prefix}_naive_mae"] = metrics["naive_sums"][split] / count
        return result

    def reproduce_shared(self, evaluated_pop, generation, seed, opt_config,
                         innovation_tracker_data, stage_schedule, param_defaults,
                         current_stage_idx=0, no_improve_count=0):
        """Exact positional interface called by PredictorOptimizer.reproduce_shared."""
        cfg = self._config(opt_config)
        if "i6n_plan" in cfg:
            return _V2(self).reproduce(evaluated_pop, generation, seed, cfg, innovation_tracker_data,
                                       stage_schedule, param_defaults, current_stage_idx,
                                       no_improve_count)
        contract = self._contract(cfg)
        _int(generation, "generation")
        _int(seed, "seed")
        _int(no_improve_count, "no_improve_count")
        if _int(current_stage_idx, "current_stage_idx") != 0:
            raise ValueError("I6N supports one fixed-support stage per horizon")
        if not isinstance(param_defaults, Mapping):
            raise ValueError("invalid param_defaults")
        settings = self._settings(cfg, param_defaults.get("population_size"))
        if dict(param_defaults) != settings or stage_schedule != self._schedule(contract, settings):
            raise ValueError("shared settings/stage schedule changed")
        tracker = deepcopy(innovation_tracker_data)
        required = {"schema", "context", "generation", "seen", "pending", "attempts", "settings"}
        if not isinstance(tracker, dict) or set(tracker) != required or tracker["schema"] != SCHEMA:
            raise ValueError("invalid I6N innovation_tracker state")
        if tracker["context"] != _digest(contract):
            raise ValueError("shared evaluation context mismatch")
        if (_int(tracker["generation"], "tracker generation") != generation
                or tracker["settings"] != settings):
            raise ValueError("shared generation/settings mismatch")
        seen = tracker["seen"]
        if (not isinstance(seen, list) or any(not isinstance(v, str) or len(v) != 64 for v in seen)
                or seen != sorted(set(seen)) or len(seen) > settings["budget"]):
            raise ValueError("invalid canonical dedup state")
        attempts = _int(tracker["attempts"], "attempts")
        if not len(seen) <= attempts <= settings["max_proposals"]:
            raise ValueError("invalid proposal attempt state")
        if not isinstance(evaluated_pop, list) or not evaluated_pop:
            raise ValueError("expected evaluated population")
        parents, identities = [], []
        for item in evaluated_pop:
            genome = self._decode(item, tracker["context"], generation, contract["horizon"])
            identities.append(item["identity"])
            if "fitness" not in item:
                raise ValueError("population includes unevaluated genome")
            fitness = item["fitness"]
            if type(fitness) in (int, float) and math.isfinite(fitness) and fitness >= 0:
                parents.append(_Individual(genome, fitness))
        if (identities != tracker["pending"] or len(set(identities)) != len(identities)
                or not set(identities) <= set(seen)
                or len(identities) > settings["population_size"]):
            raise ValueError("population does not match reserved identities")
        reason = None
        if not parents:
            reason = "no_eligible_parents"
        elif generation + 1 >= settings["n_generations"]:
            reason = "generation_budget"
        elif no_improve_count >= settings["patience"]:
            reason = "patience"
        elif len(seen) >= settings["budget"]:
            reason = "candidate_budget"
        elif attempts >= settings["max_proposals"]:
            reason = "attempt_budget"
        tracker["generation"] = generation + 1
        tracker["pending"] = []
        population = []
        if reason is None:
            population = self._propose(tracker, contract, settings, random.Random(seed), parents)
            if not population:
                reason = "attempt_budget"
        return deepcopy(dict(population=population, generation=generation + 1,
                             innovation_tracker=tracker, stage_schedule=stage_schedule,
                             param_defaults=settings, stage_idx=0, stage_advanced=False,
                             no_improve_count=no_improve_count, patience=settings["patience"],
                             converged=reason is not None, stop_reason=reason))


class _V2:
    """Plan-v2 path: sealed plan, four folds per candidate, staged size classes.

    Proposal/selection state lives in tools.i6n_search_plan (JSON, replayable);
    this adapter only binds it to the DOIN shared-mode positional interface.
    Admitted classes come from opt_config (pilot receipts owned by lane B) and
    are deliberately excluded from the context digest: admission can change
    during a run without changing what a fitness means.
    """

    def __init__(self, plugin):
        self.plugin = plugin

    def _contract(self, cfg):
        plan = plan_v2.verify_plan(deepcopy(cfg["i6n_plan"]))
        horizon = cfg.get("i6n_horizon")
        if horizon not in plan["horizons_hours"]:
            raise ValueError("i6n_horizon outside the plan")
        folds = cfg.get("i6n_folds")
        if not isinstance(folds, Mapping):
            raise ValueError("i6n_folds must bind exactly the plan folds")
        order = plan_v2.resolve_fold_ids(plan, folds)  # one lane-C fold id per plan quarter
        bound = {}
        for fold in order:
            item = folds[fold]
            context = item.get("i6n_evaluation_context") if isinstance(item, Mapping) else None
            if not isinstance(context, Mapping) or not context:
                raise ValueError(f"fold {fold} requires a nonempty i6n_evaluation_context")
            bound[fold] = dict(support=search._support(item.get("i6n_support")),
                               evaluation_context=deepcopy(dict(context)))
            if "i6n_origins_sha256" in item:  # lane C origins per split for the endpoint rows
                bound[fold]["origins_sha256"] = deepcopy(item["i6n_origins_sha256"])
        contract = dict(schema=SCHEMA_V2, plan_sha256=plan["plan_sha256"], horizon=horizon,
                        fold_order=order, folds=bound, objective="mean_over_folds_train_inner_endpoint_mae_ratio",
                        higher_is_better=False)
        _json(contract)
        return plan, contract

    @staticmethod
    def _admitted(cfg):
        admitted = cfg.get("i6n_admitted_classes", [])
        if not isinstance(admitted, (list, tuple)) or set(admitted) - set(space.CLASS_IDS):
            raise ValueError("i6n_admitted_classes must list size class ids")
        return set(admitted)

    @staticmethod
    def _settings(plan):
        s = plan["search"]
        return dict(population_size=s["population_size"], n_generations=s["generations"],
                    budget=plan["budgets"]["new_candidates_per_horizon_max"],
                    patience=s["patience"], tolerance=s["tolerance"])

    @staticmethod
    def _schedule(plan, contract):
        return [dict(name=f"i6n-v2-h{contract['horizon']}", start_gen=0,
                     end_gen=plan["search"]["generations"], patience=plan["search"]["patience"])]

    @staticmethod
    def _wire(genotype, context, generation):
        g = space.canonicalize(genotype)
        return dict(schema=SCHEMA_V2, genotype=g, genome=space.to_evaluator_genome(g),
                    identity=space.genotype_sha256(g), context=context, generation=generation)

    @staticmethod
    def _decode(item, context, generation, horizon):
        required = {"schema", "genotype", "genome", "identity", "context", "generation"}
        if (not isinstance(item, Mapping) or not required <= set(item)
                or set(item) - required - {"fitness"} or item["schema"] != SCHEMA_V2):
            raise ValueError("invalid I6N v2 shared genome schema")
        if item["context"] != context:
            raise ValueError("genome evaluation context mismatch")
        if _int(item["generation"], "genome generation") != generation:
            raise ValueError("genome generation mismatch")
        g = space.canonicalize(item["genotype"])
        if g != item["genotype"] or g["horizon_hours"] != horizon:
            raise ValueError("genotype not canonical or wrong horizon")
        if item["genome"] != space.to_evaluator_genome(g) or item["identity"] != space.genotype_sha256(g):
            raise ValueError("genome identity mismatch")
        return g

    def _population(self, state, context):
        current = [r for r in state["candidates"]
                   if r["generation"] == state["generation"] and r["disposition"] == "PENDING"]
        return [self._wire(r["genotype"], context, state["generation"]) for r in current]

    def create(self, population_size, cfg, seed):
        plan, contract = self._contract(cfg)
        if _int(seed, "seed") != plan["seed"]:
            raise ValueError("shared population seed must equal the plan seed")
        if _int(population_size, "population_size", 1) != plan["search"]["population_size"]:
            raise ValueError("population_size must equal the plan population size")
        context = _digest(contract)
        state = plan_v2.new_horizon_state(plan, contract["horizon"])
        state, batch = plan_v2.propose_generation(plan, state, self._admitted(cfg))
        if not batch:
            raise RuntimeError(f"no initial candidates: {state['status']} {state['stop_reason']}")
        settings = self._settings(plan)
        tracker = dict(schema=SCHEMA_V2, context=context, state=state)
        return deepcopy(dict(population=self._population(state, context), innovation_tracker=tracker,
                             stage_schedule=self._schedule(plan, contract), param_defaults=settings,
                             config_snapshot=dict(contract, plan_sha256=plan["plan_sha256"]),
                             generation=0, stage_idx=0, no_improve_count=0))

    def evaluate(self, genome_serialized, generation, cfg):
        plan, contract = self._contract(cfg)
        _int(generation, "generation")
        g = self._decode(genome_serialized, _digest(contract), generation, contract["horizon"])
        config = space.evaluator_config(g)
        receipts, supports = {}, {}
        try:
            for fold in contract["fold_order"]:
                bound = contract["folds"][fold]
                fold_cfg = dict(cfg)
                fold_cfg.update({k: v for k, v in cfg["i6n_folds"][fold].items()
                                 if k in ("i6n_evaluator_command", "i6n_evaluator_timeout")})
                request = dict(config=deepcopy(config), generation=generation,
                               context=deepcopy(bound["evaluation_context"]),
                               support=deepcopy(bound["support"]))
                receipts[fold] = self.plugin._evaluate(fold_cfg, request)
                supports[fold] = bound["support"]
            origins = {f: contract["folds"][f]["origins_sha256"] for f in contract["fold_order"]
                       if "origins_sha256" in contract["folds"][f]}
            if origins and len(origins) != len(contract["fold_order"]):
                raise ValueError("i6n_origins_sha256 must be given for every fold or none")
            scored = plan_v2.score_folds(plan, receipts, supports,
                                         genotype_canonical_sha256=space.genotype_sha256(g),
                                         horizon_hours=contract["horizon"],
                                         origins_sha256=origins or None)
        except Exception as exc:
            raise search.ScoreRejected(f"I6N v2 evaluator rejected candidate: {exc}") from exc
        metrics = json.loads(_json(dict(per_fold=scored["per_fold"], receipts=receipts,
                                        origins_binding=scored["origins_binding"],
                                        aggregation=scored["aggregation"])))
        result = dict(fitness=scored["fitness"], metrics=metrics, hyper_dict=config,
                      genotype_canonical_sha256=space.genotype_sha256(g))
        # Dashboard fields pool sums over folds; INNER is reported as val_*, never outer/TEST.
        for split, prefix in (("train", "train"), ("inner", "val")):
            n = sum(r["counts"][split] for r in receipts.values())
            result[f"{prefix}_mae"] = math.fsum(r["error_sums"][split] for r in receipts.values()) / n
            result[f"{prefix}_naive_mae"] = math.fsum(r["naive_sums"][split] for r in receipts.values()) / n
        return result

    def reproduce(self, evaluated_pop, generation, seed, cfg, tracker_data, stage_schedule,
                  param_defaults, current_stage_idx, no_improve_count):
        plan, contract = self._contract(cfg)
        if _int(seed, "seed") != plan["seed"] or _int(current_stage_idx, "current_stage_idx") != 0:
            raise ValueError("v2 requires the plan seed and a single stage")
        _int(no_improve_count, "no_improve_count")
        if dict(param_defaults) != self._settings(plan) or stage_schedule != self._schedule(plan, contract):
            raise ValueError("shared settings/stage schedule changed")
        tracker = deepcopy(tracker_data)
        if (not isinstance(tracker, dict) or set(tracker) != {"schema", "context", "state"}
                or tracker["schema"] != SCHEMA_V2 or tracker["context"] != _digest(contract)):
            raise ValueError("invalid I6N v2 innovation_tracker state")
        state = tracker["state"]
        if state.get("plan_sha256") != plan["plan_sha256"] or state["generation"] != _int(generation, "generation"):
            raise ValueError("shared generation/plan mismatch")
        pending = [r["genotype_canonical_sha256"] for r in state["candidates"]
                   if r["generation"] == generation and r["disposition"] == "PENDING"]
        if not isinstance(evaluated_pop, list):
            raise ValueError("expected evaluated population")
        identities = []
        for item in evaluated_pop:
            g = self._decode(item, tracker["context"], generation, contract["horizon"])
            ident = space.genotype_sha256(g)
            identities.append(ident)
            fitness = item.get("fitness")
            if type(fitness) in (int, float) and math.isfinite(fitness) and fitness >= 0:
                state = plan_v2.record_fitness(plan, state, ident, fitness)
            else:
                state = plan_v2.record_disposition(plan, state, ident, "FAILED",
                                                   "no admissible four-fold fitness from the swarm")
        if sorted(identities) != sorted(pending) or len(set(identities)) != len(identities):
            raise ValueError("population does not match reserved identities")
        if state["phase"] == "EVALUATE":
            state = plan_v2.close_generation(plan, state)
        # phase PROPOSE here means a previous WAIT_RESOURCE: re-propose, never close twice
        population = []
        if not state["stop_reason"]:
            state, batch = plan_v2.propose_generation(plan, state, self._admitted(cfg))
            population = self._population(state, tracker["context"]) if batch else []
        tracker["state"] = state
        converged = bool(state["stop_reason"])
        out = dict(population=population, generation=state["generation"], innovation_tracker=tracker,
                   stage_schedule=stage_schedule, param_defaults=self._settings(plan), stage_idx=0,
                   stage_advanced=False, no_improve_count=plan_v2.no_improve_count(plan, state),
                   patience=plan["search"]["patience"], converged=converged,
                   stop_reason=state["stop_reason"], status=state["status"])
        if converged:
            out["finalists"] = plan_v2.select_finalists(plan, state)
        return deepcopy(out)
