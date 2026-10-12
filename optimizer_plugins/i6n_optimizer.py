"""I6N DEAP adapter for shared pilots and cooperative DOIN islands.

No data or training imports. Evaluation requires an explicitly configured
receipt producer; see docs/i6n/DOIN_ADAPTER.md for the runtime boundaries.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
import time

from deap import base, tools

from tools import i6n_search as search
from tools import i6n_contamination as contamination
from tools import i6n_island_plan
from tools.i6n_contracts import feature_availability as availability
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


def _nested_tuple(value):
    """Restore ``random.Random`` state after JSON converted tuples to lists."""
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


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
    ``optimize`` is the production island path.  The historical shared methods
    remain available so retained pilot evidence stays reproducible.
    """

    # Required by app.plugin_loader; admission settings must be explicit.
    plugin_params = {}

    def __init__(self):
        self.params = {}
        self.best_fitness_so_far = float("inf")
        self.best_params_so_far = None
        self.best_train_mae_so_far = None
        self.best_train_naive_mae_so_far = None
        self.best_val_mae_so_far = None
        self.best_naive_mae_so_far = None
        self.best_test_mae_so_far = None
        self.best_test_naive_mae_so_far = None
        self._island_terminal = None
        self._runtime_statistics = {"candidate_evaluations_total": 0}
        self._candidate_durations = []

    def set_params(self, **kwargs):
        self.params.update(deepcopy(kwargs))

    def island_terminal(self):
        """Return the explicit terminal produced by the local-island run."""
        return deepcopy(self._island_terminal)

    def get_runtime_statistics(self):
        """Return a snapshot consumed by DOIN's live campaign dashboard."""
        return deepcopy(self._runtime_statistics)

    @staticmethod
    def _percentile(values, fraction):
        ordered = sorted(float(value) for value in values)
        if not ordered:
            raise ValueError("percentile requires at least one value")
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def _publish_runtime_statistics(self, *, source, total):
        snapshot = {
            "candidate_evaluations_total": int(total),
            "candidate_history_source": source,
        }
        if self._candidate_durations:
            recent = self._candidate_durations[-50:]
            mean_seconds = statistics.fmean(recent)
            snapshot.update({
                "candidates_per_hour": 3600.0 / mean_seconds,
                "candidate_seconds_median": statistics.median(recent),
                "candidate_seconds_p25": self._percentile(recent, 0.25),
                "candidate_seconds_p75": self._percentile(recent, 0.75),
            })
        self._runtime_statistics = snapshot

    def _record_island_terminal(
        self, *, settings, plan_sha256, contract_sha256,
        checkpoint_identity_sha256, generation, stop_reason, champion_sha256,
        champion_fitness, population, cache,
    ):
        if stop_reason not in {"patience", "generation_limit"}:
            raise ValueError("island terminal requires a governed stop reason")
        if not isinstance(champion_sha256, str) or not champion_sha256:
            raise ValueError("island terminal requires a champion identity")
        if type(champion_fitness) not in (int, float) or not math.isfinite(champion_fitness):
            raise ValueError("island terminal requires a finite champion fitness")
        population_ids = [space.genotype_sha256(genotype) for genotype in population]
        candidate_ledger = {
            identity: cache[identity]
            for identity in sorted(cache)
        }
        self._island_terminal = {
            "status": "COMPLETE",
            "island_id": settings["island_id"],
            "horizon": settings["horizon"],
            "seed": settings["seed"],
            "stop_reason": stop_reason,
            "terminal_generation": generation,
            "champion_sha256": champion_sha256,
            "plan_sha256": plan_sha256,
            "contract_sha256": contract_sha256,
            "checkpoint_identity_sha256": checkpoint_identity_sha256,
            "champion_fitness": float(champion_fitness),
            "final_population_sha256": _digest(population_ids),
            "candidate_ledger_sha256": _digest(candidate_ledger),
            "evaluated_candidate_count": len(candidate_ledger),
        }

    @staticmethod
    def _atomic_json(path, payload):
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".i6n-", suffix=".tmp", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _config(self, opt_config):
        if not isinstance(opt_config, Mapping):
            raise ValueError("opt_config must be a mapping")
        runtime_keys = {"i6n_evaluator", "optimization_callbacks"}
        declared_runtime = opt_config.get("_non_serializable_keys", ())
        if not isinstance(declared_runtime, (set, frozenset, list, tuple)):
            raise ValueError("_non_serializable_keys must be a collection")
        declared_runtime = set(declared_runtime)
        if (any(not isinstance(key, str) for key in declared_runtime)
                or not declared_runtime <= runtime_keys):
            raise ValueError("_non_serializable_keys contains unsupported runtime keys")
        serializable = {
            key: value for key, value in opt_config.items()
            if key not in runtime_keys and key != "_non_serializable_keys"
        }
        cfg = deepcopy(self.params)
        cfg.update(deepcopy(serializable))
        # Evaluators/callbacks are runtime capabilities, not configuration
        # values.  Preserve their identity instead of cloning stateful callables.
        for key in sorted(runtime_keys):
            if key in opt_config:
                cfg[key] = opt_config[key]
        if cfg.get("higher_is_better", False) is not False:
            raise ValueError("I6N requires higher_is_better=False (raw minimization)")
        return cfg

    def _island_settings(self, cfg):
        """Validate the non-negotiable production search envelope."""
        if cfg.get("i6n_island_mode") is not True:
            raise ValueError("i6n_island_mode=true is required for local DOIN optimization")
        horizon = cfg.get("i6n_horizon")
        if horizon not in (4, 72):
            raise ValueError("i6n_horizon must be 4 or 72")
        population = _int(cfg.get("population_size"), "population_size", 20)
        generations = _int(cfg.get("n_generations"), "n_generations", 100)
        patience = _int(cfg.get("optimization_patience"), "optimization_patience", 10)
        tolerance = cfg.get("min_relative_improvement")
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance):
            raise ValueError("min_relative_improvement must be finite")
        if float(tolerance) != 1e-8:
            raise ValueError("min_relative_improvement must equal 1e-8")
        seed = _int(cfg.get("ga_seed", cfg.get("seed")), "ga_seed")
        if seed != 6102026:
            raise ValueError("the four islands must use seed 6102026")
        island_id = cfg.get("i6n_island_id")
        if island_id is None:
            island_id = os.environ.get("I6N_ISLAND_ID", "local-test")
        if (not isinstance(island_id, str) or not island_id
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in island_id)):
            raise ValueError("i6n_island_id must contain only letters, digits, '-' or '_'")
        control_island_id = cfg.get("i6n_control_island_id", island_id)
        if (not isinstance(control_island_id, str) or not control_island_id
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                       for char in control_island_id)):
            raise ValueError(
                "i6n_control_island_id must contain only letters, digits, '-' or '_'"
            )
        return {
            "horizon": horizon,
            "population": population,
            "generations": generations,
            "patience": patience,
            "tolerance": float(tolerance),
            "seed": seed,
            "island_id": island_id,
            "control_island_id": control_island_id,
            "mutation": float(cfg.get("mutation_probability", 0.8)),
            "immigration": float(cfg.get("immigration_probability", 0.2)),
            "s2_generation": _int(cfg.get("i6n_s2_generation", 10), "i6n_s2_generation"),
            "s3_generation": _int(cfg.get("i6n_s3_generation", 25), "i6n_s3_generation"),
            "min_generations": _int(cfg.get("i6n_min_generations", 30),
                                    "i6n_min_generations", 30),
        }

    @staticmethod
    def _runtime_path(template, settings):
        """Resolve a host-local path without changing the shared config digest."""
        if template is None:
            return None
        if not isinstance(template, str) or not template:
            raise ValueError("runtime path must be a nonempty string")
        try:
            return template.format(
                island_id=settings["island_id"],
                horizon=settings["horizon"],
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError("runtime path contains an unsupported placeholder") from exc

    @staticmethod
    def _summary(result):
        return {
            key: deepcopy(result.get(key))
            for key in (
                "fitness", "hyper_dict", "genotype_canonical_sha256",
                "train_mae", "train_naive_mae", "val_mae", "val_naive_mae",
                "metrics",
            )
        }

    @staticmethod
    def _callable_identity(value):
        """Return a stable code identity for a configured runtime evaluator."""
        target = value if isinstance(value, type) or hasattr(value, "__qualname__") else type(value)
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise ValueError("i6n_evaluator has no stable module/qualified-name identity")
        return f"{module}:{qualname}"

    @staticmethod
    def _matching_identity_fields(value, markers):
        """Extract named identities even when they live inside fold bindings."""
        matched = {}

        def visit(current, path):
            if isinstance(current, Mapping):
                for key in sorted(current):
                    child = current[key]
                    child_path = path + (str(key),)
                    if any(marker in str(key).lower() for marker in markers):
                        matched[".".join(child_path)] = deepcopy(child)
                    else:
                        visit(child, child_path)
            elif isinstance(current, (list, tuple)):
                for index, child in enumerate(current):
                    visit(child, path + (str(index),))

        visit(value, ())
        return matched

    @classmethod
    def _checkpoint_identity(cls, cfg, plan, contract, settings, admitted, search_seed):
        """Bind every scientific input that makes a resumed island comparable."""
        config = {}
        for key, value in cfg.items():
            if key == "optimization_callbacks":
                continue
            if key == "i6n_evaluator":
                config["i6n_evaluator_runtime_identity"] = cls._callable_identity(value)
            else:
                config[key] = deepcopy(value)
        try:
            _json(config)
        except (TypeError, ValueError) as exc:
            raise ValueError("I6N checkpoint configuration must be canonical JSON") from exc

        features = cls._matching_identity_fields(config, ("feature", "manifest"))
        evaluator = cls._matching_identity_fields(config, ("evaluator",))
        folds = {
            "fold_order": deepcopy(contract["fold_order"]),
            "folds": deepcopy(contract["folds"]),
        }
        support = {
            fold: {
                "support": deepcopy(contract["folds"][fold]["support"]),
                "origins_sha256": deepcopy(contract["folds"][fold].get("origins_sha256")),
            }
            for fold in contract["fold_order"]
        }
        admitted_ordered = sorted(admitted)
        return {
            "schema": "i6n.island.checkpoint.identity.v1",
            "plan_sha256": plan["plan_sha256"],
            "config_sha256": _digest(config),
            "contract_sha256": _digest(contract),
            "folds_sha256": _digest(folds),
            "support_sha256": _digest(support),
            "features_sha256": _digest(features),
            "evaluator_sha256": _digest(evaluator),
            "admitted_classes": admitted_ordered,
            "admitted_classes_sha256": _digest(admitted_ordered),
            "horizon": settings["horizon"],
            "seed": settings["seed"],
            "island_id": settings["island_id"],
            "search_seed": search_seed,
        }

    @staticmethod
    def _point_in_time_admission(cfg, plan, contract):
        """Successor plans (owner order 2026-10-12) bind a gated feature manifest.

        Returns True for such plans after refusing manifest drift and any fold
        context that carries a point-in-time-contaminated identity.  Legacy
        plans (no ``feature_manifest``) are unchanged and return False.
        """
        sealed = plan.get("feature_manifest")
        if sealed is None:
            return False
        manifest = cfg.get("i6n_feature_manifest")
        if (not isinstance(manifest, Mapping) or manifest.get("sha256") != sealed["sha256"]
                or manifest.get("features") != sealed["features"]):
            raise ValueError("i6n_feature_manifest differs from the sealed plan feature manifest")
        if availability.manifest_sha256(sealed["features"]) != sealed["sha256"]:
            raise ValueError("sealed plan feature manifest digest mismatch")
        for fold in contract["fold_order"]:
            context = contract["folds"][fold]["evaluation_context"]
            if (context.get("schema") != "i6n.historical_evaluator.v3"
                    or context.get("experiment") != i6n_island_plan.EXPERIMENT_V3):
                raise ValueError(
                    f"POINT_IN_TIME_CONTAMINATED_CONTEXT: fold {fold} is not bound to the clean H72 experiment"
                )
        contamination.assert_valid_skill({"folds": contract["folds"],
                                          "feature_manifest_sha256": sealed["sha256"]})
        return True

    @staticmethod
    def _network_genotype(params, horizon, envelope):
        if not isinstance(params, Mapping):
            return None
        if params.get("_i6n_envelope") != envelope:
            return None
        try:
            genome = {key: params[key] for key in search._GENOME_KEYS}
            genotype = space.canonicalize(space.from_evaluator_genome(genome))
        except (KeyError, TypeError, ValueError, space.Infeasible):
            return None
        return genotype if genotype["horizon_hours"] == horizon else None

    @staticmethod
    def _open_classes(generation, settings, admitted):
        open_ids = ["S1"]
        if generation >= settings["s2_generation"]:
            open_ids.append("S2")
        if generation >= settings["s3_generation"]:
            open_ids.append("S3")
        return [name for name in open_ids if name in admitted]

    def _initial_island_population(self, settings, admitted, rng):
        classes = self._open_classes(0, settings, admitted)
        if "S1" not in classes:
            raise ValueError("S1 must be admitted before an island can start")
        fixed = [space.seed_genotype(settings["horizon"]),
                 space.alternative_genotype(settings["horizon"])]
        population = fixed if settings["island_id"] == settings["control_island_id"] else []
        # Non-control islands also reserve these identities so they cannot
        # accidentally repeat the governed controls through random sampling.
        seen = {space.genotype_sha256(g) for g in fixed}
        while len(population) < settings["population"]:
            candidate = space.sample_genotype(settings["horizon"], rng, classes=classes)
            identity = space.genotype_sha256(candidate)
            if identity not in seen:
                seen.add(identity)
                population.append(candidate)
        return population

    def _next_island_population(self, ranked, settings, admitted, rng):
        classes = self._open_classes(ranked[0]["generation"] + 1, settings, admitted)
        if not classes:
            raise RuntimeError("no admitted class remains open")
        population = [deepcopy(ranked[0]["genotype"])]
        seen = {space.genotype_sha256(population[0])}
        attempts = 0
        while len(population) < settings["population"] and attempts < settings["population"] * 256:
            attempts += 1
            if rng.random() < settings["immigration"]:
                candidate = space.sample_genotype(settings["horizon"], rng, classes=classes)
            else:
                tournament = rng.sample(ranked, min(3, len(ranked)))
                parent = min(tournament, key=lambda row: row["fitness"])["genotype"]
                candidate = space.mutate(parent, rng, classes=classes, exclude=seen)
                if candidate is None:
                    continue
            identity = space.genotype_sha256(candidate)
            if identity not in seen:
                seen.add(identity)
                population.append(candidate)
        if len(population) != settings["population"]:
            raise RuntimeError("unique I6N island population exhausted before reaching configured size")
        return population

    def optimize(self, predictor, preprocessor, opt_config):
        """Run one full local island; DOIN callbacks exchange champions.

        Each candidate is the governed four-fold evaluation.  The optimized
        value is the relative error against the paired naive: ratio - 1.
        """
        del predictor, preprocessor
        cfg = self._config(opt_config)
        self._island_terminal = None
        cfg.setdefault("i6n_plan", i6n_island_plan.build_plan())
        settings = self._island_settings(cfg)
        plan, contract = _V2(self)._contract(cfg)
        if (plan["seed"] != settings["seed"]
                or plan["search"]["population_size"] != settings["population"]
                or plan["search"]["generations"] != settings["generations"]
                or plan["search"]["patience"] != settings["patience"]
                or plan["search"]["min_generations_before_stop"] != settings["min_generations"]
                or float(plan["search"]["tolerance"]) != settings["tolerance"]):
            raise ValueError("runtime island settings differ from the sealed plan")
        admitted = _V2._admitted(cfg)
        gated = self._point_in_time_admission(cfg, plan, contract)
        callbacks = cfg.get("optimization_callbacks") or {}
        plan_identity = plan["plan_sha256"]
        search_seed = int.from_bytes(hashlib.sha256(_json({
            "plan_sha256": plan_identity,
            "horizon": settings["horizon"], "training_seed": settings["seed"],
            "island_id": settings["island_id"],
        }).encode("utf-8")).digest()[:8], "big")
        checkpoint_identity = self._checkpoint_identity(
            cfg, plan, contract, settings, admitted, search_seed
        )
        checkpoint_identity_sha256 = _digest(checkpoint_identity)
        contract_sha256 = checkpoint_identity["contract_sha256"]
        runtime_source = (
            f"i6n:{plan_identity}:{settings['horizon']}:{settings['island_id']}"
        )
        rng = random.Random(search_seed)
        envelope = {
            "schema": "i6n.island.champion.v1", "plan_sha256": plan["plan_sha256"],
            "context_sha256": _digest(contract), "horizon": settings["horizon"],
            "training_seed": settings["seed"], "genotype_schema": space.GENOTYPE_SCHEMA,
        }
        population = self._initial_island_population(settings, admitted, rng)
        cache = {}
        rejections = {}
        start_generation = 0
        no_improve = 0
        best_identity = None
        resume_generation_in_progress = False
        resumed_from_checkpoint = False
        checkpoint = self._runtime_path(cfg.get("optimization_resume_file"), settings)
        if checkpoint and os.path.exists(checkpoint):
            with open(checkpoint, encoding="utf-8") as handle:
                saved = json.load(handle)
            if gated:  # never resume a point-in-time-contaminated island state
                contamination.assert_valid_skill(saved.get("checkpoint_identity", {}))
            if (saved.get("schema") != "i6n.island.checkpoint.v2"
                    or saved.get("checkpoint_identity") != checkpoint_identity
                    or saved.get("checkpoint_identity_sha256") != checkpoint_identity_sha256):
                raise ValueError("island resume checkpoint identity mismatch")
            population = [space.canonicalize(g) for g in saved["population"]]
            cache = deepcopy(saved.get("cache", {}))
            rejections = deepcopy(saved.get("rejections", {}))
            if not isinstance(rejections, dict):
                raise ValueError("invalid candidate rejection checkpoint")
            start_generation = _int(saved["generation"], "checkpoint generation")
            resume_generation_in_progress = saved.get("generation_complete", True) is False
            resumed_from_checkpoint = True
            no_improve = _int(saved.get("no_improve", 0), "checkpoint no_improve")
            retained_best = saved.get("best_fitness")
            self.best_fitness_so_far = (
                float("inf") if retained_best is None else float(retained_best)
            )
            self.best_params_so_far = deepcopy(saved.get("best_params"))
            self.best_train_mae_so_far = saved.get("best_train_mae")
            self.best_train_naive_mae_so_far = saved.get("best_train_naive_mae")
            self.best_val_mae_so_far = saved.get("best_val_mae")
            self.best_naive_mae_so_far = saved.get("best_val_naive_mae")
            best_identity = saved.get("best_identity")
            rng.setstate(_nested_tuple(saved["rng_state"]))
            self._publish_runtime_statistics(source=runtime_source, total=len(cache))
            stop_reason = saved.get("stop_reason")
            if resume_generation_in_progress and stop_reason is not None:
                raise ValueError("in-progress checkpoint cannot declare a stop reason")
            if stop_reason == "patience":
                if (start_generation >= settings["min_generations"]
                        and no_improve >= settings["patience"]):
                    self._record_island_terminal(
                        settings=settings, plan_sha256=plan_identity,
                        contract_sha256=contract_sha256,
                        checkpoint_identity_sha256=checkpoint_identity_sha256,
                        generation=start_generation, stop_reason=stop_reason,
                        champion_sha256=best_identity,
                        champion_fitness=self.best_fitness_so_far,
                        population=population, cache=cache,
                    )
                    return deepcopy(self.best_params_so_far)
                if no_improve < settings["patience"]:
                    raise ValueError("checkpoint patience stop contradicts its counter")
                # A checkpoint carrying the former writer bug can claim patience
                # before the governed minimum. Resume its retained population.
                stop_reason = None
            if stop_reason == "generation_limit":
                if start_generation < settings["generations"]:
                    raise ValueError("checkpoint generation-limit stop precedes its limit")
                self._record_island_terminal(
                    settings=settings, plan_sha256=plan_identity,
                    contract_sha256=contract_sha256,
                    checkpoint_identity_sha256=checkpoint_identity_sha256,
                    generation=start_generation, stop_reason=stop_reason,
                    champion_sha256=best_identity,
                    champion_fitness=self.best_fitness_so_far,
                    population=population, cache=cache,
                )
                return deepcopy(self.best_params_so_far)
            if stop_reason is not None:
                raise ValueError("checkpoint has an unknown stop reason")

        def save_checkpoint(*, generation_value, population_value,
                            generation_complete, stop_reason=None):
            if not checkpoint:
                return
            self._atomic_json(checkpoint, {
                "schema": "i6n.island.checkpoint.v2",
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
                "plan_sha256": plan_identity,
                "horizon": settings["horizon"], "seed": settings["seed"],
                "island_id": settings["island_id"], "search_seed": search_seed,
                "generation": generation_value,
                "generation_complete": generation_complete,
                "population": population_value,
                "cache": cache, "rejections": rejections,
                "no_improve": no_improve,
                "best_fitness": (self.best_fitness_so_far
                                 if math.isfinite(self.best_fitness_so_far) else None),
                "best_params": self.best_params_so_far,
                "best_identity": best_identity, "rng_state": rng.getstate(),
                "best_train_mae": self.best_train_mae_so_far,
                "best_train_naive_mae": self.best_train_naive_mae_so_far,
                "best_val_mae": self.best_val_mae_so_far,
                "best_val_naive_mae": self.best_naive_mae_so_far,
                "stop_reason": stop_reason,
            })

        evaluator = _V2(self)
        total_eval = len(cache)
        self._publish_runtime_statistics(source=runtime_source, total=total_eval)
        if resumed_from_checkpoint and best_identity in cache:
            callback = callbacks.get("on_new_champion")
            if callback:
                retained = cache[best_identity]
                public_params = dict(
                    self.best_params_so_far,
                    _i6n_envelope=envelope,
                    _i6n_origin_island=settings["island_id"],
                    _i6n_genotype_sha256=best_identity,
                )
                callback(
                    public_params, self.best_fitness_so_far,
                    retained.get("metrics", {}),
                    start_generation, {
                        "stage": 1, "total_stages": 1,
                        "candidate": 0,
                        "total_candidates_evaluated": total_eval,
                        "resumed_checkpoint": True,
                    },
                )
        for generation in range(start_generation, settings["generations"]):
            stage = {
                "stage": 1, "total_stages": 1, "meta_mode": False,
                "total_candidates_evaluated": total_eval,
                "population_size": settings["population"],
            }
            on_start = callbacks.get("on_generation_start")
            if on_start and not (
                resume_generation_in_progress and generation == start_generation
            ):
                migrant = on_start(population, [], list(search._GENOME_KEYS), generation, stage)
                # This campaign has one search stage.  DOIN's historical stage
                # broadcast is migration metadata, never a substitute for this
                # island's generation patience/limit.
                if isinstance(migrant, Mapping) and migrant.get("_force_stage_advance"):
                    migrant = None
                migrant_g = self._network_genotype(migrant, settings["horizon"], envelope)
                if migrant_g is not None:
                    incoming_id = space.genotype_sha256(migrant_g)
                    if incoming_id not in {space.genotype_sha256(g) for g in population}:
                        population[-1] = migrant_g

            ranked = []
            best_before = self.best_fitness_so_far
            for candidate_number, genotype in enumerate(population, 1):
                identity = space.genotype_sha256(genotype)
                if identity in rejections:
                    continue
                result = cache.get(identity)
                evaluated_now = result is None
                if result is None:
                    candidate_started = time.monotonic()
                    wire = evaluator._wire(genotype, _digest(contract), generation)
                    try:
                        evaluated = evaluator.evaluate(wire, generation, cfg)
                    except search.ScoreRejected as exc:
                        rejections[identity] = {
                            "disposition": "FAILED",
                            "generation": generation,
                            "reason": str(exc),
                        }
                        save_checkpoint(
                            generation_value=generation,
                            population_value=population,
                            generation_complete=False,
                        )
                        continue
                    evaluated["fitness"] = float(evaluated["fitness"]) - 1.0
                    effective_config_sha256 = _digest(evaluated["hyper_dict"])
                    evaluated.setdefault("metrics", {}).update({
                        "relative_error_vs_naive": evaluated["fitness"],
                        "genotype_canonical_sha256": identity,
                        "effective_config_sha256": effective_config_sha256,
                        "plan_sha256": plan_identity,
                        "contract_sha256": contract_sha256,
                        "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    })
                    result = self._summary(evaluated)
                    cache[identity] = result
                    total_eval += 1
                    self._candidate_durations.append(time.monotonic() - candidate_started)
                    self._publish_runtime_statistics(
                        source=runtime_source,
                        total=total_eval,
                    )
                fitness = float(result["fitness"])
                ranked.append({"genotype": genotype, "fitness": fitness,
                               "result": result, "generation": generation})
                is_champion = fitness < self.best_fitness_so_far
                if is_champion:
                    self.best_fitness_so_far = fitness
                    self.best_params_so_far = deepcopy(result["hyper_dict"])
                    self.best_train_mae_so_far = result.get("train_mae")
                    self.best_train_naive_mae_so_far = result.get("train_naive_mae")
                    self.best_val_mae_so_far = result.get("val_mae")
                    self.best_naive_mae_so_far = result.get("val_naive_mae")
                    best_identity = identity
                info = {
                    "generation": generation, "candidate_in_gen": candidate_number,
                    "candidate_num": candidate_number,
                    "total_eval": total_eval,
                    "total_candidates_evaluated": total_eval,
                    "stage": 1, "total_stages": 1,
                    "stage_name": f"I6N-H{settings['horizon']}", "gen_in_stage": generation,
                    "n_generations_stage": settings["generations"],
                    "n_generations_total": settings["generations"],
                    "population_size": settings["population"],
                    "total_candidates": settings["population"],
                    "is_champion": is_champion,
                    "fitness": fitness, "champion_fitness": self.best_fitness_so_far,
                    "genotype_canonical_sha256": identity,
                    "effective_config_sha256": result["metrics"]["effective_config_sha256"],
                    "plan_sha256": plan_identity,
                    "contract_sha256": contract_sha256,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "parameters": result["hyper_dict"],
                    "champion_parameters": self.best_params_so_far,
                    "train_mae": result.get("train_mae"),
                    "train_naive_mae": result.get("train_naive_mae"),
                    "val_mae": result.get("val_mae"),
                    "val_naive_mae": result.get("val_naive_mae"),
                    "test_mae": None, "test_naive_mae": None,
                    "metrics": result.get("metrics", {}),
                    "no_improve_counter": no_improve,
                    "higher_is_better": False,
                    "training_seed": settings["seed"],
                    "island_id": settings["island_id"],
                }
                if evaluated_now and callbacks.get("on_candidate_evaluated"):
                    callbacks["on_candidate_evaluated"](info)
                if is_champion and callbacks.get("on_new_champion"):
                    public_params = dict(
                        self.best_params_so_far,
                        _i6n_envelope=envelope,
                        _i6n_origin_island=settings["island_id"],
                        _i6n_genotype_sha256=identity,
                    )
                    callbacks["on_new_champion"](
                        public_params, fitness, result.get("metrics", {}),
                        generation, dict(stage, candidate=candidate_number,
                                         total_candidates_evaluated=total_eval),
                    )
                if callbacks.get("on_between_candidates"):
                    signal = callbacks["on_between_candidates"](generation, candidate_number, info)
                    # A remote stage-complete signal cannot terminate the sole
                    # governed stage; only local patience or max generations can.
                # Candidate results are expensive and independently complete.
                # Persist the cache before starting the next candidate so a
                # restart never repeats a finished model fit.
                if evaluated_now:
                    save_checkpoint(
                        generation_value=generation,
                        population_value=population,
                        generation_complete=False,
                    )

            if not ranked:
                raise RuntimeError("generation has no successful candidates")
            ranked.sort(key=lambda row: (row["fitness"], space.genotype_sha256(row["genotype"])))
            # Fitness is already a relative error.  Its direct decrement is
            # therefore the relative increment governed by the 1e-8 threshold.
            relative_improvement = (float("inf") if not math.isfinite(best_before) else
                                    best_before - self.best_fitness_so_far)
            improved = relative_improvement > settings["tolerance"]
            no_improve = 0 if improved else no_improve + 1
            stage.update(generation=generation, champion_fitness=self.best_fitness_so_far,
                         champion_parameters=self.best_params_so_far,
                         total_candidates_evaluated=total_eval,
                         no_improve_counter=no_improve, patience=settings["patience"],
                         relative_improvement=relative_improvement,
                         min_relative_improvement=settings["tolerance"])
            stop = ((no_improve >= settings["patience"]
                     and generation + 1 >= settings["min_generations"])
                    or generation + 1 >= settings["generations"])
            stage.update(converged=stop,
                         stop_reason=(("patience" if no_improve >= settings["patience"] else
                                       "generation_limit") if stop else None),
                         no_improve_count=no_improve)
            if callbacks.get("on_generation_end"):
                callbacks["on_generation_end"](population, ranked[:1], list(search._GENOME_KEYS),
                                               generation, stage,
                                               {"min": ranked[0]["fitness"], "max": ranked[-1]["fitness"]})
            if not stop:
                population = self._next_island_population(ranked, settings, admitted, rng)
            save_checkpoint(
                generation_value=generation + 1,
                population_value=population,
                generation_complete=True,
                stop_reason=stage["stop_reason"] if stop else None,
            )
            resume_generation_in_progress = False
            if stop:
                self._record_island_terminal(
                    settings=settings, plan_sha256=plan_identity,
                    contract_sha256=contract_sha256,
                    checkpoint_identity_sha256=checkpoint_identity_sha256,
                    generation=generation + 1,
                    stop_reason=stage["stop_reason"],
                    champion_sha256=best_identity,
                    champion_fitness=self.best_fitness_so_far,
                    population=population, cache=cache,
                )
                break
        return deepcopy(self.best_params_so_far)

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
            try:
                result = subprocess.run(command, input=_json(request), text=True,
                                        capture_output=True, check=True, timeout=timeout,
                                        shell=False)
            except subprocess.CalledProcessError as exc:
                # Keep enough child context to diagnose governed workers without
                # echoing their JSON request or captured stdout into service logs.
                stderr = (exc.stderr or "").replace("\x00", "").strip()[-2000:]
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(
                    f"I6N evaluator command failed with exit {exc.returncode}{detail}"
                ) from None
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
                                        origins_sha256=origins or None,
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
