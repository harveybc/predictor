"""Stateful NEAT-style proposals for the modular DOIN search space.

This adapts the existing parameters-as-genes implementation.  It does not
claim to evolve a neural prediction head: genomes select modular candidate
parameters, while the existing campaign remains responsible for durable
dispatch, paired seeds, checkpoint verification and the incumbent.
"""
from __future__ import annotations

import copy
import math
import random

from optimizer_plugins.neat_optimizer import NeatGene, NeatGenome, Plugin as NeatPlugin
from tools import modular_doin_campaign as campaign
from tools import modular_search_space as search


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def normalize_restrict(space, restrict):
    """Validate a sub-space declaration and return it in canonical form.

    ``restrict`` maps a flat parameter name to ``{"choices": [subset]}`` or numeric
    ``{"low", "high"}`` inside the declared bound.  The search-space bytes (and so every
    candidate/config identity, which embeds the space digest) are NOT changed: only the
    region that proposals may visit is narrowed.  ``train.seed`` is never restricted.
    """
    result = {}
    for name, spec in sorted((restrict or {}).items()):
        declared = space["bounds"].get(name)
        if declared is None or name == "train.seed":
            raise search.SearchSpaceError(f"cannot restrict {name!r}")
        if "choices" in declared:
            allowed = {search.canonical(c) for c in declared["choices"]}
            choices = spec.get("choices") if set(spec) == {"choices"} else None
            if not choices or any(search.canonical(c) not in allowed for c in choices):
                raise search.SearchSpaceError(f"{name}: restriction must be a nonempty subset of the declared choices")
            if len({search.canonical(c) for c in choices}) != len(choices):
                raise search.SearchSpaceError(f"{name}: duplicate restricted choices")
            result[name] = {"choices": copy.deepcopy(choices)}
        else:
            if set(spec) != {"low", "high"} or not declared["low"] <= spec["low"] <= spec["high"] <= declared["high"]:
                raise search.SearchSpaceError(f"{name}: restriction must lie inside [{declared['low']}, {declared['high']}]")
            result[name] = {"low": spec["low"], "high": spec["high"]}
    return result


def restricted_spec(space, restrict, name):
    declared = space["bounds"][name]
    if name not in restrict:
        return declared
    return {**declared, **restrict[name]}


def propose_restricted(space, base, rng, restrict=None, max_tries=20000):
    """One valid flat candidate (no seed) drawn uniformly inside the restricted sub-space.

    Unlike ``modular_doin_campaign.propose`` this also draws the optional
    ``model.target_residual`` parameter (that function drops it and so can never return a
    valid candidate for a space that declares it).  Both matched arms use this draw
    rule: the random arm directly, the NEAT arm for population repair.
    """
    restrict = restrict or {}
    seed0 = space["bounds"]["train.seed"]["choices"][0]
    optional = [n for n in search.OPTIONAL_PARAMETERS if n in space["bounds"]]
    for _ in range(max_tries):
        names = [n for n in search.parameter_names() if n != "train.seed"] + optional
        flat = {n: campaign._sample(restricted_spec(space, restrict, n), rng) for n in names}
        flat["train.seed"] = seed0
        keep = (set(search.active_parameters(flat)) | set(optional))
        flat = {n: flat[n] for n in keep}
        try:
            search.validate_flat(flat, space)
            search.from_flat(flat, base, space)
        except search.SearchSpaceError:
            continue
        flat.pop("train.seed")
        return flat
    raise search.SearchSpaceError("no valid candidate inside the restricted sub-space")


class ModularNeatProposalPolicy:
    """Evolve valid flat modular candidates with deterministic resumable state."""

    SCHEMA = "modular.neat.proposal.v1"

    def __init__(self, space, default, base, *, population_size=12, seed=0, default_huber_delta=None,
                 restrict=None):
        if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        search.validate_space(space)
        self.space = copy.deepcopy(space)
        self.restrict = normalize_restrict(self.space, restrict)
        self.default = copy.deepcopy(default)
        if self.default.get("train.loss") == "huber" and "train.huber_delta" not in self.default:
            spec = self.space["bounds"]["train.huber_delta"]
            if default_huber_delta is None:
                default_huber_delta = ((spec["low"] * spec["high"]) ** 0.5 if spec.get("log")
                                       else (spec["low"] + spec["high"]) / 2)
            self.default["train.huber_delta"] = default_huber_delta
        self.base = copy.deepcopy(base)
        self._check_default_in_restriction()
        self.population_size = population_size
        self.seed = int(seed)
        self.generation = 0
        self.no_improve_count = 0
        self.best_fitness = None
        self._rng = random.Random(self.seed)
        self._prepare_codec()
        created = NeatPlugin.create_shared_population(population_size, self._neat_config,
                                                      seed=self.seed)
        self.population = created["population"]
        self.innovation_tracker = created["innovation_tracker"]
        self.stage_schedule = created["stage_schedule"]
        self.param_defaults = created["param_defaults"]
        # The declared default is always generation zero's control, not left to
        # chance.  Remaining genomes provide the evolutionary exploration.
        default_probe = {**self.default, "train.seed": self.space["bounds"]["train.seed"]["choices"][0]}
        search.from_flat(default_probe, self.base, self.space)
        self.population[0] = self._encode(self.default)
        self._repair_population()

    def _check_default_in_restriction(self):
        for name, spec in self.restrict.items():
            if name not in self.default:
                continue
            value = self.default[name]
            inside = (search.canonical(value) in {search.canonical(c) for c in spec["choices"]}
                      if "choices" in spec else spec["low"] <= value <= spec["high"])
            if not inside:
                raise search.SearchSpaceError(f"declared default {name}={value!r} lies outside the restriction")

    def _prepare_codec(self):
        self.names = [name for name in self.space["bounds"] if name != "train.seed"]
        self.choice_values = {}
        numeric = {}
        defaults = {}
        for name in self.names:
            spec = restricted_spec(self.space, self.restrict, name)
            if "choices" in spec:
                self.choice_values[name] = copy.deepcopy(spec["choices"])
                numeric[name] = (0, len(spec["choices"]) - 1)
                value = self.default.get(name, spec["choices"][0])
                defaults[name] = float(next(
                    (i for i, item in enumerate(spec["choices"]) if search.canonical(item) == search.canonical(value)),
                    0))
            else:
                numeric[name] = (spec["low"], spec["high"])
                defaults[name] = float(self.default.get(name, (spec["low"] + spec["high"]) / 2))
        self._neat_config = {
            "hyperparameter_bounds": numeric,
            **defaults,
            "higher_is_better": bool(self.base.get("objective", {}).get("higher_is_better", False)),
            "n_generations": 1000000,
            "optimization_patience": 1000000,
            "optimization_stages": [{"name": "modular", "params": "all", "generations": 1000000}],
            "neat_min_params": len(self.names),
        }

    def _decode_value(self, name, value):
        if name in self.choice_values:
            choices = self.choice_values[name]
            return copy.deepcopy(choices[max(0, min(len(choices) - 1, int(round(value))))])
        spec = restricted_spec(self.space, self.restrict, name)
        value = max(spec["low"], min(spec["high"], value))
        return int(round(value)) if spec["type"] == "int" else float(value)

    def _encoded_value(self, name, value):
        if name in self.choice_values:
            return float(next(i for i, item in enumerate(self.choice_values[name])
                              if search.canonical(item) == search.canonical(value)))
        return float(value)

    def _decode(self, serialized):
        values = copy.deepcopy(self.default)
        genome = NeatGenome.from_serializable(serialized)
        for gene in genome.genes.values():
            values[gene.param_name] = self._decode_value(gene.param_name, gene.value)
        stages = values.get("core.stage_count")
        keep = set(search.active_parameters(values)) - {"train.seed"}
        keep.update(name for name in search.OPTIONAL_PARAMETERS if name in self.space["bounds"])
        if "R3" in (values.get("branch.regime"), values.get("core.regime")):
            keep.update(search.WARM_PARAMETERS)
        result = {name: values[name] for name in keep if name in values}
        probe = {**result, "train.seed": self.space["bounds"]["train.seed"]["choices"][0]}
        search.validate_flat(probe, self.space)
        search.from_flat(probe, self.base, self.space)
        return result

    def _encode(self, flat):
        tracker = self.innovation_tracker["map"]
        values = copy.deepcopy(self.default)
        values.update(flat)
        genome = NeatGenome()
        for name in self.names:
            if name not in values:
                spec = restricted_spec(self.space, self.restrict, name)
                values[name] = spec["choices"][0] if "choices" in spec else (spec["low"] + spec["high"]) / 2
            innovation = int(tracker[name])
            genome.genes[innovation] = NeatGene(innovation, name, self._encoded_value(name, values[name]))
        return genome.to_serializable()

    def _random_valid(self):
        return propose_restricted(self.space, self.base, self._rng, self.restrict)

    def _repair_population(self):
        repaired, seen = [], set()
        for serialized in self.population:
            try:
                flat = self._decode(serialized)
            except (KeyError, ValueError, search.SearchSpaceError):
                flat = self._random_valid()
            identity = search.digest(flat)
            while identity in seen:
                flat = self._random_valid()
                identity = search.digest(flat)
            seen.add(identity)
            repaired.append(self._encode(flat))
        self.population = repaired

    def ask(self):
        """Return one unique, valid candidate per genome, with no evaluation seed."""
        return [self._decode(genome) for genome in self.population]

    def tell(self, fitness_by_digest):
        """Attach a complete generation's verified fitness and reproduce once."""
        candidates = self.ask()
        expected = {search.digest(row) for row in candidates}
        if set(fitness_by_digest) != expected:
            raise ValueError("fitness must cover exactly the current NEAT population")
        evaluated = copy.deepcopy(self.population)
        for serialized, flat in zip(evaluated, candidates):
            value = fitness_by_digest[search.digest(flat)]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("fitness values must be finite numbers")
            serialized["fitness"] = float(value)
        higher = self._neat_config["higher_is_better"]
        generation_best = (max if higher else min)(float(v) for v in fitness_by_digest.values())
        improved = (self.best_fitness is None or
                    (generation_best > self.best_fitness if higher else generation_best < self.best_fitness))
        self.no_improve_count = 0 if improved else self.no_improve_count + 1
        if improved:
            self.best_fitness = generation_best
        result = NeatPlugin.reproduce_shared(
            evaluated, self.generation, self.seed + self.generation + 1, self._neat_config,
            self.innovation_tracker, self.stage_schedule, self.param_defaults,
            current_stage_idx=0, no_improve_count=self.no_improve_count)
        if result.get("converged"):
            raise RuntimeError("modular NEAT policy converged before the requested next generation")
        self.population = result["population"]
        self.innovation_tracker = result.get("innovation_tracker", self.innovation_tracker)
        self.generation = int(result["generation"])
        self._repair_population()

    def to_state(self):
        return {
            "schema": self.SCHEMA,
            "space_sha256": search.digest(self.space),
            "default_sha256": search.digest(self.default),
            "base_sha256": search.digest(self.base),
            "normalized_default": copy.deepcopy(self.default),
            "restrict": copy.deepcopy(self.restrict),
            "restrict_sha256": search.digest(self.restrict),
            "population_size": self.population_size,
            "seed": self.seed,
            "generation": self.generation,
            "no_improve_count": self.no_improve_count,
            "best_fitness": self.best_fitness,
            "population": copy.deepcopy(self.population),
            "innovation_tracker": copy.deepcopy(self.innovation_tracker),
            "stage_schedule": copy.deepcopy(self.stage_schedule),
            "param_defaults": copy.deepcopy(self.param_defaults),
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_state(cls, space, default, base, state, restrict=None):
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported modular NEAT state")
        normalized_default = copy.deepcopy(default)
        if normalized_default.get("train.loss") == "huber" and "train.huber_delta" not in normalized_default:
            normalized_default["train.huber_delta"] = state.get("normalized_default", {}).get(
                "train.huber_delta")
        if search.digest(normalized_default) != state.get("default_sha256"):
            raise ValueError("modular NEAT default identity changed")
        for label, value, expected in (("space", space, state.get("space_sha256")),
                                       ("base", base, state.get("base_sha256"))):
            if search.digest(value) != expected:
                raise ValueError(f"modular NEAT {label} identity changed")
        stored = state.get("restrict", {})
        if search.digest(stored) != state.get("restrict_sha256", search.digest({})):
            raise ValueError("modular NEAT restriction digest mismatch")
        if restrict is not None and normalize_restrict(space, restrict) != stored:
            raise ValueError("modular NEAT restriction identity changed")
        obj = cls.__new__(cls)
        obj.restrict = normalize_restrict(space, stored)
        obj.space, obj.default, obj.base = map(copy.deepcopy, (space, normalized_default, base))
        obj.population_size = int(state["population_size"])
        obj.seed, obj.generation = int(state["seed"]), int(state["generation"])
        obj.no_improve_count = int(state["no_improve_count"])
        obj.best_fitness = state["best_fitness"]
        obj._rng = random.Random()
        obj._rng.setstate(_tuples(state["rng_state"]))
        obj._prepare_codec()
        obj.population = copy.deepcopy(state["population"])
        obj.innovation_tracker = copy.deepcopy(state["innovation_tracker"])
        obj.stage_schedule = copy.deepcopy(state["stage_schedule"])
        obj.param_defaults = copy.deepcopy(state["param_defaults"])
        obj.ask()  # validate persisted genomes without rewriting their bytes
        return obj
