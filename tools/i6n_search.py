"""Bounded, local I6N DEAP search, not a DOIN plugin or a training runner.

Only configs and callback receipts are handled here. No data, TensorFlow,
devices, global RNG seeding, DEAP creator registry, or filesystem writes.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from itertools import combinations_with_replacement, product
import json
import math
import random
import re

from deap import base, tools

from tools import i6n_model


HORIZONS = (2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72)
MULTIPLIERS = (1, 2, 4, 6)
WIDTHS = (8, 16, 32, 64, 128)
FILTER_WIDTHS = (4, 8, 16, 32, 64, 128)
INPUT_TRANSFORMS = ("identity", "difference")
MUTATION_GROUPS = (
    "family", "window", "width", "blocks", "reducers", "strides", "adapter",
    "input_transform", "output_supervision",
)
_GENOME_KEYS = frozenset((
    "horizon", "multiplier", "delta_x", "input_transform", "output_mode",
    "delta_y", "core",
))
_SPLITS = {"train", "inner"}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _horizon(horizon):
    _integer(horizon, "horizon")
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return horizon


def _divisors(horizon):
    return tuple(d for d in range(1, horizon + 1) if horizon % d == 0)


def default_genome(horizon):
    """Return a detached, explicit endpoint genome for one supported horizon."""
    return {
        "horizon": _horizon(horizon), "multiplier": 1, "delta_x": 1,
        "input_transform": "identity", "output_mode": "endpoint", "delta_y": None,
        "core": deepcopy(i6n_model.DEFAULT_CONFIG),
    }


def normalize_genome(genome):
    """Canonicalize inactive/derived genes, then validate the real core schema.

    A one-point trajectory is the endpoint, not a second experiment. Equal
    minimum-history multipliers also collapse to one genome. Invalid active
    genes fail closed; normalization does not repair reductions or widths.
    """
    if not isinstance(genome, Mapping) or set(genome) != _GENOME_KEYS:
        raise ValueError("genome requires exactly the explicit genome keys")
    g = deepcopy(dict(genome))
    h = _horizon(g["horizon"])
    multiplier = _integer(g["multiplier"], "multiplier")
    if multiplier not in MULTIPLIERS:
        raise ValueError("unsupported history multiplier")
    history = max(24, multiplier * h)
    g["multiplier"] = min(m for m in MULTIPLIERS if max(24, m * h) == history)
    delta = _integer(g["delta_x"], "delta_x")
    if h % delta or history % delta:
        raise ValueError("delta_x must divide both horizon and history")
    if g["input_transform"] not in INPUT_TRANSFORMS:
        raise ValueError("unsupported input transform")
    if g["output_mode"] not in ("endpoint", "trajectory"):
        raise ValueError("unsupported output supervision")
    if g["output_mode"] == "trajectory":
        dy = _integer(g["delta_y"], "delta_y")
        if h % dy:
            raise ValueError("trajectory delta_y must independently divide horizon")
        if dy == h:
            g["output_mode"] = "endpoint"
    if g["output_mode"] == "endpoint":
        g["delta_y"] = None
    if not isinstance(g["core"], Mapping):
        raise ValueError("core must be an explicit mapping")
    core = dict(g["core"])
    # Do not let canonicalization conceal a missing field in the core schema.
    if set(core) != set(i6n_model.DEFAULT_CONFIG):
        i6n_model.normalize_config(core)
    if core["entrance"] == "conv":
        core["transformer_blocks"] = 0
    if core["adapter"] == "none":
        core["adapter_width"] = None
    core["n_horizons"] = 1 if g["output_mode"] == "endpoint" else h // g["delta_y"]
    core = i6n_model.normalize_config(core)
    widths = [core["width"], *core["reducer_filters"]]
    if widths != sorted(widths, reverse=True):
        raise ValueError("reducer filters must be nonincreasing from entrance width")
    steps = 1 + history // delta
    for stride in core["reducer_strides"]:
        steps = (steps + stride - 1) // stride
    if steps < 2:
        raise ValueError("final temporal core must retain at least two steps")
    g["core"] = core
    return g


def effective_config(genome):
    """Model config plus exact physical-hour grids, including the current t."""
    cfg = normalize_genome(genome)
    h = cfg["horizon"]
    history = max(24, cfg["multiplier"] * h)
    samples = 1 + history // cfg["delta_x"]
    steps = samples
    for stride in cfg["core"]["reducer_strides"]:
        steps = (steps + stride - 1) // stride
    cfg.update(
        history_hours=history, samples=samples, final_steps=steps,
        input_offsets=list(range(-history, 1, cfg["delta_x"])),
        output_offsets=([h] if cfg["output_mode"] == "endpoint" else
                        list(range(cfg["delta_y"], h + 1, cfg["delta_y"]))),
    )
    return cfg


def effective_identity(genome):
    """Config identity within a fixed evaluator/support context, not a run ID."""
    payload = json.dumps({"schema": "i6n-search-v1", "config": effective_config(genome)},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def sample_genome(horizon, rng):
    """Sample only valid bounded candidates, using the supplied local RNG."""
    g = default_genome(horizon)
    g["multiplier"] = rng.choice(MULTIPLIERS)
    history = max(24, g["multiplier"] * horizon)
    g["delta_x"] = rng.choice([d for d in _divisors(horizon) if history % d == 0])
    g["input_transform"] = rng.choice(INPUT_TRANSFORMS)
    if rng.choice((False, True)):
        g.update(output_mode="trajectory", delta_y=rng.choice(_divisors(horizon)[:-1]))
    core = g["core"]
    core["entrance"] = rng.choice(("conv", "transformer", "conv_transformer"))
    core["transformer_blocks"] = 0 if core["entrance"] == "conv" else rng.choice((1, 2))
    core["width"] = rng.choice(WIDTHS)
    width, steps = core["width"], 1 + history // g["delta_x"]
    filters, strides = [], []
    for _ in range(rng.randint(1, 3)):
        width = rng.choice([f for f in FILTER_WIDTHS if f <= width])
        stride = rng.choice((1, 2)) if steps >= 3 else 1
        steps = (steps + stride - 1) // stride
        filters.append(width)
        strides.append(stride)
    core.update(reducer_filters=filters, reducer_strides=strides)
    core["adapter"] = rng.choice(("none", "causalconv", "lstm"))
    core["adapter_width"] = None if core["adapter"] == "none" else rng.choice(WIDTHS)
    return normalize_genome(g)


def _variants(g, group):
    """Finite neighborhood; invalid coupled configurations are filtered below."""
    core = g["core"]

    def changed(updates=None, **core_updates):
        child = deepcopy(g)
        child.update(updates or {})
        child["core"].update(core_updates)
        return child

    if group == "family":
        for family in ("conv", "transformer", "conv_transformer"):
            yield changed(entrance=family, transformer_blocks=(
                0 if family == "conv" else max(1, core["transformer_blocks"])))
    elif group == "window":
        for multiplier in MULTIPLIERS:
            for delta in _divisors(g["horizon"]):
                yield changed({"multiplier": multiplier, "delta_x": delta})
    elif group == "width":
        for width in WIDTHS:
            yield changed(width=width, reducer_filters=[
                min(width, f) for f in core["reducer_filters"]])
    elif group == "blocks":
        for blocks in (1, 2):
            yield changed(transformer_blocks=blocks, entrance=(
                "transformer" if core["entrance"] == "conv" else core["entrance"]))
    elif group == "reducers":
        options = [f for f in FILTER_WIDTHS if f <= core["width"]]
        for stages in (1, 2, 3):
            for widths in combinations_with_replacement(options, stages):
                yield changed(reducer_filters=list(reversed(widths)),
                              reducer_strides=(core["reducer_strides"] + [1, 1])[:stages])
    elif group == "strides":
        for strides in product((1, 2), repeat=len(core["reducer_filters"])):
            yield changed(reducer_strides=list(strides))
    elif group == "adapter":
        yield changed(adapter="none", adapter_width=None)
        for adapter in ("causalconv", "lstm"):
            for width in WIDTHS:
                yield changed(adapter=adapter, adapter_width=width)
    elif group == "input_transform":
        for transform in INPUT_TRANSFORMS:
            yield changed({"input_transform": transform})
    elif group == "output_supervision":
        yield changed({"output_mode": "endpoint", "delta_y": None})
        for delta in _divisors(g["horizon"])[:-1]:
            yield changed({"output_mode": "trajectory", "delta_y": delta})


def mutate(genome, rng, group=None):
    """Mutate a joint conditional group without modifying the parent.

    Explicit groups can have no valid alternative at a boundary (e.g. strides
    at two samples); then return the canonical parent. Unguided mutation tries
    all groups in local-random order and never spins waiting for a new genome.
    """
    g = normalize_genome(genome)
    if group is not None and group not in MUTATION_GROUPS:
        raise ValueError("unknown mutation group")
    groups = [group] if group is not None else rng.sample(MUTATION_GROUPS, len(MUTATION_GROUPS))
    for name in groups:
        options = {}
        for variant in _variants(g, name):
            try:
                candidate = normalize_genome(variant)
            except ValueError:
                continue
            if candidate != g:
                options[effective_identity(candidate)] = candidate
        if options:
            return deepcopy(rng.choice(list(options.values())))
    return g


def per_horizon_configs(seed=6102026, count=1):
    """Generate reproducible unique configs for each horizon, with finite work."""
    _integer(seed, "seed", 0)
    _integer(count, "count")
    result = {}
    for h in HORIZONS:
        rng = random.Random(f"i6n-search-v1:{seed}:{h}")
        configs = {}
        for _ in range(count * 32):
            g = sample_genome(h, rng)
            configs[effective_identity(g)] = effective_config(g)
            if len(configs) == count:
                break
        if len(configs) != count:
            raise ValueError("unique config attempt budget exhausted")
        result[h] = list(configs.values())
    return result


class ScoreRejected(ValueError):
    """Failed or incomparable callback receipt; never admitted to selection."""


def _split_mapping(value, name):
    if not isinstance(value, Mapping) or set(value) != _SPLITS:
        raise ValueError(f"{name} requires exactly train and inner")


def _number(value, name, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _support(value):
    if not isinstance(value, Mapping) or set(value) != {"counts", "naive_sums", "support_digest"}:
        raise ValueError("support requires counts, naive_sums, support_digest")
    result = deepcopy(dict(value))
    for key in result:
        _split_mapping(result[key], key)
    for split in sorted(_SPLITS):
        _integer(result["counts"][split], "count")
        _number(result["naive_sums"][split], "naive_sum", positive=True)
        digest = result["support_digest"][split]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("support_digest must be a lowercase SHA-256 hex digest")
    return result


def _score(receipt, frozen):
    required = {"status", "counts", "error_sums", "naive_sums", "support_digest", "metadata"}
    if (not isinstance(receipt, Mapping) or not required <= set(receipt)
            or set(receipt) - required - {"fitness"}):
        raise ValueError("score receipt schema mismatch")
    if receipt["status"] != "ok":
        raise ValueError("candidate did not succeed")
    actual = _support({key: receipt[key] for key in frozen})
    if actual != frozen:
        raise ValueError("candidate support/counts/naive sums do not match frozen support")
    _split_mapping(receipt["error_sums"], "error_sums")
    for value in receipt["error_sums"].values():
        _number(value, "error sum")
    if not isinstance(receipt["metadata"], Mapping):
        raise ValueError("metadata must be a JSON mapping")
    # Equal split weights, not an unweighted average of batches or weeks.
    fitness = math.fsum(receipt["error_sums"][s] / frozen["naive_sums"][s]
                        for s in sorted(_SPLITS)) / 2
    _number(fitness, "fitness")
    if "fitness" in receipt:
        reported = _number(receipt["fitness"], "reported fitness")
        if not math.isclose(reported, fitness, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reported fitness disagrees with error/naive sums")
    metrics = json.loads(json.dumps(dict(receipt), allow_nan=False))
    return fitness, metrics


class _Fitness(base.Fitness):
    weights = (-1.0,)


class _Individual:
    def __init__(self, genome, fitness):
        self.genome = deepcopy(genome)
        self.fitness = _Fitness((fitness,))


class Search:
    """Single-horizon steady-state DEAP (mu+1), mutation plus random immigrants.

    Proposal reservations consume the budget even on failure. Attempts also
    have a lifetime bound, so exhausted/duplicate neighborhoods terminate.
    Callback wall/memory limits belong to the caller; this class is synchronous
    and not thread-safe. A fixed callback and call ordering reproduce a run.
    """

    def __init__(self, horizon, *, seed, budget, support, population_size=8,
                 max_proposals=None):
        self.horizon = _horizon(horizon)
        _integer(seed, "seed", 0)
        self.budget = _integer(budget, "budget")
        self.population_size = _integer(population_size, "population_size")
        self.max_proposals = _integer(
            budget * 32 if max_proposals is None else max_proposals, "max_proposals")
        self._support = _support(support)
        self._rng = random.Random(seed)
        self.toolbox = base.Toolbox()
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("mutate", mutate, rng=self._rng)
        self.toolbox.register("candidate", self._candidate)
        self.population = []
        self.attempts = self.proposed = self.evaluations = 0
        self._seen, self._pending = set(), {}
        self._results, self._rejections = [], []

    @property
    def results(self):
        return deepcopy(self._results)

    @property
    def rejections(self):
        return deepcopy(self._rejections)

    def _candidate(self):
        if self.population and self._rng.random() < .8:
            contenders = self._rng.choices(self.population, k=3)
            parent = self.toolbox.select(contenders, 1)[0]
            return self.toolbox.mutate(parent.genome)
        return sample_genome(self.horizon, self._rng)

    def propose(self, count=1):
        """Reserve up to count unique candidates; return detached wire mappings."""
        _integer(count, "count")
        proposals = []
        while (len(proposals) < count and self.proposed < self.budget
               and self.attempts < self.max_proposals):
            self.attempts += 1
            genome = normalize_genome(self.toolbox.candidate())
            if genome["horizon"] != self.horizon:
                raise ValueError("candidate belongs to a different horizon")
            identity = effective_identity(genome)
            if identity in self._seen:
                continue
            proposal = {"identity": identity, "config": effective_config(genome)}
            self._seen.add(identity)
            self._pending[identity] = (genome, proposal)
            self.proposed += 1
            proposals.append(deepcopy(proposal))
        return proposals

    def evaluate(self, proposal, callback):
        """Evaluate a reserved proposal once; reject failed/incomparable scores."""
        if not isinstance(proposal, Mapping) or not isinstance(proposal.get("identity"), str):
            raise ValueError("expected a reserved proposal")
        pending = self._pending.get(proposal["identity"])
        if pending is None or pending[1] != proposal:
            raise ValueError("unreserved, modified, or already evaluated proposal")
        genome, reserved = self._pending.pop(proposal["identity"])
        self.evaluations += 1
        try:
            receipt = callback(deepcopy(reserved["config"]))
            fitness, metrics = _score(receipt, self._support)
        except Exception as exc:
            self._rejections.append({"identity": reserved["identity"],
                                     "error": f"{type(exc).__name__}: {exc}"})
            raise ScoreRejected(f"rejected {reserved['identity']}: {exc}") from exc
        result = dict(deepcopy(reserved), fitness=fitness, metrics=metrics)
        self._results.append(result)
        self.population = self.toolbox.select(
            self.population + [_Individual(genome, fitness)], self.population_size)
        return deepcopy(result)

    def run(self, callback):
        """Finish pending work then propose sequentially until a bound is hit."""
        for _, proposal in list(self._pending.values()):
            try:
                self.evaluate(deepcopy(proposal), callback)
            except ScoreRejected:
                pass
        while proposals := self.propose():
            try:
                self.evaluate(proposals[0], callback)
            except ScoreRejected:
                pass
        return self.results
