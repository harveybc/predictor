"""Reversible mapping between the nested modular candidate and flat optimizer parameters.

The optimizer plugin API (``optimizer.plugins``) speaks flat ``{name: value}``
dictionaries bounded by ``hyperparameter_bounds``. The modular evaluator speaks a
versioned nested candidate (``modular.candidate.v1``): ``model`` (engine config),
``evaluator`` (training settings), ``target_feature_indices`` and ``objective``.
This module is the only bridge between the two, and it is pure Python: every
invalid combination and every inactive/missing conditional parameter fails here,
before TensorFlow is imported and long before any fit.

Round trip contract (tested):
    to_flat(from_flat(flat, base), space) == flat
    from_flat(to_flat(nested, space), base) == nested   (for nested in the space)

Flat names are stable identifiers, each bound to an explicit nested location:

    branch.grouping_size   -> model.branches (contiguous feature blocks of this size)
    branch.plugin          -> model.branches[*].plugin (uniform)
    branch.channels        -> model.branches[*].params.channels (uniform)
    branch.kernel_size     -> model.branches[*].params.kernel_size (uniform)
    branch.dilation_rate   -> model.branches[*].params.dilation_rate (engine-capability gated)
    branch.regime          -> model.branches[*].regime (+ donor from base donors map)
    model.branch_steps     -> model.branch_steps (shared time resolution)
    model.output_steps     -> model.output_steps
    model.output_channels  -> model.output_channels
    core.d_model/heads/blocks/ff_dim/dropout/kernel_size -> model.core.params.*
    core.stage_count       -> len(model.core.params.stage_channels)
    core.stage_channels_i  -> model.core.params.stage_channels[i] (i < stage_count; last is output_channels)
    core.time_factor_i     -> model.core.params.time_factors[i] (i < stage_count)
    core.regime            -> model.core.regime (+ donor)
    train.learning_rate/weight_decay/loss/patience/min_delta/max_epochs/batch_size -> evaluator.*
    train.huber_delta      -> evaluator.huber_delta, ACTIVE ONLY when train.loss == 'huber'
    train.seed             -> evaluator.seed (paired seeds; set by the queue, never searched)

``weight_decay`` is AdamW decoupled decay: the evaluator has no learning-rate
schedule, so "learning rate/decay" maps to learning_rate and weight_decay.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

SCHEMA = "modular.search_space.v1"
CANDIDATE_SCHEMA = "modular.candidate.v1"

# Parameters the pinned engine accepts per plugin. A flat value that needs a key
# outside this table is refused before fit (not silently dropped).
ENGINE_CAPABILITIES = {
    # Corrected owner design (da4ce7b4 / lane A integrated commit): branches keep the full
    # window grid (branch_steps == window), fusion (B, window, sum widths), PE after fusion,
    # causal Transformer blocks, residual Conv1D reduction stages. Same parameter keys.
    "modular_temporal.v2_full_grid": {
        "full_grid": True,
        "branch": {"causal_conv1d": {"channels", "kernel_size"}},
        "core": {"transformer_conv": {"d_model", "heads", "blocks", "ff_dim", "dropout",
                                      "stage_channels", "time_factors", "kernel_size"}},
    },
    # Superseded engine (556c5f3e lineage): branches compressed 24 -> branch_steps.
    "modular_temporal.v1": {
        "branch": {"causal_conv1d": {"channels", "kernel_size"}},
        "core": {"transformer_conv": {"d_model", "heads", "blocks", "ff_dim", "dropout",
                                      "stage_channels", "time_factors", "kernel_size"}},
    },
}

LOSSES = ("huber", "mae", "mse")
# Optional flat parameters: a space may omit them (older campaigns); when declared they are always active.
OPTIONAL_PARAMETERS = {"model.target_residual": ("none", "seasonal_naive_24")}
REGIMES = ("R0", "R1", "R2")


class SearchSpaceError(ValueError):
    """Invalid space, invalid combination or bad conditional parameter (pre-fit)."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _fail(message):
    raise SearchSpaceError(message)


# ---------------------------------------------------------------- space --

def parameter_names(max_stages=4):
    names = ["branch.grouping_size", "branch.plugin", "branch.channels", "branch.kernel_size",
             "branch.dilation_rate", "branch.regime", "model.branch_steps", "model.output_steps",
             "model.output_channels", "core.d_model", "core.heads", "core.blocks", "core.ff_dim",
             "core.dropout", "core.kernel_size", "core.stage_count", "core.regime"]
    names += [f"core.stage_channels_{i}" for i in range(max_stages)]
    names += [f"core.time_factor_{i}" for i in range(max_stages)]
    names += ["train.learning_rate", "train.weight_decay", "train.loss", "train.huber_delta",
              "train.patience", "train.min_delta", "train.max_epochs", "train.batch_size", "train.seed"]
    return names


def validate_space(space):
    """A space is {schema, engine, bounds{name: spec}}; spec is choices or low/high."""
    if not isinstance(space, dict) or space.get("schema") != SCHEMA:
        _fail(f"search space schema must be {SCHEMA}")
    if space.get("engine") not in ENGINE_CAPABILITIES:
        _fail("unknown engine capability identity")
    bounds = space.get("bounds")
    expected = set(parameter_names()) | (set(bounds or {}) & set(OPTIONAL_PARAMETERS))
    if not isinstance(bounds, dict) or set(bounds) != expected:
        missing = expected ^ set(bounds or {})
        _fail(f"bounds must declare every flat parameter exactly: {sorted(missing)}")
    for name, spec in bounds.items():
        if not isinstance(spec, dict):
            _fail(f"{name}: spec must be an object")
        if "choices" in spec:
            if set(spec) - {"choices"} or not isinstance(spec["choices"], list) or not spec["choices"]:
                _fail(f"{name}: choices must be a nonempty list")
            if len({canonical(c) for c in spec["choices"]}) != len(spec["choices"]):
                _fail(f"{name}: duplicate choices")
        else:
            if set(spec) - {"low", "high", "type", "log"} or spec.get("type") not in ("int", "float"):
                _fail(f"{name}: numeric spec needs type int/float and low/high")
            low, high = spec.get("low"), spec.get("high")
            for v in (low, high):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    _fail(f"{name}: bounds must be finite numbers")
            if spec["type"] == "int" and (type(low) is not int or type(high) is not int):
                _fail(f"{name}: integer bounds must be integers")
            if low > high or (spec.get("log") and low <= 0):
                _fail(f"{name}: invalid bounds")
    return space


def _in_bounds(name, value, spec):
    if "choices" in spec:
        if canonical(value) not in {canonical(c) for c in spec["choices"]}:
            _fail(f"{name}={value!r} outside declared choices {spec['choices']}")
        return
    if spec["type"] == "int":
        if type(value) is not int:
            _fail(f"{name} must be an integer")
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{name} must be a finite number")
    if not spec["low"] <= value <= spec["high"]:
        _fail(f"{name}={value} outside [{spec['low']}, {spec['high']}]")


def active_parameters(flat):
    """Names that must be present given the values of the governing parameters."""
    stages = flat.get("core.stage_count")
    names = [n for n in parameter_names() if not n.startswith(("core.stage_channels_", "core.time_factor_"))
             and n != "train.huber_delta"]
    if stages in (3, 4):
        # the last stage's channels are fixed to model.output_channels, so not a free parameter
        names += [f"core.stage_channels_{i}" for i in range(stages - 1)]
        names += [f"core.time_factor_{i}" for i in range(stages)]
    if flat.get("train.loss") == "huber":
        names.append("train.huber_delta")
    return names


def validate_flat(flat, space):
    """Bounds, conditionals and every cross-parameter constraint the engine enforces."""
    validate_space(space)
    if not isinstance(flat, dict):
        _fail("flat parameters must be an object")
    optional = {k: v for k, v in flat.items() if k in OPTIONAL_PARAMETERS}
    for name in optional:
        if name not in space["bounds"]:
            _fail(f"{name} is not declared by this search space")
        _in_bounds(name, optional[name], space["bounds"][name])
    flat = {k: v for k, v in flat.items() if k not in OPTIONAL_PARAMETERS}
    for name in OPTIONAL_PARAMETERS:
        if name in space["bounds"] and name not in optional:
            _fail(f"active parameters missing: ['{name}']")
    if flat.get("core.stage_count") not in (3, 4):
        _fail("core.stage_count must be 3 or 4")
    if flat.get("train.loss") not in LOSSES:
        _fail(f"train.loss must be one of {LOSSES}")
    active = set(active_parameters(flat))
    inactive = set(flat) - active
    if inactive:
        _fail(f"conditional parameters inactive for this candidate: {sorted(inactive)}")
    missing = active - set(flat)
    if missing:
        _fail(f"active parameters missing: {sorted(missing)}")
    for name in sorted(active):
        _in_bounds(name, flat[name], space["bounds"][name])
    _cross_constraints(flat, space)
    return flat


def _cross_constraints(flat, space):
    caps = ENGINE_CAPABILITIES[space["engine"]]
    plugin = flat["branch.plugin"]
    if plugin not in caps["branch"]:
        _fail(f"branch plugin {plugin!r} not provided by engine {space['engine']}")
    if flat["branch.dilation_rate"] != 1 and "dilation_rate" not in caps["branch"][plugin]:
        _fail(f"branch.dilation_rate={flat['branch.dilation_rate']} unsupported by engine {space['engine']}")
    if flat["core.d_model"] % flat["core.heads"]:
        _fail("core.d_model must be divisible by core.heads")
    if not 0 <= flat["core.dropout"] < 1:
        _fail("core.dropout must be in [0, 1)")
    stages = flat["core.stage_count"]
    channels = [flat[f"core.stage_channels_{i}"] for i in range(stages - 1)] + [flat["model.output_channels"]]
    widths = [flat["core.d_model"], *channels]
    if any(a <= b for a, b in zip(widths, widths[1:])):
        _fail(f"compression channels must strictly decrease from d_model: {widths}")
    factors = [flat[f"core.time_factor_{i}"] for i in range(stages)]
    if flat["model.branch_steps"] % flat["model.output_steps"]:
        _fail("model.branch_steps must be divisible by model.output_steps")
    if math.prod(factors) != flat["model.branch_steps"] // flat["model.output_steps"]:
        _fail(f"time factors {factors} must multiply to branch_steps/output_steps")
    length = flat["model.branch_steps"]
    for f in factors:
        if length % f:
            _fail("each temporal stage requires exact divisibility")
        length //= f


# -------------------------------------------------------------- mapping --

def _branches(feature_names, size, plugin, params, regime, donors):
    branches = []
    for start in range(0, len(feature_names), size):
        name = f"branch_{start // size}"
        spec = {"name": name, "features": feature_names[start:start + size], "plugin": plugin,
                "params": copy.deepcopy(params), "regime": regime, "donor": None}
        if regime != "R0":
            donor = (donors or {}).get(f"{size}:{name}")
            if not donor:
                _fail(f"regime {regime} requires a declared donor for branch {name} at grouping {size}")
            spec["donor"] = donor
        branches.append(spec)
    return branches


def from_flat(flat, base, space):
    """Build the nested modular.candidate.v1 from flat parameters and a fixed base.

    base = {feature_names, window, sample_hours, horizons, target_feature_indices,
            objective, evaluator_fixed{max_updates, max_seconds}, donors{...}}
    """
    validate_flat(flat, space)
    names = list(base["feature_names"])
    if flat["branch.grouping_size"] > len(names):
        _fail("branch.grouping_size exceeds feature count")
    if base["window"] % flat["model.branch_steps"]:
        _fail("window must be divisible by model.branch_steps")
    if ENGINE_CAPABILITIES[space["engine"]].get("full_grid") and flat["model.branch_steps"] != base["window"]:
        _fail("full-grid engine: branches preserve the window, model.branch_steps must equal window")
    if base["sample_hours"] * base["window"] < 24:
        _fail("window must cover at least 24 physical hours")
    params = {"channels": flat["branch.channels"], "kernel_size": flat["branch.kernel_size"]}
    if flat["branch.dilation_rate"] != 1:
        params["dilation_rate"] = flat["branch.dilation_rate"]
    donors = base.get("donors") or {}
    stages = flat["core.stage_count"]
    core = {"plugin": "transformer_conv", "regime": flat["core.regime"], "donor": None,
            "params": {"d_model": flat["core.d_model"], "heads": flat["core.heads"],
                       "blocks": flat["core.blocks"], "ff_dim": flat["core.ff_dim"],
                       "dropout": flat["core.dropout"], "kernel_size": flat["core.kernel_size"],
                       "stage_channels": [flat[f"core.stage_channels_{i}"] for i in range(stages - 1)]
                       + [flat["model.output_channels"]],
                       "time_factors": [flat[f"core.time_factor_{i}"] for i in range(stages)]}}
    if core["regime"] != "R0":
        key = f"core:{flat['branch.grouping_size']}"
        if not donors.get(key):
            _fail(f"core regime {core['regime']} requires a declared core donor {key}")
        core["donor"] = donors[key]
    evaluator = {"learning_rate": flat["train.learning_rate"], "weight_decay": flat["train.weight_decay"],
                 "loss": flat["train.loss"], "patience": flat["train.patience"],
                 "min_delta": flat["train.min_delta"], "max_epochs": flat["train.max_epochs"],
                 "batch_size": flat["train.batch_size"], "seed": flat["train.seed"]}
    if flat["train.loss"] == "huber":
        evaluator["huber_delta"] = flat["train.huber_delta"]
    fixed = base.get("evaluator_fixed", {})
    if set(fixed) - {"max_updates", "max_seconds"}:
        _fail("evaluator_fixed may only carry max_updates/max_seconds")
    evaluator.update(fixed)
    model = {"window": base["window"], "sample_hours": base["sample_hours"],
             "feature_names": names, "horizons": list(base["horizons"]),
             "target_count": len(base["target_feature_indices"]),
             "branch_steps": flat["model.branch_steps"], "output_steps": flat["model.output_steps"],
             "output_channels": flat["model.output_channels"],
             "branches": _branches(names, flat["branch.grouping_size"], flat["branch.plugin"],
                                   params, flat["branch.regime"], donors),
             "core": core, "fusion": {"plugin": "sequence_concat", "params": {}},
             "head": {"plugin": "forecast", "params": {}}}
    meta = {"schema": CANDIDATE_SCHEMA, "search_space_sha256": digest(space)}
    binding = base.get("donor_binding")
    used = [b["donor"] for b in model["branches"] if b["donor"]] + ([core["donor"]] if core["donor"] else [])
    if used and binding:
        missing = [d for d in used if d not in binding["donors"]]
        if missing:
            _fail(f"donor binding lacks {len(missing)} declared donors (e.g. {missing[0]})")
        meta["donor_binding"] = {**{k: binding[k] for k in ("index_sha256", "amendment_sha256", "required_contract")},
                                 "donors": {d: binding["donors"][d] for d in used}}
    residual = flat.get("model.target_residual", "none")
    if residual == "seasonal_naive_24":
        model["target_residual"] = {"kind": "seasonal_naive", "period": 24,
                                    "target_features": [names[i] for i in base["target_feature_indices"]]}
    nested = {"modular_candidate": meta,
              "model": model, "evaluator": evaluator,
              "target_feature_indices": list(base["target_feature_indices"]),
              "objective": dict(base["objective"])}
    return json.loads(canonical(nested))


def to_flat(nested, space):
    """Invert from_flat; refuse any nested candidate the flat space cannot express."""
    validate_space(space)
    meta = nested.get("modular_candidate", {})
    if meta.get("schema") != CANDIDATE_SCHEMA or meta.get("search_space_sha256") != digest(space):
        _fail("candidate schema/search-space identity mismatch")
    model, ev = nested["model"], nested["evaluator"]
    branches = model["branches"]
    size = len(branches[0]["features"])
    names = model["feature_names"]
    uniform = {canonical((b["plugin"], b["params"], b["regime"])) for b in branches}
    if len(uniform) != 1:
        _fail("branches are not uniform; the flat space cannot express this candidate")
    expected = [names[i:i + size] for i in range(0, len(names), size)]
    if [b["features"] for b in branches] != expected or [b["name"] for b in branches] != [
            f"branch_{i}" for i in range(len(expected))]:
        _fail("branches are not contiguous feature blocks; not expressible")
    bp = branches[0]["params"]
    if set(bp) - {"channels", "kernel_size", "dilation_rate"}:
        _fail("branch params outside the flat space")
    params = model["core"]["params"]
    stages = len(params["stage_channels"])
    flat = {"branch.grouping_size": size, "branch.plugin": branches[0]["plugin"],
            "branch.channels": bp["channels"], "branch.kernel_size": bp["kernel_size"],
            "branch.dilation_rate": bp.get("dilation_rate", 1), "branch.regime": branches[0]["regime"],
            "model.branch_steps": model["branch_steps"], "model.output_steps": model["output_steps"],
            "model.output_channels": model["output_channels"],
            "core.d_model": params["d_model"], "core.heads": params["heads"], "core.blocks": params["blocks"],
            "core.ff_dim": params["ff_dim"], "core.dropout": params["dropout"],
            "core.kernel_size": params["kernel_size"], "core.stage_count": stages,
            "core.regime": model["core"]["regime"],
            "train.learning_rate": ev["learning_rate"], "train.weight_decay": ev["weight_decay"],
            "train.loss": ev["loss"], "train.patience": ev["patience"], "train.min_delta": ev["min_delta"],
            "train.max_epochs": ev["max_epochs"], "train.batch_size": ev["batch_size"], "train.seed": ev["seed"]}
    for i in range(stages - 1):
        flat[f"core.stage_channels_{i}"] = params["stage_channels"][i]
    if params["stage_channels"][-1] != model["output_channels"]:
        _fail("last compression stage must equal output_channels")
    for i in range(stages):
        flat[f"core.time_factor_{i}"] = params["time_factors"][i]
    if ev["loss"] == "huber":
        flat["train.huber_delta"] = ev["huber_delta"]
    elif "huber_delta" in ev:
        _fail("huber_delta present while loss is not huber")
    if "model.target_residual" in space["bounds"]:
        flat["model.target_residual"] = "seasonal_naive_24" if model.get("target_residual") else "none"
    validate_flat(flat, space)
    return flat


def config_identity(nested):
    """Configuration identity excluding the seed: paired seeds share it."""
    stripped = copy.deepcopy(nested)
    stripped["evaluator"].pop("seed", None)
    return digest(stripped)
