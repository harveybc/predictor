"""I6-N v2 joint search space: exact plan domain, canonical genotype, identities.

Lane E, plan af0c3570 ``I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md`` section 3.
No data, TensorFlow, devices, filesystem writes or global RNG. Every function
is pure; callers pass their own ``random.Random``.

Single-sourced identity
-----------------------
``genotype_sha256(g)`` (published as ``genotype_canonical_sha256``) is the
frozen v1 evaluator identity ``tools.i6n_search.effective_identity`` of the
exact evaluator genome this genotype maps to. The v2 -> v1 mapping is
injective on the v2 domain, so lane A's receipts, lane B's ledger and the 48
completed endpoint candidates all share one canonical hash. ``tools/i6n_search.py``
is deliberately left byte-identical so that hash stays reproducible.

Infeasible genotypes raise :class:`Infeasible` with every typed reason found.
Nothing is repaired into a different experiment; only inactive genes and the
endpoint alias (trajectory k=1 == endpoint) are canonicalized. Plan v1.1:
strides are a free per-layer 1|2 list; none/progressive/final/mixed are
derived labels (and seed choices), not the admissible domain.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from itertools import combinations_with_replacement, product
import json

from tools import i6n_model
from tools import i6n_search as _v1


GENOTYPE_SCHEMA = "i6n.genotype.v2.1"  # plan v1.1: per-layer reducer_strides gene
EFFECTIVE_SCHEMA = "i6n.effective_config.v2"

SHORT_HORIZONS = (2, 3, 4, 5, 6, 12)
LONG_HORIZONS = (18, 24, 36, 48, 60, 72)
HORIZONS = SHORT_HORIZONS + LONG_HORIZONS
HISTORY_MULTIPLIERS = (1, 2, 4, 6)
MIN_HISTORY_HOURS = 24
INPUT_TRANSFORMS = ("identity", "difference")
SUPERVISIONS = ("endpoint", "trajectory")
TRAJECTORY_K = (2, 3, 4)
ENTRANCES = ("conv", "transformer", "conv_transformer")
ENTRANCE_WIDTHS = (8, 16, 32, 64, 128)
TRANSFORMER_BLOCKS = (1, 2)
REDUCER_COUNTS = (1, 2, 3)
REDUCER_FILTERS = (4, 8, 16, 32, 64, 128)
STRIDE_PATTERNS = ("none", "progressive", "final")  # named labels/seeds, not the domain
REDUCER_STRIDES = (1, 2)  # per reducer layer (plan v1.1 ruling)
ADAPTERS = ("none", "causalconv", "lstm")
ADAPTER_WIDTHS = (8, 16, 32, 64)
MIN_FINAL_STEPS = 2  # inherited model rule: the temporal core keeps t and >=1 past step

GENE_KEYS = (
    "schema", "horizon_hours", "history_hours", "delta_x_hours", "input_transform",
    "supervision", "trajectory_k", "entrance", "width", "transformer_blocks",
    "reducer_filters", "reducer_strides", "adapter", "adapter_width",
)

SIZE_CLASSES = (
    {"class_id": "S1", "max_samples": 25, "max_width": 32, "min_delta_x_hours": 2,
     "requires_admitted_cost_pilot": True,
     "note": "downsampled first class (dx>=2), samples<=25, every width<=32"},
    {"class_id": "S2", "max_samples": 55, "max_width": 64, "min_delta_x_hours": 1,
     "requires_admitted_cost_pilot": True,
     "note": "samples<=55 (hourly allowed), every width<=64"},
    {"class_id": "S3", "max_samples": 433, "max_width": 128, "min_delta_x_hours": 1,
     "requires_admitted_cost_pilot": True,
     "note": "up to 433 samples / width 128; pilot must exercise the 433-sample contract"},
)
CLASS_IDS = tuple(c["class_id"] for c in SIZE_CLASSES)
_CLASS = {c["class_id"]: c for c in SIZE_CLASSES}

MUTATION_GROUPS = ("window", "input_transform", "supervision", "family", "width",
                   "reducers", "strides", "adapter")

REASON_CODES = (
    "SCHEMA_MISMATCH", "HORIZON_NOT_IN_DOMAIN", "HISTORY_NOT_IN_DOMAIN",
    "DELTA_X_NOT_COMMON_DIVISOR", "INPUT_TRANSFORM_NOT_IN_DOMAIN", "SUPERVISION_NOT_IN_DOMAIN",
    "TRAJECTORY_K_NOT_IN_DOMAIN", "TRAJECTORY_DY_NOT_INTEGER", "ENTRANCE_NOT_IN_DOMAIN",
    "WIDTH_NOT_IN_DOMAIN", "TRANSFORMER_BLOCKS_NOT_IN_DOMAIN", "REDUCER_COUNT_NOT_IN_DOMAIN",
    "REDUCER_FILTER_NOT_IN_DOMAIN", "REDUCER_FILTERS_INCREASING", "REDUCER_STRIDES_NOT_IN_DOMAIN",
    "REDUCER_STRIDES_LENGTH_MISMATCH",
    "ADAPTER_NOT_IN_DOMAIN", "ADAPTER_WIDTH_NOT_IN_DOMAIN", "FINAL_STEPS_BELOW_MINIMUM",
    "MODEL_CONTRACT_REJECTED", "EVALUATOR_CONTRACT_REJECTED",
)
# Plan dispositions that are NOT domain infeasibility (the genotype is valid).
ADMISSION_CODES = ("CLASS_NOT_OPEN", "CLASS_BUDGET_EXHAUSTED", "DUPLICATE_IDENTITY")


class Infeasible(ValueError):
    """Typed infeasibility; ``reasons`` is a list of {code, gene, detail}."""

    def __init__(self, reasons):
        self.reasons = [dict(r) for r in reasons]
        super().__init__("infeasible genotype: " + ", ".join(
            f"{r['code']}({r['gene']})" for r in self.reasons))


def _is_int(value):
    return type(value) is int


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


def _hex64(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")
    return value


# --- domain helpers -------------------------------------------------------------

def history_domain(h):
    """L = max(24, m*h), m in {1,2,4,6}, deduplicated and sorted."""
    return tuple(sorted({max(MIN_HISTORY_HOURS, m * h) for m in HISTORY_MULTIPLIERS}))


def delta_x_domain(h, history):
    """Integer-hour divisors common to h and L, including 1."""
    return tuple(d for d in range(1, h + 1) if h % d == 0 and history % d == 0)


def trajectory_k_domain(h):
    """k in {2,3,4} with integer dy = h/k (k=1 is the endpoint)."""
    return tuple(k for k in TRAJECTORY_K if h % k == 0)


def stride_list(pattern, n_reducers):
    """Right-anchored stride per reducer for a named pattern."""
    if pattern == "none":
        return [1] * n_reducers
    if pattern == "progressive":
        return [2] * n_reducers
    if pattern == "final":
        return [1] * (n_reducers - 1) + [2]
    raise ValueError(f"unknown stride pattern {pattern!r}")


def stride_label(strides):
    """Derived label: none / progressive / final / mixed (one reducer [2] is final)."""
    n = len(strides)
    if strides == [1] * n:
        return "none"
    if strides == [1] * (n - 1) + [2]:
        return "final"
    if strides == [2] * n:
        return "progressive"
    return "mixed"


def _steps(samples, strides):
    out = [samples]
    for s in strides:
        out.append((out[-1] + s - 1) // s)  # right-anchored: keeps the t slot
    return out


# --- canonicalization ---------------------------------------------------------------

def canonicalize(raw):
    """Return the canonical genotype or raise :class:`Infeasible` (never repairs)."""
    if not isinstance(raw, Mapping) or set(raw) != set(GENE_KEYS):
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise Infeasible([{"code": "SCHEMA_MISMATCH", "gene": "*", "detail":
                           f"missing={sorted(set(GENE_KEYS) - keys)} unknown={sorted(map(str, keys - set(GENE_KEYS)))}"}])
    if raw["schema"] != GENOTYPE_SCHEMA:
        raise Infeasible([{"code": "SCHEMA_MISMATCH", "gene": "schema", "detail": repr(raw["schema"])}])
    g = deepcopy(dict(raw))
    reasons = []

    def bad(code, gene, detail):
        reasons.append({"code": code, "gene": gene, "detail": detail})

    h, L, dx = g["horizon_hours"], g["history_hours"], g["delta_x_hours"]
    h_ok = _is_int(h) and h in HORIZONS
    if not h_ok:
        bad("HORIZON_NOT_IN_DOMAIN", "horizon_hours", f"{h!r} not in {HORIZONS}")
    L_ok = h_ok and _is_int(L) and L in history_domain(h)
    if h_ok and not L_ok:
        bad("HISTORY_NOT_IN_DOMAIN", "history_hours", f"{L!r} not in {history_domain(h)}")
    dx_ok = L_ok and _is_int(dx) and dx in delta_x_domain(h, L)
    if L_ok and not dx_ok:
        bad("DELTA_X_NOT_COMMON_DIVISOR", "delta_x_hours", f"{dx!r} not in {delta_x_domain(h, L)}")
    elif not L_ok and not (_is_int(dx) and dx >= 1):
        bad("DELTA_X_NOT_COMMON_DIVISOR", "delta_x_hours", f"{dx!r} is not a positive integer")

    if not (isinstance(g["input_transform"], str) and g["input_transform"] in INPUT_TRANSFORMS):
        bad("INPUT_TRANSFORM_NOT_IN_DOMAIN", "input_transform", repr(g["input_transform"]))

    sup, k = g["supervision"], g["trajectory_k"]
    if not (isinstance(sup, str) and sup in SUPERVISIONS):
        bad("SUPERVISION_NOT_IN_DOMAIN", "supervision", repr(sup))
    elif sup == "trajectory":
        if _is_int(k) and k == 1:
            g["supervision"] = "endpoint"  # endpoint alias: not another fit
        elif not (_is_int(k) and k in TRAJECTORY_K):
            bad("TRAJECTORY_K_NOT_IN_DOMAIN", "trajectory_k", f"{k!r} not in {TRAJECTORY_K}")
        elif h_ok and h % k:
            bad("TRAJECTORY_DY_NOT_INTEGER", "trajectory_k", f"h={h} not divisible by k={k}")
    if g["supervision"] == "endpoint":
        g["trajectory_k"] = None

    ent, width = g["entrance"], g["width"]
    ent_ok = isinstance(ent, str) and ent in ENTRANCES
    if not ent_ok:
        bad("ENTRANCE_NOT_IN_DOMAIN", "entrance", repr(ent))
    width_ok = _is_int(width) and width in ENTRANCE_WIDTHS
    if not width_ok:
        bad("WIDTH_NOT_IN_DOMAIN", "width", f"{width!r} not in {ENTRANCE_WIDTHS}")
    if ent == "conv":
        g["transformer_blocks"] = 0
    elif ent_ok and not (_is_int(g["transformer_blocks"]) and g["transformer_blocks"] in TRANSFORMER_BLOCKS):
        bad("TRANSFORMER_BLOCKS_NOT_IN_DOMAIN", "transformer_blocks",
            f"{g['transformer_blocks']!r} not in {TRANSFORMER_BLOCKS} for {ent}")

    filters = g["reducer_filters"]
    filters_ok = isinstance(filters, (list, tuple)) and len(filters) in REDUCER_COUNTS
    if not filters_ok:
        bad("REDUCER_COUNT_NOT_IN_DOMAIN", "reducer_filters", repr(filters))
    else:
        g["reducer_filters"] = filters = list(filters)
        for i, f in enumerate(filters):
            if not (_is_int(f) and f in REDUCER_FILTERS):
                bad("REDUCER_FILTER_NOT_IN_DOMAIN", f"reducer_filters[{i}]", repr(f))
                filters_ok = False
        if filters_ok and width_ok:
            chain = [width, *filters]
            if chain != sorted(chain, reverse=True):
                bad("REDUCER_FILTERS_INCREASING", "reducer_filters",
                    f"entrance width then reducers must be non-increasing: {chain}")
    strides = g["reducer_strides"]
    if not isinstance(strides, (list, tuple)) or not all(_is_int(x) and x in REDUCER_STRIDES for x in strides):
        bad("REDUCER_STRIDES_NOT_IN_DOMAIN", "reducer_strides", f"{strides!r}: each layer 1 or 2")
    else:
        g["reducer_strides"] = strides = list(strides)
        if isinstance(filters, list) and len(filters) in REDUCER_COUNTS and len(strides) != len(filters):
            bad("REDUCER_STRIDES_LENGTH_MISMATCH", "reducer_strides",
                f"{len(strides)} strides for {len(filters)} reducers")

    adapter = g["adapter"]
    if not (isinstance(adapter, str) and adapter in ADAPTERS):
        bad("ADAPTER_NOT_IN_DOMAIN", "adapter", repr(adapter))
    elif adapter == "none":
        g["adapter_width"] = None
    elif not (_is_int(g["adapter_width"]) and g["adapter_width"] in ADAPTER_WIDTHS):
        bad("ADAPTER_WIDTH_NOT_IN_DOMAIN", "adapter_width", f"{g['adapter_width']!r} not in {ADAPTER_WIDTHS}")

    if not reasons:
        steps = _steps(1 + L // dx, g["reducer_strides"])
        if steps[-1] < MIN_FINAL_STEPS:
            bad("FINAL_STEPS_BELOW_MINIMUM", "reducer_strides",
                f"right-anchored steps {steps} end below {MIN_FINAL_STEPS}")
    if not reasons:
        try:
            i6n_model.normalize_config(model_core(g, _checked=True))
        except ValueError as exc:
            bad("MODEL_CONTRACT_REJECTED", "*", str(exc))
    if not reasons:
        try:
            genome = to_evaluator_genome(g, _checked=True)
            if _v1.normalize_genome(genome) != genome:
                raise ValueError("evaluator genome is not v1-canonical")
        except ValueError as exc:
            bad("EVALUATOR_CONTRACT_REJECTED", "*", str(exc))
    if reasons:
        raise Infeasible(reasons)
    return g


def check(raw):
    """Non-raising form: FEASIBLE with identity, or INFEASIBLE with typed reasons."""
    try:
        g = canonicalize(raw)
    except Infeasible as exc:
        return {"status": "INFEASIBLE", "reasons": exc.reasons}
    return {"status": "FEASIBLE", "genotype": g, "genotype_canonical_sha256": genotype_sha256(g)}


# --- derived values, model core, evaluator mapping ---------------------------------------

def derived(genotype):
    g = canonicalize(genotype)
    h, L, dx = g["horizon_hours"], g["history_hours"], g["delta_x_hours"]
    strides = list(g["reducer_strides"])
    steps = _steps(1 + L // dx, strides)
    dy = None if g["supervision"] == "endpoint" else h // g["trajectory_k"]
    outputs = [h] if dy is None else list(range(dy, h + 1, dy))
    return {
        "horizon_hours": h, "history_hours": L, "delta_x_hours": dx,
        "n_samples": 1 + L // dx, "input_offsets_hours": list(range(-L, 1, dx)),
        "delta_y_hours": dy, "output_offsets_hours": outputs, "n_heads": len(outputs),
        "reducer_strides": strides, "stride_pattern": stride_label(strides),
        "steps_per_stage": steps, "final_steps": steps[-1],
        "t_retained": True,
    }


def model_core(genotype, _checked=False):
    """Config for ``tools.i6n_model.build_model`` (n_horizons = number of heads)."""
    g = genotype if _checked else canonicalize(genotype)
    k = g["trajectory_k"]
    return {
        "entrance": g["entrance"], "width": g["width"],
        "transformer_blocks": g["transformer_blocks"],
        "reducer_filters": list(g["reducer_filters"]),
        "reducer_strides": list(g["reducer_strides"]),
        "adapter": g["adapter"], "adapter_width": g["adapter_width"],
        "n_horizons": 1 if k is None else k,
    }


def to_evaluator_genome(genotype, _checked=False):
    """Exact v1 genome consumed by lane A's evaluator (INTERFACE_A A.1)."""
    g = genotype if _checked else canonicalize(genotype)
    h, L = g["horizon_hours"], g["history_hours"]
    multiplier = min(m for m in HISTORY_MULTIPLIERS if max(MIN_HISTORY_HOURS, m * h) == L)
    k = g["trajectory_k"]
    return {
        "horizon": h, "multiplier": multiplier, "delta_x": g["delta_x_hours"],
        "input_transform": g["input_transform"],
        "output_mode": "endpoint" if k is None else "trajectory",
        "delta_y": None if k is None else h // k,
        "core": model_core(g, _checked=True),
    }




def from_evaluator_genome(genome):
    """Structural v1 -> v2 translation WITHOUT repair; canonicalize() types the result."""
    g1 = _v1.normalize_genome(genome)
    core, h = g1["core"], g1["horizon"]
    dy = g1["delta_y"]
    return {
        "schema": GENOTYPE_SCHEMA, "horizon_hours": h,
        "history_hours": max(MIN_HISTORY_HOURS, g1["multiplier"] * h),
        "delta_x_hours": g1["delta_x"], "input_transform": g1["input_transform"],
        "supervision": g1["output_mode"],
        "trajectory_k": None if dy is None else (h // dy if h % dy == 0 else None),
        "entrance": core["entrance"], "width": core["width"],
        "transformer_blocks": core["transformer_blocks"],
        "reducer_filters": list(core["reducer_filters"]),
        "reducer_strides": list(core["reducer_strides"]),
        "adapter": core["adapter"], "adapter_width": core["adapter_width"],
    }


# --- identities ---------------------------------------------------------------------------

def genotype_sha256(genotype):
    """Canonical genotype identity (published as genotype_canonical_sha256)."""
    return _v1.effective_identity(to_evaluator_genome(canonicalize(genotype), _checked=True))


def evaluator_config(genotype):
    """The exact ``config`` placed in lane A's evaluator request."""
    return _v1.effective_config(to_evaluator_genome(canonicalize(genotype), _checked=True))


def effective_config_sha256(genotype):
    """sha256 of canonical JSON (sort_keys, ',' ':' separators, UTF-8) of evaluator_config."""
    return _sha(evaluator_config(genotype))


def search_candidate_sha256(genotype, plan_sha256):
    """Fold-independent search candidate: {plan_sha256, horizon_hours, genotype id}."""
    g = canonicalize(genotype)
    return _sha({"plan_sha256": _hex64(plan_sha256, "plan_sha256"),
                 "horizon_hours": g["horizon_hours"],
                 "genotype_canonical_sha256": genotype_sha256(g)})


FIT_BINDING_KEYS = frozenset(("experiment", "fold_id", "context_sha256", "support_sha256"))


def fit_id(genotype, binding):
    """One physical fit unit (lane B formula, derived from E's canonical config).

    sha256(canonical {experiment, fold, effective_config_sha256, context_sha256,
    support_sha256}). Plan independent, so a verified terminal is reusable
    across plans/nodes/directories only when every component matches.
    """
    if not isinstance(binding, Mapping) or set(binding) != FIT_BINDING_KEYS:
        raise ValueError(f"fit binding requires exactly {sorted(FIT_BINDING_KEYS)}")
    for key in ("experiment", "fold_id"):
        if not isinstance(binding[key], str) or not binding[key]:
            raise ValueError(f"{key} must be a nonempty string")
    return _sha({"experiment": binding["experiment"], "fold": binding["fold_id"],
                 "effective_config_sha256": effective_config_sha256(genotype),
                 "context_sha256": _hex64(binding["context_sha256"], "context_sha256"),
                 "support_sha256": _hex64(binding["support_sha256"], "support_sha256")})


def inactive_genes(genotype):
    g = canonicalize(genotype)
    out = {}
    if g["entrance"] == "conv":
        out["transformer_blocks"] = 0
    if g["supervision"] == "endpoint":
        out["trajectory_k"] = None
    if g["adapter"] == "none":
        out["adapter_width"] = None
    return out


def class_of(genotype):
    g = canonicalize(genotype)
    n = 1 + g["history_hours"] // g["delta_x_hours"]
    widest = max(g["width"], g["adapter_width"] or 0, *g["reducer_filters"])
    for c in SIZE_CLASSES:
        if n <= c["max_samples"] and widest <= c["max_width"] and g["delta_x_hours"] >= c["min_delta_x_hours"]:
            return c["class_id"]
    # Unreachable for canonical genotypes: S3 admits every domain value (lane A review E-A4).
    raise AssertionError("domain genotype outside every size class")


def effective_config(genotype):
    """Complete candidate record: genotype, active/inactive genes, derived, configs, ids."""
    g = canonicalize(genotype)
    inactive = inactive_genes(g)
    return {
        "schema": EFFECTIVE_SCHEMA, "genotype": g,
        "genotype_canonical_sha256": genotype_sha256(g),
        "effective_config_sha256": effective_config_sha256(g),
        "active_genes": sorted(k for k in GENE_KEYS if k != "schema" and k not in inactive),
        "inactive_genes": inactive, "derived": derived(g), "model_core": model_core(g),
        "size_class": class_of(g), "evaluator_genome": to_evaluator_genome(g),
        "evaluator_config": evaluator_config(g),
    }


# --- seeds -----------------------------------------------------------------------------------

def _seed_window(h):
    s1 = _CLASS["S1"]
    for L in history_domain(h):
        for dx in delta_x_domain(h, L):
            if dx >= s1["min_delta_x_hours"] and 1 + L // dx <= s1["max_samples"]:
                return L, dx
    raise ValueError(f"no downsampled S1 window for h={h}")


def _base(h, L, dx, **changes):
    g = {"schema": GENOTYPE_SCHEMA, "horizon_hours": h, "history_hours": L, "delta_x_hours": dx,
         "input_transform": "identity", "supervision": "endpoint", "trajectory_k": None,
         "entrance": "conv", "width": 8, "transformer_blocks": 0, "reducer_filters": [4],
         "reducer_strides": [1], "adapter": "none", "adapter_width": None}
    g.update(changes)
    return canonicalize(g)


def seed_genotype(h):
    """Minimal cheap seed: Conv8 -> Conv4, stride 1, no Transformer/LSTM, endpoint.

    Window: the smallest history with a downsampled S1 grid, then its finest dx.
    A seed, not a winner or a permanent capacity limit.
    """
    if not (_is_int(h) and h in HORIZONS):
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return _base(h, *_seed_window(h))


def alternative_genotype(h):
    """Small non-equivalent alternative: Conv16 -> Conv8 -> Conv4, same window."""
    seed = seed_genotype(h)
    return _base(h, seed["history_hours"], seed["delta_x_hours"], width=16, reducer_filters=[8, 4],
                 reducer_strides=[1, 1])


# --- sampling and mutation ----------------------------------------------------------------------

def _open(max_class=None, classes=None):
    if classes is None:
        if max_class not in CLASS_IDS:
            raise ValueError(f"max_class must be one of {CLASS_IDS}")
        classes = CLASS_IDS[:CLASS_IDS.index(max_class) + 1]
    classes = [c for c in CLASS_IDS if c in set(classes)]
    if not classes:
        raise ValueError("no open size class")
    return classes


def sample_genotype(h, rng, max_class=None, *, classes=None, max_tries=2000,
                    predicate=None):
    """Sample a feasible genotype inside the open classes (all earlier genes free)."""
    open_ids = _open(max_class, classes)
    limits = [_CLASS[c] for c in open_ids]
    windows = [(L, dx) for L in history_domain(h) for dx in delta_x_domain(h, L)
               if any(1 + L // dx <= c["max_samples"] and dx >= c["min_delta_x_hours"] for c in limits)]
    max_width = max(c["max_width"] for c in limits)
    by_L = {}
    for L, dx in windows:
        by_L.setdefault(L, []).append(dx)
    for _ in range(max_tries):
        L = rng.choice(sorted(by_L))
        dx = rng.choice(by_L[L])
        entrance = rng.choice(ENTRANCES)
        width = rng.choice([w for w in ENTRANCE_WIDTHS if w <= max_width])
        n_red = rng.choice(REDUCER_COUNTS)
        filters, top = [], width
        for _ in range(n_red):
            top = rng.choice([f for f in REDUCER_FILTERS if f <= top])
            filters.append(top)
        adapter = rng.choice(ADAPTERS)
        ks = trajectory_k_domain(h)
        trajectory = bool(ks) and rng.random() < .5
        raw = {"schema": GENOTYPE_SCHEMA, "horizon_hours": h, "history_hours": L, "delta_x_hours": dx,
               "input_transform": rng.choice(INPUT_TRANSFORMS),
               "supervision": "trajectory" if trajectory else "endpoint",
               "trajectory_k": rng.choice(ks) if trajectory else None,
               "entrance": entrance, "width": width,
               "transformer_blocks": 0 if entrance == "conv" else rng.choice(TRANSFORMER_BLOCKS),
               "reducer_filters": filters,
               "reducer_strides": [rng.choice(REDUCER_STRIDES) for _ in filters],
               "adapter": adapter,
               "adapter_width": None if adapter == "none" else rng.choice(
                   [w for w in ADAPTER_WIDTHS if w <= max_width])}
        try:
            g = canonicalize(raw)
        except Infeasible:
            continue
        if class_of(g) in open_ids and (predicate is None or predicate(g)):
            return g
    raise ValueError("sampling attempt bound exhausted")


def _variants(g, group):
    def changed(**updates):
        child = deepcopy(g)
        child.update(updates)
        return child

    h = g["horizon_hours"]
    if group == "window":
        for L in history_domain(h):
            for dx in delta_x_domain(h, L):
                yield changed(history_hours=L, delta_x_hours=dx)
    elif group == "input_transform":
        for t in INPUT_TRANSFORMS:
            yield changed(input_transform=t)
    elif group == "supervision":
        yield changed(supervision="endpoint", trajectory_k=None)
        for k in trajectory_k_domain(h):
            yield changed(supervision="trajectory", trajectory_k=k)
    elif group == "family":
        yield changed(entrance="conv", transformer_blocks=0)
        for ent in ENTRANCES[1:]:
            for blocks in TRANSFORMER_BLOCKS:
                yield changed(entrance=ent, transformer_blocks=blocks)
    elif group == "width":
        for w in ENTRANCE_WIDTHS:
            yield changed(width=w)
            # joint conditional proposal: reducers capped by the new entrance width
            yield changed(width=w, reducer_filters=[min(w, f) for f in g["reducer_filters"]])
    elif group == "reducers":
        options = [f for f in REDUCER_FILTERS if f <= g["width"]]
        old = g["reducer_strides"]
        for count in REDUCER_COUNTS:
            # keep existing per-layer strides where layers survive; new layers stride 1
            strides = (old + [1] * count)[:count]
            for combo in combinations_with_replacement(options, count):
                yield changed(reducer_filters=sorted(combo, reverse=True), reducer_strides=list(strides))
    elif group == "strides":
        for strides in product(REDUCER_STRIDES, repeat=len(g["reducer_filters"])):
            yield changed(reducer_strides=list(strides))
    elif group == "adapter":
        yield changed(adapter="none", adapter_width=None)
        for a in ADAPTERS[1:]:
            for w in ADAPTER_WIDTHS:
                yield changed(adapter=a, adapter_width=w)
    else:
        raise ValueError(f"unknown mutation group {group!r}")


def mutate(genotype, rng, max_class=None, *, classes=None, group=None, exclude=(), filtered=None,
           predicate=None):
    """Joint conditional mutation inside the open classes.

    Returns a new canonical genotype, or None when no admissible distinct
    variant exists (never the parent, never a duplicate filler). Infeasible
    and class-closed variants are appended to ``filtered`` with typed reasons.
    """
    parent = canonicalize(genotype)
    open_ids = _open(max_class, classes)
    if group is not None and group not in MUTATION_GROUPS:
        raise ValueError("unknown mutation group")
    groups = [group] if group is not None else rng.sample(MUTATION_GROUPS, len(MUTATION_GROUPS))
    blocked = set(exclude) | {genotype_sha256(parent)}
    for name in groups:
        options = {}
        for variant in _variants(parent, name):
            try:
                child = canonicalize(variant)
            except Infeasible as exc:
                if filtered is not None:
                    filtered.append({"group": name, "reasons": exc.reasons})
                continue
            ident = genotype_sha256(child)
            if ident in blocked or ident in options:
                continue
            cls = class_of(child)
            if cls not in open_ids:
                if filtered is not None:
                    filtered.append({"group": name, "genotype_canonical_sha256": ident, "reasons": [
                        {"code": "CLASS_NOT_OPEN", "gene": "*", "detail": f"{cls} not in {open_ids}"}]})
                continue
            if predicate is not None and not predicate(child):
                if filtered is not None:
                    filtered.append({"group": name, "genotype_canonical_sha256": ident, "reasons": [
                        {"code": "CAMPAIGN_CONSTRAINT", "gene": "*",
                         "detail": "candidate is outside the sealed campaign subdomain"}]})
                continue
            options[ident] = child
        if options:
            return deepcopy(options[rng.choice(sorted(options))])
    return None


# --- legacy (v1) completed candidates ------------------------------------------------------------

LEGACY_FOLD_REUSE_REASONS = (
    "EVALUATION_CONTEXT_NOT_A_2023_QUARTER_FOLD",  # one inner interval before week 2024-01-07
    "FIT_ID_CANNOT_MATCH_ANY_PLAN_FOLD",           # fit_id binds fold, context and support
)


def legacy_map(genome_v1, stored_identity=None):
    """Map one completed v1 endpoint candidate onto the v2 canonical identity.

    The genotype identity is preserved when it lies in the v2 domain; its
    scores are never reused as fold scores, and its legacy context is never
    refit (the 48 fits are historical engineering).
    """
    v1_identity = _v1.effective_identity(genome_v1)
    rec = {"v1_identity": v1_identity,
           "v1_identity_recomputed": None if stored_identity is None else stored_identity == v1_identity,
           "fold_score_reuse": "NONE", "fold_reuse_reasons": list(LEGACY_FOLD_REUSE_REASONS),
           "refit_of_legacy_context": "FORBIDDEN"}
    try:
        g = canonicalize(from_evaluator_genome(genome_v1))
    except Infeasible as exc:
        rec.update(genotype_status="OUT_OF_V2_DOMAIN", reasons=exc.reasons, v2_genotype=None,
                   genotype_canonical_sha256=None, size_class=None)
        return rec
    rec.update(genotype_status="MAPPED", reasons=[], v2_genotype=g,
               genotype_canonical_sha256=genotype_sha256(g), size_class=class_of(g))
    return rec
