#!/usr/bin/env python3
"""The representation-utility harness, causal by construction: design ready for review, NOT an
experiment (L5, M2–M4).

What it measures, and only that: the PREDICTIVE LOSS of a fixed, small probe model (ridge for
a return target, logistic for a direction target) fed with a branch of features, compared
PAIRWISE on the same emittable rows between the raw branch and a transformed branch (and, when
the sealed protocol declares it, an augmented raw+transformed branch). The difference of losses
is a difference of losses: never information in bits, never mutual information, never trading
utility.

Causality is not a flag. A representation is built here, per block, by the REAL operator:
`fit` on that block's training prefix only, `transform` with its own fresh state over the
series, validated by the operator contract (`emitted_at`, `available`). Every feature a row
consumes must have been emitted at or before that row's decision instant — a late output is
aligned to the later row it is first available for, never cured by a larger purge. Rows are
identified by observation id; discordant ids or times refuse. A prefix-consistency check
re-transforms the series cut at sampled decision rows and demands the same outputs: a
representation whose prefix outputs move when the tail changes is refused as non-causal.
Eligibility is read from the verified matrix's per-cell record for (unit, variable,
operator, spec digest): a control that is not in that record, however spectacular its loss,
is never scored.

Inference uses scipy's Student t over walk-forward validation blocks with a purge of
horizon + reach + window; blocks that fall short of the minimum make the whole contrast
INSUFFICIENT_ROWS (no silent denominator); ADVANCES is emitted only under a protocol whose
false-advance rate was measured beforehand on a dependent generator (calibration record in
the sealed protocol); otherwise the result is descriptive (`INCONCLUSIVE_UNCALIBRATED`).

Budgets are observed, not declared: `run_isolated` runs a contrast in a child process under
`df_isolated_runner` (wall, CPU, memory ceilings enforced during the work); an exhausted
budget is RESOURCE_EXCEEDED with the measured cost and no partial score.

The reserved holdout is adjudicated once per (reserve identity, protocol), write-once, in the
state directory — never in an arbitrary new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOT_COMPARABLE = "NOT_COMPARABLE"
INSUFFICIENT_ROWS = "INSUFFICIENT_ROWS"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
RESOURCE_EXCEEDED = "RESOURCE_EXCEEDED"
ADVANCES = "ADVANCES"
DOES_NOT_ADVANCE = "DOES_NOT_ADVANCE"
INCONCLUSIVE_UNCALIBRATED = "INCONCLUSIVE_UNCALIBRATED"
REFUSED = "REFUSED"
SCORE_UNVERIFIED = "SCORE_UNVERIFIED"
CONTRAST_SCHEMA = "df_utility_contrast.v1"
TARGETS = {"direction": "logistic", "return": "ridge"}
BRANCHES = ("raw", "transformed", "augmented", "raw_wide")   # raw_wide: the capacity control
HOLDOUT_STATE = Path("~/.local/state/crispdm-data-foundation/utility_holdout").expanduser()


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --- the protocol, validated and sealed -----------------------------------------------------------

class ProtocolRefusal(ValueError):
    """A protocol that cannot be sealed."""


@dataclass(frozen=True)
class Protocol:
    target: str
    horizon: int
    model: str
    window: int
    n_blocks: int
    margin: float
    seed: int
    family: tuple                    # the sealed contrast identities; comparisons = len(family)
    min_rows_per_block: int = 30
    alpha: float = 0.05
    ridge_lambda: float = 1.0
    logistic_steps: int = 200
    branches: tuple = ("raw", "transformed")
    blocks_policy: str = "all_or_insufficient"
    inference: str = "block_t"
    calibration: dict | None = None  # the full record `calibrate` returns, validated below
    calibration_plan: dict | None = None  # {"generator", "n_sims", "n", "bound_confidence"}
    prefix_checks: int = 8           # decision rows re-transformed from a cut series
    schema: str = "df_utility_protocol.v3"

    def __post_init__(self):
        problems = []
        if self.target not in TARGETS:
            problems.append(f"target {self.target!r} not in {sorted(TARGETS)}")
        elif self.model != TARGETS[self.target]:
            problems.append(f"target {self.target!r} pairs with model {TARGETS[self.target]!r}, "
                            f"not {self.model!r}")
        for name, lo in (("horizon", 1), ("window", 1), ("n_blocks", 3),
                         ("min_rows_per_block", 5), ("prefix_checks", 1)):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < lo:
                problems.append(f"{name} must be an integer >= {lo}, got {v!r}")
        import math
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool) \
                or not math.isfinite(self.alpha) or not (0 < self.alpha < 1):
            problems.append("alpha must be a finite number in (0, 1)")
        if not isinstance(self.margin, (int, float)) or isinstance(self.margin, bool) \
                or not math.isfinite(self.margin) or self.margin < 0:
            problems.append("margin must be a finite non-negative number")
        for name in ("ridge_lambda",):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                problems.append(f"{name} must be a finite non-negative number")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            problems.append("seed must be an integer")
        if not isinstance(self.logistic_steps, int) or self.logistic_steps < 1:
            problems.append("logistic_steps must be a positive integer")
        if self.calibration_plan is not None:
            plan = self.calibration_plan
            need = {"generator", "n_sims", "n", "bound_confidence"}
            if not isinstance(plan, dict) or set(plan) != need:
                problems.append(f"calibration_plan must carry exactly {sorted(need)}")
            else:
                if plan["generator"] not in GENERATORS or not GENERATORS[plan["generator"]]["null"]:
                    problems.append("calibration_plan.generator must be a null of no effect")
                if not isinstance(plan["n_sims"], int) or plan["n_sims"] < 1:
                    problems.append("calibration_plan.n_sims must be a positive integer")
                if not isinstance(plan["n"], int) or plan["n"] < 1:
                    problems.append("calibration_plan.n must be a positive integer")
                c = plan["bound_confidence"]
                if not isinstance(c, (int, float)) or not math.isfinite(c) or not (0 < c < 1):
                    problems.append("calibration_plan.bound_confidence must be in (0, 1)")
        if not self.branches or any(b not in BRANCHES for b in self.branches) \
                or len(set(self.branches)) != len(self.branches):
            problems.append(f"branches must be distinct members of {BRANCHES}")
        if "raw" not in self.branches:
            problems.append("the raw branch is mandatory")
        if not isinstance(self.family, tuple) or not self.family \
                or any(not isinstance(c, str) or not c for c in self.family) \
                or len(set(self.family)) != len(self.family):
            problems.append("family must be a non-empty tuple of distinct contrast ids")
        if self.blocks_policy != "all_or_insufficient":
            problems.append("blocks_policy must be 'all_or_insufficient'")
        if self.inference != "block_t":
            problems.append("inference must be 'block_t'")
        if self.calibration is not None:
            problems += calibration_record_problems(self.calibration)
        if problems:
            raise ProtocolRefusal("; ".join(problems))

    @property
    def comparisons(self) -> int:
        return len(self.family)

    @property
    def alpha_adjusted(self) -> float:
        return self.alpha / self.comparisons

    def sealed(self) -> dict:
        doc = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
        doc["comparisons"] = self.comparisons
        doc["alpha_adjusted"] = self.alpha_adjusted
        doc["protocol_sha256"] = sha_obj(doc)
        return doc

    def base_sha256(self) -> str:
        """The protocol's identity WITHOUT its calibration record: what a record is bound to."""
        doc = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()
               if k != "calibration"}
        return sha_obj(doc)

    def with_calibration(self, record: dict) -> "Protocol":
        """The full record, nothing dropped: a consumer must be able to recount its rate."""
        return Protocol(**{**self.__dict__, "calibration": dict(record)})


#: P1 — a record names the branch pair it simulated, the widths it consumed and the rows
#: policy; a v1 record (the pilot's) is the raw/transformed pair by construction.
ROWS_POLICY = "PAIRED_EMITTABLE_ROWS_ALL_BLOCKS_OR_INSUFFICIENT"
CALIBRATION_PAIR_KEYS = ("branch_a", "branch_b", "widths", "rows_policy")
#: Q2 — v3 adds the canonical key of the scientific COMPUTATION (no family/unit labels), the
#: numeric dependencies, and the source (measured here, or a verified shared record)
CALIBRATION_COMPUTATION_KEYS = ("computation", "computation_sha256", "numeric_dependencies")
GENERATOR_PARAMS = {"white_null": {"phi": 0.0}, "ar1_features_independent_target": {"phi": 0.6},
                    "ar1_null": {"phi": 0.6}}


def numeric_dependencies() -> dict:
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}


SCIENTIFIC_HELPER_MODULES = ("df_d3_contract", "df_d3_operators")


def _normalised_code(code) -> tuple:
    """A code object without its file name and line numbers: bytecode, constants (nested code
    normalised recursively), names, variables, arity and flags — what executes, wherever the
    file lives."""
    consts = tuple(_normalised_code(c) if hasattr(c, "co_code") else repr(c) for c in code.co_consts)
    return (code.co_code, consts, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags)


def _code_objects_sha256(cls) -> str:
    """Digest of the executing code of every function defined on the operator's class hierarchy
    (bytecode, constants, names, and the code of nested functions), object excluded, independent
    of file location. A method replaced in memory or edited on disk changes it; a declaration
    left unchanged does not hide it."""
    h = hashlib.sha256()
    for klass in [k for k in cls.__mro__ if k is not object]:
        h.update(klass.__name__.encode())
        for name, member in sorted(vars(klass).items()):
            fn = getattr(member, "__func__", member) if isinstance(member, (staticmethod, classmethod)) else member
            code = getattr(fn, "__code__", None)
            if code is None:
                continue
            h.update(name.encode())
            h.update(repr(_normalised_code(code)).encode())
    return h.hexdigest()


def scientific_code_identity(operator, harness: str | None = None) -> dict:
    """The local scientific code a calibration actually executes: the operator's implementation
    (code objects of its class hierarchy), the module files that define it and the helpers it
    inherits (contract, operators), the harness, and the numeric environment. Administrative
    labels are absent; the numeric scope is declared honestly (version strings do not prove
    cross-CPU bit identity)."""
    import inspect
    module_file = Path(inspect.getfile(type(operator))).resolve()
    helpers = {}
    for name in SCIENTIFIC_HELPER_MODULES:
        path = HERE / f"{name}.py"
        helpers[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"schema": "df_utility_scientific_code.v1",
            "operator_class": type(operator).__name__,
            "operator_code_sha256": _code_objects_sha256(type(operator)),
            "operator_module": module_file.name,
            "operator_module_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
            "helper_modules_sha256": helpers,
            "harness_sha256": harness or harness_sha256(),
            "numeric_dependencies": numeric_dependencies(),
            "scope": {"numeric_portability": "SAME_ENVIRONMENT_ONLY",
                      "note": "version strings identify the environment that computed; they do not "
                              "demonstrate bit identity across CPUs or BLAS builds"}}


def computation_key(protocol: Protocol, operator, plan: dict, *, branch_a: str, branch_b: str, seed: int,
                    harness: str | None = None) -> dict:
    """The canonical key of ONE calibration computation: everything the simulations depend on,
    nothing they do not. Family ids and unit names are administrative labels and are absent;
    alpha enters through alpha_adjusted (the effective threshold), the multiplicity only through
    it. Two families that differ only in labels share a key; any scientific difference breaks it."""
    contract = _load("df_d3_contract")
    return {"schema": "df_utility_calibration_computation.v2",
            "harness_sha256": harness or harness_sha256(), "numeric_dependencies": numeric_dependencies(),
            "code_identity": scientific_code_identity(operator, harness=harness),
            "generator": plan["generator"], "generator_params": dict(GENERATOR_PARAMS.get(plan["generator"], {})),
            "seed": int(seed), "n": int(plan["n"]), "n_sims": int(plan["n_sims"]),
            "bound_confidence": float(plan["bound_confidence"]),
            "operator": {"kind": operator.KIND, "spec_sha256": contract.spec_sha256(operator.describe()),
                         "params": dict(operator.params)},
            "branch_a": branch_a, "branch_b": branch_b,
            "widths": {"a": branch_width(branch_a, protocol.window), "b": branch_width(branch_b, protocol.window)},
            "target": protocol.target, "horizon": protocol.horizon, "model": protocol.model,
            "window": protocol.window, "n_blocks": protocol.n_blocks, "min_rows_per_block": protocol.min_rows_per_block,
            "ridge_lambda": protocol.ridge_lambda, "logistic_steps": protocol.logistic_steps,
            "prefix_checks": protocol.prefix_checks, "blocks_policy": protocol.blocks_policy,
            "inference": protocol.inference, "margin": protocol.margin, "alpha_adjusted": protocol.alpha_adjusted,
            "failure_policy": FAILURE_POLICY, "rows_policy": ROWS_POLICY}


def computation_sha256(key: dict) -> str:
    return sha_obj(key)


def _as_legacy_key(expected: dict, recorded: dict) -> dict:
    """When the record's key predates code binding (computation v1), compare on its own terms —
    the code identity is dropped from the expectation and its schema kept — so a historical
    record still binds to its contract while never gaining the code-bound guarantee."""
    if "code_identity" not in recorded:
        return {**{k: v for k, v in expected.items() if k != "code_identity"}, "schema": recorded.get("schema")}
    if recorded.get("code_identity", {}).get("harness_sha256") and expected.get("code_identity"):
        expected = {**expected, "code_identity": {**expected["code_identity"],
                                                  "harness_sha256": recorded["code_identity"].get("harness_sha256"),
                                                  "numeric_dependencies": recorded["code_identity"].get("numeric_dependencies")}}
    return expected


def rekeyed(rec: dict) -> dict:
    """A v3 record whose computation key is rebuilt from its own fields (for a record edited
    on purpose in a test or a migration; the digest then seals the edited key)."""
    if "computation" not in rec:
        return dict(rec)
    key = dict(rec["computation"])
    for name in ("generator", "seed", "n", "n_sims", "bound_confidence", "operator", "branch_a", "branch_b", "widths",
                 "target", "model", "window", "n_blocks", "margin", "alpha_adjusted", "rows_policy", "harness_sha256",
                 "numeric_dependencies"):
        key[name] = rec[name]
    if "code_identity" in key:
        key["code_identity"] = {**key["code_identity"], "harness_sha256": rec["harness_sha256"],
                                "numeric_dependencies": rec["numeric_dependencies"]}
    return {**rec, "computation": key, "computation_sha256": computation_sha256(key)}

CALIBRATION_KEYS = ("schema", "generator", "null", "plan", "n_sims", "scored", "failed", "advances",
                    "false_advance_rate", "upper_bound", "bound_confidence", "alpha_adjusted",
                    "seed", "n", "operator", "protocol_base_sha256", "family", "margin", "n_blocks",
                    "window", "target", "model", "per_sim", "per_sim_sha256", "harness_sha256",
                    "cost")


#: O1 — the failure policy, declared before use: a simulation that fails under the null
#: (insufficient rows, refusal) is not dropped from the denominator; for the decision it is
#: counted AS an advance (worst case), so selective failures can only make the bound larger.
FAILURE_POLICY = "WORST_CASE_FAILED_COUNTED_AS_ADVANCES"
SIM_SCORED_KEYS = {"index", "seed", "outcome", "delta_mean", "delta_lower"}
SIM_FAILED_KEYS = {"index", "seed", "outcome", "why"}
SIM_FAILED_OUTCOMES = {INSUFFICIENT_ROWS, REFUSED}


def derive_calibration(rec) -> dict:
    """Everything a decision needs, DERIVED from the per-simulation records — never read from
    the summary (O1). Returns counts, rate, the Clopper–Pearson bound over scored, the decision
    bound under FAILURE_POLICY, and the list of problems found while validating each simulation
    and comparing the record's summaries with the derived values."""
    import math
    problems = []
    sims = rec.get("per_sim")
    n_sims = rec.get("n_sims")
    margin = rec.get("margin")
    if not isinstance(sims, list):
        return {"problems": ["per_sim must be a list"], "scored": 0, "failed": 0, "advances": 0,
                "false_advance_rate": float("nan"), "upper_bound": float("nan"),
                "decision_bound": float("nan"), "failure_policy": FAILURE_POLICY}
    if not isinstance(n_sims, int) or isinstance(n_sims, bool) or len(sims) != n_sims:
        problems.append("per_sim must list every simulation (partial denominator)")
    indices, seeds = [], []
    scored = failed = advances = 0
    margin_ok = isinstance(margin, (int, float)) and not isinstance(margin, bool) and math.isfinite(margin)
    for k, x in enumerate(sims):
        if not isinstance(x, dict) or "outcome" not in x or "index" not in x or "seed" not in x:
            problems.append(f"simulation {k}: not a record with index, seed and outcome")
            continue
        indices.append(x["index"])
        seeds.append(x["seed"])
        if not isinstance(x["index"], int) or isinstance(x["index"], bool) or x["index"] != k:
            problems.append(f"simulation {k}: index {x['index']!r} is not the expected {k}")
        if not isinstance(x["seed"], int) or isinstance(x["seed"], bool) or x["seed"] < 0:
            problems.append(f"simulation {k}: seed is not a non-negative integer")
        out = x["outcome"]
        if out in (ADVANCES, DOES_NOT_ADVANCE):
            if set(x) != SIM_SCORED_KEYS:
                problems.append(f"simulation {k}: a scored simulation carries exactly {sorted(SIM_SCORED_KEYS)}")
                continue
            finite = all(isinstance(x[key], (int, float)) and not isinstance(x[key], bool)
                         and math.isfinite(x[key]) for key in ("delta_mean", "delta_lower"))
            if not finite:
                problems.append(f"simulation {k}: delta is not finite")
                continue
            scored += 1
            if margin_ok:
                advanced = x["delta_lower"] > margin
                if advanced != (out == ADVANCES):
                    problems.append(f"simulation {k}: outcome {out} is not consistent with "
                                    f"delta_lower {x['delta_lower']!r} and margin {margin!r}")
                advances += int(out == ADVANCES)
        elif out in SIM_FAILED_OUTCOMES:
            if set(x) != SIM_FAILED_KEYS:
                problems.append(f"simulation {k}: a failed simulation carries exactly {sorted(SIM_FAILED_KEYS)}")
            failed += 1
        else:
            problems.append(f"simulation {k}: unknown outcome {out!r}")
    if len(set(indices)) != len(indices):
        problems.append("duplicate simulation index")
    if len(set(seeds)) != len(seeds):
        problems.append("duplicate simulation seed")
    conf = rec.get("bound_confidence")
    conf_ok = isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0 < conf < 1
    rate = advances / scored if scored else float("nan")
    upper = clopper_pearson_upper(advances, scored, conf) if scored and conf_ok else float("nan")
    total = scored + failed
    decision = clopper_pearson_upper(advances + failed, total, conf) if total and conf_ok else float("nan")
    derived = {"scored": scored, "failed": failed, "advances": advances, "false_advance_rate": rate,
               "upper_bound": upper, "decision_bound": decision, "failure_policy": FAILURE_POLICY,
               "n_sims": len(sims)}
    for name in ("scored", "failed", "advances"):
        if rec.get(name) != derived[name]:
            problems.append(f"{name} {rec.get(name)!r} is not the derived {derived[name]} "
                            f"(recounted from per_sim)")
    for name in ("false_advance_rate", "upper_bound"):
        v = rec.get(name)
        d = derived[name]
        if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                and (math.isnan(d) and math.isnan(v) or abs(float(v) - d) <= 1e-12)):
            problems.append(f"{name} {v!r} is not the derived {d!r}")
    derived["problems"] = problems
    return derived


def calibration_record_problems(rec) -> list:
    """Everything a calibration record must carry to be consumed; a missing or non-finite
    element makes the record unusable (N2). Counts, rate and bound are re-derived from the
    simulations and compared with the summaries (O1)."""
    import math
    problems = []
    k1, k2 = set(CALIBRATION_KEYS), set(CALIBRATION_KEYS) | set(CALIBRATION_PAIR_KEYS)
    k3 = k2 | set(CALIBRATION_COMPUTATION_KEYS)
    if not isinstance(rec, dict) or set(rec) not in (k1, k2, k3):
        return [f"calibration must carry exactly {list(CALIBRATION_KEYS)} (+ {list(CALIBRATION_PAIR_KEYS)} in v2, "
                f"+ {list(CALIBRATION_COMPUTATION_KEYS)} in v3)"]
    v2 = set(rec) in (k2, k3)
    v3 = set(rec) == k3
    code_bound = v3 and isinstance(rec.get("computation"), dict) and "code_identity" in rec["computation"]
    expected_schema = "df_utility_calibration.v4" if code_bound else "df_utility_calibration.v3" if v3 \
        else "df_utility_calibration.v2" if v2 else "df_utility_calibration.v1"
    if rec["schema"] != expected_schema:
        problems.append("calibration schema does not match its keys")
    if v3:
        key = rec["computation"]
        if not isinstance(key, dict) or computation_sha256(key) != rec["computation_sha256"]:
            problems.append("computation key digest does not seal the key")
        else:
            checks = {"generator": rec["generator"], "seed": rec["seed"], "n": rec["n"], "n_sims": rec["n_sims"],
                      "bound_confidence": rec["bound_confidence"], "operator": rec["operator"],
                      "branch_a": rec["branch_a"], "branch_b": rec["branch_b"], "widths": rec["widths"],
                      "target": rec["target"], "model": rec["model"], "window": rec["window"], "n_blocks": rec["n_blocks"],
                      "margin": rec["margin"], "alpha_adjusted": rec["alpha_adjusted"], "rows_policy": rec["rows_policy"],
                      "harness_sha256": rec["harness_sha256"], "numeric_dependencies": rec["numeric_dependencies"],
                      "failure_policy": FAILURE_POLICY}
            for name, v in checks.items():
                if key.get(name) != v:
                    problems.append(f"computation key {name} disagrees with the record")
    if v2:
        a, b = rec["branch_a"], rec["branch_b"]
        if a not in BRANCHES or b not in BRANCHES or a == b:
            problems.append("branch pair must be two distinct declared branches")
        elif isinstance(rec.get("window"), int):
            if rec["widths"] != {"a": branch_width(a, rec["window"]), "b": branch_width(b, rec["window"])}:
                problems.append("widths are not the pair's widths at this window")
        if rec["rows_policy"] != ROWS_POLICY:
            problems.append(f"rows policy must be {ROWS_POLICY}")
    if rec["generator"] not in GENERATORS:
        problems.append(f"unknown generator {rec['generator']!r}")
    elif not GENERATORS[rec["generator"]]["null"] or rec["null"] is not True:
        problems.append("a calibration generator must be a null of no effect")
    for name in ("n_sims", "scored", "failed", "advances", "n", "n_blocks", "window"):
        v = rec[name]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            problems.append(f"{name} must be a non-negative integer")
    if isinstance(rec["n_sims"], int) and rec["n_sims"] < 1:
        problems.append("zero simulations calibrate nothing")
    if all(isinstance(rec[k], int) for k in ("n_sims", "scored", "failed")) \
            and rec["scored"] + rec["failed"] != rec["n_sims"]:
        problems.append("scored + failed must equal n_sims (no simulation vanishes)")
    if isinstance(rec["scored"], int) and rec["scored"] < 1:
        problems.append("no scored simulation: the rate has no denominator")
    for name in ("false_advance_rate", "upper_bound", "bound_confidence", "alpha_adjusted", "margin"):
        v = rec[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) \
                or v < 0 or (name != "margin" and v > 1):
            problems.append(f"{name} must be a finite number in [0, 1]")
    plan = rec["plan"]
    need = {"generator", "n_sims", "n", "bound_confidence"}
    if not isinstance(plan, dict) or set(plan) != need:
        problems.append(f"plan must carry exactly {sorted(need)}")
    else:
        for name in ("generator", "n_sims", "n", "bound_confidence"):
            if plan[name] != rec[name]:
                problems.append(f"{name} {rec[name]!r} disagrees with the sealed plan {plan[name]!r}")
    if not isinstance(rec["per_sim"], list) or len(rec["per_sim"]) != rec.get("n_sims"):
        problems.append("per_sim must list every simulation")
    elif sha_obj(rec["per_sim"]) != rec["per_sim_sha256"]:
        problems.append("per_sim digest does not seal the simulations")
    if not isinstance(rec["operator"], dict) or set(rec["operator"]) != {"kind", "spec_sha256", "params"}:
        problems.append("operator identity must carry kind, spec_sha256 and params")
    if not isinstance(rec["harness_sha256"], str) or len(rec["harness_sha256"]) != 64:
        problems.append("harness_sha256 must name the code that calibrated")
    problems += derive_calibration(rec)["problems"]
    return problems


# --- the observation contract -----------------------------------------------------------------------

def series(values, *, ids=None, timestamps=None, available_at=None, period_seconds=1,
           target_values=None) -> dict:
    """A series with observation identity: ids strictly increasing, timestamps non-decreasing,
    available_at >= timestamp. Gaps are kept as NaN; nothing is imputed."""
    v = np.asarray(values, dtype=float)
    n = v.size
    ids = np.asarray(ids if ids is not None else np.arange(n))
    ts = np.asarray(timestamps if timestamps is not None else np.arange(n) * period_seconds, dtype=float)
    av = np.asarray(available_at if available_at is not None else ts, dtype=float)
    if not (ids.size == ts.size == av.size == n):
        raise ValueError("ids, timestamps, available_at and values must have one length")
    if n > 1 and not (np.diff(ids) > 0).all():
        raise ValueError("observation ids must be strictly increasing")
    if n > 1 and not (np.diff(ts) >= 0).all():
        raise ValueError("timestamps must not go backwards")
    if (av < ts).any():
        raise ValueError("an observation cannot be available before its timestamp")
    out = {"values": v, "ids": ids, "timestamps": ts, "available_at": av,
           "period_seconds": float(period_seconds)}
    if target_values is not None:
        tv = np.asarray(target_values, dtype=float)
        if tv.size != n:
            raise ValueError("target_values must have the series' length")
        out["target_values"] = tv                  # the label's source when it is not x
    return out


def _as_operator_input(s: dict, upto: int | None = None) -> dict:
    n = s["values"].size if upto is None else upto
    return {"values": [float(x) if not np.isnan(x) else float("nan") for x in s["values"][:n]],
            "timestamps": [float(t) for t in s["timestamps"][:n]],
            "available_at": [float(a) for a in s["available_at"][:n]],
            "period_seconds": s["period_seconds"]}


# --- eligibility from the verified matrix's cells ----------------------------------------------------

def eligibility_record(cells_path: Path) -> dict:
    """The per-cell verdicts a verifier sealed (`MATRIX.verified.*.cells.json`)."""
    doc = json.loads(Path(cells_path).read_text(encoding="utf-8"))
    if doc.get("schema") != "d3_mechanics_cells.v1" or doc.get("verified") is not True:
        raise ValueError("eligibility must come from a VERIFIED matrix's cells record")
    return {"freeze_sha256": doc["freeze_sha256"], "design_sha256": doc["design_sha256"],
            "cells": {(c["unit"], c["variable"], c["operator"]): c for c in doc["cells"]}}


def eligible(record: dict, *, unit: str, variable: str, operator) -> tuple:
    cell = record["cells"].get((unit, variable, operator.KIND))
    if cell is None:
        return False, f"({unit}, {variable}, {operator.KIND}) is not in the verified record"
    contract = _load("df_d3_contract")
    spec_sha = contract.spec_sha256(operator.describe())
    if cell.get("spec_sha256") != spec_sha:
        return False, "the operator's declaration is not the one the record was sealed on"
    if cell["verdict"] != "MECHANICALLY_ACCEPTED":
        return False, f"verdict {cell['verdict']} for that cell"
    return True, cell


# --- representations by the real operator, per block -------------------------------------------------

def represent(operator, s: dict, train_end: int) -> dict:
    """fit on the training prefix only; transform the series with a fresh state; contract-
    validated output (values, available, emitted_at)."""
    contract = _load("df_d3_contract")
    x = _as_operator_input(s)
    train = _as_operator_input(s, upto=max(2, train_end))
    state = operator.fit(train)
    out = contract.validate_output(operator.transform(x, state), spec=operator.describe(), x=x)
    return {"values": np.asarray(out["values"], dtype=float),
            "available": np.asarray(out["available"], dtype=bool),
            "emitted_at": np.asarray(out["emitted_at"], dtype=float), "state": state}


def prefix_consistent(operator, s: dict, state, rep: dict, rows, decision) -> tuple:
    """At sampled decision rows t, transforming the series CUT at t must give the same
    outputs (value, availability, emission) for every output emitted by decision(t)."""
    contract = _load("df_d3_contract")
    for t in rows:
        cut = _as_operator_input(s, upto=int(t) + 1)
        out = contract.validate_output(operator.transform(cut, state), spec=operator.describe(),
                                       x=cut)
        for i in range(int(t) + 1):
            if rep["available"][i] and rep["emitted_at"][i] <= decision[t]:
                same = (bool(out["available"][i]) == bool(rep["available"][i])
                        and float(out["emitted_at"][i]) == float(rep["emitted_at"][i])
                        and (np.isnan(out["values"][i]) and np.isnan(rep["values"][i])
                             or out["values"][i] == rep["values"][i]))
                if not same:
                    return False, {"decision_row": int(t), "output": i}
    return True, None


# --- labels and features by identity ------------------------------------------------------------------

def label(s: dict, protocol: Protocol) -> np.ndarray:
    x = s["target_values"] if "target_values" in s else s["values"]
    n = x.size
    h = protocol.horizon
    out = np.full(n, np.nan)
    if h >= n:
        return out
    future = x[h:] - x[:-h]
    if protocol.target == "direction":
        out[:n - h] = np.where(np.isnan(future), np.nan, (future > 0).astype(float))
    else:
        out[:n - h] = future
    return out


def _lags_by_emission(values, available, emitted_at, decision, window) -> tuple:
    """Row t takes the `window` most recent outputs i <= t that are available and were
    emitted at or before decision[t]. Fewer than `window` such outputs: not emittable."""
    n = values.size
    X = np.full((n, window), np.nan)
    ok = np.zeros(n, dtype=bool)
    first = np.full(n, -1, dtype=int)                  # the earliest row a feature row consumes
    for t in range(n):
        taken = 0
        i = t
        while i >= 0 and taken < window:
            if available[i] and emitted_at[i] <= decision[t]:
                X[t, taken] = values[i]
                taken += 1
                first[t] = i
            elif available[i] and emitted_at[i] > decision[t] and i == t:
                pass                                   # the newest output is not out yet
            i -= 1
        ok[t] = taken == window
    return X, ok, first


def branch_width(branch: str, window: int) -> int:
    """Feature columns a branch consumes: raw and transformed `window`, augmented and its
    capacity control raw_wide 2·window (equal total width by construction)."""
    if branch in ("raw", "transformed"):
        return int(window)
    if branch in ("augmented", "raw_wide"):
        return 2 * int(window)
    raise ValueError(f"unknown branch {branch!r}")


def features(branch: str, s: dict, rep: dict | None, protocol: Protocol) -> tuple:
    """(X, emittable, support_start): the feature matrix, the rows that have every lag, and for
    each row the earliest row it consumed (gaps, delays and emission alignment included)."""
    decision = s["available_at"]                       # the decision for row t happens when
    raw_avail = ~np.isnan(s["values"])                 # observation t is itself available
    if branch in ("raw", "raw_wide"):
        # raw_wide is the capacity control for `augmented`: raw lags of the SAME total width
        # (2·window), so raw+R is compared against raw of equal dimensionality, never raw alone
        return _lags_by_emission(np.nan_to_num(s["values"]), raw_avail, s["available_at"],
                                 decision, branch_width(branch, protocol.window))
    if rep is None:
        raise ValueError("a transformed branch needs a representation")
    Xr, okr, fr = _lags_by_emission(np.nan_to_num(rep["values"]), rep["available"],
                                    rep["emitted_at"], decision, protocol.window)
    if branch == "transformed":
        return Xr, okr, fr
    if branch == "augmented":
        Xa, oka, fa = _lags_by_emission(np.nan_to_num(s["values"]), raw_avail, s["available_at"],
                                        decision, protocol.window)
        return np.hstack([Xa, Xr]), oka & okr, np.minimum(fa, fr)
    raise ValueError(f"unknown branch {branch!r}")


# --- the probe models, fitted inside the training block ---------------------------------------------

def _standardise(Xtr, Xva):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd


def fit_predict(Xtr, ytr, Xva, protocol: Protocol) -> np.ndarray:
    Xtr, Xva = _standardise(Xtr, Xva)
    Xtr1 = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    Xva1 = np.hstack([Xva, np.ones((Xva.shape[0], 1))])
    if protocol.model == "ridge":
        lam = protocol.ridge_lambda * np.eye(Xtr1.shape[1])
        lam[-1, -1] = 0.0
        beta = np.linalg.solve(Xtr1.T @ Xtr1 + lam, Xtr1.T @ ytr)
        return Xva1 @ beta
    beta = np.zeros(Xtr1.shape[1])
    for _ in range(protocol.logistic_steps):
        p = 1.0 / (1.0 + np.exp(-(Xtr1 @ beta)))
        grad = Xtr1.T @ (p - ytr) / Xtr1.shape[0] \
            + protocol.ridge_lambda * np.r_[beta[:-1], 0.0] / Xtr1.shape[0]
        beta -= 0.1 * grad
    return 1.0 / (1.0 + np.exp(-(Xva1 @ beta)))


def loss(pred, y, protocol: Protocol) -> float:
    if protocol.target == "direction":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(np.abs(pred - y)))


# --- blocks -------------------------------------------------------------------------------------------

def blocks(rows: np.ndarray, protocol: Protocol, support_start: np.ndarray, train_reach: int) -> list:
    """Walk-forward: validation slices over the second half of the emittable rows; training
    rows are those whose consumed rows (label to t + horizon, representation to t + reach)
    end strictly BEFORE the earliest row the validation slice consumes (`support_start`, per
    row, from the features actually built: widths, gaps, delays, emission). The boundary is
    recorded by row identity, never as a number alone. Every block must reach the minimum or
    the contrast is INSUFFICIENT_ROWS (blocks_policy all_or_insufficient)."""
    idx = np.asarray(rows)
    support = np.asarray(support_start)
    if idx.size < protocol.n_blocks * protocol.min_rows_per_block * 2:
        return []
    cuts = np.linspace(idx.size // 2, idx.size, protocol.n_blocks + 1).astype(int)
    out = []
    for k in range(protocol.n_blocks):
        va = idx[cuts[k]:cuts[k + 1]]
        if va.size == 0:
            return []
        earliest = int(support[va].min())
        tr = idx[idx + int(train_reach) < earliest]
        out.append({"block": k, "train": tr, "validation": va,
                    "purge": int(va[0] - tr.max()) if tr.size else None,
                    "boundary": {"min_validation_support_row": earliest,
                                 "max_train_consumed_row": int(tr.max() + train_reach) if tr.size else None}})
    return out


def boundary_violations(scheme: list) -> list:
    """Blocks whose training consumption reaches the validation support (by row identity)."""
    return [b["block"] for b in scheme
            if b["boundary"]["max_train_consumed_row"] is None
            or b["boundary"]["max_train_consumed_row"] >= b["boundary"]["min_validation_support_row"]]


# --- inference ----------------------------------------------------------------------------------------

def t_interval_lower(deltas: np.ndarray, alpha: float) -> tuple:
    from scipy.stats import t as student_t
    d = np.asarray(deltas, dtype=float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(d.size))
    t_crit = float(student_t.ppf(1 - alpha / 2, d.size - 1))
    return mean, se, t_crit, mean - t_crit * se


# --- one contrast ---------------------------------------------------------------------------------------

def contrast(s: dict, operator, protocol: Protocol, *, contrast_id: str, eligibility: dict,
             unit: str, variable: str, branch_a: str = "raw", branch_b: str = "transformed",
             train_fraction: float = 0.5) -> dict:
    """Loss(branch_a) − loss(branch_b), paired per block. Refuses before any scoring when the
    contrast is not in the sealed family, a branch is not declared, or the representation is
    not eligible for this exact cell; refuses as non-causal when the prefix check fails."""
    t0 = time.process_time()
    w0 = time.monotonic()
    if contrast_id not in protocol.family:
        return {"outcome": REFUSED, "why": f"contrast {contrast_id!r} is not in the sealed family"}
    for b in (branch_a, branch_b):
        if b not in protocol.branches:
            return {"outcome": REFUSED, "why": f"branch {b!r} is not declared by the protocol"}
    needs_rep = branch_b not in ("raw", "raw_wide")
    rep_meta = None
    if needs_rep:
        if operator is None:
            return {"outcome": REFUSED, "why": "a transformed branch needs an operator"}
        ok, why = eligible(eligibility, unit=unit, variable=variable, operator=operator)
        if not ok:
            return {"outcome": REFUSED, "why": f"not eligible: {why}"}
        rep_meta = {"operator": operator.KIND, "spec_sha256": why.get("spec_sha256"),
                    "reach_right": int(operator.reach_right(s["values"].size))}
    y = label(s, protocol)
    n = s["values"].size
    coverage = {"n": int(n), "inputs_missing": int(np.isnan(s["values"]).sum())}
    # a first pass with a train-only fit at the split decides the emittable rows and blocks
    rep0 = represent(operator, s, int(n * train_fraction)) if needs_rep else None
    Xa, oka, fa = features(branch_a, s, rep0, protocol)
    Xb, okb, fb = features(branch_b, s, rep0, protocol)
    emittable = oka & okb & ~np.isnan(y)
    coverage.update(rows_a=int(oka.sum()), rows_b=int(okb.sum()), rows_paired=int(emittable.sum()))
    reach = rep_meta["reach_right"] if rep_meta else 0
    train_reach = max(protocol.horizon, reach)         # label to t+h, representation to t+reach
    support = {"branch_a": branch_width(branch_a, protocol.window),
               "branch_b": branch_width(branch_b, protocol.window), "operator_reach": int(reach)}
    rows = np.flatnonzero(emittable)
    scheme = blocks(rows, protocol, np.minimum(fa, fb), train_reach)
    purge = min((b["purge"] for b in scheme if b["purge"] is not None), default=None)
    if not scheme or any(b["train"].size < protocol.min_rows_per_block
                         or b["validation"].size < protocol.min_rows_per_block for b in scheme):
        return {"outcome": INSUFFICIENT_ROWS, "coverage": coverage, "purge": purge, "support": support,
                "blocks": len(scheme), "policy": protocol.blocks_policy}
    rng = np.random.default_rng(protocol.seed)
    deltas, per_block = [], []
    for b in scheme:
        tr, va = b["train"], b["validation"]
        train_end = int(tr.max()) + 1
        if needs_rep:
            rep = represent(operator, s, train_end)           # fit on THIS block's train only
            sample = rng.choice(va, size=min(protocol.prefix_checks, va.size), replace=False)
            consistent, where = prefix_consistent(operator, s, rep["state"], rep, sample,
                                                  s["available_at"])
            if not consistent:
                return {"outcome": REFUSED, "why": "representation is not causal: its prefix "
                                                   "outputs changed when the series was cut",
                        "where": where, "block": b["block"]}
        else:
            rep = None
        Xa, oka, fa = features(branch_a, s, rep, protocol)
        Xb, okb, fb = features(branch_b, s, rep, protocol)
        both = oka & okb & ~np.isnan(y)
        va_k = va[both[va]]
        # the boundary is re-derived from THIS block's features (the representation was refit)
        earliest = int(np.minimum(fa, fb)[va_k].min()) if va_k.size else b["boundary"]["min_validation_support_row"]
        tr_k = tr[both[tr]]
        tr_k = tr_k[tr_k + train_reach < earliest]
        boundary = {"min_validation_support_row": earliest,
                    "max_train_consumed_row": int(tr_k.max() + train_reach) if tr_k.size else None}
        if tr_k.size < protocol.min_rows_per_block or va_k.size < protocol.min_rows_per_block:
            return {"outcome": INSUFFICIENT_ROWS, "coverage": coverage, "purge": purge, "support": support,
                    "block": b["block"], "policy": protocol.blocks_policy}
        if boundary["max_train_consumed_row"] >= boundary["min_validation_support_row"]:
            return {"outcome": REFUSED, "why": "training consumption reaches the validation support",
                    "block": b["block"], "boundary": boundary}
        la = loss(fit_predict(Xa[tr_k], y[tr_k], Xa[va_k], protocol), y[va_k], protocol)
        lb = loss(fit_predict(Xb[tr_k], y[tr_k], Xb[va_k], protocol), y[va_k], protocol)
        deltas.append(la - lb)
        per_block.append({"block": b["block"], "loss_a": la, "loss_b": lb, "delta": la - lb,
                          "train_rows": int(tr_k.size), "validation_rows": int(va_k.size),
                          "boundary": boundary, "purge": int(va_k[0] - tr_k.max()),
                          "train_ids": [int(s["ids"][tr_k[0]]), int(s["ids"][tr_k[-1]])],
                          "validation_ids": [int(s["ids"][va_k[0]]), int(s["ids"][va_k[-1]])]})
    mean, se, t_crit, lower = t_interval_lower(np.asarray(deltas), protocol.alpha_adjusted)
    cost = {"cpu_seconds": round(time.process_time() - t0, 3),
            "wall_seconds": round(time.monotonic() - w0, 3)}
    base = {"schema": CONTRAST_SCHEMA, "contrast_id": contrast_id, "branch_a": branch_a,
            "branch_b": branch_b,
            "representation": rep_meta,
            "loss_name": "log_loss" if protocol.target == "direction" else "mae",
            "delta_mean": mean, "delta_se": se, "delta_lower": lower, "t_crit": t_crit,
            "alpha_adjusted": protocol.alpha_adjusted, "margin": protocol.margin,
            "blocks": per_block, "blocks_used": len(deltas), "purge": purge, "support": support,
            "rows_policy": ROWS_POLICY, "coverage": coverage,
            "cost": cost, "protocol_sha256": protocol.sealed()["protocol_sha256"],
            "note": "a difference of predictive losses of the probe model; not information, "
                    "not mutual information, not trading utility"}
    supported, why = calibration_supports(protocol, operator, n, branch_a=branch_a, branch_b=branch_b) \
        if operator is not None else (False, "no operator to calibrate against")
    if not supported:
        return {**base, "outcome": INCONCLUSIVE_UNCALIBRATED,
                "why": f"{why}; the delta is descriptive, not a decision"}
    return {**base, "outcome": ADVANCES if lower > protocol.margin else DOES_NOT_ADVANCE}


# --- calibration: the false-advance rate under a dependent null --------------------------------------

def _ar1_null(n: int, rng, phi: float = 0.6) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0, 1.0, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return np.cumsum(x)


GENERATORS = {
    "white_null": {"null": True, "what": "independent N(0,1) increments, cumulated: the "
                                          "exchangeable null (dependent levels, no effect)"},
    "ar1_features_independent_target": {
        "null": True, "what": "features from AR(1) (phi 0.6) increments cumulated — dependent, "
                              "structured — with the label taken from an INDEPENDENT white "
                              "series: dependence with no effect"},
    "ar1_null": {"null": False, "what": "AR(1) increments cumulated, label from the same series: "
                                        "structured, a POSITIVE-CONTROL-like diagnostic, not a "
                                        "null of no effect"},
}


def _white_null(n: int, rng, phi: float = 0.0) -> np.ndarray:
    return np.cumsum(rng.normal(0, 1.0, n))


def _make_series(generator: str, n: int, rng) -> dict:
    if generator == "white_null":
        return series(_white_null(n, rng))
    if generator == "ar1_null":
        return series(_ar1_null(n, rng))
    if generator == "ar1_features_independent_target":
        return series(_ar1_null(n, rng), target_values=np.cumsum(rng.normal(0, 1.0, n)))
    raise ValueError(f"unknown generator {generator!r}")


def clopper_pearson_upper(k: int, n: int, confidence: float) -> float:
    from scipy.stats import beta
    if n <= 0:
        return float("nan")
    return 1.0 if k >= n else float(beta.ppf(confidence, k + 1, n - k))


def sims_required_for_zero(alpha: float, confidence: float) -> int:
    """The smallest n such that 0 advances in n gives an upper bound <= alpha."""
    import math
    return int(math.ceil(math.log(1 - confidence) / math.log(1 - alpha)))


def calibrate(protocol: Protocol, operator, *, plan: dict | None = None, seed: int | None = None,
              n_sims: int | None = None, n: int | None = None, generator: str | None = None,
              bound_confidence: float | None = None, branch_a: str = "raw",
              branch_b: str = "transformed") -> dict:
    """The false-advance rate of THIS protocol with THIS operator at THIS length under a null of
    no effect, from the sealed plan (`protocol.calibration_plan`) or explicit arguments for
    diagnostics. Every simulation is kept (index, seed, outcome, delta); the rate is
    advances / scored with failed simulations counted apart; the Clopper–Pearson upper bound at
    the plan's confidence is what a decision is gated on, not the point estimate. A structured
    generator (`ar1_null`) is a diagnostic and is refused as a calibration null."""
    plan = plan or protocol.calibration_plan or {}
    generator = generator or plan.get("generator")
    n_sims = n_sims if n_sims is not None else plan.get("n_sims")
    n = n if n is not None else plan.get("n")
    bound_confidence = bound_confidence if bound_confidence is not None else plan.get("bound_confidence", 0.95)
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}")
    if not isinstance(n_sims, int) or n_sims < 1 or not isinstance(n, int) or n < 1:
        raise ValueError("n_sims and n must be positive integers, declared before running")
    seed = seed if seed is not None else protocol.seed + 1000
    rng = np.random.default_rng(seed)
    contract = _load("df_d3_contract")
    spec_sha = contract.spec_sha256(operator.describe())
    fake = {"freeze_sha256": "calibration", "design_sha256": "calibration",
            "cells": {("cal", "v0", operator.KIND): {"verdict": "MECHANICALLY_ACCEPTED",
                                                    "spec_sha256": spec_sha}}}
    # the protocol under calibration: no record, no gate — outcomes are read from the interval
    proto = Protocol(**{**protocol.__dict__, "calibration": None})
    t0 = time.process_time()
    per_sim, advances, scored, failed = [], 0, 0, 0
    for k in range(n_sims):
        sim_seed = int(rng.integers(0, 2 ** 31 - 1))
        s = _make_series(generator, n, np.random.default_rng(sim_seed))
        out = contrast(s, operator, proto, contrast_id=proto.family[0], eligibility=fake,
                       unit="cal", variable="v0", branch_a=branch_a, branch_b=branch_b)
        if "delta_lower" in out:
            scored += 1
            advanced = out["delta_lower"] > proto.margin
            advances += int(advanced)
            per_sim.append({"index": k, "seed": sim_seed, "outcome": "ADVANCES" if advanced
                            else "DOES_NOT_ADVANCE", "delta_mean": out["delta_mean"],
                            "delta_lower": out["delta_lower"]})
        else:
            failed += 1
            per_sim.append({"index": k, "seed": sim_seed, "outcome": out["outcome"],
                            "why": out.get("why")})
    rate = advances / scored if scored else float("nan")
    this_harness = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key = computation_key(protocol, operator, {"generator": generator, "n_sims": n_sims, "n": n,
                                               "bound_confidence": bound_confidence},
                          branch_a=branch_a, branch_b=branch_b, seed=seed, harness=this_harness)
    return {"schema": "df_utility_calibration.v4", "generator": generator,
            "null": bool(GENERATORS[generator]["null"]),
            "computation": key, "computation_sha256": computation_sha256(key),
            "numeric_dependencies": key["numeric_dependencies"],
            "branch_a": branch_a, "branch_b": branch_b,
            "widths": {"a": branch_width(branch_a, protocol.window), "b": branch_width(branch_b, protocol.window)},
            "rows_policy": ROWS_POLICY,
            "plan": {"generator": generator, "n_sims": n_sims, "n": n,
                     "bound_confidence": bound_confidence},
            "n_sims": n_sims, "scored": scored, "failed": failed, "advances": advances,
            "false_advance_rate": rate,
            "upper_bound": clopper_pearson_upper(advances, scored, bound_confidence) if scored else float("nan"),
            "bound_confidence": bound_confidence, "alpha_adjusted": protocol.alpha_adjusted,
            "seed": seed, "n": n,
            "operator": {"kind": operator.KIND, "spec_sha256": spec_sha,
                         "params": dict(operator.params)},
            "protocol_base_sha256": protocol.base_sha256(), "family": list(protocol.family),
            "margin": protocol.margin, "n_blocks": protocol.n_blocks, "window": protocol.window,
            "target": protocol.target, "model": protocol.model,
            "per_sim": per_sim, "per_sim_sha256": sha_obj(per_sim),
            "harness_sha256": this_harness,
            "cost": {"cpu_seconds": round(time.process_time() - t0, 3)}}


def harness_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def record_pair(rec: dict) -> tuple:
    return (rec.get("branch_a", "raw"), rec.get("branch_b", "transformed"))


def calibration_supports(protocol: Protocol, operator, n: int, *, record: dict | None = None,
                         harness_sha256: str | None = None, branch_a: str = "raw",
                         branch_b: str = "transformed") -> tuple:
    """Does the sealed record support a decision for THIS contrast? Scope and identity are
    checked (operator declaration, protocol base, family, length, multiplicity, the sealed
    plan, margin, blocks, window, target/model and the harness code that applied), then the
    DERIVED decision bound — recounted from the simulations under FAILURE_POLICY, never the
    bound the record states — against alpha_adjusted (O1). `record` overrides the protocol's
    own record (a verifier re-checking a stored one); `harness_sha256` names the code that
    applied to the run being verified (default: this file)."""
    rec = record if record is not None else protocol.calibration
    if rec is None:
        return False, "no calibration record"
    problems = calibration_record_problems(rec)
    if problems:
        return False, "calibration record unusable: " + "; ".join(problems)
    contract = _load("df_d3_contract")
    spec_sha = contract.spec_sha256(operator.describe())
    if rec["operator"]["kind"] != operator.KIND or rec["operator"]["spec_sha256"] != spec_sha \
            or rec["operator"]["params"] != dict(operator.params):
        return False, "calibrated for another operator declaration"
    if record_pair(rec) != (branch_a, branch_b):
        return False, (f"calibrated for another branch pair {record_pair(rec)}, this contrast is "
                       f"({branch_a}, {branch_b})")
    if "computation" in rec:
        # v3/v4: the scientific computation must be THIS contrast's; family/unit labels do not
        # count. A legacy v3 record (no code identity) keeps its historical status: it is compared
        # on its own fields and never acquires the code-bound guarantee.
        expected = computation_key(protocol, operator, rec["plan"], branch_a=branch_a, branch_b=branch_b,
                                   seed=rec["seed"], harness=harness_sha256 or globals()["harness_sha256"]())
        expected["numeric_dependencies"] = rec["computation"].get("numeric_dependencies")
        expected = _as_legacy_key(expected, rec["computation"])
        if computation_sha256(expected) != rec["computation_sha256"]:
            differing = sorted(k for k in set(expected) | set(rec["computation"]) if expected.get(k) != rec["computation"].get(k))
            return False, f"calibrated for another computation ({', '.join(differing)})"
    else:
        if rec["protocol_base_sha256"] != protocol.base_sha256():
            return False, "calibrated for another protocol"
        if list(rec["family"]) != list(protocol.family):
            return False, "calibrated for another contrast family"
    if rec["n"] != int(n):
        return False, f"calibrated at length {rec['n']}, this series has {n}"
    if rec["alpha_adjusted"] != protocol.alpha_adjusted:
        return False, "calibrated for another multiplicity"
    if protocol.calibration_plan is None or dict(rec["plan"]) != dict(protocol.calibration_plan):
        return False, "calibrated under another plan (generator, simulations, length or confidence)"
    for name in ("margin", "n_blocks", "window", "target", "model"):
        if rec[name] != getattr(protocol, name):
            return False, f"calibrated for another {name}"
    expected_code = harness_sha256 or globals()["harness_sha256"]()
    if rec["harness_sha256"] != expected_code:
        return False, "calibrated under another harness code"
    derived = derive_calibration(rec)
    if derived["problems"]:
        return False, "calibration record unusable: " + "; ".join(derived["problems"])
    if derived["decision_bound"] > protocol.alpha_adjusted:
        why = (f"derived upper bound {derived['decision_bound']:.4f} exceeds alpha_adjusted "
               f"{protocol.alpha_adjusted:.4f} ({derived['advances']}/{derived['scored']}"
               f"{' + ' + str(derived['failed']) + ' failed counted as advances (worst case)' if derived['failed'] else ''}"
               f" at {rec['bound_confidence']:.0%})")
        return False, why
    return True, None


# --- observed budgets: one contrast in an isolated child -----------------------------------------------

def verified_score(attempt_dir: Path, result: dict, verified: dict, job: dict, *,
                   allow_legacy_schema: bool = False) -> tuple:
    """The scientific result is the file the child named and the runner re-hashed — never the
    process summary (N1). Missing, altered, discordant or non-finite: a typed refusal, no
    fabricated zero."""
    name = (result or {}).get("output_file")
    if not name:
        return None, {"outcome": SCORE_UNVERIFIED, "why": "the child named no output file"}
    path = Path(attempt_dir) / name
    if not path.is_file():
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"{name} is absent"}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (verified or {}).get("output_sha256"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "the output's bytes are not the ones "
                                                           "the child declared and the runner verified",
                      "declared": result.get("output_sha256"),
                      "verified": (verified or {}).get("output_sha256"), "found": digest}
    try:
        score = json.loads(body)
    except ValueError:
        return None, {"outcome": SCORE_UNVERIFIED, "why": "the output is not JSON"}
    legacy = allow_legacy_schema and "schema" not in score and "delta_mean" in score
    preparatory = score.get("schema") in ("df_utility_calibration.v1", "df_utility_calibration.v2",
                                          "df_utility_calibration.v3", "df_utility_calibration.v4",
                                          "d3_mechanics_cells.v1")
    if not legacy and not preparatory and score.get("schema") != CONTRAST_SCHEMA \
            and score.get("outcome") not in (REFUSED, INSUFFICIENT_ROWS):
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"schema {score.get('schema')!r}"}
    if preparatory:
        if str(score.get("schema", "")).startswith("df_utility_calibration.v"):
            problems = calibration_record_problems(score)
            if job.get("branch_a") or job.get("branch_b"):
                if record_pair(score) != (job.get("branch_a", "raw"), job.get("branch_b", "transformed")):
                    return None, {"outcome": SCORE_UNVERIFIED, "why": "calibration is for another branch pair"}
            if problems:
                return None, {"outcome": SCORE_UNVERIFIED, "why": "calibration record unusable "
                              "(bound or counts not derived from its simulations): " + "; ".join(problems)}
            if job.get("operator") and score["operator"]["kind"] != job["operator"]:
                return None, {"outcome": SCORE_UNVERIFIED, "why": "calibration is for another operator"}
            base = (job.get("protocol") or {})
            if base and "protocol_sha256" in base:
                try:
                    proto_job = Protocol(**{k: (tuple(v) if isinstance(v, list) else v)
                                            for k, v in base.items()
                                            if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")})
                except (ProtocolRefusal, TypeError):
                    proto_job = None
                if proto_job is not None and "computation" in score:
                    # v3: the job's computation (labels apart) must be the record's; the harness
                    # that applies is judged at support time, not here
                    ops = _load("df_d3_operators")
                    expected = computation_key(proto_job, ops.build(job["operator"]), job.get("plan") or score["plan"],
                                               branch_a=job.get("branch_a", "raw"), branch_b=job.get("branch_b", "transformed"),
                                               seed=job.get("seed", score["seed"]), harness=score["computation"].get("harness_sha256"))
                    expected["numeric_dependencies"] = score["computation"].get("numeric_dependencies")
                    expected = _as_legacy_key(expected, score["computation"])
                    if computation_sha256(expected) != score.get("computation_sha256"):
                        return None, {"outcome": SCORE_UNVERIFIED, "why": "calibration is for another computation"}
                elif proto_job is not None and score["protocol_base_sha256"] != proto_job.base_sha256():
                    return None, {"outcome": SCORE_UNVERIFIED, "why": "calibration is for another protocol"}
            if job.get("plan") and dict(score["plan"]) != dict(job["plan"]):
                return None, {"outcome": SCORE_UNVERIFIED, "why": "calibration is for another plan"}
        return score, None
    if score.get("contrast_id", job.get("contrast_id")) != job.get("contrast_id"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "contrast identity differs"}
    expected_proto = (job.get("protocol") or {}).get("protocol_sha256")
    if "protocol_sha256" in score and score["protocol_sha256"] != expected_proto:
        return None, {"outcome": SCORE_UNVERIFIED, "why": "protocol identity differs"}
    for key in ("delta_mean", "delta_se", "delta_lower"):
        if key in score and not (isinstance(score[key], (int, float))
                                 and np.isfinite(score[key])):
            return None, {"outcome": SCORE_UNVERIFIED, "why": f"{key} is not finite"}
    if score.get("outcome") != (result or {}).get("outcome"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "the summary's outcome is not the file's"}
    return score, None


def _job_binding_refusal(attempt_dir: Path, job: dict):
    """A resumed attempt is the attempt of THE job it recorded (O2): the job the caller brings
    must equal the one written before the child started, or nothing is reused."""
    path = Path(attempt_dir) / "job.json"
    if not path.is_file():
        return {"outcome": SCORE_UNVERIFIED, "why": "the attempt records no job to bind to"}
    try:
        recorded = json.loads(path.read_text())
    except ValueError:
        return {"outcome": SCORE_UNVERIFIED, "why": "the attempt's recorded job is not JSON"}
    recorded.pop("attempt_dir", None)
    current = json.loads(json.dumps({k: v for k, v in job.items() if k != "attempt_dir"}, default=_jsonable))
    if recorded != current:
        differing = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
        return {"outcome": SCORE_UNVERIFIED, "why": f"the job differs from the recorded attempt's job "
                                                     f"({', '.join(differing)}); nothing is reused"}
    return None


def run_isolated(job: dict, *, attempt_dir: Path, assigned_bytes: int, wall_seconds: float,
                 cpu_seconds: float, before_run=None) -> dict:
    """The contrast in a child process under df_isolated_runner: ceilings enforced during the
    work, cost measured, RESOURCE_EXCEEDED with no partial score when a ceiling is hit. The
    score is the verified output file (N1). `before_run`, when given, is called just before
    the child starts (governance's before_run) and may refuse by raising."""
    IR = _load("df_isolated_runner")
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prior = attempt_dir / "outcome.json"
    if prior.is_file():
        # a completed attempt is never re-run: the recorded outcome is re-verified and returned.
        # O2: one truthful outcome — when the evidence no longer verifies, or the attempt was
        # made for another job, the outcome IS SCORE_UNVERIFIED; the old summary is history.
        recorded = json.loads(prior.read_text())
        history = dict(recorded.get("summary") or {})
        refusal = _job_binding_refusal(attempt_dir, job)
        score = None
        if refusal is None and recorded.get("status") == "COMPLETED":
            result = json.loads((attempt_dir / "result.json").read_text()) \
                if (attempt_dir / "result.json").is_file() else None
            score, refusal = verified_score(attempt_dir, result, recorded.get("verified"), job)
        if refusal is not None:
            return {"outcome": SCORE_UNVERIFIED, "reason": refusal["why"], "cost": history.get("cost", {}),
                    "score": None, "output_sha256": None, "resumed": True, "refusal": refusal,
                    "history": history}
        return {**history, "score": score, "resumed": True}
    if before_run is not None:
        before_run(job)
    job_file = attempt_dir / "job.json"
    job_file.write_text(json.dumps({**job, "attempt_dir": str(attempt_dir)}, default=_jsonable))
    task = IR.Task(argv=[sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                         str(job_file)], name=f"utility-{job.get('contrast_id', 'c')}",
                   attempt_dir=attempt_dir, assigned_bytes=assigned_bytes,
                   wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                   mechanism=IR.detect_mechanism())
    task.start()
    task.wait()
    status, reason, verified = IR.classify(task.outcome, attempt_dir)
    result = None
    if (attempt_dir / "result.json").is_file():
        result = json.loads((attempt_dir / "result.json").read_text())
    cost = {"cpu_seconds": task.outcome.get("cpu_seconds"),
            "wall_seconds": task.outcome.get("wall_seconds"),
            "peak_rss_bytes": task.outcome.get("child_maxrss_bytes"),
            "cgroup_memory_peak": task.outcome.get("cgroup_memory_peak"),
            "started_at": task.outcome.get("started_at"), "ended_at": task.outcome.get("ended_at")}
    if status != "COMPLETED":
        summary = {"outcome": RESOURCE_EXCEEDED if status == "RESOURCE_EXCEEDED" else "UNCERTAIN",
                   "reason": reason, "cost": cost, "score": None, "output_sha256": None}
    else:
        score, refusal = verified_score(attempt_dir, result, verified, job)
        summary = {"outcome": (score.get("outcome", "COMPLETED") if score else SCORE_UNVERIFIED),
                   "reason": reason,
                   "cost": cost, "score": score, "output_sha256": verified.get("output_sha256"),
                   **({"refusal": refusal} if refusal else {})}
    prior.write_text(json.dumps({"status": status, "verified": verified,
                                 "summary": {k: v for k, v in summary.items() if k != "score"}},
                                default=_jsonable))
    return summary


def _finish(adir: Path, name: str, doc: dict, outcome, extra: dict | None = None, body: bytes | None = None) -> int:
    body = body if body is not None else json.dumps(doc, sort_keys=True, default=_jsonable).encode()
    (adir / name).write_bytes(body)
    result = {"status": "COMPLETED", "reason": "", "output_file": name,
              "output_sha256": hashlib.sha256(body).hexdigest(), "rows_written": 1,
              "outcome": outcome if outcome is not None else "COMPLETED", **(extra or {})}
    tmp = adir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, adir / "result.json")
    return 0


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o).__name__)


def worker_main(job_file: Path) -> int:
    """Child protocol: result.json at the end, nothing partial."""
    job = json.loads(Path(job_file).read_text())
    adir = Path(job["attempt_dir"])
    if job.get("slow_seconds"):                         # the controlled slow model
        end = time.monotonic() + float(job["slow_seconds"])
        acc = 0.0
        while time.monotonic() < end:
            acc += float(np.sum(np.random.default_rng(1).normal(size=20000) ** 2))
    ops = _load("df_d3_operators")
    proto = Protocol(**{k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in job["protocol"].items()
                        if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")})
    operator = ops.build(job["operator"]) if job.get("operator") else None
    kind = job.get("kind", "contrast")
    if kind == "calibrate":
        # preparatory work under the same ceilings: the record is the child's output. With a
        # cache directory (Q2), a verified shared record of the SAME computation is reused byte
        # for byte and the source is declared; otherwise the simulations run here and the record
        # is offered to the cache.
        pair = (job.get("branch_a", "raw"), job.get("branch_b", "transformed"))
        seed = job.get("seed") if job.get("seed") is not None else proto.seed + 1000
        source = {"kind": "MEASURED_HERE"}
        cache = None
        if job.get("cache_dir"):
            CC = _load("df_utility_calibration_cache")
            cache = CC.CalibrationCache(Path(job["cache_dir"]))
            key = computation_key(proto, operator, job["plan"], branch_a=pair[0], branch_b=pair[1], seed=seed)
            t0 = time.process_time()
            body, why = cache.lookup(key)
            if body is not None:
                cache.note_consumer(computation_sha256(key), consumer={"attempt_dir": str(adir), "contrast_id": job.get("contrast_id"),
                                                                     "protocol_key": job.get("protocol_key")},
                                    kind="HIT", verification_seconds=round(time.process_time() - t0, 3))
                (adir / "calibration.json").write_bytes(body)
                record = json.loads(body)
                source = {"kind": "CACHE_HIT", "computation_sha256": computation_sha256(key),
                          "producer": why, "verification_seconds": round(time.process_time() - t0, 3),
                          "simulations_run_here": 0}
                return _finish(adir, "calibration.json", record, record.get("upper_bound"), extra={"calibration_source": source},
                               body=body)
            source = {"kind": "MISS_PRODUCED", "computation_sha256": computation_sha256(key), "why_miss": why}
        record = calibrate(proto, operator, plan=job["plan"], seed=seed, branch_a=pair[0], branch_b=pair[1])
        body = json.dumps(record, sort_keys=True, default=_jsonable).encode()
        if cache is not None:
            stored = cache.store(record["computation_sha256"], body,
                                 producer={"attempt_dir": str(adir), "contrast_id": job.get("contrast_id"),
                                           "protocol_key": job.get("protocol_key"), "cost_cpu_seconds": record["cost"]["cpu_seconds"]})
            source["stored"] = stored
        return _finish(adir, "calibration.json", record, record.get("upper_bound"), extra={"calibration_source": source}, body=body)
    if kind == "mechanics":
        battery = _load("df_d3_acceptance")
        contract = _load("df_d3_contract")
        s = series(job["series"]["values"])
        xin = _as_operator_input(s)
        train = battery.prefix(xin, max(2, s["values"].size // 2))
        cells = []
        for k in job["operators"]:
            op = ops.build(k)
            report = battery.run_battery(op, xin, train=train, twin=ops.twin_of(op),
                                         resource_contract=job.get("resource_contract"))
            cells.append({"unit": job["unit"], "variable": job["variable"], "operator": k,
                          "verdict": report["verdict"],
                          "spec_sha256": contract.spec_sha256(op.describe()),
                          "failed": report["failed"], "undecided": report["undecided"]})
        record = {"schema": "d3_mechanics_cells.v1", "run_id": job.get("run_id", "rehearsal"),
                  "verified": True, "freeze_sha256": job.get("freeze_sha256", "rehearsal-mechanics"),
                  "design_sha256": _load("df_d3_design").D3_DESIGN_CURRENT["design_sha256"],
                  "cells": cells}
        return _finish(adir, "cells.json", record, None)
    s = series(job["series"]["values"], ids=job["series"].get("ids"),
               timestamps=job["series"].get("timestamps"),
               available_at=job["series"].get("available_at"),
               period_seconds=job["series"].get("period_seconds", 1))
    record = eligibility_record(Path(job["eligibility"])) if job.get("eligibility") else \
        {"freeze_sha256": job["eligibility_inline"]["freeze_sha256"],
         "design_sha256": job["eligibility_inline"]["design_sha256"],
         "cells": {tuple(c["key"]): c["cell"] for c in job["eligibility_inline"]["cells"]}}
    out = contrast(s, operator, proto, contrast_id=job["contrast_id"], eligibility=record,
                   unit=job["unit"], variable=job["variable"],
                   branch_a=job.get("branch_a", "raw"), branch_b=job.get("branch_b", "transformed"))
    body = json.dumps(out, sort_keys=True, default=_jsonable).encode()
    (adir / "contrast.json").write_bytes(body)
    result = {"status": "COMPLETED", "reason": "", "output_file": "contrast.json",
              "output_sha256": hashlib.sha256(body).hexdigest(), "rows_written": 1,
              "outcome": out["outcome"]}
    tmp = adir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, adir / "result.json")
    return 0


# --- the reserved holdout, bound to the reserve identity ---------------------------------------------

def adjudicate_holdout(reserve: dict, protocol: Protocol, run, *, state_dir: Path = HOLDOUT_STATE) -> dict:
    """One adjudication per (reserve identity, protocol). The reserve is named by its campaign
    digest or dataset digests; the marker lives in the state directory, write-once."""
    if not isinstance(reserve, dict) or not any(reserve.get(k) for k in
                                                 ("campaign_sha256", "dataset_sha256s")):
        raise SystemExit("REFUSED: a reserve is identified by campaign_sha256 or dataset_sha256s")
    reserve_sha = sha_obj(reserve)
    proto_sha = protocol.sealed()["protocol_sha256"]
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / f"{reserve_sha}.{proto_sha}.json"
    if marker.exists():
        raise SystemExit("REFUSED: this reserve was already adjudicated under this protocol; a "
                         "second use is a second look")
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"reserve": reserve, "reserve_sha256": reserve_sha, "protocol_sha256": proto_sha,
                   "at": time.time()}, handle)
    return run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    parser.error("this module is a harness; import it, or run a governed job through "
                 "df_utility_run.py")


if __name__ == "__main__":
    raise SystemExit(main())
