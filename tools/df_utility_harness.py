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
BRANCHES = ("raw", "transformed", "augmented")
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
    calibration: dict | None = None  # {"generator", "n_sims", "seed", "false_advance_rate"}
    prefix_checks: int = 8           # decision rows re-transformed from a cut series
    schema: str = "df_utility_protocol.v2"

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
        if not (0 < self.alpha < 1):
            problems.append("alpha must be in (0, 1)")
        if not isinstance(self.margin, (int, float)) or self.margin < 0:
            problems.append("margin must be a non-negative number")
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
            need = {"generator", "n_sims", "seed", "false_advance_rate", "alpha_adjusted"}
            if not isinstance(self.calibration, dict) or set(self.calibration) != need:
                problems.append(f"calibration must carry exactly {sorted(need)}")
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

    def with_calibration(self, record: dict) -> "Protocol":
        keep = ("generator", "n_sims", "seed", "false_advance_rate", "alpha_adjusted")
        return Protocol(**{**self.__dict__, "calibration": {k: record[k] for k in keep}})


# --- the observation contract -----------------------------------------------------------------------

def series(values, *, ids=None, timestamps=None, available_at=None, period_seconds=1) -> dict:
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
    return {"values": v, "ids": ids, "timestamps": ts, "available_at": av,
            "period_seconds": float(period_seconds)}


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
    x = s["values"]
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
    for t in range(n):
        taken = 0
        i = t
        while i >= 0 and taken < window:
            if available[i] and emitted_at[i] <= decision[t]:
                X[t, taken] = values[i]
                taken += 1
            elif available[i] and emitted_at[i] > decision[t] and i == t:
                pass                                   # the newest output is not out yet
            i -= 1
        ok[t] = taken == window
    return X, ok


def features(branch: str, s: dict, rep: dict | None, protocol: Protocol) -> tuple:
    decision = s["available_at"]                       # the decision for row t happens when
    raw_avail = ~np.isnan(s["values"])                 # observation t is itself available
    if branch == "raw":
        return _lags_by_emission(np.nan_to_num(s["values"]), raw_avail, s["available_at"],
                                 decision, protocol.window)
    if rep is None:
        raise ValueError("a transformed branch needs a representation")
    Xr, okr = _lags_by_emission(np.nan_to_num(rep["values"]), rep["available"],
                                rep["emitted_at"], decision, protocol.window)
    if branch == "transformed":
        return Xr, okr
    if branch == "augmented":
        Xa, oka = _lags_by_emission(np.nan_to_num(s["values"]), raw_avail, s["available_at"],
                                    decision, protocol.window)
        return np.hstack([Xa, Xr]), oka & okr
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

def blocks(rows: np.ndarray, protocol: Protocol, purge: int) -> list:
    """Walk-forward: validation slices over the second half of the emittable rows, training
    rows ending at least `purge` samples before the slice. Every block must reach the
    minimum or the contrast is INSUFFICIENT_ROWS (blocks_policy all_or_insufficient)."""
    idx = np.asarray(rows)
    if idx.size < protocol.n_blocks * protocol.min_rows_per_block * 2:
        return []
    cuts = np.linspace(idx.size // 2, idx.size, protocol.n_blocks + 1).astype(int)
    out = []
    for k in range(protocol.n_blocks):
        va = idx[cuts[k]:cuts[k + 1]]
        if va.size == 0:
            return []
        tr = idx[idx <= va[0] - purge]
        out.append({"block": k, "train": tr, "validation": va, "purge": int(purge)})
    return out


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
    needs_rep = "raw" not in (branch_a, branch_b) or branch_b != "raw"
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
    Xa, oka = features(branch_a, s, rep0, protocol)
    Xb, okb = features(branch_b, s, rep0, protocol)
    emittable = oka & okb & ~np.isnan(y)
    coverage.update(rows_a=int(oka.sum()), rows_b=int(okb.sum()), rows_paired=int(emittable.sum()))
    reach = rep_meta["reach_right"] if rep_meta else 0
    purge = protocol.horizon + reach + protocol.window
    rows = np.flatnonzero(emittable)
    scheme = blocks(rows, protocol, purge)
    if not scheme or any(b["train"].size < protocol.min_rows_per_block
                         or b["validation"].size < protocol.min_rows_per_block for b in scheme):
        return {"outcome": INSUFFICIENT_ROWS, "coverage": coverage, "purge": purge,
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
        Xa, oka = features(branch_a, s, rep, protocol)
        Xb, okb = features(branch_b, s, rep, protocol)
        both = oka & okb & ~np.isnan(y)
        tr_k = tr[both[tr]]
        va_k = va[both[va]]
        if tr_k.size < protocol.min_rows_per_block or va_k.size < protocol.min_rows_per_block:
            return {"outcome": INSUFFICIENT_ROWS, "coverage": coverage, "purge": purge,
                    "block": b["block"], "policy": protocol.blocks_policy}
        la = loss(fit_predict(Xa[tr_k], y[tr_k], Xa[va_k], protocol), y[va_k], protocol)
        lb = loss(fit_predict(Xb[tr_k], y[tr_k], Xb[va_k], protocol), y[va_k], protocol)
        deltas.append(la - lb)
        per_block.append({"block": b["block"], "loss_a": la, "loss_b": lb, "delta": la - lb,
                          "train_rows": int(tr_k.size), "validation_rows": int(va_k.size),
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
            "blocks": per_block, "blocks_used": len(deltas), "purge": purge, "coverage": coverage,
            "cost": cost, "protocol_sha256": protocol.sealed()["protocol_sha256"],
            "note": "a difference of predictive losses of the probe model; not information, "
                    "not mutual information, not trading utility"}
    cal = protocol.calibration
    if cal is None or cal["false_advance_rate"] > protocol.alpha_adjusted \
            or cal["alpha_adjusted"] != protocol.alpha_adjusted:
        return {**base, "outcome": INCONCLUSIVE_UNCALIBRATED,
                "why": "no calibration record supports this interval as confirmatory; the "
                       "delta is descriptive"}
    return {**base, "outcome": ADVANCES if lower > protocol.margin else DOES_NOT_ADVANCE}


# --- calibration: the false-advance rate under a dependent null --------------------------------------

def _ar1_null(n: int, rng, phi: float = 0.6) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0, 1.0, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return np.cumsum(x)


GENERATORS = {"ar1_null": "AR(1) increments (phi 0.6), cumulated: dependent, structured",
              "white_null": "independent N(0,1) increments, cumulated: the exchangeable null"}


def _white_null(n: int, rng, phi: float = 0.0) -> np.ndarray:
    return np.cumsum(rng.normal(0, 1.0, n))


def calibrate(protocol: Protocol, operator, *, n_sims: int, seed: int, n: int = 1500,
              generator: str = "white_null", eligibility: dict | None = None) -> dict:
    """The rate at which a contrast under this protocol says ADVANCES under a null. The
    exchangeable null (`white_null`) is the one that supports the interval: under it neither
    branch carries anything, so every ADVANCES is false. `ar1_null` is dependent AND
    structured — a diagnostic of what dependence does, not a null of no effect. Diagnostic
    seeds only; the record is sealed into the protocol before any real contrast."""
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}")
    rng = np.random.default_rng(seed)
    proto = protocol.with_calibration({"generator": generator, "n_sims": n_sims, "seed": seed,
                                       "false_advance_rate": 0.0,
                                       "alpha_adjusted": protocol.alpha_adjusted})
    fake = {"freeze_sha256": "calibration", "design_sha256": "calibration", "cells": {}}
    contract = _load("df_d3_contract")
    fake["cells"][("cal", "v0", operator.KIND)] = {
        "verdict": "MECHANICALLY_ACCEPTED", "spec_sha256": contract.spec_sha256(operator.describe())}
    advances = 0
    scored = 0
    make = _ar1_null if generator == "ar1_null" else _white_null
    for k in range(n_sims):
        s = series(make(n, rng))
        out = contrast(s, operator, proto, contrast_id=proto.family[0], eligibility=fake,
                       unit="cal", variable="v0")
        if out["outcome"] in (ADVANCES, DOES_NOT_ADVANCE):
            scored += 1
            advances += int(out["outcome"] == ADVANCES)
    rate = advances / scored if scored else float("nan")
    return {"generator": generator, "n_sims": n_sims, "seed": seed,
            "false_advance_rate": rate, "alpha_adjusted": protocol.alpha_adjusted,
            "scored": scored, "advances": advances}


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
    if not legacy and score.get("schema") != CONTRAST_SCHEMA and score.get("outcome") not in (
            REFUSED, INSUFFICIENT_ROWS):
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"schema {score.get('schema')!r}"}
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
        # a completed attempt is never re-run: the recorded outcome is re-verified and returned
        recorded = json.loads(prior.read_text())
        result = json.loads((attempt_dir / "result.json").read_text()) \
            if (attempt_dir / "result.json").is_file() else None
        score, refusal = verified_score(attempt_dir, result, recorded.get("verified"), job) \
            if recorded.get("status") == "COMPLETED" else (None, None)
        return {**recorded["summary"], "score": score, "resumed": True,
                **({"refusal": refusal} if refusal else {})}
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
        summary = {"outcome": score["outcome"] if score else SCORE_UNVERIFIED, "reason": reason,
                   "cost": cost, "score": score, "output_sha256": verified.get("output_sha256"),
                   **({"refusal": refusal} if refusal else {})}
    prior.write_text(json.dumps({"status": status, "verified": verified,
                                 "summary": {k: v for k, v in summary.items() if k != "score"}},
                                default=_jsonable))
    return summary


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
    s = series(job["series"]["values"], ids=job["series"].get("ids"),
               timestamps=job["series"].get("timestamps"),
               available_at=job["series"].get("available_at"),
               period_seconds=job["series"].get("period_seconds", 1))
    operator = ops.build(job["operator"]) if job.get("operator") else None
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
