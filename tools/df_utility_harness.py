#!/usr/bin/env python3
"""The representation-utility harness: design ready for review, NOT an experiment (L5).

What it measures, and only that: the PREDICTIVE LOSS of a fixed, small probe model (ridge
for a regression target, logistic for a direction target) fed with a branch of features,
compared PAIRWISE on the same emittable rows between the raw branch and a transformed branch
(and, when the hypothesis asks, an augmented raw+transformed branch). The difference of
losses is a difference of losses. It is never information in bits, never mutual information,
never trading utility.

Everything the estimate depends on is declared in a sealed `Protocol` before any row is
scored: target, horizon, model, window, walk-forward blocks, purge, margin, seeds, capacity
control, the number of comparisons the multiplicity correction covers, the abstention rules.
Windows are built by observation identity: a feature row at index t consumes the branch's
outputs available at t (their emission instant is the decision instant); its label consumes
samples up to t + horizon; the purge between a training block and its validation block is
at least the label horizon plus the representation's reach, so no training label overlaps a
validation feature or label. Normalisation and every fit happen inside the training block.

The holdout split is reserved: the harness scores `development` blocks; a `holdout`
adjudication is a single write-once act that refuses to run twice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

NOT_COMPARABLE = "NOT_COMPARABLE"
INSUFFICIENT_ROWS = "INSUFFICIENT_ROWS"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
ADVANCES = "ADVANCES"
DOES_NOT_ADVANCE = "DOES_NOT_ADVANCE"
REFUSED = "REFUSED"


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --- the protocol -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Protocol:
    """Sealed before scoring. `comparisons` is the number of (variable, representation)
    contrasts the multiplicity correction covers; `margin` is in loss units."""
    target: str                      # "direction" (log-loss) or "return" (MAE)
    horizon: int                     # label consumes samples up to t + horizon
    model: str                       # "logistic" or "ridge"
    window: int                      # lags per branch (capacity control: equal per branch)
    n_blocks: int                    # walk-forward validation blocks over development rows
    margin: float                    # the delta must clear this to ADVANCE
    seed: int
    comparisons: int                 # for Bonferroni over the whole predeclared family
    min_rows_per_block: int = 30
    alpha: float = 0.05
    ridge_lambda: float = 1.0
    logistic_steps: int = 200
    branches: tuple = ("raw", "transformed")   # optionally "augmented"
    budget_cpu_seconds_per_contrast: float = 60.0
    schema: str = "df_utility_protocol.v1"

    def sealed(self) -> dict:
        doc = asdict(self)
        doc["branches"] = list(self.branches)
        doc["protocol_sha256"] = sha_obj(doc)
        return doc


# --- inputs ---------------------------------------------------------------------------------------

def label(x: np.ndarray, protocol: Protocol) -> np.ndarray:
    """The label of row t uses x[t + horizon]; rows without it are NaN."""
    n = x.size
    h = protocol.horizon
    out = np.full(n, np.nan)
    if h <= 0 or h >= n:
        return out
    future = x[h:] - x[:-h]
    if protocol.target == "direction":
        out[:n - h] = (future > 0).astype(float)
    elif protocol.target == "return":
        out[:n - h] = future
    else:
        raise ValueError(f"unknown target {protocol.target!r}")
    return out


def lag_matrix(values: np.ndarray, available: np.ndarray, window: int) -> tuple:
    """Row t = the `window` outputs available at t (t, t-1, ...). A row is emittable only
    when every lag is available: no imputation, no future."""
    n = values.size
    X = np.full((n, window), np.nan)
    ok = np.zeros(n, dtype=bool)
    for t in range(window - 1, n):
        seg_v = values[t - window + 1:t + 1][::-1]
        seg_a = available[t - window + 1:t + 1]
        if seg_a.all():
            X[t] = seg_v
            ok[t] = True
    return X, ok


def branch_features(branch: str, x: np.ndarray, rep: dict | None, protocol: Protocol) -> tuple:
    """Features for one branch. `rep` = {"values", "available", "reach_right", "accepted"}.
    Capacity control: raw and transformed get the same `window` columns; augmented gets both
    (a separate contrast: does adding R to raw help?), declared as such."""
    avail_x = ~np.isnan(x)
    if branch == "raw":
        return lag_matrix(np.nan_to_num(x), avail_x, protocol.window)
    if rep is None:
        raise ValueError("a transformed branch needs a representation")
    Xr, okr = lag_matrix(np.nan_to_num(np.asarray(rep["values"], dtype=float)),
                         np.asarray(rep["available"], dtype=bool), protocol.window)
    if branch == "transformed":
        return Xr, okr
    if branch == "augmented":
        Xraw, okraw = lag_matrix(np.nan_to_num(x), avail_x, protocol.window)
        return np.hstack([Xraw, Xr]), okraw & okr
    raise ValueError(f"unknown branch {branch!r}")


# --- the probe models (fit inside the training block only) -------------------------------------

def _standardise(Xtr: np.ndarray, Xva: np.ndarray) -> tuple:
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
    if protocol.model == "logistic":
        beta = np.zeros(Xtr1.shape[1])
        lr = 0.1
        for _ in range(protocol.logistic_steps):
            p = 1.0 / (1.0 + np.exp(-(Xtr1 @ beta)))
            grad = Xtr1.T @ (p - ytr) / Xtr1.shape[0] + protocol.ridge_lambda * np.r_[beta[:-1], 0.0] / Xtr1.shape[0]
            beta -= lr * grad
        return 1.0 / (1.0 + np.exp(-(Xva1 @ beta)))
    raise ValueError(f"unknown model {protocol.model!r}")


def loss(pred: np.ndarray, y: np.ndarray, protocol: Protocol) -> float:
    if protocol.target == "direction":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(np.abs(pred - y)))


# --- blocks, purge, pairing -----------------------------------------------------------------------

def blocks(n_rows_index: np.ndarray, protocol: Protocol, purge: int) -> list:
    """Walk-forward blocks over the emittable row indices (time order): block k validates on
    its slice and trains on every emittable row that ends at least `purge` samples before
    the slice starts. Non-overlapping validation slices are the statistical unit."""
    idx = np.asarray(n_rows_index)
    if idx.size < protocol.n_blocks * protocol.min_rows_per_block * 2:
        return []
    cuts = np.linspace(idx.size // 2, idx.size, protocol.n_blocks + 1).astype(int)
    out = []
    for k in range(protocol.n_blocks):
        va = idx[cuts[k]:cuts[k + 1]]
        if va.size == 0:
            continue
        start = va[0]
        tr = idx[idx <= start - purge]
        out.append({"block": k, "train": tr, "validation": va, "purge": int(purge)})
    return out


def contrast(x: np.ndarray, rep: dict | None, protocol: Protocol, *, branch_a="raw",
             branch_b="transformed", cpu_seconds: float | None = None) -> dict:
    """One paired contrast: loss(branch_a) - loss(branch_b) per block on the SAME rows."""
    if rep is not None and not rep.get("accepted", False):
        return {"outcome": REFUSED, "why": "the representation is not MECHANICALLY_ACCEPTED in "
                                          "a verified matrix; utility is never measured on it"}
    if cpu_seconds is not None and cpu_seconds > protocol.budget_cpu_seconds_per_contrast:
        return {"outcome": BUDGET_EXHAUSTED, "cpu_seconds": cpu_seconds,
                "budget": protocol.budget_cpu_seconds_per_contrast}
    y = label(x, protocol)
    Xa, oka = branch_features(branch_a, x, rep, protocol)
    Xb, okb = branch_features(branch_b, x, rep, protocol)
    emittable = oka & okb & ~np.isnan(y)          # paired: the same rows for both branches
    coverage = {"n": int(x.size), "rows_a": int(oka.sum()), "rows_b": int(okb.sum()),
                "rows_paired": int(emittable.sum()),
                "inputs_missing": int(np.isnan(x).sum())}
    reach = int((rep or {}).get("reach_right", 0))
    purge = protocol.horizon + reach + protocol.window
    rows = np.flatnonzero(emittable)
    scheme = blocks(rows, protocol, purge)
    if not scheme:
        return {"outcome": INSUFFICIENT_ROWS, "coverage": coverage, "purge": purge}
    deltas = []
    per_block = []
    for b in scheme:
        tr, va = b["train"], b["validation"]
        if tr.size < protocol.min_rows_per_block or va.size < protocol.min_rows_per_block:
            per_block.append({"block": b["block"], "outcome": INSUFFICIENT_ROWS,
                              "train_rows": int(tr.size), "validation_rows": int(va.size)})
            continue
        la = loss(fit_predict(Xa[tr], y[tr], Xa[va], protocol), y[va], protocol)
        lb = loss(fit_predict(Xb[tr], y[tr], Xb[va], protocol), y[va], protocol)
        deltas.append(la - lb)
        per_block.append({"block": b["block"], "loss_a": la, "loss_b": lb, "delta": la - lb,
                          "train_rows": int(tr.size), "validation_rows": int(va.size)})
    if len(deltas) < 2:
        return {"outcome": INSUFFICIENT_ROWS, "coverage": coverage, "blocks": per_block,
                "purge": purge}
    d = np.asarray(deltas)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    # Bonferroni over the predeclared family; a t-interval over blocks (the statistical unit)
    alpha = protocol.alpha / max(1, protocol.comparisons)
    t_crit = _t_quantile(1 - alpha / 2, d.size - 1)
    lower = mean - t_crit * se
    outcome = ADVANCES if lower > protocol.margin else DOES_NOT_ADVANCE
    return {"outcome": outcome, "branch_a": branch_a, "branch_b": branch_b,
            "loss_name": "log_loss" if protocol.target == "direction" else "mae",
            "delta_mean": mean, "delta_se": se, "delta_lower": lower, "t_crit": t_crit,
            "alpha_adjusted": alpha, "margin": protocol.margin, "blocks": per_block,
            "blocks_used": int(d.size), "purge": purge, "coverage": coverage,
            "note": "a difference of predictive losses of the probe model; not information, "
                    "not mutual information, not trading utility"}


def _t_quantile(p: float, df: int) -> float:
    """Student t quantile without scipy: Cornish-Fisher from the normal quantile."""
    z = _norm_quantile(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def _norm_quantile(p: float) -> float:
    # Acklam's rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_norm_quantile(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --- the reserved holdout: one adjudication, write-once ----------------------------------------

def adjudicate_holdout(marker_dir: Path, protocol: Protocol, run) -> dict:
    """The reserved split is scored ONCE. A marker is written before scoring; a second call
    refuses. `run` is the callable that scores the holdout; it is never called otherwise."""
    marker_dir = Path(marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / "HOLDOUT_USED.json"
    if marker.exists():
        raise SystemExit("REFUSED: the reserved holdout was already adjudicated once; a second "
                         "use is a second look")
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"protocol_sha256": protocol.sealed()["protocol_sha256"]}, handle)
    return run()
