#!/usr/bin/env python3
"""Controls of known utility for the contrast instrument (Q3). Fabricated fixtures only.

Before another sweep, the instrument must show it separates what it claims to measure. Each
control binds a requirement, the estimand the contrast reports (paired loss difference A − B
of the probe model), a generator with a JUSTIFIED differential advantage (or none), the loss
and the criterion expected — declared here, before running, never tuned to pass:

  positive_H_T   next increment = a · z[t−1] + ε, z = |x − median16| / MAD16 (the very score
                 `mad_extremes_trailing` emits, unsigned); 4 raw lags cannot compute a 16-row
                 median/MAD → the transformed branch should ADVANCE over raw.
  positive_H_A   same generator; raw+R (augmented) against raw of equal width (raw_wide, 8 lags,
                 still short of the 16-row support) → should ADVANCE.
  null_contrast  white null (independent increments) → DOES_NOT_ADVANCE, delta near zero.
  info_loss      next increment = b · (x[t−1] − x[t−2]) + ε: signed momentum; the unsigned z
                 loses the sign → transformed alone should be WORSE than raw (delta < 0).
  future_leak    a representation emitting x[t+1] − x[t] as if at t → REFUSED before scoring
                 by the prefix check, even under a forged eligibility.

Calibration (the false-advance rate under a null) and positive control (sensitivity to a
justified effect) are distinct roles and are reported apart. Power/error are counted over a
predeclared number of replicates under a predeclared CPU budget; a budget exhausted stops
the run and reports the partial count. Nothing here opens a selection, a reserve or a campaign.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")
ops = _load("df_d3_operators")
contract = _load("df_d3_contract")

OPERATOR = "mad_extremes_trailing"
W = 16                                    # the operator's trailing window (its declared params)
A_DRIFT = 1.0                             # predeclared effect sizes — not tuned
B_MOMENTUM = 0.5
NOISE = 1.0

CONTROLS = {
    "positive_H_T": {"generator": "mad_extremeness_drift", "pair": ("raw", "transformed"),
                     "requirement": "the transformed branch must ADVANCE when the next increment depends on the "
                                    "operator's own unsigned extremeness score, which 4 raw lags cannot compute",
                     "estimand": "paired MAE(raw) − MAE(transformed) over 4 walk-forward blocks, lower bound at 1−α/2",
                     "expected": {"outcome": H.ADVANCES, "min_fraction": 5 / 6}},
    "positive_H_A": {"generator": "mad_extremeness_drift", "pair": ("raw_wide", "augmented"),
                     "requirement": "raw+R must ADVANCE over raw of the same width (8 lags, below the 16-row support)",
                     "estimand": "paired MAE(raw_wide) − MAE(augmented), lower bound at 1−α/2",
                     "expected": {"outcome": H.ADVANCES, "min_fraction": 5 / 6}},
    "null_contrast": {"generator": "white_null", "pair": ("raw", "transformed"),
                      "requirement": "no advance under independent increments",
                      "estimand": "paired MAE(raw) − MAE(transformed)",
                      "expected": {"outcome": H.DOES_NOT_ADVANCE, "min_fraction": 5 / 6, "max_advances": 1}},
    "info_loss": {"generator": "signed_momentum", "pair": ("raw", "transformed"),
                  "requirement": "when the signal is the signed last increment, the unsigned score must lose: "
                                 "delta < 0 and no advance",
                  "estimand": "paired MAE(raw) − MAE(transformed); sign of the delta",
                  "expected": {"outcome": H.DOES_NOT_ADVANCE, "delta_sign": -1, "min_fraction": 5 / 6}},
    "future_leak": {"generator": "white_null", "pair": ("raw", "transformed"),
                    "requirement": "a representation that looks one row ahead is refused before any scoring",
                    "estimand": "none (refusal)",
                    "expected": {"outcome": H.REFUSED, "why_contains": "not causal", "min_fraction": 1.0}},
}


class FutureLeakOperator(ops.DeltaRunLength):
    """Emits x[t+1] − x[t] as if at t, declaring itself causal (the reviewer's control)."""
    KIND = "future_leak_control"

    def transform(self, x, state):
        v = np.asarray(x["values"], dtype=float)
        out = np.zeros(v.size)
        out[:-1] = v[1:] - v[:-1]
        avail = np.ones(v.size, dtype=bool)
        avail[-1] = False
        avail[0] = False
        return self._pack(x, out.tolist(), avail.tolist())


def _z(x, t):
    w = x[t - W:t]
    med = np.median(w)
    mad = np.median(np.abs(w - med)) or 1e-12
    return abs(x[t - 1] - med) / mad


def generate(name: str, n: int, rng) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0, NOISE, n)
    if name == "white_null":
        return np.cumsum(e)
    for t in range(1, n):
        drive = 0.0
        if name == "mad_extremeness_drift" and t > W:
            drive = A_DRIFT * min(3.0, _z(x, t))
        elif name == "signed_momentum" and t > 1:
            drive = B_MOMENTUM * (x[t - 1] - x[t - 2])
        x[t] = x[t - 1] + drive + e[t]
    return x


def _protocol(n_sims_conf=0.5):
    fam = ("ctl__v0__mad_extremes_trailing__transformed", "ctl__v0__mad_extremes_trailing__augmented")
    plan = {"generator": "white_null", "n": 0, "bound_confidence": n_sims_conf,
            "n_sims": H.sims_required_for_zero(0.05 / len(fam), n_sims_conf)}
    return fam, plan


def run_controls(*, n: int, replicates: int, seed0: int = 100, budget_cpu_seconds: float, operator: str = OPERATOR,
                 controls: tuple = tuple(CONTROLS)) -> dict:
    t0 = time.process_time()
    op = ops.build(operator)
    fam, plan = _protocol()
    plan = {**plan, "n": int(n)}
    base = H.Protocol(target="return", horizon=1, model="ridge", window=4, n_blocks=4, margin=0.0, seed=7,
                      family=fam, min_rows_per_block=30, calibration_plan=plan,
                      branches=("raw", "transformed", "augmented", "raw_wide"))
    elig = {"freeze_sha256": "c" * 64, "design_sha256": "c" * 64,
            "cells": {("ctl", "v0", op.KIND): {"verdict": "MECHANICALLY_ACCEPTED", "spec_sha256": contract.spec_sha256(op.describe())}}}
    out = {"schema": "df_utility_controls.v1", "operator": operator, "n": n, "replicates": replicates, "seed0": seed0,
           "budget_cpu_seconds": budget_cpu_seconds, "effect_sizes": {"a_drift": A_DRIFT, "b_momentum": B_MOMENTUM, "noise": NOISE},
           "calibration_role": "fixture calibration (white null, low confidence) decides ADVANCES/DOES_NOT_ADVANCE; "
                               "it is NOT a positive control and the positive controls are NOT calibrations",
           "calibrations": {}, "controls": {}, "stopped": None}
    protocols = {}
    for pair in {c["pair"] for c in CONTROLS.values()}:
        rec = H.calibrate(base, op, plan=plan, seed=11, branch_a=pair[0], branch_b=pair[1])
        protocols[pair] = base.with_calibration(rec)
        out["calibrations"]["/".join(pair)] = {"advances": rec["advances"], "scored": rec["scored"],
                                               "upper_bound": rec["upper_bound"], "supports": H.calibration_supports(
                                                   base, op, n, record=rec, branch_a=pair[0], branch_b=pair[1])[0]}
    for name in controls:
        spec = CONTROLS[name]
        pair = spec["pair"]
        rows = []
        for r in range(replicates):
            if time.process_time() - t0 > budget_cpu_seconds:
                out["stopped"] = f"BUDGET_EXHAUSTED at {name} replicate {r}"
                break
            rng = np.random.default_rng(seed0 + r)
            s = H.series(generate(spec["generator"], n, rng))
            operator_used = FutureLeakOperator() if name == "future_leak" else op
            elig_used = {**elig, "cells": {("ctl", "v0", operator_used.KIND): {"verdict": "MECHANICALLY_ACCEPTED",
                                                                              "spec_sha256": contract.spec_sha256(operator_used.describe())}}}
            res = H.contrast(s, operator_used, protocols[pair], contrast_id=fam[0] if pair[1] == "transformed" else fam[1],
                             eligibility=elig_used, unit="ctl", variable="v0", branch_a=pair[0], branch_b=pair[1])
            rows.append({"replicate": r, "seed": seed0 + r, "outcome": res.get("outcome"), "delta_mean": res.get("delta_mean"),
                         "delta_lower": res.get("delta_lower"), "why": res.get("why"),
                         "cpu_seconds": (res.get("cost") or {}).get("cpu_seconds")})
        exp = spec["expected"]
        hits = [x for x in rows if x["outcome"] == exp["outcome"]
                and (exp.get("delta_sign") is None or (x["delta_mean"] is not None and np.sign(x["delta_mean"]) == exp["delta_sign"]))
                and (exp.get("why_contains") is None or exp["why_contains"] in (x["why"] or ""))]
        advances = sum(1 for x in rows if x["outcome"] == H.ADVANCES)
        met = bool(rows) and len(rows) == replicates and len(hits) / replicates >= exp["min_fraction"] \
            and advances <= exp.get("max_advances", replicates)
        out["controls"][name] = {"spec": {k: v for k, v in spec.items() if k != "expected"}, "expected": exp, "rows": rows,
                                 "hits": len(hits), "advances": advances, "of": replicates, "completed": len(rows),
                                 "met": met}
        if out["stopped"]:
            break
    out["cpu_seconds"] = round(time.process_time() - t0, 2)
    out["instrument_verdict"] = "SEPARATES" if all(c["met"] for c in out["controls"].values()) and not out["stopped"] \
        and len(out["controls"]) == len(controls) else "DOES_NOT_SEPARATE_OR_INCOMPLETE"
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--seed0", type=int, default=100)
    parser.add_argument("--budget-cpu-seconds", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; controls are never written over")
    out = run_controls(n=args.n, replicates=args.replicates, seed0=args.seed0, budget_cpu_seconds=args.budget_cpu_seconds)
    args.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"verdict": out["instrument_verdict"], "stopped": out["stopped"], "cpu_seconds": out["cpu_seconds"],
                      "calibrations": out["calibrations"],
                      "controls": {k: (v["hits"], v["advances"], v["of"], v["met"]) for k, v in out["controls"].items()}}, indent=1))
    return 0 if out["instrument_verdict"] == "SEPARATES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
