#!/usr/bin/env python3
"""MOD-E0-DEV numerical design sheet (RP2): every number with its derivation, alternative and
sensitivity; cells of the pilot; budget; review table. Sealed before any outcome.

    python tools/df_mod_e0_design.py --out DESIGN.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("df_mod_e0")

DESIGN_SCHEMA = "df_mod_e0_design.v1"
REPLICATES = (1, 2, 3)                 # independent trajectories: distinct generator seeds -> distinct thetas, AR and noise draws
RANDOM_ASSIGNMENTS = 3                 # predefined redistributions per H2 condition (same sizes, never a relabeling)
H3_LEVEL = 2                           # heterogeneity level at which H3 pairs are generated (groups identifiable: ARI 1 at h >= 2 in smoke)
BUDGET = {"aggregate_cpu_seconds": 14400.0, "headroom": 0.25, "counts": "cost pilot + every child + failures + verification",
          "gpu": "none unless in the authorised cost design; not used", "prior_adequacy_seconds_separately": 45.0 + 31.0}


def derivations() -> dict:
    b = E.boundaries()
    rows_train = b["train"][1] - b["train"][0]
    return {
        "sampling": {"unit": "1 sample = 1 time step of the generator; all variables share the interval",
                     "why": "H2 forbids changing sampling or volume across levels; a common grid keeps information equal"},
        "period_A": {"value": E.P_A, "why": f"P_A = {E.P_A} samples gives {rows_train / E.P_A:.0f} cycles in training and "
                                            f"{E.WINDOW / E.P_A:.0f} cycles in the context; small enough for many cycles, large enough "
                                            "for the detector's receptive field (17) to see a fraction of a cycle",
                     "alternatives": [12, 48], "sensitivity": "declared, not run in this pilot"},
        "heterogeneity_levels": {"levels": list(E.LEVELS),
                                 "phi": {h: E.group_params(h)["A"]["phi"] for h in E.LEVELS},
                                 "phi_B": {h: E.group_params(h)["B"]["phi"] for h in E.LEVELS},
                                 "period_B": {h: E.group_params(h)["B"]["period"] for h in E.LEVELS},
                                 "why": "h scales BOTH persistence gap (0.24 h) and period gap (50 % h) so that level 0 is a homogeneous "
                                        "population (grouping arbitrary by construction) and level 3 separates groups by profile alone; "
                                        "amplitude, sampling, p, N and noise are constant across h (H2 rule)",
                                 "control": "level 0 is the internal null of H2: e(0) has no mechanism to favour profiles"},
        "cross_dependence": {"beta": E.BETA, "tau": E.TAU,
                             "why": f"tau = {E.TAU} < window and > horizon: the dependence is causal and inside the context; beta = {E.BETA} "
                                    "gives the oracle a visible gain over the marginal oracle (smoke: cross-correlation 0.68 at lag tau under "
                                    "r = 1 vs 0.00 under r = 0, marginal variance equal within 4 %)",
                             "r0_construction": "the lagged partner term is replaced by an independent N(0,1) draw scaled to the same variance; "
                                                "marginals preserved in distribution (ML04 checks variance, ACF and quantiles across replicates); "
                                                "no temporal shuffle"},
        "context": {"window": E.WINDOW, "why": f"W = {E.WINDOW} = 2 cycles of P_A and {E.WINDOW / E.group_params(3)['B']['period']:.2f} cycles of "
                                              "P_B at h = 3; covers tau and the branch receptive field",
                    "receptive_field": E.RECEPTIVE_FIELD, "alternatives": [24, 96], "in_cycles_A": E.WINDOW / E.P_A},
        "horizon": {"value": E.HORIZON, "why": "one step, direct, per variable (multi-output head); multi-step is E1 work"},
        "volume": {"n_total": E.N_TOTAL, "boundaries": b, "rows_train": rows_train, "rows_validation": E.VALIDATION_ROWS,
                   "rows_test": E.TEST_ROWS - E.HORIZON, "purge": b["purge"],
                   "unique_values": E.N_TOTAL * E.P_VARS, "windows_train": rows_train, "mini_batches_per_epoch": math.ceil(rows_train / E.TRAINING["batch"]),
                   "max_updates": E.TRAINING["max_updates"],
                   "why": "purge = W + H derived from the consumed support; windows are overlapping (not independent replicates); "
                          "independence comes from replicates (generator seeds), not from windows, seeds of the optimiser or origins"},
        "mase": {"denominator": "train MAE of the seasonal naive with the variable's declared period; shared by every arm",
                 "zero_policy": "a zero denominator (deterministic clean control) is NO_APLICA before any test score; MAE reported"},
        "architecture": {"per_branch": ["Conv1D(16,3,causal,relu) x2 (detector)", "Conv1D(16,3,causal,d=2) + Conv1D(16,3,causal,d=4) (integrator)",
                                        "TimeDistributed Dense(8,relu) (adapter)"],
                         "fusion_sequence": "Concatenate(channels) -> Conv1D(16,3,causal) core -> last position -> Dense(p)",
                         "fusion_summary": "GlobalAveragePooling1D per branch -> Concatenate -> Dense(32) core -> Dense(p)",
                         "trainable_params": {"H2 full model": 6312, "H3 sequence (fusion+core+head)": 920, "H3 summary": 808,
                                              "tolerance": "15 % (declared in this pilot)"},
                         "optimizer": E.TRAINING, "why": "widths follow the proposal's illustration scaled down for CPU; every width is a "
                                                       "sensitivity alternative, not an optimum"},
        "estimands": {"H2": "e(h) = MASE(profiles, h) - MASE(random, h), averaged over replicates and the predefined random assignments; "
                            "slope of e over h by least squares with equal weights",
                      "H3": "d_r = MASE(sequence, r) - MASE(summary, r); gamma = d_1 - d_0",
                      "sign": "negative favours the method; no margin applied in the pilot; the pilot reports precision, not support"},
        "precision_rule": "the pilot's replicate SD of e(h), d_r and gamma feeds a simulation-based sample-size rule for E0-CONF "
                          "(bootstrap over replicates); not claimed confirmatory",
        "controls": {"positive_learning": "the modular model must beat the naive forecaster and approach the causal oracle on the "
                                          "development split at h = 3, r = 1 (ML07); measured with the real receiver",
                     "grouping_positive": "profile grouping recovers the latent groups at h >= 2 (ARI = 1) and not at h = 0",
                     "null_H2": "level 0", "null_H3": "r = 0", "adverse": "a shuffled-time control is NOT used as the r = 0 arm"},
        "information_not_seen": "latent group labels, generator parameters, thetas, the oracle and the future are never inputs",
    }


def cells(design: dict) -> list:
    out = []
    for h in design["levels"]:
        for seed in design["replicates"]:
            out.append({"cell_id": f"H2__h{h}__s{seed}__profiles", "hypothesis": "H2", "level": h, "r": 1, "seed": seed, "arm": "profiles"})
            for k in range(design["random_assignments"]):
                out.append({"cell_id": f"H2__h{h}__s{seed}__random_{k}", "hypothesis": "H2", "level": h, "r": 1, "seed": seed, "arm": f"random_{k}"})
    for r in (0, 1):
        for seed in design["replicates"]:
            out.append({"cell_id": f"H3__r{r}__s{seed}__extractor", "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed, "arm": "extractor"})
            for arm in ("sequence", "summary"):
                out.append({"cell_id": f"H3__r{r}__s{seed}__{arm}", "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed, "arm": arm,
                            "depends_on": f"H3__r{r}__s{seed}__extractor"})
    return out


REVIEW = [
    ("ML01", "generator reproduces equations, population and distribution; labels/metadata/latent group never features", "tests"),
    ("ML02", "future/validation/test changes leave fit, groups and past outputs unchanged; roles by identity/time", "tests"),
    ("ML03", "H2 changes the assignment, not names; sizes, models, contexts, budget equal per arm", "tests + cell records"),
    ("ML04", "H3 preserves marginals in distribution and changes the relevant dependence; no shuffle control", "tests"),
    ("ML05", "H3 arms use identical frozen extractor activations; freezing verified by weights", "tests + cell records"),
    ("ML06", "group -> detector/integrator/adapter -> fusion -> core/head keeps temporal shape; real gradients", "tests"),
    ("ML07", "positive learning with the real receiver; oracle and naive references; curves distinguish lack of training from lack of utility", "tests + pilot"),
    ("ML08", "early stopping by validation, real weight restore, reload parity, test never selects", "tests + cell records"),
    ("ML09", "metrics from arrays; shared denominators; zero policy; empty/non-finite refuse", "tests + verify"),
    ("ML10", "e(h), d0/d1/gamma with the declared sign; replicates/seeds/windows never inflate the unit", "tests + verify"),
    ("ML11", "child reads delivered bytes; CPU updates observed; complete costs and limits", "runner tests + pilot"),
    ("ML12", "a real failure leaves a terminal/outbox; reconciliation by population and content", "runner tests + verify"),
]


def build(max_updates: int | None = None, successor_of: str | None = None, reason: str | None = None) -> dict:
    """`max_updates`: a smaller DEV stage sealed BEFORE its scores (RP4) keeping both contrasts,
    every condition and replicate; recorded as a successor with its reason."""
    training = dict(E.TRAINING)
    if max_updates is not None:
        training["max_updates"] = int(max_updates)
        training["stage_note"] = reason or "smaller DEV stage: update allowance reduced before any score"
    doc = {"schema": DESIGN_SCHEMA, "experiment": "MOD-E0-DEV", "proposal": "P-MOD", "hypotheses": ["H2", "H3"],
           "classification": "DEVELOPMENT_ONLY_NO_RESERVED_CONFIRMATION", "levels": list(E.LEVELS), "replicates": list(REPLICATES),
           "random_assignments": RANDOM_ASSIGNMENTS, "h3_level": H3_LEVEL, "p": E.P_VARS, "window": E.WINDOW, "horizon": E.HORIZON,
           "n_total": E.N_TOTAL, "training": training, "successor_of": successor_of, "successor_reason": reason,
           "derivations": derivations(), "budget": BUDGET,
           "review_table": [{"id": i, "obligation": o, "evidence": e, "status": "NOT_TESTED"} for i, o, e in REVIEW],
           "reserve": "not generated, not opened: E0-CONF uses distinct generator parameter ranges and seeds fixed later",
           "design_sha256": ""}
    doc["cells"] = cells(doc)
    doc["cells_total"] = len(doc["cells"])
    doc["design_sha256"] = E.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--successor-of", default=None)
    parser.add_argument("--reason", default=None)
    args = parser.parse_args(argv)
    doc = build(args.max_updates, args.successor_of, args.reason)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; a design is never written over")
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"design_sha256": doc["design_sha256"], "cells": doc["cells_total"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
