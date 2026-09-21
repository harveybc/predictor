#!/usr/bin/env python3
"""FIN-LOSS-OPT: the executable protocol of the financial loss/optimizer comparison — designed, not run.

Owner's decision (FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21): MAE/Huber x Adam/AdamW is MANDATORY
on the actual financial forecasting task before any financial recipe is fixed; the household result
selects nothing for trading; improvements of 1e-5 and 1e-6 in MAE_z are resolutions of interest,
not thresholds of relevance. This module freezes what the policy asks to be frozen and computes the
parts that must come from data before a run — and refuses to be mistaken for the run.

  task_freeze()        the governed resource, its coverage, the two horizons and the weekly folds
  delta_candidates()   Huber deltas as fractions of a residual scale computed on ROLLING TRAIN
                       origins only; a change after the last train origin cannot move them
  search_grid()        one bounded, equal budget per loss family; LR/decay candidates declared, not
                       derived from a scaling law
  receivers()          a compact modular receiver and a larger one motivated by the business configs,
                       with measured reach and parameter count
  seal()               the design, with the benchmark contract embedded; state NOT_STARTED

    python tools/df_fin_loss_opt_design.py --seal DESIGN.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "fin_loss_opt_design.v1"
RESOURCE = "market_data/forex/g10/eurusd/1h.parquet"
LAKE = "financial_files"


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def task_freeze() -> dict:
    """The financial task, fixed from the governed resource — not from a remembered configuration."""
    return {
        "lake": LAKE, "resource": RESOURCE,
        "coverage_read_from_the_lake": {"rows": 129873, "t_min": "2005-01-03 01:00:00", "t_max": "2025-12-31 16:00:00",
                                        "time_column": "datetime", "read_at": "2026-09-21 via /api/v1/coverage"},
        "reserve": {"policy": "deny_from 2025-01-01 on financial_files", "rule": "never read; no selection query"},
        "target": {"column": "close", "construction": "hourly bar close as served", "transform_for_evaluation": "zscore_train per fold"},
        "horizons": {"short": {"steps": 6, "hours": 6, "why": "the owner's recalled 6 h is an antecedent; 6 hourly steps is "
                                                                "declared here as the short horizon and justified as one "
                                                                "quarter of a trading day at this resolution"},
                     "long": {"steps": 72, "hours": 72, "why": "the recalled ~3 days becomes 72 hourly steps; both are "
                                                                 "declared BEFORE any score and are not recovered configs"}},
        "folds": {"scheme": "weekly walk-forward", "week": "Monday 00:00 UTC .. Sunday 23:00 UTC (the 13C definition)",
                  "retraining": "weekly", "history_window_candidates_weeks": [52, 104, 208],
                  "why_candidates": "four years is the owner's antecedent, not a minimum; 52/104/208 weeks are declared "
                                    "candidates judged by regime coverage and learning curves on DEV folds only",
                  "dev_folds": "the last 26 weeks before 2025-01-01 as DEV; the reserve stays closed"},
        "missing_policy": "non-finite bars withdrawn; weekend gaps stay gaps",
        "naive": "persistence at each horizon on identical rows",
    }


def delta_candidates(y: np.ndarray, *, train_end: int, horizon: int, window: int = 24*20) -> dict:
    """Huber delta candidates from a residual scale obtained CAUSALLY inside train.

    The scale is the median absolute h-step change over rolling origins that end before `train_end`
    — a persistence residual, the only residual available before any model exists — expressed in
    standardised units of the train target. Rows after `train_end` are never touched, so a change
    there cannot move a candidate. Candidates are fractions of that scale, plus delta = 1 so the
    default arm stays in the grid.
    """
    y = np.asarray(y, dtype=np.float64)
    train = y[:train_end+1]
    sd = float(np.std(train))
    if not np.isfinite(sd) or sd <= 0:
        return {"status": "DEGENERATE_SCALE", "candidates": []}
    changes = np.abs(train[horizon:]-train[:-horizon])
    rolling = [np.median(changes[i:i+window]) for i in range(0, max(1, changes.size-window), window)]
    scale_z = float(np.median(rolling))/sd
    fractions = (0.25, 0.5, 1.0, 2.0)
    cands = [{"delta_z": round(scale_z*f, 6), "fraction_of_scale": f} for f in fractions] + [{"delta_z": 1.0, "fraction_of_scale": None}]
    below = [float(np.mean(changes/sd <= c["delta_z"])) for c in cands]
    for c, b in zip(cands, below):
        c["fraction_of_train_residuals_below_delta"] = round(b, 4)
        c["delta_in_original_units"] = c["delta_z"]*sd
    return {"status": "MEASURED", "scale_source": "rolling median absolute h-step change over train origins only",
            "train_end": int(train_end), "horizon": int(horizon), "scale_z": scale_z, "sd_train": sd,
            "candidates": cands,
            "note": "a single delta=1 does not dismiss Huber; changing the target scale changes the physical "
                    "threshold, which is why delta travels in z and is re-derived per fold"}


def search_grid() -> dict:
    return {"families": ["mae", "huber"], "optimizers": ["adam", "adamw"],
            "budget_per_loss_family": 12, "budget_unit": "DEV fits per family per horizon (equal, declared)",
            "learning_rates": [0.0005, 0.001, 0.003],
            "weight_decays_adamw": [0.0, 0.001, 0.004, 0.01],
            "delta": "from delta_candidates() per fold",
            "defaults_arm": {"adam": {"lr": 0.001, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
                             "adamw": {"lr": 0.001, "weight_decay": 0.004, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
                             "huber": {"delta_z": 1.0}, "note": "the installed version's explicit defaults; not the owner's "
                                                                "historical optimum, which is not recovered"},
            "decay_note": "the cumulative product of (1 - lr_t*decay) describes shrinkage due to decay ONLY; it orients "
                          "candidates with width, data and duration and is not an optimal-value formula; muP scaling "
                          "rules do not transfer to a network without that parameterisation",
            "selection": "validation MAE_z on DEV folds only; the reserve is never queried",
            "weight_groups_adamw": "decay applied to kernels; bias and normalisation parameters excluded (declared)"}


def receivers() -> dict:
    """Two receivers: the compact modular one and a larger one motivated by the business configs."""
    P = _module("df_e1_pilot")
    E = _module("df_mod_e0")
    tf = E._tf()
    compact = P._model_for_target([0, 0, 0, 0, 0], 60, 5, 3, 1, core="tcn_w")
    n_compact = int(sum(int(w.shape.num_elements()) for w in compact.trainable_weights))
    return {"compact_modular": {"window": 60, "channels": "OHLC + volume proxy (5)", "core": "tcn_w", "parameters": n_compact,
                                "reach": P.model_reach(60, "tcn_w")},
            "larger_business_receiver": {"window": 144, "channels": "the business configs' feature sets (phase 3/4 window 144)",
                                         "core": "tcn_w with width chosen by a declared rule at pilot",
                                         "parameters": "measured at pilot; capacity declared, not required to reach a million",
                                         "reach": P.model_reach(144, "tcn_w")},
            "rule": "both receivers run every recipe; no recipe is compared across receivers"}


def fl_matrix() -> list:
    return [
        {"id": "FL01", "requirement": "real factorial", "test": "tests/test_fin_loss_opt_acceptance.py::test_FL01_*", "state": "GREEN (mechanism) / runner pending"},
        {"id": "FL02", "requirement": "fair comparison", "test": "::test_FL02_*", "state": "GREEN on the household factorial; financial pairing pending"},
        {"id": "FL03", "requirement": "no leak", "test": "::test_FL03_*", "state": "GREEN (prefix invariance, leaky control fails) / fold-change rule xfail"},
        {"id": "FL04", "requirement": "marginal precision", "test": "::test_FL04_*", "state": "GREEN arrays+json / terminal+warehouse xfail"},
        {"id": "FL05", "requirement": "observed training", "test": "::test_FL05_*", "state": "GREEN on the household factorial / min_delta+floor xfail"},
        {"id": "FL06", "requirement": "hyperparameters", "test": "::test_FL06_*", "state": "GREEN (delta from causal train scale; grid declared)"},
        {"id": "FL07", "requirement": "statistical inference", "test": "::test_FL07_*", "state": "GREEN (both signs kept) / temporal blocks xfail"},
        {"id": "FL08", "requirement": "complete closure", "test": "::test_FL08_*", "state": "GREEN on the household factorial / financial delivery xfail"},
    ]


def seal() -> dict:
    B = _module("df_benchmark_contract")
    E = _module("df_mod_e0")
    ours = B.fx_eurusd_1h_ours()
    # the typed block (RP67): no literature contract exists for this task, so the decision is NOT_COMPARABLE with a
    # PLANNED reference to be re-executed under this contract; never a paper's number
    contract = ours.to_design_block(comparability=B.planned_reference(
        ours, why="no literature reference has been re-executed on this task; no contract of another work is registered"))
    design = {"schema": SCHEMA, "purpose": "FIN_LOSS_OPT", "state": "DESIGNED_NOT_STARTED",
              "task": task_freeze(), "recipes": ["mae_adam", "mae_adamw", "huber_adam", "huber_adamw"],
              "search": search_grid(), "receivers": receivers(), "benchmark_contract": contract,
              "reporting": {"scale": "MAE_z = mean|yhat-y|/sigma_train per fold, shared by every arm; RMSE_z; skill = 1 - "
                                     "MAE_z_model/MAE_z_naive; original units secondarily",
                            "precision": "predictions and targets saved without rounding; metrics accumulated in float64 and "
                                         "cross-checked by an independent calculator; dtype, reload and backend variation recorded",
                            "inference": "paired by week/origin and seed; intervals respecting temporal blocks; multiplicity declared"},
              "hosts": {"blocks": "all four recipes of a (seed, horizon) on one host; seeds spread across hosts only when the "
                                  "wrapper and environment are identical", "no_padding": True},
              "acceptance": fl_matrix(),
              "what_this_is_not": "not a run, not a winner, not a trading loss; the household factorial is a preserved "
                                  "antecedent (12 fits, one task) and selects nothing financial"}
    design["design_sha256"] = E.sha_obj(design)
    return design


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", type=Path, required=True)
    a = ap.parse_args(argv)
    d = seal()
    _module("df_d3_campaign").write_once(a.seal, d)
    print(json.dumps({"design_sha256": d["design_sha256"], "state": d["state"],
                      "horizons": {k: v["steps"] for k, v in d["task"]["horizons"].items()},
                      "compact_receiver_parameters": d["receivers"]["compact_modular"]["parameters"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
