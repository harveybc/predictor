#!/usr/bin/env python3
"""FIN-LOSS-OPT v2 (RP70): the executable protocol of the financial loss/optimizer comparison — designed, not run.

Owner's decision (FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21): MAE/Huber x Adam/AdamW is MANDATORY on the
actual financial forecasting task before any financial recipe is fixed; improvements of 1e-5 and 1e-6 in
MAE_z are resolutions of interest. Musashi's review found the v1 design not executable as declared
(row-step horizons over weekend gaps; zero and rounded Huber deltas; no enumerated candidates; receivers
"chosen at pilot"). Everything executable now lives in tools/df_fin_task.py and tools/df_fin_runner.py;
this module freezes the design around them and refuses to be mistaken for the run.

    python tools/df_fin_loss_opt_design.py --seal DESIGN.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "fin_loss_opt_design.v2"


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def task_freeze() -> dict:
    T = _module("df_fin_task")
    return {
        "lake": T.LAKE, "resource": T.RESOURCE, "time_column": T.TIME_COLUMN,
        "coverage_read_from_the_lake": {"rows": 129873, "t_min": "2005-01-03 01:00:00", "t_max": "2025-12-31 16:00:00",
                                        "read_at": "2026-09-21 via /api/v1/coverage (re-read this round)"},
        "reserve": {"policy": f"deny_from {T.HOLDOUT} on financial_files", "rule": "never read; deliveries are bounded ranges before it"},
        "target": {"column": T.TARGET, "construction": "hourly bar close as served", "transform_for_evaluation": "zscore with the fold's train sigma"},
        "horizons": {k: {"hours": v["hours"], "mapping": "ELAPSED TIME: the bar labelled exactly origin + hours; absent -> excluded with reason "
                                                        "(weekend, holiday, missing intraday bar); never the next retained row", "why": v["why"]}
                     for k, v in T.HORIZONS.items()},
        "availability": {"declared": "a bar's label is the time it is complete (lake contract: event time = available time); the decision at t "
                                     "reads labels <= t only",
                         "not_established_by_the_file": ["intrabar finality", "producer publication delay", "timezone of the labels",
                                                         "release cutoffs", "historical availability"],
                         "rule": "recorded as limitations; nothing about the producer is invented"},
        "folds": {"scheme": "weekly walk-forward by week identity (Monday 00:00 .. Sunday 23:00 on the label's clock)",
                  "dev": "the last 26 complete weeks strictly before the reserve", "validation": "the week before the test week",
                  "history_window_candidates_weeks": [52, 104, 208], "primary_history_weeks": 52,
                  "purge": "the horizon in hours between train targets and validation origins, and between validation targets and test origins",
                  "identities": "derived at prepare from the delivered bars (tools/df_fin_task.dev_folds), recorded per fold"},
        "missing_policy": "non-finite bars withdrawn from pairs; weekend/holiday gaps stay gaps; an origin without its elapsed-time target bar is excluded",
        "input_window": "the last W RETAINED bars ending at the origin (row identity; declared — the window is not an elapsed-time span)",
        "naive": "persistence: the origin's close, on identical pairs",
    }


def fl_matrix() -> list:
    t = "tests/test_fin_loss_opt_acceptance.py"
    return [
        {"id": "FL01", "requirement": "real factorial", "test": f"{t}::test_FL01_*", "state": "executable: enumerated candidates, real loss/optimizer components"},
        {"id": "FL02", "requirement": "fair comparison", "test": f"{t}::test_FL02_*", "state": "executable: identical pairs per fold, elapsed-time targets, metric oracle"},
        {"id": "FL03", "requirement": "no leak", "test": f"{t}::test_FL03_*", "state": "executable: fold-change isolation of scaler, sigma, deltas and pairs; leaky control fails"},
        {"id": "FL04", "requirement": "marginal precision", "test": f"{t}::test_FL04_*", "state": "executable: 1e-6 through arrays, json, terminal and the disposable warehouse"},
        {"id": "FL05", "requirement": "observed training", "test": f"{t}::test_FL05_*", "state": "executable: optimizer iterations, events, min_delta 0, float64 floor"},
        {"id": "FL06", "requirement": "hyperparameters", "test": f"{t}::test_FL06_*", "state": "executable: deltas from train pairs, full precision, declared fallbacks, 12 per family"},
        {"id": "FL07", "requirement": "statistical inference", "test": f"{t}::test_FL07_*", "state": "executable: both signs, moving-block interval over folds, multiplicity declared"},
        {"id": "FL08", "requirement": "complete closure", "test": f"{t}::test_FL08_*", "state": "executable: governance before reading, terminal artifacts in the warehouse, tamper detected"},
    ]


def seal() -> dict:
    B = _module("df_benchmark_contract")
    E = _module("df_mod_e0")
    T = _module("df_fin_task")
    ours = B.fx_eurusd_1h_ours()
    contract = ours.to_design_block(comparability=B.planned_reference(
        ours, why="no literature reference has been re-executed on this task; no contract of another work is registered"))
    design = {"schema": SCHEMA, "purpose": "FIN_LOSS_OPT", "state": "DESIGNED_NOT_STARTED",
              "task": task_freeze(), "candidates": T.candidate_allocation(), "receivers": T.receivers(),
              "delta_rules": T.delta_candidates.__doc__.strip(),
              "recipe": _module("df_fin_runner").RECIPE, "benchmark_contract": contract,
              "runner": "tools/df_fin_runner.py (prepare / execute / close); acceptance FL01-FL08 on synthetic bars through the disposable stack",
              "reporting": {"scale": "MAE_z = mean|yhat-y|/sigma_train per fold, shared by every candidate of the fold; original units secondarily",
                            "precision": "predictions and targets saved without rounding; metrics in float64; the float64 floor at the observed "
                                         "MAE_z recorded per cell; min_delta 0",
                            "inference": "paired by fold (week) between each family's validation-selected candidate; moving-block bootstrap "
                                         "interval; multiplicity = candidates compared per family, declared; no best-test",
                            "resolvable": "differences down to the float64 floor at ~0.5 (about 1e-16) survive arrays, json and the warehouse; "
                                          "what is DISTINGUISHABLE by inference is bounded by the fold-to-fold spread, reported, not assumed"},
              "hosts": {"blocks": "all candidates of a (fold, seed) on one host", "no_padding": True},
              "acceptance": fl_matrix(),
              "what_this_is_not": "not a run, not a winner, not a trading loss; the household factorial selects nothing financial"}
    design["design_sha256"] = E.sha_obj(design)
    return design


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", type=Path, required=True)
    a = ap.parse_args(argv)
    d = seal()
    _module("df_d3_campaign").write_once(a.seal, d)
    print(json.dumps({"design_sha256": d["design_sha256"], "state": d["state"],
                      "horizons": {k: v["hours"] for k, v in d["task"]["horizons"].items()},
                      "candidate_populations": d["candidates"]["totals"],
                      "receivers": {k: v.get("parameters") for k, v in d["receivers"].items() if isinstance(v, dict)}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
