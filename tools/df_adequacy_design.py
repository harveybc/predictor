#!/usr/bin/env python3
"""Frozen adequacy design: context x model x task x data volume on single-frequency signals (S1).

Separates four questions and answers only the first two here: (a) can the learner predict the
raw task, (b) how much history does it need. (c) representation utility and (d) transfer to
prediction/RL and weekly trading are NOT part of this pilot; its completion authorises no
selection. Everything below is fixed before any training: units, tasks with their own baseline
and oracle, models, contexts, training lengths, chronological boundaries with a purge derived
from the consumed support, seeds, optimizer, early-stopping rule, maximum updates, the (absent)
tuning allowance, error criteria, budgets, and the ML review table (requirement, executable
check, evidence, status, owner, next action) with every status NOT_TESTED until measured.

    python tools/df_adequacy_design.py --bank BANK --out DESIGN.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DESIGN_SCHEMA = "df_adequacy_design.v2"
UNITS = ("sinusoid__white__snr10__none__n2048__v1__seed12", "sinusoid__white__snr10__none__n2048__v1__seed13")
TASKS = {
    "clean_next_level": {"input": "clean", "label": "clean[t+1]", "baseline": "persistence: clean[t]",
                         "oracle": "analytic recurrence with the KNOWN period: 2cos(2pi/P)*clean[t] - clean[t-1] (diagnostic; truth-derived)",
                         "loss": "MAE", "denominator": "MAE of the persistence baseline on the same test rows"},
    "clean_increment": {"input": "clean", "label": "clean[t+1] - clean[t]", "baseline": "zero change",
                        "oracle": "recurrence increment: (2cos(2pi/P)-1)*clean[t] - clean[t-1] (diagnostic; truth-derived)",
                        "loss": "MAE", "denominator": "MAE of the zero-change baseline on the same test rows"},
    "observed_increment": {"input": "observed", "label": "observed[t+1] - observed[t]", "baseline": "zero change",
                           "oracle": "CONDITIONAL oracle (T1): next_clean by the known recurrence minus the CURRENT observation, "
                                     "2cos(2pi/P)*clean[t] - clean[t-1] - observed[t]; its residual is the next noise sample only "
                                     "(diagnostic; truth-derived). The old oracle (clean increment) is kept as oracle_old for history",
                           "loss": "MAE", "denominator": "MAE of the zero-change baseline on the same test rows",
                           "noise_floor": "expected Gaussian noise-only MAE sigma*sqrt(2/pi) under the declared generator, reported "
                                          "apart from the realised sample MAE of the oracle; neither is a samplewise lower bound",
                           "limitation": "the future noise sample is independent of the past: zero error is impossible; "
                                         "adequacy is judged against the conditional oracle, never against zero"},
}
MODELS = {
    "ridge": {"family": "linear", "lambda": 1.0, "inputs": "W standardised lags", "state": "none",
              "note": "closed form on train-only standardised features; the historical probe, kept as baseline",
              "limitation": "parameter count varies with the lag count (W + 1): a context change is also a capacity change for ridge"},
    "causal_conv1d": {"family": "tcn", "filters": 16, "kernel": 3, "dilations": "FIXED across W: 1,2,4,...,128 (receptive field 511 >= 256)",
                      "head": "last position -> Dense(1)", "activation": "relu", "state": "none", "padding": "causal",
                      "note": "the same graph and parameter count at every W (T1): a context ablation for the CNN; effective support = min(W, RF)"},
    "causal_conv1d_variable_depth": {"family": "tcn", "dilations": "1,2,4,... until the receptive field covers W (2 layers at W=4, 8 at W=256)",
                                     "role": "separate evidence (the S1 configuration): depth changes with W, so it is a joint "
                                             "architecture/context experiment; NOT in the factorial, no budget"},
    "lstm": {"family": "recurrent", "units": 32, "head": "Dense(1) on the final state", "state_policy": "RESET_PER_WINDOW",
             "note": "state never carries across windows, so validation/test never leak into training state"},
}
CONTEXTS = (4, 8, 128, 256)
TRAIN_LENGTHS = (256, 512, 768)          # 1024 does not fit n = 2048 with W = 256 and the purges (staged proposal)
HORIZON = 1
N = 2048
TEST = (1664, 2048)              # the untouched final test: 384 rows, chronologically last
INNER_VALIDATION_ROWS = 96
SEEDS = (1,)
TRAINING = {"optimizer": "adam", "learning_rate": 1e-3, "batch": 64, "max_epochs": 200, "max_updates": 3000,
            "early_stopping": {"monitor": "inner validation MAE", "patience": 10, "restore_best": True},
            "tuning_allowance": "NONE: no hyperparameter is changed after seeing a result",
            "scaling": "input and label standardised with TRAIN-ONLY mean and sd per cell"}
CRITERIA = {
    "raw_task_adequacy": "skill = 1 - MAE_model / MAE_baseline on the final test; a clean task is ADEQUATE when skill >= 0.5 "
                         "at some context, INADEQUATE when no context reaches 0.1; observed_increment is judged against the "
                         "oracle floor: ADEQUATE when MAE_model <= 1.1 * MAE_oracle at some context",
    "context_need": "the smallest W (by learning curve over W) at which the criterion is met; reported with W/P and the "
                    "actual consumed span (W-1)/P and the effective receptive field",
    "data_sufficiency": "learning curve over training length: SATURATED when the best two lengths differ by < 5 % of "
                        "MAE_baseline, UNSATURATED otherwise; declared honestly, never assumed",
    "diagnostics": "each cell is classified FITTED / UNDERFIT / OVERFIT / OPTIMIZATION_FAILURE from its curves and "
                   "observed updates (rules in df_adequacy_models.diagnose)",
    "stopping": "fixed factorial; no cell is added, repeated or re-seeded after seeing results; incomplete cells are reported",
}
BUDGET = {"aggregate_cpu_seconds": 14400.0, "counts": "this successor's pilots + every child + failures + all workers",
          "prior_pilot_seconds_separately": 31.0, "headroom": 0.25, "gpu": "none",
          "cost_pilot": "one child per model at W=4 and at W=256 (L=768), 200-update ceiling, no test access; startup/evaluation "
                        "overhead separated from optimizer work; projection with 25 % headroom must fit before dispatch"}

REVIEW_ROWS = [
    ("question_estimand", "distinct questions: raw skill (this pilot), representation utility (utildev-v1, relative), "
                          "denoising, forecasting/RL/policy utility (not tested); decision supported: whether ridge/W=4 was an "
                          "adequate probe and what context/model the raw task needs",
     "tasks and criteria sealed in this design; utildev-v1 kept at its original scope", "DESIGN.json + 12C"),
    ("target_noise", "clean vs observed targets with exact row/horizon identity, irreducible future noise, meaningful "
                     "baseline and a truth-derived oracle per task",
     "tests: label identity, recurrence oracle, noise-only control; per-cell oracle floor", "tests/test_df_adequacy.py; cell arrays"),
    ("context_support", "time span, periods, lookback, emission time, receptive field/state, equal information across arms",
     "per cell: W, (W-1)/P, RF, state policy; every model of a cell reads the same rows and labels", "cell records"),
    ("data_sufficiency", "usable training/evaluation rows after purge, independent series, regime coverage, learning curves",
     "learning curves over nested L in {256, 512, 768} at one cutoff and over W on shared rows (1024 needs a longer series: staged proposal); two independent units; regimes: one SNR, one perturbation (limited)",
     "curves in the report"),
    ("model_adequacy", "actual graph, optimizer updates, train/validation curves, capacity and context ablations, "
                       "underfit vs overfit vs optimisation failure",
     "per cell: layers, parameters, RF, observed updates, curves, diagnosis; tests: updates observed, reload parity",
     "cell records; tests"),
    ("temporal_validity", "train-only fitting and scaling, chronological nested validation, label-overlap purge, "
                          "prefix/future/missingness/restart tests on the deployed path",
     "tests: train-only scaling, forward held-out, future perturbation, restart parity; purge = W + h between splits",
     "tests; DESIGN.json"),
    ("comparison_fairness", "raw skill first; matched decision times and targets; information span vs input width, "
                            "capacity, compute and tuning allowance",
     "identical validation/test rows across W, model and L (asserted); fixed CNN graph across W; ridge capacity varies with W (declared); no tuning allowance", "DESIGN.json"),
    ("statistics", "practical effect, uncertainty, multiplicity, replication, calibration scope, predefined stopping",
     "skill with block-wise dispersion over the test; fixed factorial; no calibration transferred from ridge/W=4",
     "report"),
    ("business_transfer", "forecasting and RL tracked separately; weekly cutoff, release, costs, next-week evaluation",
     "not part of this pilot; protocol designed in 12E (S4)", "12E"),
    ("independent_evidence", "losses recomputed from predictions and labels, rows linked to governed inputs, content "
                             "reconciled, negative tests per guarantee, fresh vs retained queries labelled",
     "df_adequacy_verify recomputes every loss from arrays and compares with terminal/warehouse content", "verify receipts"),
]


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def receptive_field(kernel: int, dilations: list) -> int:
    return 1 + sum((kernel - 1) * d for d in dilations)


def conv_dilations(window: int, kernel: int = 3) -> list:
    """Dilations 1, 2, 4, ... until the receptive field covers the window (the variable-depth
    configuration of S1, kept as separate evidence)."""
    dilations = [1]
    while receptive_field(kernel, dilations) < window:
        dilations.append(dilations[-1] * 2)
    return dilations


def conv_dilations_fixed(kernel: int = 3) -> list:
    """The ONE CNN graph of the factorial (T1): enough receptive field for the largest context."""
    return conv_dilations(max(CONTEXTS), kernel)


def boundaries(window: int, train_length: int, horizon: int = HORIZON) -> dict:
    """Shared decision rows (T1): validation and test rows are the SAME for every W, model and L;
    the purge between splits is the largest allowed support + h, so no cell's consumption reaches
    the next split; training histories are nested, ending at one cutoff and starting L rows before."""
    purge = max(CONTEXTS) + horizon
    test_start, test_end = TEST
    val_end = test_start - purge
    val_start = val_end - INNER_VALIDATION_ROWS
    train_end = val_start - purge
    train_start = train_end - train_length
    if train_start - (window - 1) < 0:
        raise ValueError(f"W={window}, L={train_length}: the first training row would consume rows before 0")
    return {"train": [train_start, train_end], "validation": [val_start, val_end], "test": [test_start, test_end],
            "purge": purge, "cutoff": train_end, "consumed_first_row": train_start - (window - 1)}


def geometry_identity() -> dict:
    """The decision-row anchors every cell must share (asserted at preparation)."""
    b = boundaries(min(CONTEXTS), min(TRAIN_LENGTHS))
    return {"validation": tuple(b["validation"]), "test": tuple(b["test"]), "cutoff": b["cutoff"], "purge": b["purge"]}


def unit_metadata(bank: Path, unit: str) -> dict:
    rec = json.loads((Path(bank) / unit / "UNIT.json").read_text())
    pv = rec["clean_params"]["per_variable"][0]
    return {"unit": unit, "period": pv["period"], "amplitude": pv["amplitude"], "phase": pv["phase"],
            "noise_sd": rec["noise_model"]["scale_per_variable"][0], "n": rec["n_samples"],
            "digests": {"observed": rec["digests"]["observed_signal"], "clean": rec["digests"]["clean_signal"]}}


def factorial_models(design: dict) -> list:
    """The models of the factorial: separate-evidence configurations (variable-depth CNN) excluded."""
    models = design["models"]
    if isinstance(models, dict):
        return [m for m, spec in models.items() if not str(spec.get("role", "")).startswith("separate evidence")]
    return list(models)


def cells(design: dict) -> list:
    out = []
    for unit in design["units"]:
        for task in design["tasks"]:
            for model in factorial_models(design):
                for window in design["contexts"]:
                    for length in design["train_lengths"]:
                        for seed in design["seeds"]:
                            out.append({"cell_id": f"{unit.split('__')[-1]}__{task}__{model}__W{window}__L{length}__s{seed}",
                                        "unit": unit, "task": task, "model": model, "window": window,
                                        "train_length": length, "seed": seed})
    return out


def build(bank: Path) -> dict:
    units = {u: unit_metadata(bank, u) for u in UNITS}
    contexts = {}
    for u, meta in units.items():
        for w in CONTEXTS:
            contexts[f"{u.split('__')[-1]}__W{w}"] = {"W": w, "W_over_P": w / meta["period"],
                                                       "consumed_span_over_P": (w - 1) / meta["period"],
                                                       "covers_two_periods": (w - 1) / meta["period"] >= 2.0,
                                                       "conv_dilations_fixed": conv_dilations_fixed(),
                                                       "conv_receptive_field": receptive_field(3, conv_dilations_fixed()),
                                                       "conv_effective_support": min(w, receptive_field(3, conv_dilations_fixed())),
                                                       "variable_depth_layers_S1": len(conv_dilations(w))}
    doc = {"schema": DESIGN_SCHEMA, "purpose": "DEVELOPMENT_ADEQUACY_PILOT", "classification": "NON_GOVERNING",
           "questions": {"a_raw_task": "can the learner predict the raw task?", "b_context": "how much history does it need?",
                         "c_representation": "NOT in this pilot", "d_transfer": "NOT in this pilot"},
           "units": list(UNITS), "unit_metadata": units, "tasks": TASKS, "models": MODELS, "contexts": list(CONTEXTS),
           "context_coverage": contexts, "train_lengths": list(TRAIN_LENGTHS), "horizon": HORIZON,
           "horizon_semantics": "one-step direct; one-period and multi-period tasks are a separate staged proposal",
           "seeds": list(SEEDS), "n": N, "boundaries": {f"W{w}__L{l}": boundaries(w, l) for w in CONTEXTS for l in TRAIN_LENGTHS},
           "geometry": {"shared_decision_rows": {k: list(v) if isinstance(v, tuple) else v for k, v in geometry_identity().items()},
                        "rule": "validation and test rows identical across W, model and L; purge = max(W) + h; nested training "
                                "histories ending at one cutoff; a cell whose prepared rows differ refuses"},
           "exposure_ledger": {"test_rows": list(TEST),
                               "history": ["adequacy-v1 cost pilot (S3) SCORED the test rows of three cells: DEVELOPMENT diagnostic, "
                                           "not an untouched confirmation"],
                               "rule": "cost pilots have no test access (the accessor fails); cells score the test descriptively; "
                                       "any proposed model/context is selected from inner validation only; independent final "
                                       "confirmation is future work"},
           "cnn_graph": {"dilations": conv_dilations_fixed(), "receptive_field": receptive_field(3, conv_dilations_fixed()),
                         "padding": "causal", "fixed_across_W": True},
           "training": TRAINING, "criteria": CRITERIA, "budget": BUDGET,
           "review_table": [{"requirement": r, "executable_check": c, "evidence": e, "status": "NOT_TESTED",
                             "owner": "Satoshi", "next_action": "measure in S3; Musashi reviews"} for r, c, e, _ in REVIEW_ROWS],
           "preserved": "utildev-v1 (18 non-advances, 18 inconclusive) and utilinst-v1 stay at their original scope",
           "execution": "NONE here; df_adequacy_run executes it governed under the budget",
           "design_sha256": ""}
    doc["cells_total"] = len(cells(doc))
    doc["design_sha256"] = sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = build(args.bank)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; a design is never written over")
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"design_sha256": doc["design_sha256"], "cells": doc["cells_total"],
                      "context_coverage": {k: (round(v["consumed_span_over_P"], 3), v["conv_receptive_field"]) for k, v in doc["context_coverage"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
