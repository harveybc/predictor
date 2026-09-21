#!/usr/bin/env python3
"""Phase 2 of the E1 diagnostic — three questions frozen apart, sealed before anything runs.

    Q1  calendar information      does the wall-clock position (hour, weekday as sin/cos) add information
                                  the model can use at 60 minutes ahead?
    Q2  daily context             does history beyond the hour add information — and is that separable
                                  from the deeper core a longer window silently brings?
    Q3  training volume           does more history (rows), with the evaluation FIXED, change the error —
                                  counting rows, windows, labels and repeated exposure apart?

What stays fixed everywhere the question allows it: the architecture family (ARCH-A branches, tcn_w
core), the training recipe (MAE loss, Adam, lr 3e-3, batch 64, patience 3, restore best, 4 000
updates — the household CONTINUITY reference, which selects no trading loss), the forecast task
(W60_h60, one household), the validation population (the successor run's 10 020 evaluation
origins), the scaler grain, and three paired seeds.

The confound the order names, measured at seal time by building the models: `core_dilations(W)`
derives the core's depth from the window, so W=60 gives 5 blocks and 8 127 parameters while W=1440
gives 10 blocks and 12 047. A longer window is therefore information AND model. Q2 separates them
factorially instead of pretending otherwise:

    Q2.a  daily LAG channel      + one input channel, the target at t+h-1440 (one day before the label),
                                 known at t because t+h-1440 <= t; same window, same depth. This is
                                 the information-only arm: one channel, its parameter delta measured.
    Q2.b  long window, own depth W=1440 with the depth the rule gives (10 blocks): information + model.
    Q2.c  long window, fixed depthW=1440 with the core CLAMPED to W=60's five dilations: the history
                                 is present in the window but beyond the core's reach — a null for
                                 "seeing more rows" without "reaching them".
    Q2.d  short window, deep coreW=60 with ten blocks: the model change alone, no extra history.

Q1 adds four calendar channels (+1 788 parameters at W=60): its control is Q1.b, the same four
channels filled with a seed-fixed permutation of themselves across rows — same capacity, no
calendar information. Q3 grows history BACKWARDS from the same DEV validation week, fixing the
evaluation, and reports unique raw rows, distinct windows, labels and mean exposure per row apart;
the train-only scaler is re-fitted per volume arm and the change of its parameters is reported, so
a changed objective in raw units is never hidden inside a "volume-only" comparison.

Nothing here trains. The cost pilot and the campaign run only after this design is reviewed.

    python tools/df_e1_phase2_design.py --seal DESIGN.json [--source-run ROOT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "df_e1_phase2_design.v1"
SEEDS = (1, 2, 3)
DAY = 1440


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _params(model) -> int:
    return int(sum(int(w.shape.num_elements()) for w in model.trainable_weights))


def capacity_table(assignment: list, p: int, j: int) -> dict:
    """Depth, reach and trainable parameters for every (window, channels) this phase touches — measured."""
    P = _module("df_e1_pilot")
    out = {}
    for W in (60, 1440):
        out[f"W{W}_p{p}"] = {"window": W, "channels": p, "dilations": P.core_dilations(W),
                             "blocks": len(P.core_dilations(W)), "core_reach": P.core_reach(W, "tcn_w"),
                             "model_reach": P.model_reach(W, "tcn_w"),
                             "parameters": _params(P._model_for_target(assignment, W, p, j, 1, core="tcn_w"))}
    cal = assignment + [max(assignment)+1]*4
    out["W60_p11_calendar"] = {"window": 60, "channels": p+4, "dilations": P.core_dilations(60),
                               "parameters": _params(P._model_for_target(cal, 60, p+4, j, 1, core="tcn_w"))}
    lag = assignment + [max(assignment)+1]
    out["W60_p8_daily_lag"] = {"window": 60, "channels": p+1, "dilations": P.core_dilations(60),
                               "parameters": _params(P._model_for_target(lag, 60, p+1, j, 1, core="tcn_w"))}
    return out


def daily_lag_availability(h: int, lag: int = DAY) -> dict:
    """The identity of the past sample the lag channel carries, and why it is available at t."""
    return {"channel": f"y(t + h - {lag})", "horizon": h, "lag": lag,
            "sample_index_relative_to_origin": h-lag,
            "available_at_decision": (h-lag) <= 0,
            "reading": f"the label is y(t+{h}); one day before it is row t+{h}-{lag} = t{h-lag:+d}, which is "
                       f"{'at or before' if (h-lag) <= 0 else 'AFTER'} the origin t",
            "delayed_observation_rule": "if the panel published a row later than its label, the lag would "
                                        "have to be taken from the PUBLICATION time; this archive declares "
                                        "no delay (UNKNOWN), so the wall-clock label is used and said so"}


def volume_counts(train_origins: int, window: int, horizon: int, rows_in_slice: int) -> dict:
    """Rows, windows, labels and exposure counted apart — never 'windows' presented as 'observations'."""
    exposure = window*train_origins/max(1, rows_in_slice)
    return {"unique_raw_rows_in_train_span": rows_in_slice, "distinct_windows": train_origins,
            "labels": train_origins, "mean_exposures_per_raw_row": exposure,
            "reading": "consecutive windows overlap by W-1 rows; a row inside the span is seen up to W times; "
                       "distinct windows are NOT independent observations"}


def seal(source_run: Path) -> dict:
    E = _module("df_mod_e0")
    P1 = _module("df_e1_phase1")
    B = _module("df_benchmark_contract")
    source = Path(source_run)
    src = json.loads((source/"DESIGN.json").read_text())
    data_json = json.loads((source/"DATA.json").read_text())
    assignment = src["graph"]["assignment"]
    p, j = len(data_json["input_columns"]), int(data_json["target_channel"])
    caps = capacity_table(assignment, p, j)
    train_rows = int(data_json["slice_rows"][1]-data_json["slice_rows"][0])
    base_train = int(data_json["enumerator"]["train"]["admissible"])
    recipe = {"loss": "mae", "optimizer": "adam", "learning_rate": 0.003, "batch": 64, "monitor": "val_mae",
              "patience_epochs": 3, "restore_best": True, "max_updates": 4000,
              "role": "HOUSEHOLD CONTINUITY REFERENCE (MAE+Adam); it selects no trading loss — FIN-LOSS-OPT does"}
    ours = B.household_ours()
    # the typed block (RP67): validated, digested and decided from fields; the runner's require()+bind() refuse otherwise
    contract = ours.to_design_block(comparability={**B.decide(ours, B.gasparin_2019()), "against": "gasparin_2019"})
    design = {
        "schema": SCHEMA, "purpose": "E1_DIAGNOSTIC_PHASE_2", "phase": "DEVELOPMENT", "state": "SEALED_NOT_EXECUTED",
        "source_run": {"root": str(source), "design_sha256": src["design_sha256"],
                       "data_sha256": data_json["data_sha256"], "panel_sha256": data_json["panel_sha256"],
                       "evaluation_origins": int(data_json["enumerator"]["validation"]["admissible"])},
        "held_constant": ["architecture family (ARCH-A branches, tcn_w core) except where the question moves it",
                          "training recipe (continuity reference)", "forecast task W60_h60", "validation population",
                          "scaler grain (train windows)", "three paired seeds", "no pretraining"],
        "recipe": recipe, "benchmark_contract": contract,
        "capacity_measured_at_seal": caps,
        "confound": {"statement": "core_dilations(W) derives depth from the window: W=1440 gives "
                                  f"{caps['W1440_p7']['blocks']} blocks and {caps['W1440_p7']['parameters']} parameters "
                                  f"against {caps['W60_p7']['blocks']} and {caps['W60_p7']['parameters']} at W=60; a longer "
                                  "window is information AND model, and Q2 separates them factorially"},
        "questions": {
            "Q1_calendar": {
                "arms": {"Q1.a_calendar": {"channels": "+4: hour and weekday sin/cos from the row's own label "
                                                       "(tools/df_e1_calendar.py)", "window": 60,
                                           "parameters": caps["W60_p11_calendar"]["parameters"]},
                         "Q1.b_permuted_calendar_control": {"channels": "+4: the same four channels, permuted across "
                                                            "rows by a seed-fixed permutation", "window": 60,
                                                            "parameters": caps["W60_p11_calendar"]["parameters"],
                                                            "why": "same capacity, no calendar information"},
                         "Q1.0_baseline": {"channels": "the 7 declared", "window": 60,
                                           "parameters": caps["W60_p7"]["parameters"]}},
                "calendar_declaration": {"clock": "NAIVE_WALL_CLOCK", "timezone": "UNKNOWN (local French time presumed)",
                                         "dst": "none inside the DEV slice (2009-08-23..09-27, measured in RP59); "
                                                "reported per slice, never assumed",
                                         "known_at_decision": "each row's features read that row's label only",
                                         "publication": "static archive, no revision"},
                "acceptance": ["prefix invariance through tools/df_e1_calendar.py (future values and future "
                               "timestamps change nothing before them)", "a shift-minus-one calendar control fails the same test",
                               "numeric/missing/unparseable/absent timestamp refused", "parameter delta measured and matched by Q1.b"]},
            "Q2_daily_context": {
                "arms": {"Q2.a_daily_lag_channel": {"window": 60, "channels": "+1: y(t+h-1440)",
                                                    "parameters": caps["W60_p8_daily_lag"]["parameters"],
                                                    "availability": daily_lag_availability(60)},
                         "Q2.b_long_window_own_depth": {"window": 1440, "dilations": caps["W1440_p7"]["dilations"],
                                                        "parameters": caps["W1440_p7"]["parameters"],
                                                        "reach": caps["W1440_p7"]["model_reach"]},
                         "Q2.c_long_window_fixed_depth": {"window": 1440, "dilations": caps["W60_p7"]["dilations"],
                                                          "reach": "60 of 1440: the history is in the window but beyond "
                                                                   "the core's reach — a null for seeing without reaching",
                                                          "parameters": "measured at pilot (a clamped-depth builder is a "
                                                                        "runner change gated by its acceptance test)"},
                         "Q2.d_short_window_deep_core": {"window": 60, "dilations": caps["W1440_p7"]["dilations"],
                                                         "reach": "capped at 60 by the window",
                                                         "parameters": "measured at pilot (same builder change)"},
                         "Q2.0_baseline": {"window": 60, "parameters": caps["W60_p7"]["parameters"]}},
                "train_side_evidence_for_a_daily_period": {
                    "spectral_share_24h": 0.0674807313099007, "spectral_share_12h": 0.10105418574133604,
                    "source": "E1_TASKS uci_235 train_only_periodicities (train span only)",
                    "autocorrelation_1440_min": 0.31718580057541035, "autocorrelation_60_min": 0.4030024231300615,
                    "source_autocorr": "RP59 data audit, consumed train rows",
                    "reading": "a daily component exists in train but is weaker than the hourly one; 24 h is a "
                               "hypothesis to test, not an assumed period"},
                "acceptance": ["the lag channel's sample identity t+h-1440 <= t is asserted for the horizon",
                               "prefix invariance of the lag channel (a future perturbation leaves it unchanged)",
                               "the clamped-depth builder reproduces W=60's dilations at W=1440 and its reach is measured",
                               "parameter counts of every arm are recorded at seal or at pilot, never assumed"]},
            "Q3_volume": {
                "arms": {f"Q3.{k}_train_{days}d": {"train_days": days, "evaluation": "FIXED: the same 7-day DEV validation week",
                                                   "history": "grown BACKWARDS from the current DEV train span"}
                         for k, days in enumerate((28, 56, 112), start=1)},
                "counting": volume_counts(base_train, 60, 60, train_rows),
                "scaler_rule": "train-only scaler re-fitted per volume arm; its mean/sd reported per arm so a changed "
                               "raw-unit objective is never hidden inside a volume-only comparison; a Huber delta in "
                               "standardised units would change with the scaler and is NOT used in this phase",
                "prior_history": "the family train span holds 1 452 681 rows before the DEV slice; 112 days need "
                                 "161 280, available; if a later arm needed more than the span, a successor task "
                                 "and common evaluation are declared before any score",
                "acceptance": ["rows, windows, labels and exposure counted apart", "the evaluation origins are byte-identical "
                               "across volume arms", "the scaler of each arm is recorded and differs as stated"]},
        },
        "training_adequacy": {
            "observed_in_phase_1_and_the_factorial": "7 of 12 factorial fits and 4 of 9 phase-1 fits stopped at 4 000 updates",
            "learning_curves": "train and validation loss per epoch, saved per cell", "validation_cadence": "every epoch",
            "patience": 3, "restore_check": "restored predictions must reproduce the best validation MAE (as df_e1_huber does)",
            "ceiling_policy": "a fit whose best epoch is within one epoch of the ceiling is CENSORED; a pre-declared second "
                              "tier (8 000 updates) may be run for the SAME cells only if declared before the first tier's "
                              "scores are read; more windows under a fixed budget are not more training",
            "claims": "finite-budget comparisons only; no convergence claim from a ceiling"},
        "cells": [{"cell_id": f"{arm}_s{s}", "question": q, "arm": arm, "seed": s}
                  for q, block in (("Q1", ("Q1.0_baseline", "Q1.a_calendar", "Q1.b_permuted_calendar_control")),
                                   ("Q2", ("Q2.a_daily_lag_channel", "Q2.b_long_window_own_depth",
                                           "Q2.c_long_window_fixed_depth", "Q2.d_short_window_deep_core")),
                                   ("Q3", ("Q3.2_train_56d", "Q3.3_train_112d")))
                  for arm in block for s in SEEDS],
        "pilot_cost_plan": {"pilot_updates": 200, "per_arm": "one 200-update pilot per arm on seed 1 measures seconds per "
                                                              "update; W=1440 arms are projected from their own pilot, never from W=60",
                            "reference": "phase-1 measured 0.130 s/update at W=60 for the core recipe (omega)",
                            "ceiling": "14 400 aggregate CPU seconds per round including closure; reserve 2 000 for closure",
                            "stop_rule": "BUDGET_LIMITED before any outcome if projection exceeds the ceiling; no cell cut after a score"},
        "resource_assignment": {"blocks": "all arms of one seed on ONE host (seed = block), so optimiser and feature choice are "
                                          "never confounded with CPU; seeds 1/2/3 -> omega/dragon/gamma only if every host "
                                          "runs the same wrapper and environment (workers currently run no crispdm-run scope: "
                                          "fix that first or run all seeds on omega)",
                                "no_padding": "no cell is repeated merely to occupy a machine"},
        "acceptance_before_runner_changes": ["tests/test_df_e1_calendar.py", "tests/test_df_e1_phase2_acceptance.py",
                                             "the causal battery's negative controls stay DETECTED",
                                             "governance before consumption, observed update counts, per-cell artifacts, "
                                             "outbox and warehouse content checks unchanged from phase 1"],
        "reading_rules": ["three seeds on one task are development evidence", "R0/R1/R2 return only after Q1-Q3 with the "
                          "adequate objective common to every arm", "no pretraining moved in this phase"],
    }
    design["design_sha256"] = E.sha_obj(design)
    return design


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", type=Path, required=True)
    ap.add_argument("--source-run", type=Path,
                    default=Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser())
    a = ap.parse_args(argv)
    design = seal(a.source_run)
    _module("df_d3_campaign").write_once(a.seal, design)
    print(json.dumps({"design_sha256": design["design_sha256"], "cells": len(design["cells"]),
                      "capacity": {k: v.get("parameters") for k, v in design["capacity_measured_at_seal"].items()},
                      "state": design["state"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
