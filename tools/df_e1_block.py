#!/usr/bin/env python3
"""RP68/RP72: the block runner — the ACTUAL feature paths, enumeration, scaling, cadence and governance.

A block is a set of paired cells (arm x seed) on the household DEV task (W60/h60, the successor run's
rows), sealed before anything runs and executed under governance, one campaign and delivery per unit,
one terminal with artifacts per unit. Everything the phase-2 design declares happens HERE, in code
that the tests drive through this consumer:

  features       base (the 7 declared channels), calendar (+4: hour/weekday sin-cos from each row's own
                 label, tools/df_e1_calendar.py), randomised_calendar_control (the same 4 channels from
                 the label plus a per-row hashed random offset: same capacity, deterministic, prefix-
                 stable, chance association measured), daily_lag (+1: y(t+h-1440), a row at or before
                 t, read from the panel rows before the slice)
  windows        W=60 or W=1440 over the SAME origins: the panel is read with a left pad so a long
                 window never withdraws an origin the short one keeps
  crop           `crop=60` cuts the raw input to its last 60 rows BEFORE the extractor: an exact
                 information null for the long window (same weights, same padding, same 60 rows).
                 The clamped-dilation long model is NOT a null: it reaches 67 samples (measured) and is
                 named local_support_67 — extra context, declared as such
  enumeration    admissible origins from ROW IDENTITIES: grid-consecutive support, finite inputs on the
                 window rows, finite label at t+h, finite lag where the arm uses it; one COMMON
                 admissible evaluation mask across the block's arms, derived before any score
  scaling        the COMMON train-only scaler of the source run (28 d, W60 windows) for every arm and
                 every volume tier; calendar channels are on the circle and take mean 0 / sd 1; the lag
                 takes the target's scaler. One evaluation sigma for the whole block
  cadence        validation every `validate_every` OBSERVED optimizer updates, patience counted in
                 validation events, restore best, both fixed across arms; reaching the update ceiling
                 is CENSORED wherever the best checkpoint fell
  volume         train history grown BACKWARDS from the fixed DEV validation week; counts of unique
                 support rows, train-only rows, targets, windows and exposures from the identities
  cost pilot     on a declared subset INSIDE train (validation = the last 7 train days, purged); it never
                 reads the DEV validation; it measures seconds/update, seconds/validation event, peak RSS
  closure        arrays vs source rows, reload parity, terminal artifacts vs warehouse, three baselines on
                 identical rows (persistence, daily seasonal, train-only constant); no fit at closure

    seal     python tools/df_e1_block.py seal --block DEV_MATCHED --out DESIGN.json
    prepare  python tools/df_e1_block.py prepare --root ROOT --run-id ID --api-key-file KEY
    pilot    python tools/df_e1_block.py pilot   --root ROOT --run-id ID --api-key-file KEY
    execute  python tools/df_e1_block.py execute --root ROOT --run-id ID --api-key-file KEY
    close    python tools/df_e1_block.py close   --root ROOT --warehouse-token-file T
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "df_e1_block_design.v1"
SCHEMA_DATA = "df_e1_block_data.v1"
#: The origin policies a design may declare. A preparation is read against the policy its OWN
#: design carries; "UNDECLARED" is not a policy, it is the absence of one (designs sealed before
#: the intersection existed carry no key at all) and is refused the moment the held origins differ.
TRAIN_POPULATIONS = ("COMMON_INTERSECTION", "PER_ARM_ADMISSIBLE")
DAY = 1440
W0, H0 = 60, 60
SEEDS = (1, 2, 3)
LAKE, RESOURCE = "public_panels", "uci_235_individual_household_power/panel.parquet"
SOURCE_RUN = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()

# every arm the phase can run; a block picks some of them. `features`, `window`, `dilations`, `crop`
# and `train_days` are the ONLY things an arm may move; the recipe, rows, scaler and seeds are the block's
ARMS = {
    "modular_w60":            {"family": "modular", "window": 60,   "features": "base"},
    "gru_adapted_w60":        {"family": "gru",     "window": 60,   "features": "base"},
    "calendar":               {"family": "modular", "window": 60,   "features": "calendar"},
    "gru_calendar_w60":       {"family": "gru",     "window": 60,   "features": "calendar",
                               "role": "the adapted GRU with the same 4 calendar channels appended to its 7 inputs (+600 parameters: 3 x 50 x 4)"},
    "randomised_calendar_control": {"family": "modular", "window": 60, "features": "randomised_calendar",
                               "role": "CAPACITY_CONTROL: the same 4 channels built from each row's label plus a per-row random offset "
                                       "drawn from a hash of (seed, panel row id): deterministic, prefix-stable, no calendar information; "
                                       "its finite-sample association with the true clock is measured, not assumed zero"},
    "daily_lag":              {"family": "modular", "window": 60,   "features": "daily_lag"},
    "long_window_own_depth":  {"family": "modular", "window": 1440, "features": "base"},
    "long_window_crop60":     {"family": "modular", "window": 1440, "features": "base", "crop": 60,
                               "role": "EXACT_INFORMATION_NULL: the raw input is cropped to its last 60 rows before the extractor"},
    "long_window_local_support_67": {"family": "modular", "window": 1440, "features": "base", "dilations": [1, 2, 4, 8, 16],
                               "role": "EXTRA_CONTEXT_67_SAMPLES (measured): NOT a null; the clamped core still reaches "
                                       "branch 5 + core 63 - 1 = 67 raw samples"},
    "short_window_deep_core": {"family": "modular", "window": 60,   "features": "base", "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]},
    "volume_56d":             {"family": "modular", "window": 60,   "features": "base", "train_days": 56},
    "volume_112d":            {"family": "modular", "window": 60,   "features": "base", "train_days": 112},
}
BLOCKS = {
    "DEV_MATCHED": {"arms": ["modular_w60", "gru_adapted_w60"], "question": "matched architecture comparison: the modular continuity "
                                                                       "model against the adapted literature GRU on identical rows, "
                                                                       "inputs, split, transform and monitoring schedule"},
    "Q1_CALENDAR": {"arms": ["calendar", "randomised_calendar_control"], "question": "does the wall-clock position add information at h60? "
                                                                          "baseline modular_w60 reused from DEV_MATCHED when its contract matches"},
    "Q2_CONTEXT":  {"arms": ["modular_w60", "daily_lag", "long_window_own_depth", "long_window_crop60", "long_window_local_support_67",
                             "short_window_deep_core"], "question": "information beyond the hour, separated from depth and from padding",
                    "train_population": "COMMON_INTERSECTION",
                    "why": "a long window or a lag withdraws the train origins whose support reaches a non-finite padded row; every arm of "
                           "this block, the baseline included, trains on the SAME origins so context is never conflated with volume"},
    "Q3_VOLUME":   {"arms": ["volume_56d", "volume_112d"], "question": "more history with the evaluation, scaler and cadence FIXED"},
    "Q2_CONTEXT_BOUNDED": {"arms": ["modular_w60", "daily_lag", "long_window_crop60", "short_window_deep_core"],
                    "question": "the part of the Q2_CONTEXT question a 30 GiB host can fit: does a causal daily-lag channel, or a "
                                "deeper dilated core at W60, change error against the W60 baseline and against the W1440 exact-crop "
                                "information null? The two W1440 FULL-DEPTH arms of Q2_CONTEXT (long_window_own_depth, "
                                "long_window_local_support_67) are NOT in this block, so this block does NOT separate context from "
                                "depth and does NOT answer the Q2_CONTEXT question: it measures the three arms whose retained cost "
                                "pilots fit this host, and leaves the long-window treatment unmeasured",
                    "train_population": "COMMON_INTERSECTION",
                    "why": "the W1440 crop arm withdraws the train origins whose support reaches a padded row and the lag arm those "
                           "whose daily lookup is non-finite; every arm of this block, the baseline included, trains on the SAME "
                           "origins so an input difference is never conflated with volume",
                    "primary_factorial": ["modular_w60", "daily_lag"],
                    "secondary_control": ["long_window_crop60"],
                    "informed_by": "RESTRICTED BEFORE ANY SCORE from the arms of Q2_CONTEXT v1 (design "
                                   "6d1aaecaf27c581c709a745a4f976c2e9dcc05594815b2ee1a9747595f4398b1, state "
                                   "BUDGET_LIMITED_BEFORE_ANY_OUTCOME, zero cells fitted), on the two RETAINED cost pilots' own "
                                   "measurements and on nothing else: long_window_own_depth 4.165 CPU s per update and 8 458 399 744 B "
                                   "peak RSS, long_window_local_support_67 4.443 CPU s per update and 10 279 276 544 B peak RSS, i.e. "
                                   "17 507 s and 18 406 s per cell at the 4 000-update ceiling and a resident set that would take this "
                                   "host's memory away from its owner; the three arms kept cost 287 s, 312 s and 488 s per cell at the "
                                   "ceiling with peak RSS under 1 GiB. The restriction is a resource declaration made before any score "
                                   "of any cell existed, never a removal after a score was seen (reading rule 3); no arm, seed, "
                                   "recipe, scaler, row, cadence or ceiling of Q2_CONTEXT v1 is otherwise changed"},
    "CONTEXT_DAILY_LAG": {"arms": ["modular_w60", "daily_lag"],
                          "question": "does a causal daily-lag channel y(t+h-1440) add predictive information to the W60 receiver, without paying for "
                                      "W1440 or changing receiver depth? (RP87; scoped as THIS feature addition, not an abstract information-only effect)",
                          "train_population": "COMMON_INTERSECTION",
                          "why": "the lag withdraws the train origins whose lag row is non-finite; both arms train on the SAME origins",
                          "primary_factorial": ["modular_w60", "daily_lag"],
                          "inherited_control": {"arm": "long_window_crop60", "inherits_from": "modular_w60",
                                                "declaration": "the exact-crop control is the SAME computation as modular_w60 (the W1440 input cropped to its last 60 rows "
                                                               "before the extractor: identical initial weights, identical tensors, identical training prefix — proved by "
                                                               "tests/test_df_e1_block.py::test_RP87_the_exact_crop_control_is_training_equivalent...); it inherits the "
                                                               "baseline's measurement and is NOT fitted again",
                                                "capacity": "8 127 parameters, equal to modular_w60; the lag arm has 8 208 (+81: one more input column in the group-0 detector, one head row, one skip row; measured at seal)"},
                          "lag_declaration": {"channel": "y(t + h - 1440) at origin t, i.e. the target's value at the panel row labelled 1440 minutes before the LABEL row t+h",
                                              "refers_to_timestamp": "label(t) + h - 1440 minutes = label(t) - 1380 minutes: a row 23 hours BEFORE the decision",
                                              "availability": "at every decision t the row t-1380 is in the past by construction (h=60 <= 1440); the runner reads it from the "
                                                              "padded panel rows and withdraws the origin when it is non-finite; no centered interpolation, no future fill",
                                              "not_live_evidence": "the archive declares no publication delay; UNKNOWN is not zero delay (stated)"},
                          "informed_by": "prior DEV results (RP72, RP79); NOT confirmatory", "tier": "TIER2"},
    "ARCH_X_CALENDAR": {"arms": ["modular_w60", "gru_adapted_w60", "calendar", "gru_calendar_w60", "randomised_calendar_control"],
                        "question": "architecture {modular, adapted GRU} x inputs {original 7, original + real calendar}, three paired seeds "
                                    "(12 cells), plus the modular randomised-calendar control (3 cells): an architecture difference separated "
                                    "from an input-information difference, with the equal-capacity calendar check retained",
                        "primary_factorial": ["modular_w60", "gru_adapted_w60", "calendar", "gru_calendar_w60"],
                        "secondary_control": ["randomised_calendar_control"],
                        "informed_by": "prior DEV results (DEV_MATCHED, Q1, Q3 of RP72) and the stopping behaviour they showed; NOT confirmatory",
                        "tier": "TIER2"},
}
RECIPE = {"loss": "mae", "optimizer": "adam", "learning_rate": 0.003, "batch": 64, "max_updates": 4000,
          "validate_every_updates": 200, "patience_events": 3, "restore_best": True, "min_delta": 0.0,
          "monitor": "validation MAE in scaled units (equals MAE_z with the common sigma)",
          "checkpoint_opportunities": 20,
          "role": "HOUSEHOLD CONTINUITY REFERENCE (MAE+Adam) with validation in OBSERVED updates; it selects no trading loss"}
PILOT = {"max_updates": 200, "validate_every_updates": 50, "patience_events": 3,
         "population": "TRAIN_SUBSET: trains on train origins before the last 7 train days (purged W+h); validates on the last 7 "
                       "train days; the DEV validation is never read by a pilot"}
LIMITS = {"child_cpu_seconds": 3600, "child_wall_seconds": 4800, "parallel_children": 3, "campaign_cpu_seconds": 14400,
          "closure_reserve_seconds": 2000}

RECIPE_TIER2 = {**RECIPE, "patience_events": 10,
                "role": "HOUSEHOLD DEV SENSITIVITY TIER (RP79): the same cadence (every 200 observed updates), ceiling (4 000) and monitor; "
                        "patience 10 events = 2 000 non-improving updates, approximately the former three passes (~1 881); no per-arm "
                        "patience or budget tuning; no promise that longer patience improves error",
                "checkpoint_opportunities": 20}
TIERS = {"TIER1": {"recipe": RECIPE, "why": "RP66-RP73 blocks: patience 3 events (600 non-improving updates)"},
         "TIER2": {"recipe": RECIPE_TIER2, "why": "RP79: patience 10 events (2 000 non-improving updates); informed by prior DEV behaviour"}}


class BlockRefusal(SystemExit):
    pass


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj) -> str:
    return _module("df_mod_e0").sha_obj(obj)


def write(path: Path, value):
    _module("df_d3_campaign").write_once(Path(path), value)


# --- the sealed design ---------------------------------------------------------------------------------------

def arm_spec(arm: str) -> dict:
    if arm not in ARMS:
        raise BlockRefusal(f"REFUSED: unknown arm {arm!r}")
    return {"arm": arm, "crop": None, "dilations": None, "train_days": 28, "role": "ARM", **ARMS[arm]}


def seal(block: str, *, source_run: Path = SOURCE_RUN, seeds=SEEDS, reuse: dict | None = None, contract=None,
         limits: dict | None = None, recipe: dict | None = None, tier: str | None = None) -> dict:
    """The block, sealed by content: arms, seeds, recipe, rows, scaler identity and the benchmark contract.
    `contract` replaces the household registry contract ONLY for tests on synthetic panels (recorded as such)."""
    if block not in BLOCKS:
        raise BlockRefusal(f"REFUSED: unknown block {block!r}")
    if recipe is None and BLOCKS[block].get("tier"):
        recipe, tier = TIERS[BLOCKS[block]["tier"]]["recipe"], BLOCKS[block]["tier"] + ": " + TIERS[BLOCKS[block]["tier"]]["why"]
    B = _module("df_benchmark_contract")
    src_design = json.loads((Path(source_run)/"DESIGN.json").read_text())
    data_json = json.loads((Path(source_run)/"DATA.json").read_text())
    ours = contract if contract is not None else B.household_ours()
    arms = [arm_spec(a) for a in BLOCKS[block]["arms"]]
    if block == "DEV_MATCHED":
        comp = {**B.decide(ours, B.gasparin_2019()), "against": "gasparin_2019",
                "rule": "the adapted GRU is measured under OUR contract; Table 5 stays in the source notes"}
    else:
        contrast = ("permitted_inputs" if block in ("Q1_CALENDAR", "Q2_CONTEXT", "Q2_CONTEXT_BOUNDED") else "split_rule")
        theirs = B.replace(ours, task_id=f"{ours.task_id}.{block}", varying_factors=(contrast,),
                           estimand=BLOCKS[block]["question"])
        ours_c = B.replace(ours, varying_factors=(contrast,), estimand=BLOCKS[block]["question"])
        comp = {**B.decide(ours_c, theirs), "against": "the block's own baseline under one declared estimand"}
        ours = ours_c
    pad = max([a["window"]-W0 for a in arms] + [DAY-H0 if any(a["features"] == "daily_lag" for a in arms) else 0] + [0])
    max_days = max(a["train_days"] for a in arms)
    lo, hi = data_json["slice_rows"]
    train_end = lo + 28*DAY
    if hi != train_end + 7*DAY:
        raise BlockRefusal("REFUSED: the source slice is not 28 d train + 7 d validation")
    design = {
        "schema": SCHEMA, "purpose": f"E1_BLOCK_{block}", "block": block, "phase": "DEVELOPMENT",
        "state": "SEALED_NOT_EXECUTED", "question": BLOCKS[block]["question"],
        "train_population": BLOCKS[block].get("train_population", "PER_ARM_ADMISSIBLE"),
        "train_population_why": BLOCKS[block].get("why", "every arm of this block admits the same origins by construction (W60, 28 d, base inputs)"),
        "source_run": {"root": str(source_run), "design_sha256": src_design["design_sha256"],
                       "data_sha256": data_json["data_sha256"], "panel_sha256": data_json["panel_sha256"],
                       "slice_rows": [lo, hi], "train_end_row": train_end, "evaluation_origins": data_json["enumerator"]["validation"]["admissible"],
                       "input_columns": data_json["input_columns"], "target_channel": data_json["target_channel"],
                       "scaler": data_json["scaler"], "graph_assignment": src_design["graph"]["assignment"]},
        "rows": {"lo": lo - max_days*DAY + 28*DAY - pad, "pad_rows": pad, "widest_train_days": max_days, "train_end": train_end, "hi": hi,
                 "reading": "the panel is read from lo (pad + the widest volume tier before the DEV train span) to hi; origins are "
                            "enumerated inside [train span, validation week] from row identities; padded rows are support only"},
        "arms": arms, "seeds": list(seeds), "recipe": recipe or RECIPE, "pilot": PILOT, "limits": limits or LIMITS,
        "tier": tier or "TIER1: " + TIERS["TIER1"]["why"],
        "factorial": {k: BLOCKS[block][k] for k in ("primary_factorial", "secondary_control", "informed_by", "inherited_control", "lag_declaration") if k in BLOCKS[block]},
        "capacity": capacity(arms, src_design["graph"]["assignment"], len(data_json["input_columns"]), int(data_json["target_channel"])),
        "contract_source": "REGISTRY household_W60_h60" if contract is None else "DECLARED_OVERRIDE (synthetic test panel)",
        "scaler_rule": "COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels "
                       "mean 0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd",
        "common_evaluation_rule": "the intersection over the block's arms of admissible validation origins, with a finite label "
                                  "and a finite daily lookup, derived at prepare BEFORE any score; every arm scores on it",
        "cells": [{"cell_id": f"{a['arm']}_s{s}", "arm": a["arm"], "seed": s} for s in seeds for a in arms],
        "pilots": [{"cell_id": f"pilot_{a['arm']}", "arm": a["arm"], "seed": int(seeds[0]), **PILOT, "role": "COST_PILOT"} for a in arms],
        "reuse": reuse or {},
        "baselines": {"persistence": "y(t)", "daily_seasonal": "y(t+h-1440)", "train_constant": "mean of the train labels of the "
                      "28 d tier; computed on the common evaluation set at closure, no fit, no terminal"},
        "benchmark_contract": ours.to_design_block(comparability=comp),
        "source_code": {name: sha_file(HERE/name) for name in ("df_e1_block.py", "df_gru_reference.py", "df_e1_calendar.py",
                                                             "df_e1_pilot.py", "df_mod_e0.py", "df_e1_governed.py")},
        "reading_rules": ["three seeds on one task are development evidence", "a fit that reached the update ceiling is CENSORED "
                          "wherever its best checkpoint fell", "no cell is removed after its score is seen",
                          "a published number under another protocol never enters the comparison column"],
    }
    design["design_sha256"] = sha_obj(design)
    return design


def reuse_record(root: Path) -> dict:
    """The baseline reused from a CLOSED block: its design digest, recipe and the accepted terminals of modular_w60.
    Reuse is legitimate only when the complete contract matches (rows, inputs, scaler, recipe, cadence, seeds)."""
    root = Path(root)
    d = json.loads((root/"DESIGN.json").read_text())
    rep = json.loads((root/"REPORT.json").read_text())
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    if not rep.get("verified"):
        raise BlockRefusal("REFUSED: the block to reuse from is not closed and verified")
    if d["recipe"] != RECIPE or d["source_run"]["data_sha256"] != json.loads((Path(d["source_run"]["root"])/"DATA.json").read_text())["data_sha256"]:
        raise BlockRefusal("REFUSED: the baseline's recipe or rows differ; it cannot be reused")
    units = {c["cell_id"]: {"terminal_sha256": receipts[c["cell_id"]]["terminal_sha256"], "campaign_sha256": receipts[c["cell_id"]]["campaign_sha256"]}
             for c in d["cells"] if c["arm"] == "modular_w60"}
    return {"baseline_arm": "modular_w60", "run_root": str(root), "block": d["block"], "design_sha256": d["design_sha256"],
            "recipe": d["recipe"], "seeds": d["seeds"], "units": units,
            "rule": "reused because rows, inputs, scaler, recipe, cadence and seeds are identical; the baseline is NOT re-trained"}


def code_drift(design: dict) -> dict:
    """Which scientific source files differ now from the ones the design was sealed with."""
    return {name: {"sealed": digest, "now": sha_file(HERE/name)} for name, digest in design["source_code"].items() if sha_file(HERE/name) != digest}


def training_equivalence(data: dict, design: dict, *, updates: int = 20, seed: int = 1) -> dict:
    """RP87: the exact-crop control (W1440 input cropped to its last 60 rows) trained on the SAME origins with the SAME seed must
    follow the SAME training prefix as modular_w60 (batch losses and weights); measured, not assumed from shapes."""
    tf = _module("df_mod_e0")._tf()
    R = design["recipe"]
    j, h = int(data["target_channel"][0]), int(data["horizon"][0])
    m, sd = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    X, asg = arm_inputs(data, arm_spec("modular_w60"), assignment=design["source_run"]["graph_assignment"])
    origins = data.get("train_origins__modular_w60", data[[k for k in data if k.startswith("train_origins__")][0]])
    origins = origins[origins >= 1439]                                      # both windows must fit
    out = {}
    for name, spec in (("modular_w60", arm_spec("modular_w60")), ("crop60", arm_spec("long_window_crop60"))):
        tr = Batches(X, data["Y"], origins, int(spec["window"]), h, j, int(R["batch"]), mean=m, sd=sd, shuffle=True, seed=seed)
        model = build_model(spec, asg, X.shape[1], j, seed)
        tf.keras.utils.set_random_seed(seed)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(R["learning_rate"])), loss=R["loss"])
        losses = []
        for i in range(updates):
            xb, yb = tr[i % len(tr)]
            losses.append(float(model.train_on_batch(xb, yb, return_dict=True)["loss"]))
        out[name] = {"initial_weights": None, "losses": losses, "final_weights_sha256": weight_hash(model), "parameters": n_params(model)}
    a, b = out["modular_w60"], out["crop60"]
    diff = max(abs(x-y) for x, y in zip(a["losses"], b["losses"]))
    return {"updates": updates, "origins": int(origins.size), "max_abs_loss_difference": diff, "final_weights_equal": a["final_weights_sha256"] == b["final_weights_sha256"],
            "equivalent": diff <= 1e-6 and a["final_weights_sha256"] == b["final_weights_sha256"], "losses_modular": a["losses"], "losses_crop": b["losses"],
            "reading": "identical batch losses and identical weights after the same updates on the same rows: the crop control is the baseline's computation"}


def capacity(arms: list, assignment: list, p: int, j: int) -> dict:
    """Parameters per arm, built; how added channels change each architecture's parameter count is stated, not assumed."""
    out = {}
    for a in arms:
        extra = {"calendar": 4, "randomised_calendar": 4, "daily_lag": 1}.get(a["features"], 0)
        asg = list(assignment) + ([max(assignment)+1]*4 if extra == 4 else [assignment[j]] if extra == 1 else [])
        out[a["arm"]] = {"channels": p+extra, "parameters": n_params(build_model(a, asg, p+extra, j, 1))}
    for a in arms:
        base = next((b for b in arms if b["family"] == a["family"] and b["features"] == "base" and b["window"] == a["window"]), None)
        if base and a["arm"] != base["arm"]:
            out[a["arm"]]["parameter_delta_vs_base_inputs"] = out[a["arm"]]["parameters"] - out[base["arm"]]["parameters"]
            out[a["arm"]]["how"] = ("a 4-channel calendar group adds one ARCH-A branch (detector + adapter) and 4 head/skip rows" if a["family"] == "modular"
                                    else "4 input columns add 3 x units x 4 GRU kernel rows; the recurrent kernel, biases and readout are unchanged")
    out["initialization"] = ("every arm is built after set_random_seed(seed): arms of one seed are paired by seed, not weight-identical in their "
                             "shared parts (a wider input changes the shapes and the draw order); the initial weights digest is recorded per cell")
    return out


def validate(design: dict, *, strict_code: bool = True) -> dict:
    """The design, checked against its own digest, its arms, the code and the prepared source it binds to.
    `strict_code=False` is for CLOSURE only: fits already ran under the sealed code; the closure records the drift."""
    B = _module("df_benchmark_contract")
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    if design.get("schema") != SCHEMA or sha_obj(body) != design.get("design_sha256"):
        raise BlockRefusal("REFUSED: design digest/schema mismatch")
    for a in design["arms"]:
        if arm_spec(a["arm"]) != a:
            raise BlockRefusal(f"REFUSED: arm {a['arm']} differs from the registry")
    drift = code_drift(design)
    if drift and strict_code:
        raise BlockRefusal(f"REFUSED: scientific source changed: {sorted(drift)}")
    contract = B.require(design, purpose=f"the {design['block']} block")
    B.bind(contract, json.loads((Path(design["source_run"]["root"])/"DATA.json").read_text()), purpose=f"the {design['block']} block")
    return contract


# --- features and enumeration from row identities ----------------------------------------------------------------

def calendar_channels(frame, ts_format: str) -> np.ndarray:
    """The 4 calendar channels through the production path (each row's own label, nothing after it)."""
    C = _module("df_e1_calendar")
    spec = C.CalendarSpec(timestamp_column="timestamp_label", ts_format=ts_format, step_seconds=60)
    return C.build(frame, spec)["features"]


def hash_uniform(ids: np.ndarray, seed: int, stream: int) -> np.ndarray:
    """A deterministic uniform in [0, 1) per (seed, stream, id): a row's draw never depends on any other row."""
    with np.errstate(over="ignore"):                                              # 64-bit wraparound is the mixer
        s = np.asarray([seed], dtype=np.uint64)*np.uint64(0x9E3779B97F4A7C15) + np.asarray([stream], dtype=np.uint64)*np.uint64(0xBF58476D1CE4E5B9)
        x = ids.astype(np.uint64) + s
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def randomised_calendar_channels(ts, panel_rows: np.ndarray, seed: int = 20260921) -> np.ndarray:
    """The CAPACITY control: hour/weekday sin-cos of (label + a per-row random offset). The offset is a hash of
    (seed, panel row id), so the channel is deterministic, stable under prefix extension and carries no calendar
    information; its chance association with the true clock on a finite slice is measured, never declared zero."""
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy()/60.0
    weekday = ts.dt.dayofweek.to_numpy() + hour/24.0
    hour = (hour + 24.0*hash_uniform(panel_rows, seed, 1)) % 24.0
    weekday = (weekday + 7.0*hash_uniform(panel_rows, seed, 2)) % 7.0
    two_pi = 2.0*math.pi
    return np.stack([np.sin(two_pi*hour/24.0), np.cos(two_pi*hour/24.0),
                     np.sin(two_pi*weekday/7.0), np.cos(two_pi*weekday/7.0)], axis=1)


def daily_lag_channel(Y: np.ndarray, *, h: int, lag: int = DAY) -> np.ndarray:
    """At row r the channel holds y(r + h - lag): a row at or before r for h <= lag; NaN where that row is not in the array."""
    if h > lag:
        raise BlockRefusal(f"REFUSED: a lag of {lag} at horizon {h} would read a row AFTER the origin")
    out = np.full(Y.shape[0], np.nan)
    shift = lag - h
    out[shift:] = Y[:Y.shape[0]-shift]
    return out


def admissible_origins(ts_ns: np.ndarray, inputs_finite: np.ndarray, label_finite: np.ndarray, *, W: int, h: int,
                       lo: int, hi: int, step_seconds: int = 60) -> np.ndarray:
    """Origins t in [lo, hi) whose support rows t-W+1..t+h are grid-consecutive, whose window inputs are finite
    and whose label at t+h is finite — from the rows' own identities, nothing assumed."""
    n = ts_ns.shape[0]
    ok = np.ones(n, dtype=bool)
    ok[1:] = np.diff(ts_ns) == step_seconds*1_000_000_000
    ok[0] = True
    cg = np.concatenate([[0], np.cumsum(ok)])
    cf = np.concatenate([[0], np.cumsum(inputs_finite)])
    t = np.arange(max(lo, W-1), min(hi, n-h))
    if t.size == 0:
        return t
    span_ok = (cg[t+h+1]-cg[t-W+2]) == (W+h-1)
    fin = (cf[t+1]-cf[t-W+1]) == W
    return t[span_ok & fin & label_finite[t+h]]


def _coverage(origins: np.ndarray, back: int, forward: int, n: int) -> np.ndarray:
    """How many windows [t-back, t+forward] cover each row, by a difference array (O(n), no W x n index)."""
    d = np.zeros(n+1, dtype=np.int64)
    np.add.at(d, np.clip(origins-back, 0, n), 1)
    np.add.at(d, np.clip(origins+forward+1, 0, n), -1)
    return np.cumsum(d)[:n]


def counts_from_identities(train_origins: np.ndarray, eval_origins: np.ndarray, W: int, h: int, n_rows: int) -> dict:
    """Unique support rows, train-only rows, targets, windows and exposures — from the identities."""
    train_origins, eval_origins = np.asarray(train_origins, dtype=np.int64), np.asarray(eval_origins, dtype=np.int64)
    cov = _coverage(train_origins, W-1, 0, n_rows)
    support = cov > 0
    targets = np.zeros(n_rows, dtype=bool)
    targets[train_origins+h] = True
    val_support = _coverage(eval_origins, W-1, h, n_rows) > 0 if eval_origins.size else np.zeros(n_rows, dtype=bool)
    train_rows = support | targets
    train_only = train_rows & ~val_support
    return {"distinct_windows": int(train_origins.size), "labels": int(targets.sum()),
            "unique_support_rows": int(support.sum()), "unique_train_rows_incl_targets": int(train_rows.sum()),
            "train_only_rows_excluding_validation_support": int(train_only.sum()),
            "rows_shared_with_validation_support": int((train_rows & val_support).sum()),
            "mean_exposures_per_support_row": float(cov[support].mean()) if support.any() else 0.0,
            "reading": "consecutive windows overlap by W-1 rows; distinct windows are NOT independent observations; validation "
                       "support rows are excluded from the train-only count"}


def feasibility_train_before(arm: str, ts_ns, inputs_finite, label_finite, train_end_local: int) -> np.ndarray:
    """The W60/28 d baseline's own admissible train origins (before any block-level intersection): the binding to the source."""
    return admissible_origins(ts_ns, inputs_finite, label_finite, W=W0, h=H0, lo=train_end_local - 28*DAY + W0 - 1, hi=train_end_local - (W0+H0))


def train_population_report(design: dict, rec: dict, *, origins: dict | None = None) -> dict:
    """The origins a preparation actually HOLDS, read against the policy its design declares.

    `rec` is the preparation's own BLOCK_DATA record; `counts_from_identities[arm]["labels"]` is the
    post-intersection count, i.e. the origins the arm would really train on. `origins`, when given,
    is the arm -> train-origin array mapping (from BLOCK_DATA.npz or from prepare() itself) and makes
    the check exact instead of count-only: two arms can hold the same NUMBER of different origins.
    """
    policy = design.get("train_population", "UNDECLARED")
    arms = [a["arm"] for a in design["arms"]]
    counts = rec.get("counts_from_identities") or {}
    feasibility = rec.get("feasibility") or {}
    binding = rec.get("binding_to_source") or {}
    held = {arm: (counts.get(arm) or {}).get("labels") for arm in arms}
    identical = None
    if origins is not None and all(arm in origins for arm in arms):
        first = np.asarray(origins[arms[0]])
        identical = all(np.array_equal(first, np.asarray(origins[a])) for a in arms)
    return {
        "declared_policy": policy,
        "held_train_origins": held,
        "held_identical_counts": len(set(held.values())) == 1 and None not in held.values(),
        "held_identical_origins": identical,
        "before_intersection_recorded": {
            arm: (feasibility.get(arm) or {}).get("train_admissible_before_intersection") for arm in arms
        },
        "common_train_origins_recorded": binding.get("common_train_origins"),
    }


def validate_train_population(design: dict, rec: dict, *, origins: dict | None = None) -> dict:
    """Refuse a preparation whose declared origin policy disagrees with the origins it HOLDS.

    Q2_CONTEXT v1 is why this exists. Its sealed design (`6d1aaecaf27c581c...`, sealed before the
    commit that introduced the intersection) carries no `train_population` key at all, while its
    retained BLOCK_DATA holds PER-ARM train origins -- 40 020 / 38 700 / 38 700 / 38 700 / 40 080 --
    and its record carries neither `train_admissible_before_intersection` nor `common_train_origins`.
    The block catalogue today declares `COMMON_INTERSECTION` for Q2_CONTEXT, so an audit reading the
    catalogue would believe the sealed preparation honoured it. Had a cell ever been fitted on those
    bytes, the input contrast would have been confounded with train volume: different arms trained on
    different origins while the block claimed they shared them. No cell was, so nothing published
    moves -- and this refusal is what keeps it that way.

    Returns the report on agreement; raises BlockRefusal, naming the disagreement, otherwise.
    """
    report = train_population_report(design, rec, origins=origins)
    policy, held = report["declared_policy"], report["held_train_origins"]
    shown = ", ".join(f"{arm}={held[arm]}" for arm in held)
    if None in held.values():
        raise BlockRefusal(
            f"REFUSED: the preparation records no held train-origin count for every declared arm ({shown})")
    if policy != "UNDECLARED" and policy not in TRAIN_POPULATIONS:
        raise BlockRefusal(f"REFUSED: unknown train_population {policy!r}; the vocabulary is {list(TRAIN_POPULATIONS)}")
    if policy == "UNDECLARED":
        # Not a policy: the absence of one. Harmless while the arms happen to hold the same origins,
        # unusable the moment they do not, because no declaration says which reading is intended.
        if not report["held_identical_counts"] or report["held_identical_origins"] is False:
            raise BlockRefusal(
                "REFUSED: the design declares no train_population and the preparation holds per-arm train "
                f"origins ({shown}); it cannot be read as a common-origin comparison, and an input contrast "
                "over it would be confounded with train volume")
        return report
    if policy == "COMMON_INTERSECTION":
        if not report["held_identical_counts"]:
            raise BlockRefusal(
                f"REFUSED: the design declares train_population COMMON_INTERSECTION but the preparation holds "
                f"per-arm train origins ({shown}); the input contrast would be confounded with train volume")
        if report["held_identical_origins"] is False:
            raise BlockRefusal(
                "REFUSED: the design declares train_population COMMON_INTERSECTION and the per-arm train-origin "
                "counts agree, but the origin SETS are not identical")
        missing = [arm for arm, value in report["before_intersection_recorded"].items() if value is None]
        if missing:
            raise BlockRefusal(
                "REFUSED: the design declares train_population COMMON_INTERSECTION but the preparation records no "
                f"train_admissible_before_intersection for {missing}; there is no evidence the intersection was applied")
        common = report["common_train_origins_recorded"]
        if common is None:
            raise BlockRefusal(
                "REFUSED: the design declares train_population COMMON_INTERSECTION but the preparation's "
                "binding_to_source records no common_train_origins")
        if int(common) != int(next(iter(held.values()))):
            raise BlockRefusal(
                f"REFUSED: the recorded common_train_origins {common} is not the count the arms hold ({shown})")
    return report


def prepare(design: dict, root: Path, *, frame=None) -> dict:
    """Read the delivered panel rows the block declares, build every channel, enumerate every arm's origins from
    row identities, derive the COMMON evaluation mask, apply the COMMON scaler — before any score.
    `frame` is a pandas frame for tests on synthetic panels; the governed path reads the delivery."""
    import pandas as pd
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate(design)
    src = design["source_run"]
    rows = design["rows"]
    if frame is None:
        G = _module("df_e1_governed")
        delivered = G.require_delivery(root, design, "prepare")["delivery"]
        if delivered["sha256"] != src["panel_sha256"]:
            raise BlockRefusal("REFUSED: the delivered panel is not the source run's panel")
        frame = pd.read_parquet(delivered["path"])
    lo, hi, pad = int(rows["lo"]), int(rows["hi"]), int(rows["pad_rows"])
    if lo < 0 or hi > len(frame):
        raise BlockRefusal(f"REFUSED: rows [{lo}, {hi}) are not inside the panel of {len(frame)} rows")
    part = frame.iloc[lo:hi].reset_index(drop=True)
    cols = src["input_columns"]
    j = int(src["target_channel"])
    ts_format = "%d/%m/%Y %H:%M:%S"
    C = _module("df_e1_calendar")
    ts = C.parse_labels(part, C.CalendarSpec(timestamp_column="timestamp_label", ts_format=ts_format))
    ts_ns = ts.to_numpy().astype("datetime64[ns]").astype(np.int64)
    X = part[cols].to_numpy(dtype=np.float64)
    Y = X[:, j].copy()
    mean, sd = np.asarray(src["scaler"]["mean"], dtype=np.float64), np.asarray(src["scaler"]["sd"], dtype=np.float64)
    if mean.shape != (len(cols),) or not (np.isfinite(sd).all() and (sd > 0).all()):
        raise BlockRefusal("REFUSED: the common scaler is not a finite positive per-channel scaler of the declared columns")
    Xs = ((X-mean)/sd).astype(np.float32)
    cal = calendar_channels(part, ts_format).astype(np.float32)
    cal_rand = randomised_calendar_channels(ts, np.arange(lo, hi)).astype(np.float32)
    chance = {C.FEATURES[k]: float(np.corrcoef(cal[:, k], cal_rand[:, k])[0, 1]) for k in range(4)}
    lag_raw = daily_lag_channel(Y, h=H0)
    lag = ((lag_raw-mean[j])/sd[j]).astype(np.float32)
    inputs_finite = np.isfinite(X).all(axis=1)
    label_finite = np.isfinite(Y)
    train_end_local = int(rows["train_end"]) - lo
    hi_local = hi - lo
    origins, counts, feasibility = {}, {}, {}
    for a in design["arms"]:
        W = int(a["window"])
        need_lag = a["features"] == "daily_lag"
        fin = inputs_finite & np.isfinite(lag_raw) if need_lag else inputs_finite
        t_lo = train_end_local - int(a["train_days"])*DAY
        train = admissible_origins(ts_ns, fin, label_finite, W=W, h=H0, lo=t_lo + W0 - 1, hi=train_end_local - (W0+H0))
        val = admissible_origins(ts_ns, fin, label_finite, W=W, h=H0, lo=train_end_local, hi=hi_local)
        origins[a["arm"]] = {"train": train, "validation": val}
        feasibility[a["arm"]] = {"train_candidates": [t_lo + W0 - 1, train_end_local - (W0+H0)], "train_admissible": int(train.size),
                                 "validation_admissible": int(val.size), "window": W, "needs_lag": need_lag}
    # a block whose arms withdraw different train origins (long windows, lags over padded rows) trains EVERY arm on the
    # intersection, so context is never conflated with volume; the per-arm admissible counts stay recorded above
    train_common = None
    if design.get("train_population") == "COMMON_INTERSECTION":
        for a in design["arms"]:
            t = origins[a["arm"]]["train"]
            train_common = t if train_common is None else np.intersect1d(train_common, t)
        for a in design["arms"]:
            feasibility[a["arm"]]["train_admissible_before_intersection"] = int(origins[a["arm"]]["train"].size)
            origins[a["arm"]]["train"] = train_common
    # the COMMON evaluation set: admissible for every arm, finite label, finite daily lookup — as the source did
    common = None
    for a in design["arms"]:
        v = origins[a["arm"]]["validation"]
        common = v if common is None else np.intersect1d(common, v)
    lookup = common + H0 - DAY
    common = common[(lookup >= 0) & np.isfinite(Y[np.clip(lookup, 0, None)])]
    # binding to the source run: the 28 d baseline enumeration must reproduce the source's populations exactly
    with np.load(Path(src["root"])/"DATA.npz", allow_pickle=False) as z:
        src_train = z["train_origins"] + (src["slice_rows"][0]-lo)
        src_eval = z["eval_origins"] + (src["slice_rows"][0]-lo)
        src_Y = z["Y"]
    base = next((a for a in design["arms"] if a["window"] == W0 and a["train_days"] == 28 and a["features"] in ("base", "calendar", "randomised_calendar")), None)
    binding = {"baseline_arm_checked": base["arm"] if base else None}
    if base is not None:
        base_train = origins[base["arm"]]["train"] if train_common is None else feasibility_train_before(base["arm"], ts_ns, inputs_finite, label_finite, train_end_local)
        binding["train_origins_equal_source"] = bool(np.array_equal(base_train, src_train))
        if not binding["train_origins_equal_source"]:
            raise BlockRefusal("REFUSED: the block's enumeration does not reproduce the source run's train origins")
        if train_common is not None:
            binding["common_train_origins"] = int(train_common.size)
            binding["common_train_subset_of_source"] = bool(np.isin(train_common, src_train).all())
    if not np.array_equal(src_Y, Y[src["slice_rows"][0]-lo: src["slice_rows"][1]-lo], equal_nan=True):
        raise BlockRefusal("REFUSED: the target rows are not the source run's")
    binding["common_evaluation_equals_source"] = bool(np.array_equal(common, src_eval))
    binding["common_evaluation_subset_of_source"] = bool(np.isin(common, src_eval).all())
    if not binding["common_evaluation_subset_of_source"]:
        raise BlockRefusal("REFUSED: the common evaluation set is not inside the source run's evaluation origins")
    for a in design["arms"]:
        counts[a["arm"]] = counts_from_identities(origins[a["arm"]]["train"], common, int(a["window"]), H0, Y.shape[0])
    payload = {"Xs": Xs, "Y": Y, "calendar": cal, "calendar_randomised": cal_rand, "lag": lag, "lag_raw": lag_raw, "ts_ns": ts_ns,
               "scaler_mean": mean, "scaler_sd": sd, "target_channel": np.array([j]), "horizon": np.array([H0]),
               "row_offset": np.array([lo]), "train_end_local": np.array([train_end_local]), "common_eval": common}
    for arm, o in origins.items():
        payload[f"train_origins__{arm}"] = o["train"]
        payload[f"validation_origins__{arm}"] = o["validation"]
    np.savez(root/"BLOCK_DATA.npz", **payload)
    grid = C.grid_report(ts, C.CalendarSpec(timestamp_column="timestamp_label", ts_format=ts_format))
    rec = {"schema": SCHEMA_DATA, "design_sha256": design["design_sha256"], "rows": {"lo": lo, "hi": hi, "pad": pad, "n": int(hi-lo)},
           "input_columns": cols, "target_channel": j, "horizon": H0, "scaler": {"mean": mean.tolist(), "sd": sd.tolist(),
           "source": "COMMON: the source run's train-only scaler; calendar mean 0 / sd 1; lag = target scaler"},
           "sigma_evaluation": float(sd[j]), "grid": grid,
           "randomised_calendar_control": {"seed": 20260921, "chance_association_corr_with_true_calendar": chance,
                                           "reading": "finite-sample association of the control with the true clock; reported, not assumed zero"}, "feasibility": feasibility, "counts_from_identities": counts,
           "common_evaluation": {"n": int(common.size), "first_row": int(common.min()) if common.size else None,
                                 "last_row": int(common.max()) if common.size else None},
           "binding_to_source": binding, "data_sha256": sha_file(root/"BLOCK_DATA.npz")}
    # the origins this preparation HOLDS must agree with the policy the design declares, checked on the
    # arrays themselves before the record is written: a preparation that contradicts its own design never seals
    validate_train_population(design, rec, origins={arm: o["train"] for arm, o in origins.items()})
    (root/"BLOCK_DATA.json").write_text(json.dumps(rec, indent=1))
    return rec


def load_data(root: Path, design: dict) -> dict:
    rec = json.loads((Path(root)/"BLOCK_DATA.json").read_text())
    if rec["design_sha256"] != design["design_sha256"] or sha_file(Path(root)/"BLOCK_DATA.npz") != rec["data_sha256"]:
        raise BlockRefusal("REFUSED: BLOCK_DATA belongs to another design or was altered")
    with np.load(Path(root)/"BLOCK_DATA.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    # a retained preparation is read at CONSUMPTION too: nothing is fitted on origins that contradict the
    # design's declared policy, whatever the preparation was sealed by and whenever it was sealed
    validate_train_population(design, rec, origins={
        a["arm"]: data[f"train_origins__{a['arm']}"] for a in design["arms"]
        if f"train_origins__{a['arm']}" in data})
    return data


def arm_inputs(data: dict, spec: dict, *, assignment: list) -> tuple:
    """The channel matrix an arm consumes and its group assignment; the extra channels are declared groups."""
    Xs = data["Xs"]
    p0 = Xs.shape[1]
    j = int(data["target_channel"][0])
    if spec["features"] == "base":
        return Xs, list(assignment)
    if spec["features"] == "calendar":
        return np.concatenate([Xs, data["calendar"]], axis=1), list(assignment) + [max(assignment)+1]*4
    if spec["features"] == "randomised_calendar":
        return np.concatenate([Xs, data["calendar_randomised"]], axis=1), list(assignment) + [max(assignment)+1]*4
    if spec["features"] == "daily_lag":
        return np.concatenate([Xs, data["lag"][:, None]], axis=1), list(assignment) + [assignment[j]]
    raise BlockRefusal(f"REFUSED: unknown feature set {spec['features']!r}")


# --- models ---------------------------------------------------------------------------------------------------

def build_modular(assignment: list, W: int, p: int, j: int, seed: int, *, dilations: list | None = None, crop: int | None = None):
    """ARCH-A branches + tcn_w core, with EXPLICIT dilations and an optional raw-input crop before the extractor."""
    E = _module("df_mod_e0")
    P = _module("df_e1_pilot")
    tf = E._tf()
    tf.keras.utils.set_random_seed(int(seed))
    dil = list(dilations) if dilations is not None else P.core_dilations(crop or W)
    inp = tf.keras.Input(shape=(W, p), name="x")
    x = tf.keras.layers.Lambda(lambda t, c=crop: t[:, -c:, :], name=f"crop_last_{crop}")(inp) if crop else inp
    groups = sorted(set(assignment))
    branches = []
    for g in groups:
        idx = [k for k in range(p) if assignment[k] == g]
        sub = tf.keras.layers.Lambda(lambda t, idx=idx: tf.gather(t, idx, axis=2), name=f"g{g}_select")(x)
        branches.append(E.branch_extractor(tf, sub, f"g{g}", "A"))
    joint = tf.keras.layers.Concatenate(axis=2, name="fusion_seq")(branches) if len(branches) > 1 else branches[0]
    h = joint
    for i, d in enumerate(dil, start=1):
        h = E._tcn_block(tf, h, f"core_tcn{i}", d)
    read = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="core_last")(h)
    delta = tf.keras.layers.Dense(p, name="head")(read)
    last_x = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_observation")(x)
    full = tf.keras.layers.Add(name="persistence_skip")([last_x, delta])
    out = tf.keras.layers.Lambda(lambda t: t[:, j:j+1], name="target_readout")(full)
    return tf.keras.Model(inp, out, name=f"modular_A_tcnw_W{W}_crop{crop}_d{len(dil)}")


def build_model(spec: dict, assignment: list, p: int, j: int, seed: int):
    if spec["family"] == "gru":
        return _module("df_gru_reference").build(int(spec["window"]), p, j, seed)
    return build_modular(assignment, int(spec["window"]), p, j, seed, dilations=spec.get("dilations"), crop=spec.get("crop"))


def n_params(model) -> int:
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def weight_hash(model) -> str:
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(str((w.shape, str(w.dtype))).encode())
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


# --- training in observed updates ------------------------------------------------------------------------------

def _gather(Xs, origins, W):
    idx = origins[:, None] - W + 1 + np.arange(W)[None, :]
    return Xs[idx]


class Batches:
    """Windows gathered per batch from the scaled channel matrix; reshuffled per pass from (seed, pass)."""
    def __init__(self, Xs, Y, origins, W, h, j, batch, *, mean, sd, shuffle, seed):
        self.Xs, self.Y, self.o, self.W, self.h, self.j, self.batch = Xs, Y, np.asarray(origins), W, h, j, batch
        self.m, self.s, self.shuffle, self.seed, self.epoch = float(mean), float(sd), shuffle, int(seed), 0
        self.perm = np.arange(self.o.size)
        self._reshuffle()

    def _reshuffle(self):
        if self.shuffle:
            self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.o.size)

    def __len__(self):
        return math.ceil(self.o.size/self.batch)

    def __getitem__(self, i):
        o = self.o[self.perm[i*self.batch:(i+1)*self.batch]]
        y = ((self.Y[o+self.h]-self.m)/self.s).astype(np.float32)[:, None]
        return _gather(self.Xs, o, self.W), y

    def on_epoch_end(self):
        self.epoch += 1
        self._reshuffle()


def predict(model, ds) -> np.ndarray:
    return np.concatenate([np.asarray(model.predict_on_batch(ds[i][0])) for i in range(len(ds))], axis=0)[:, 0]


def evaluate_mae(model, ds) -> float:
    pred = predict(model, ds)
    y = np.concatenate([ds[i][1][:, 0] for i in range(len(ds))])
    return float(np.mean(np.abs(pred.astype(np.float64)-y.astype(np.float64))))


def fit_by_updates(model, train, val, *, max_updates: int, validate_every: int, patience: int, lr: float, seed: int,
                   loss="mae", min_delta: float = 0.0, optimizer=None) -> dict:
    """The loop: one optimizer update per batch, validation every `validate_every` OBSERVED updates, patience in
    validation events, restore the best weights; the ceiling is CENSORING wherever the best event fell.
    The monitor is the validation MAE of the predictions (scaled units), whatever loss the arm trains on."""
    tf = _module("df_mod_e0")._tf()
    tf.keras.utils.set_random_seed(int(seed))
    opt = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    updates, i, events = 0, 0, []
    best, best_weights, best_event = math.inf, model.get_weights(), 0
    running, patience_expired = [], False
    t_fit, t_val = time.process_time(), 0.0
    while updates < max_updates:
        x, y = train[i]
        logs = model.train_on_batch(x, y, return_dict=True)
        running.append(float(logs["loss"]))
        i += 1
        updates += 1
        if i >= len(train):
            train.on_epoch_end()
            i = 0
        if updates % validate_every == 0 or updates == max_updates:
            t0 = time.process_time()
            v = evaluate_mae(model, val)
            t_val += time.process_time()-t0
            events.append({"update": updates, "val_mae_scaled": v, "train_loss_mean_since_last": float(np.mean(running))})
            running = []
            if v < best - min_delta:
                best, best_weights, best_event = v, model.get_weights(), len(events)
            elif len(events)-best_event >= patience:
                patience_expired = True
                break
    iterations = int(opt.iterations.numpy())
    budget_reached = updates >= max_updates                      # true whatever branch broke the loop (RP77/A5)
    triggers = {"budget_reached": bool(budget_reached), "patience_expired": bool(patience_expired)}
    stop = "+".join([n for n, on in (("UPDATE_BUDGET", budget_reached), ("EARLY_STOPPING", patience_expired)) if on]) or "LOOP_ENDED"
    t_r = time.process_time()
    model.set_weights(best_weights)
    restored = evaluate_mae(model, val)
    t_restore = time.process_time()-t_r
    fit_cpu = time.process_time()-t_fit
    return {"updates": updates, "optimizer_iterations": iterations, "updates_are_optimizer_iterations": iterations == updates,
            "validate_every_updates": validate_every, "validation_events": len(events), "events": events,
            "best_event": best_event, "best_update": events[best_event-1]["update"] if best_event else None,
            "best_val_mae_scaled": best, "restored_val_mae_scaled": restored,
            "restore_verified": bool(abs(restored-best) <= 1e-6*max(1.0, abs(best))),
            "stop_reason": stop, "triggers": triggers, "patience_events": patience, "min_delta": min_delta,
            "censoring": {"verdict": "CENSORED_BY_BUDGET" if budget_reached else "STOPPED_ON_VALIDATION",
                          "rule": "reaching the update ceiling is censoring wherever the best checkpoint fell and whether or not "
                                  "patience expired on the same update; early stopping never claims convergence"},
            "cpu": {"train_update_seconds": fit_cpu - t_val - t_restore, "validation_seconds": t_val, "restore_seconds": t_restore,
                    "loop_total_seconds": fit_cpu},
            "fit_cpu_seconds": fit_cpu, "validation_cpu_seconds": t_val}


# --- one cell (child process) ------------------------------------------------------------------------------------

def run_cell(design: dict, data: dict, cell: dict, out_dir: Path, *, pilot: bool) -> dict:
    cpu0, wall0 = time.process_time(), time.monotonic()
    setup0 = time.process_time()
    spec = next(a for a in design["arms"] if a["arm"] == cell["arm"])
    R = design["recipe"]
    W, h, j = int(spec["window"]), int(data["horizon"][0]), int(data["target_channel"][0])
    X, assignment = arm_inputs(data, spec, assignment=design["source_run"]["graph_assignment"])
    p = X.shape[1]
    Y = data["Y"]
    m, sd = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    train_o = data[f"train_origins__{spec['arm']}"]
    eval_o = data["common_eval"]
    train_end = int(data["train_end_local"][0])
    if pilot:
        # the cost pilot lives INSIDE train: it validates on the last 7 train days and never reads the DEV validation
        cut = train_end - 7*DAY
        pilot_train, pilot_val = train_o[train_o + h < cut - W], train_o[train_o >= cut]      # purge = the arm's own window
        train_o, eval_o = pilot_train, pilot_val
        max_updates, every, patience = int(cell["max_updates"]), int(cell["validate_every_updates"]), int(cell["patience_events"])
    else:
        max_updates, every, patience = int(R["max_updates"]), int(R["validate_every_updates"]), int(R["patience_events"])
    seed = int(cell["seed"])
    tr = Batches(X, Y, train_o, W, h, j, int(R["batch"]), mean=m, sd=sd, shuffle=True, seed=seed)
    va = Batches(X, Y, eval_o, W, h, j, int(R["batch"]), mean=m, sd=sd, shuffle=False, seed=seed)
    model = build_model(spec, assignment, p, j, seed)
    initial = weight_hash(model)
    setup_cpu = time.process_time()-setup0
    training = fit_by_updates(model, tr, va, max_updates=max_updates, validate_every=every, patience=patience,
                              lr=float(R["learning_rate"]), seed=seed, loss=R["loss"], min_delta=float(R["min_delta"]))
    replay0 = time.process_time()
    pred = predict(model, va).astype(np.float64)*sd + m
    y, naive = Y[eval_o+h], Y[eval_o]
    H = _module("df_e1_huber")
    score = H.metrics(pred, y, naive, sd)
    if not np.isclose(score["mae_z"], training["best_val_mae_scaled"], rtol=2e-5, atol=2e-6):
        raise BlockRefusal("REFUSED: the restored predictions do not reproduce the best validation MAE")
    out_dir.mkdir(parents=True, exist_ok=False)
    model.save_weights(out_dir/"weights.weights.h5")
    fresh = build_model(spec, assignment, p, j, seed)
    fresh.load_weights(out_dir/"weights.weights.h5")
    reload_pred = predict(fresh, va).astype(np.float64)*sd + m
    np.testing.assert_allclose(pred, reload_pred, atol=1e-6, rtol=1e-6)
    absolute = eval_o + int(data["row_offset"][0])
    np.savez(out_dir/"arrays.npz", pred=pred, y=y, naive=naive, origins=eval_o, origins_panel_rows=absolute, reload_pred=reload_pred)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    record = {"schema": "df_e1_block_cell.v1", "cell": cell, "arm_spec": spec, "design_sha256": design["design_sha256"],
              "pilot": pilot, "population": {"train_origins": int(train_o.size), "evaluation_origins": int(eval_o.size),
                                             "evaluation_is_common_dev_set": not pilot},
              "channels": p, "assignment": assignment, "parameters": n_params(model), "initial_weights_sha256": initial,
              "final_weights_sha256": weight_hash(model), "training": training, "scores": score,
              "target_mean": m, "target_sd": sd, "reload_max_error": float(np.max(np.abs(pred-reload_pred))),
              "cost": {"cpu_seconds": time.process_time()-cpu0, "wall_seconds": time.monotonic()-wall0,
                       "setup_cpu_seconds": setup_cpu, "train_update_cpu_seconds": training["cpu"]["train_update_seconds"],
                       "validation_cpu_seconds": training["cpu"]["validation_seconds"], "restore_cpu_seconds": training["cpu"]["restore_seconds"],
                       "final_predict_and_replay_cpu_seconds": time.process_time()-replay0,
                       "seconds_per_update": training["cpu"]["train_update_seconds"]/max(1, training["updates"]),
                       "seconds_per_validation_event": training["validation_cpu_seconds"]/max(1, training["validation_events"]),
                       "accounting": "seconds_per_update = TRAIN-UPDATE CPU only (validation, restore, setup and replay apart; RP77/A6)",
                       "peak_rss_bytes": int(ru.ru_maxrss)*1024, "host": os.uname().nodename},
              "arrays_sha256": sha_file(out_dir/"arrays.npz"), "weights_file_sha256": sha_file(out_dir/"weights.weights.h5")}
    write(out_dir/"cell.json", record)
    return record


def child(root: Path, unit: str) -> dict:
    design = json.loads((Path(root)/"DESIGN.json").read_text())
    validate(design)
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["child_cpu_seconds"], LIMITS["child_cpu_seconds"]+5))
    G = _module("df_e1_governed")
    delivered = G.require_delivery(root, design, unit)["delivery"]
    if delivered["sha256"] != design["source_run"]["panel_sha256"]:
        raise BlockRefusal("REFUSED: the unit's delivery is not the source panel")
    data = load_data(root, design)
    cell = next(c for c in design["pilots"] + design["cells"] if c["cell_id"] == unit)
    return run_cell(design, data, cell, Path(root)/"attempts"/unit, pilot=cell.get("role") == "COST_PILOT")


# --- governance: prepare, pilot, execute --------------------------------------------------------------------------

def _acquire(a, design, unit):
    G = _module("df_e1_governed")
    return G.acquire(run_id=a.run_id, root=a.root, lake=a.lake, resource=a.resource, unit_id=unit, gov_url=a.gov_url,
                     api_key_file=a.api_key_file, design_sha256=design["design_sha256"], cache_dir=Path(a.root)/"cache",
                     expect_sha256=design["source_run"]["panel_sha256"])


def _terminal_for(a, design, unit, cell, started, ok, rec, exit_code, wall):
    U = _module("df_utility_run")
    root = Path(a.root)
    cost = {"wall_seconds": wall}
    if rec:
        cost["cpu_seconds"] = rec["cost"]["cpu_seconds"]
    ms = [] if not rec else [U._metric(f"e1.block.{k}", float(v), "kW" if k.endswith("kw") else "dimensionless",
                                       split="validation" if not rec["pilot"] else "train_subset_pilot", horizon=H0)
                             for k, v in rec["scores"].items() if k != "rows"]
    artifacts = [] if not rec else [{"role": role, "sha256": sha_file(root/"attempts"/unit/f), "bytes": (root/"attempts"/unit/f).stat().st_size}
                                    for role, f in (("predictions", "arrays.npz"), ("weights", "weights.weights.h5"), ("record", "cell.json"))]
    terminal = U._terminal(status="COMPLETED" if ok else "FAILED", reason=None if ok else f"child exited {exit_code}; see the retained log",
                           cost=cost, metrics=ms, started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": unit,
                                 "arm": cell["arm"], "seed": str(cell["seed"]), "role": cell.get("role", "ARM"),
                                 "design_sha256": design["design_sha256"], "contract_sha256": design["benchmark_contract"]["contract_sha256"]})
    terminal["artifacts"] = artifacts
    return terminal


def governance_modules():
    """The legacy dynamic loader publishes a module before executing its body: complete the governance imports on
    the parent thread BEFORE children acquire in parallel (the huber runner learned this the hard way)."""
    G = _module("df_e1_governed")
    G._load("governed_run")
    G._load("df_e1_receipts")
    return G, _module("df_utility_run")


def run_units(a, design, units, *, parallel: int) -> list:
    G, U = governance_modules()
    root = Path(a.root)
    results = []

    def one(cell):
        unit = cell["cell_id"]
        term_path, rec_path = root/"TERMINALS"/f"{unit}.json", root/"attempts"/unit/"cell.json"
        if term_path.exists():
            # an accepted, recorded unit is REUSED, never re-trained to repair a receipt; anything else is a refusal
            receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text()).get("units", {}) if (root/"TERMINAL_RECEIPTS.json").is_file() else {}
            held = json.loads(term_path.read_text())
            if held.get("status") == "COMPLETED" and rec_path.is_file() and unit in receipts:
                print(json.dumps({"unit": unit, "reused": True}), flush=True)
                return {"unit": unit, "ok": True, "record": json.loads(rec_path.read_text()), "terminal_sha256": receipts[unit].get("terminal_sha256"), "reused": True}
            raise BlockRefusal(f"REFUSED: {unit} already has a terminal that is not an accepted COMPLETED record; experiments are not silently repeated")
        started = U._z(U.now_iso())
        _acquire(a, design, unit)
        wall = time.monotonic()
        with open(root/f"{unit}.log", "x") as log:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--root", str(root), "--unit", unit],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                                           "TF_CPP_MIN_LOG_LEVEL": "3"}, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=LIMITS["child_wall_seconds"])
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "WALL_TIMEOUT"
        rec_path = root/"attempts"/unit/"cell.json"
        ok = code == 0 and rec_path.exists()
        rec = json.loads(rec_path.read_text()) if ok else None
        terminal = _terminal_for(a, design, unit, cell, started, ok, rec, code, time.monotonic()-wall)
        (root/"TERMINALS").mkdir(exist_ok=True)
        write(root/"TERMINALS"/f"{unit}.json", terminal)
        reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file,
                                     outbox_dir=str(root/"outbox"), started_at=started)
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise BlockRefusal(f"REFUSED: the terminal of {unit} was not accepted: {reported['flushed']['failures']}")
        print(json.dumps({"unit": unit, "ok": ok, "mae_z": rec["scores"]["mae_z"] if rec else None,
                          "stop": rec["training"]["stop_reason"] if rec else None,
                          "cpu": rec["cost"]["cpu_seconds"] if rec else None}), flush=True)
        return {"unit": unit, "ok": ok, "record": rec, "terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256")}

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(0, len(units), parallel):
            batch = list(pool.map(one, units[start:start+parallel]))
            results += batch
            if not all(r["ok"] for r in batch):
                raise BlockRefusal("REFUSED: a child failed; the remaining cells are not started")
    return results


def spent_cpu(root: Path) -> float:
    return float(sum(json.loads(p.read_text())["cost"]["cpu_seconds"] for p in (Path(root)/"attempts").glob("*/cell.json")))


def pilot_costs(rec: dict) -> dict:
    """Per-arm unit costs from a pilot record, validation counted ONCE (RP77/A6): the train-update rate is the loop CPU
    minus validation and restore; old records without the split are re-derived from fit minus validation."""
    tr, cost = rec["training"], rec["cost"]
    cpu = tr.get("cpu") or {"train_update_seconds": tr["fit_cpu_seconds"]-tr["validation_cpu_seconds"],
                            "validation_seconds": tr["validation_cpu_seconds"], "restore_seconds": 0.0}
    return {"seconds_per_update": cpu["train_update_seconds"]/max(1, tr["updates"]),
            "seconds_per_validation_event_pilot": tr["validation_cpu_seconds"]/max(1, tr["validation_events"]),
            "restore_seconds_pilot": cpu.get("restore_seconds", 0.0), "setup_seconds_pilot": cost.get("setup_cpu_seconds", 0.0),
            "pilot_validation_rows": rec["population"]["evaluation_origins"], "peak_rss_bytes": cost["peak_rss_bytes"],
            "accounting": "train-update CPU / updates; validation per event; restore and setup once"}


def projection(design: dict, pilots: list) -> dict:
    """What the block would cost at the ceiling, from each arm's OWN pilot; validation counted once, restore and setup
    once per cell; the ceiling is never assumed reached early."""
    R = design["recipe"]
    events = math.ceil(R["max_updates"]/R["validate_every_updates"])
    per_arm = {r["record"]["cell"]["arm"]: pilot_costs(r["record"]) for r in pilots}
    n_eval = design["source_run"]["evaluation_origins"]
    cells = {}
    for c in design["cells"]:
        pa = per_arm[c["arm"]]
        val_scale = n_eval/max(1, pa["pilot_validation_rows"])
        cells[c["cell_id"]] = (pa["seconds_per_update"]*R["max_updates"] + pa["seconds_per_validation_event_pilot"]*val_scale*events
                               + pa["restore_seconds_pilot"]*val_scale + pa["setup_seconds_pilot"])
    total = sum(cells.values())
    return {"per_arm": per_arm, "per_cell_at_ceiling_seconds": cells, "total_at_ceiling_seconds": total,
            "with_headroom_25_percent": total*1.25, "closure_reserve_seconds": LIMITS["closure_reserve_seconds"],
            "accounting": "train-update rate x ceiling + validation events (scaled to the evaluation rows) + restore + setup, each once",
            "assumption": "every cell runs to the ceiling with every validation event; early stopping can only lower it"}


def recost(root: Path) -> dict:
    """Recalculate a retained pilot report with the corrected accounting; the old report is preserved."""
    root = Path(root)
    design = json.loads((root/"DESIGN.json").read_text())
    old = json.loads((root/"REPORT.pilot.json").read_text())
    pilots = [{"record": json.loads((root/"attempts"/c["cell_id"]/"cell.json").read_text())} for c in design["pilots"]
              if (root/"attempts"/c["cell_id"]/"cell.json").is_file()]
    proj = projection(design, pilots)
    doc = {"schema": "df_e1_block_pilot_report.corrected.v1", "design_sha256": design["design_sha256"], "corrected_on": "2026-09-21 (RP77/A6)",
           "old_total_at_ceiling_seconds": old["projection"]["total_at_ceiling_seconds"], "corrected_projection": proj,
           "old_decision": old["decision"], "decision_unchanged": (proj["with_headroom_25_percent"] + LIMITS["closure_reserve_seconds"] <= LIMITS["campaign_cpu_seconds"]) == (old["decision"] == "EXECUTE"),
           "reading": "arithmetic correction of a retained pilot, not a new measurement; validation was counted inside the update rate and again per event"}
    (root/"REPORT.pilot.corrected.json").write_text(json.dumps(doc, indent=1, default=str))
    return doc


def scan_stops(root: Path) -> dict:
    """Re-derive both stop triggers from every retained cell's saved events; corrections are written beside the records,
    the records themselves are never rewritten and nothing is retrained (RP77/A5)."""
    root = Path(root)
    design = json.loads((root/"DESIGN.json").read_text())
    out = {"schema": "df_e1_block_stop_corrections.v1", "design_sha256": design["design_sha256"], "units": {}, "corrected": []}
    for c in design["pilots"] + design["cells"]:
        p = root/"attempts"/c["cell_id"]/"cell.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        tr = r["training"]
        ceiling = int(c.get("max_updates", design["recipe"]["max_updates"]))
        patience = int(tr.get("patience_events", 0))
        events = tr.get("events") or []
        best = int(tr.get("best_event") or 0)
        budget = tr["updates"] >= ceiling
        expired = bool(events) and (len(events)-best >= patience) and not (tr.get("stop_reason") == "UPDATE_BUDGET" and len(events)-best < patience)
        verdict = "CENSORED_BY_BUDGET" if budget else "STOPPED_ON_VALIDATION"
        entry = {"updates": tr["updates"], "ceiling": ceiling, "events": len(events), "best_event": best, "patience": patience,
                 "recorded_stop_reason": tr.get("stop_reason"), "recorded_verdict": tr["censoring"]["verdict"],
                 "derived_triggers": {"budget_reached": bool(budget), "patience_expired": bool(expired)}, "derived_verdict": verdict}
        if verdict != tr["censoring"]["verdict"] or (budget and expired and "+" not in str(tr.get("stop_reason"))):
            out["corrected"].append(c["cell_id"])
            entry["correction"] = "both triggers apply; the recorded metadata named one"
        out["units"][c["cell_id"]] = entry
    (root/"STOP_CORRECTIONS.json").write_text(json.dumps(out, indent=1))
    return out


def profile(root: Path, arm: str, *, batches: int = 5, seed: int = 1) -> dict:
    """Where the CPU goes for one arm on the prepared data: window gather, forward+backward, validation predict, build —
    measured separately, so no bottleneck is asserted from a total (RP77/A6)."""
    root = Path(root)
    design = json.loads((root/"DESIGN.json").read_text())
    validate(design, strict_code=False)
    data = load_data(root, design)
    spec = next(a for a in design["arms"] if a["arm"] == arm)
    R = design["recipe"]
    W, h, j = int(spec["window"]), int(data["horizon"][0]), int(data["target_channel"][0])
    X, assignment = arm_inputs(data, spec, assignment=design["source_run"]["graph_assignment"])
    m, sd = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    tr = Batches(X, data["Y"], data[f"train_origins__{arm}"], W, h, j, int(R["batch"]), mean=m, sd=sd, shuffle=True, seed=seed)
    t0 = time.process_time(); model = build_model(spec, assignment, X.shape[1], j, seed); t_build = time.process_time()-t0
    tf = _module("df_mod_e0")._tf()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(R["learning_rate"])), loss=R["loss"])
    xb, yb = tr[0]; model.train_on_batch(xb, yb)                                        # warm-up (graph tracing) excluded
    t_gather = t_step = t_pred = 0.0
    for i in range(1, batches+1):
        t0 = time.process_time(); xb, yb = tr[i]; t_gather += time.process_time()-t0
        t0 = time.process_time(); model.train_on_batch(xb, yb); t_step += time.process_time()-t0
        t0 = time.process_time(); model.predict_on_batch(xb); t_pred += time.process_time()-t0
    per = {"gather_seconds_per_batch": t_gather/batches, "forward_backward_seconds_per_batch": t_step/batches,
           "predict_seconds_per_batch": t_pred/batches, "build_seconds": t_build}
    total = per["gather_seconds_per_batch"]+per["forward_backward_seconds_per_batch"]
    doc = {"schema": "df_e1_block_profile.v1", "arm": arm, "window": W, "channels": int(X.shape[1]), "batch": int(R["batch"]), "batches": batches,
           "per_batch": per, "share_of_a_train_update": {"gather": per["gather_seconds_per_batch"]/total, "forward_backward": per["forward_backward_seconds_per_batch"]/total},
           "parameters": n_params(model), "reading": "a measurement of where one update's CPU goes; it prescribes no optimization"}
    (root/f"PROFILE_{arm}.json").write_text(json.dumps(doc, indent=1))
    return doc


def cmd_prepare(a):
    design = json.loads((Path(a.root)/"DESIGN.json").read_text())
    validate(design)
    G, U = governance_modules()
    root = Path(a.root)
    started = U._z(U.now_iso())
    _acquire(a, design, "prepare")
    t0 = time.process_time()
    try:
        rec = prepare(design, root)
    except BaseException as exc:
        G.report_failed(root, "prepare", f"prepare refused: {str(exc)[:200]}", gov_url=a.gov_url, api_key_file=a.api_key_file)
        raise
    # the prepare unit closes like any other: a COMPLETED terminal with the prepared data's digests as artifacts
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": time.process_time()-t0, "cpu_seconds": time.process_time()-t0},
                           metrics=[U._metric("e1.block.common_evaluation_rows", rec["common_evaluation"]["n"], "rows", split="validation", horizon=H0)],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": "prepare",
                                 "role": "PREPARATION", "design_sha256": design["design_sha256"]})
    terminal["artifacts"] = [{"role": r, "sha256": sha_file(root/f), "bytes": (root/f).stat().st_size} for r, f in (("data", "BLOCK_DATA.npz"), ("record", "BLOCK_DATA.json"))]
    (root/"TERMINALS").mkdir(exist_ok=True)
    write(root/"TERMINALS"/"prepare.json", terminal)
    reported = G.report_terminal(root, "prepare", terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root/"outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise BlockRefusal(f"REFUSED: the prepare terminal was not accepted: {reported['flushed']['failures']}")
    print(json.dumps({k: rec[k] for k in ("common_evaluation", "binding_to_source", "feasibility")}, indent=1))


def cmd_pilot(a):
    design = json.loads((Path(a.root)/"DESIGN.json").read_text())
    validate(design)
    load_data(a.root, design)
    results = run_units(a, design, design["pilots"], parallel=min(LIMITS["parallel_children"], len(design["pilots"])))
    proj = projection(design, results)
    fits = proj["with_headroom_25_percent"] + spent_cpu(a.root) + LIMITS["closure_reserve_seconds"] <= LIMITS["campaign_cpu_seconds"]
    doc = {"schema": "df_e1_block_pilot_report.v1", "design_sha256": design["design_sha256"], "spent_cpu_seconds": spent_cpu(a.root),
           "projection": proj, "fits_the_ceiling": fits, "decision": "EXECUTE" if fits else "BUDGET_LIMITED_BEFORE_ANY_OUTCOME"}
    (Path(a.root)/"REPORT.pilot.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(doc, indent=1))
    return 0 if fits else 2


def cmd_execute(a):
    design = json.loads((Path(a.root)/"DESIGN.json").read_text())
    validate(design)
    load_data(a.root, design)
    if a.decision_from:
        pilot = json.loads((Path(a.decision_from)/"REPORT.pilot.json").read_text())   # a worker executes under the coordinator's pilot decision
    else:
        pilot = json.loads((Path(a.root)/"REPORT.pilot.json").read_text())
    if pilot["decision"] != "EXECUTE":
        raise BlockRefusal("REFUSED: the cost pilot did not project inside the ceiling")
    cells = [c for c in design["cells"] if not a.seeds or c["seed"] in a.seeds]           # a host block = the cells of its seeds
    run_units(a, design, cells, parallel=a.parallel or LIMITS["parallel_children"])
    print(json.dumps({"host": os.uname().nodename, "cells": [c["cell_id"] for c in cells], "spent_cpu_seconds": spent_cpu(a.root)}))


def merge(into: Path, sources: list) -> dict:
    """Bring the cells a worker ran into the coordinator's root: attempts, terminal payloads, receipts and deliveries per unit,
    verified against the terminal's artifact digests; the prepared data must be byte-identical (the portability check)."""
    import shutil
    into = Path(into)
    design = json.loads((into/"DESIGN.json").read_text())
    here = json.loads((into/"BLOCK_DATA.json").read_text())
    out = {"schema": "df_e1_block_merge.v1", "into": str(into), "units": {}, "problems": []}
    for src in sources:
        src = Path(src)
        d2 = json.loads((src/"DESIGN.json").read_text())
        if d2["design_sha256"] != design["design_sha256"]:
            out["problems"].append(f"{src}: another design"); continue
        there = json.loads((src/"BLOCK_DATA.json").read_text())
        if there["data_sha256"] != here["data_sha256"]:
            out["problems"].append(f"{src}: the prepared data differ (portability): {there['data_sha256'][:12]} vs {here['data_sha256'][:12]}"); continue
        rc_src = json.loads((src/"TERMINAL_RECEIPTS.json").read_text()).get("units", {}) if (src/"TERMINAL_RECEIPTS.json").is_file() else {}
        dl_src = json.loads((src/"DELIVERIES.json").read_text()) if (src/"DELIVERIES.json").is_file() else {"units": {}}
        rc_path, dl_path = into/"TERMINAL_RECEIPTS.json", into/"DELIVERIES.json"
        rc = json.loads(rc_path.read_text()) if rc_path.is_file() else {"schema": "df_e1_terminal_receipts.v1", "units": {}}
        dl = json.loads(dl_path.read_text()) if dl_path.is_file() else {"schema": dl_src.get("schema"), "units": {}}
        for tpath in sorted((src/"TERMINALS").glob("*.json")):
            unit = tpath.stem
            if unit == "prepare" or not (src/"attempts"/unit/"cell.json").is_file():
                continue
            terminal = json.loads(tpath.read_text())
            arts = {x["role"]: x["sha256"] for x in terminal.get("artifacts", [])}
            ok = all(sha_file(src/"attempts"/unit/f) == arts.get(r) for r, f in (("predictions", "arrays.npz"), ("record", "cell.json"), ("weights", "weights.weights.h5")))
            if not ok or unit not in rc_src:
                out["problems"].append(f"{unit}: artifacts do not match the terminal or no receipt"); continue
            if (into/"attempts"/unit).exists():
                out["problems"].append(f"{unit}: already present in the coordinator root; not overwritten"); continue
            shutil.copytree(src/"attempts"/unit, into/"attempts"/unit)
            (into/"TERMINALS").mkdir(exist_ok=True)
            shutil.copy2(tpath, into/"TERMINALS"/f"{unit}.json")
            rc["units"][unit] = rc_src[unit]
            if unit in dl_src.get("units", {}):
                dl["units"][unit] = dl_src["units"][unit]
            out["units"][unit] = {"from": str(src), "host": json.loads((src/"attempts"/unit/"cell.json").read_text())["cost"].get("host")}
        rc_path.write_text(json.dumps(rc, indent=1)); dl_path.write_text(json.dumps(dl, indent=1, default=str))
    (into/"MERGE.json").write_text(json.dumps(out, indent=1))
    return out


# --- closure -------------------------------------------------------------------------------------------------------

def baselines(data: dict, design: dict) -> dict:
    """Three DISTINCT references on the common evaluation rows: persistence, daily seasonal persistence, train-only constant."""
    H = _module("df_e1_huber")
    h, j = int(data["horizon"][0]), int(data["target_channel"][0])
    Y, o = data["Y"], data["common_eval"]
    sd = float(data["scaler_sd"][j])
    y, naive = Y[o+h], Y[o]
    base_arm = next((a["arm"] for a in design["arms"] if a["train_days"] == 28), None)
    if base_arm is not None:
        train_o = data[f"train_origins__{base_arm}"]
    else:                                                   # no 28 d arm in this block: the 28 d tier IS the source run's train origins
        src = design["source_run"]
        with np.load(Path(src["root"])/"DATA.npz", allow_pickle=False) as z:
            train_o = z["train_origins"] + (src["slice_rows"][0] - int(data["row_offset"][0]))
    const = float(np.mean(Y[train_o+h]))
    out = {}
    for name, pred in (("persistence", naive), ("daily_seasonal", Y[o+h-DAY]), ("train_constant", np.full(o.size, const))):
        out[name] = {**H.metrics(pred, y, naive, sd), "definition": design["baselines"][name]}
    return out


def replay_cell(root: Path, unit: str) -> dict:
    """RP86: a READ-BOUND checkpoint replay in a fresh process — rebuild the model, load the saved weights, predict the common
    evaluation windows and compare with the stored predictions under the existing rule allclose(atol=1e-6, rtol=1e-6)."""
    code = f"""
import json, sys, importlib.util, numpy as np
from pathlib import Path
spec = importlib.util.spec_from_file_location("df_e1_block", {str(Path(__file__).resolve())!r}); K = importlib.util.module_from_spec(spec); sys.modules["df_e1_block"] = K; spec.loader.exec_module(K)
root = Path({str(root)!r}); unit = {unit!r}
design = json.loads((root/"DESIGN.json").read_text()); data = K.load_data(root, design)
rec = json.loads((root/"attempts"/unit/"cell.json").read_text()); spec_a, seed = rec["arm_spec"], int(rec["cell"]["seed"])
X, asg = K.arm_inputs(data, spec_a, assignment=design["source_run"]["graph_assignment"])
j, h = int(data["target_channel"][0]), int(data["horizon"][0]); m, sd = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
ds = K.Batches(X, data["Y"], data["common_eval"], int(spec_a["window"]), h, j, int(design["recipe"]["batch"]), mean=m, sd=sd, shuffle=False, seed=seed)
model = K.build_model(spec_a, asg, X.shape[1], j, seed); model.load_weights(root/"attempts"/unit/"weights.weights.h5")
pred = K.predict(model, ds).astype(np.float64)*sd + m
with np.load(root/"attempts"/unit/"arrays.npz", allow_pickle=False) as z: stored, y = z["pred"], z["y"]
print(json.dumps({{"unit": unit, "allclose_1e_6": bool(np.allclose(pred, stored, atol=1e-6, rtol=1e-6)), "max_abs_prediction_difference": float(np.max(np.abs(pred-stored))),
                   "mae_z_replayed": float(np.mean(np.abs(pred-y))/sd), "mae_z_stored": float(np.mean(np.abs(stored-y))/sd)}}))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3", "OMP_NUM_THREADS": "1"})
    if proc.returncode:
        return {"unit": unit, "allclose_1e_6": False, "error": proc.stderr[-600:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def replay_identity(root: Path, unit: str, design: dict, cell: dict) -> dict:
    """RP90: the identity a cached replay must match, computed from the BYTES on disk now — the checkpoint that a replay
    would load, the arrays it would compare with, the sealed design and the replay code itself — plus the cell."""
    folder = Path(root)/"attempts"/unit
    return {"unit": unit, "cell": {"arm": cell.get("arm"), "seed": cell.get("seed")}, "design_sha256": design.get("design_sha256"),
            "weights_file_sha256": sha_file(folder/"weights.weights.h5") if (folder/"weights.weights.h5").is_file() else None,
            "arrays_sha256": sha_file(folder/"arrays.npz") if (folder/"arrays.npz").is_file() else None,
            "replay_code_sha256": sha_file(Path(__file__).resolve())}


def _replay_evidence_binds(entry: dict, unit: str, ident: dict, rec: dict, row: dict | None) -> bool:
    """An ADOPTED external replay (Musashi's fresh-process replays) binds only when the checkpoint and arrays bytes on disk are
    the accepted record's, the entry names this unit, and its stored MAE_z equals the one recomputed independently now."""
    try:
        return (entry.get("unit") == unit and ident["weights_file_sha256"] is not None
                and ident["weights_file_sha256"] == rec.get("weights_file_sha256") and ident["arrays_sha256"] == rec.get("arrays_sha256")
                and row is not None and row.get("model_error_z") is not None
                and abs(float(entry.get("mae_z_stored")) - float(row["model_error_z"])) <= 1e-12 and bool(entry.get("allclose_1e_6")))
    except (TypeError, ValueError):
        return False


def close(a) -> dict:
    """RP82: the closure CONSUMES the authoritative verification (tools/df_closure_table.verify_run) — rows, preparation
    custody, design identity, denominator — and adds what only the block knows (reload parity, paired initial weights,
    censoring); a failed verification emits no summary, no paired contrast and no baselines."""
    root = Path(a.root)
    design = json.loads((root/"DESIGN.json").read_text())
    validate(design, strict_code=False)
    data = load_data(root, design)
    T = _module("df_closure_table")
    B = _module("df_benchmark_contract")
    C = _module("df_mod_e0_close")
    token = Path(a.warehouse_token_file).read_text().strip().strip('"').strip("'") if getattr(a, "warehouse_token_file", None) else None
    warehouse = (lambda campaign: C.warehouse_terminals(a.warehouse_url, token, campaign)) if token else None
    verification = T.verify_run(root, label=root.name, registry=B.registry(), warehouse=warehouse)
    rows_by_unit = {r["unit"]: r for r in verification["rows"]}
    expected = [c["cell_id"] for c in design["cells"]]
    problems = list(verification["problems"])
    if warehouse is None:
        problems.append("closure without a warehouse read: no accepted custody, nothing is verified")
    missing = [u for u in expected if u not in rows_by_unit]
    if missing:
        problems.append(f"population: registered cells without a scored row: {missing}")
    inits, rows, replays = {}, [], {}
    adopted = getattr(a, "replay_evidence", None)
    adopted_rows = {}
    if adopted:
        doc = json.loads(Path(adopted).read_text())
        adopted_rows = {c["cell_id"]: {"adopted_from": str(adopted), **(c.get("replay") or {})} for c in doc.get("cells", []) if not c.get("problems")}
    # RP90 (Musashi RP89 #1): a cached replay is reused ONLY when the replay identity recomputed from the ACTUAL bytes now on
    # disk (checkpoint, arrays, design, replay code) equals the one recorded at replay time; a record's claimed digest is a
    # claim, never the cache key. Entries without an identity (older REPLAYS.json) are never reused.
    prior = json.loads((root/"REPLAYS.json").read_text()) if (root/"REPLAYS.json").is_file() else {}
    report_paired = None
    for cell in design["cells"]:
        unit = cell["cell_id"]
        folder = root/"attempts"/unit
        if not (folder/"cell.json").is_file():
            continue
        rec = json.loads((folder/"cell.json").read_text())
        if rec["design_sha256"] != design["design_sha256"] or rec["cell"] != cell:
            problems.append(f"{unit}: cell/design mismatch in the record")
        with np.load(folder/"arrays.npz", allow_pickle=False) as z:
            if not np.allclose(z["pred"], z["reload_pred"], rtol=1e-6, atol=1e-6):
                problems.append(f"{unit}: reload parity")
        # 2026-09-26: the key must name EVERY field that changes the graph, or two arms that are different models collide
        # and the check reports a false "unpaired" problem. `dilations` and `crop` change the built graph (and therefore the
        # parameter count and the draw order), so they belong in the key exactly like `family`, `channels` and `window`.
        # What the check still enforces is its real intent: two arms with the SAME graph and the same seed, differing only in
        # the DATA they are fed (calendar vs randomised_calendar_control), must start from the same initial weights.
        inits.setdefault((cell["seed"], rec["arm_spec"]["family"], rec["channels"], rec["arm_spec"]["window"],
                          tuple(rec["arm_spec"].get("dilations") or ()), rec["arm_spec"].get("crop")),
                         set()).add(rec["initial_weights_sha256"])
        r = rows_by_unit.get(unit)
        # the checkpoint CONSUMED is hashed now and bound to the accepted record's claim (the record is in the accepted chain)
        ident = replay_identity(root, unit, design, cell)
        if ident["weights_file_sha256"] != rec.get("weights_file_sha256"):
            problems.append(f"{unit}: the checkpoint bytes on disk are not the accepted record's checkpoint: CHANGED CHECKPOINT")
        if ident["arrays_sha256"] != rec.get("arrays_sha256"):
            problems.append(f"{unit}: the arrays on disk are not the record's arrays: CHANGED ARRAYS (replay identity)")
        cached = prior.get(unit) or {}
        # RP86: a read-bound fresh-process replay for every NEWLY measured artifact; a replay of IDENTICAL bytes is not repeated
        if cached.get("identity") == ident and "allclose_1e_6" in cached:
            replays[unit] = {**{k: v for k, v in cached.items() if k != "identity"},
                             "adopted_from": "REPLAYS.json (identical checkpoint, arrays, design and replay-code bytes, hashed now)"}
        elif unit in adopted_rows and _replay_evidence_binds(adopted_rows[unit], unit, ident, rec, r):
            replays[unit] = {**adopted_rows[unit], "bound_by": "actual checkpoint and arrays bytes == accepted record; stored MAE_z == independently recomputed"}
        elif not getattr(a, "skip_replay", False):
            replays[unit] = replay_cell(root, unit)
        else:
            replays[unit] = {"skipped": True, "scope": "in-process reload parity only; no fresh-process replay for this closure"}
        if unit in replays and "allclose_1e_6" in replays[unit] and not replays[unit].get("allclose_1e_6"):
            problems.append(f"{unit}: fresh-process checkpoint replay exceeds allclose(1e-6, 1e-6)")
        replays[unit]["identity"] = ident
        rows.append({**cell, "mae_z": r["model_error_z"] if r else None, "mae_kw": r["model_error"] if r else None,
                     "naive_mae_z": r["naive_error_z"] if r else None, "skill_vs_naive": (r["skill_vs_naive"] or {}).get("value") if r else None,
                     "verified": bool(r and r.get("verified")), "custody": (r or {}).get("custody"), "parameters": rec["parameters"],
                     "updates": rec["training"]["updates"], "validation_events": rec["training"]["validation_events"], "best_update": rec["training"]["best_update"],
                     "stop": rec["training"]["stop_reason"], "triggers": rec["training"].get("triggers"), "censoring": rec["training"]["censoring"]["verdict"],
                     "host": rec["cost"].get("host"), "cpu_seconds": rec["cost"]["cpu_seconds"], "peak_rss_bytes": rec["cost"]["peak_rss_bytes"],
                     "reload_max_error": rec["reload_max_error"], "initial_weights_sha256": rec["initial_weights_sha256"],
                     "environment": rec.get("environment")})
    unpaired = [k for k, v in inits.items() if len(v) != 1]
    if unpaired:
        problems.append(f"unpaired initial weights within a seed for the same graph: {unpaired}")
    verified_all = not problems and all(r["verified"] for r in rows) and len(rows) == len(expected)
    (root/"REPLAYS.json").write_text(json.dumps({u: r for u, r in replays.items() if "allclose_1e_6" in r}, indent=1, default=str))
    disposition = B.disposition(design)
    report = {"schema": "df_e1_block_report.v3", "design_sha256": design["design_sha256"], "block": design["block"], "paired": None,
              # RP90: every block of the household task is HISTORICAL_DEV_ONLY — its summary is a preserved measurement, not a
              # selection; nothing here proposes an architecture, optimizer, loss or policy
              "disposition": disposition, "active_selection": None,
              "verification": {k: verification[k] for k in ("design_identity", "preparation_custody", "denominator", "verified_units", "unverified_units")},
              "common_evaluation_rows": int(data["common_eval"].size), "sigma_evaluation": verification["denominator"]["sd_used"],
              "rows": rows, "replays": replays, "problems": problems, "verified": verified_all, "spent_cpu_seconds": spent_cpu(root),
              "closure_code_drift": code_drift(design) or "none: closed under the sealed code",
              "scope": "DEVELOPMENT; paired seeds within host blocks; one previously inspected DEV validation week; no test rows read"}
    if verified_all:
        arms = sorted({r["arm"] for r in rows})
        report["summary"] = {arm: {"n_seeds": len(v := [r["mae_z"] for r in rows if r["arm"] == arm]), "mean_mae_z": float(np.mean(v)),
                                   "sd_mae_z_ddof1": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                   "mean_mae_kw": float(np.mean([r["mae_kw"] for r in rows if r["arm"] == arm])),
                                   "censored_fits": sum(1 for r in rows if r["arm"] == arm and r["censoring"] == "CENSORED_BY_BUDGET"),
                                   "hosts": sorted({r["host"] for r in rows if r["arm"] == arm})} for arm in arms}
        if len(arms) == 2:
            a0, a1 = arms
            d = [next(r["mae_z"] for r in rows if r["arm"] == a1 and r["seed"] == s) - next(r["mae_z"] for r in rows if r["arm"] == a0 and r["seed"] == s)
                 for s in design["seeds"] if any(r["arm"] == a1 and r["seed"] == s for r in rows) and any(r["arm"] == a0 and r["seed"] == s for r in rows)]
            report["paired"] = {"difference": f"{a1} - {a0} in MAE_z, paired by seed within host blocks", "values": d, "mean": float(np.mean(d)) if d else None,
                                "sd_ddof1": float(np.std(d, ddof=1)) if len(d) > 1 else None, "signs": {"positive": sum(x > 0 for x in d), "negative": sum(x < 0 for x in d)},
                                "reading": "three seed/host blocks; both signs are reported; no interval is claimed from n=3"}
        report["baselines"] = baselines(data, design)
    else:
        report["summary"] = None
        report["paired"] = None
        report["baselines"] = None
        report["reading"] = "closure FAILED: no verified comparator, no selected arm and no scientific proposal are emitted"
    (root/"REPORT.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("summary", "paired", "problems", "verified")}, indent=1, default=str))
    if not verified_all:
        raise BlockRefusal(f"REFUSED: closure failed: {problems[:5]}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["seal", "prepare", "pilot", "execute", "child", "close", "recost", "scan", "profile", "merge", "equivalence"])
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="execute only the cells of these seeds (a host block)")
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--decision-from", type=Path, default=None, help="the coordinator root whose REPORT.pilot.json authorises execution")
    ap.add_argument("--from", dest="sources", type=Path, action="append", default=[])
    ap.add_argument("--replay-evidence", type=Path, default=None, help="an independent fresh-process replay to ADOPT for already-replayed cells (not repeated)")
    ap.add_argument("--skip-replay", action="store_true", help="tests only: no fresh-process replay")
    ap.add_argument("--arm")
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--block")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--unit")
    ap.add_argument("--run-id")
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--lake", default=LAKE)
    ap.add_argument("--resource", default=RESOURCE)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path)
    ap.add_argument("--reuse-from", type=Path, help="a CLOSED block root whose modular_w60 cells are this block's baseline (contract must match)")
    a = ap.parse_args(argv)
    if a.command == "seal":
        d = seal(a.block, source_run=a.source_run, reuse=reuse_record(a.reuse_from) if a.reuse_from else None)
        write(a.out, d)
        print(json.dumps({"design_sha256": d["design_sha256"], "block": d["block"], "cells": len(d["cells"]), "pilots": len(d["pilots"])}, indent=1))
        return 0
    if a.command == "child":
        child(a.root, a.unit)
        return 0
    if a.command == "equivalence":
        design = json.loads((a.root/"DESIGN.json").read_text()); data = load_data(a.root, design)
        doc = training_equivalence(data, design, updates=a.updates); (a.root/"EQUIVALENCE.json").write_text(json.dumps(doc, indent=1))
        print(json.dumps({k: doc[k] for k in ("updates", "origins", "max_abs_loss_difference", "final_weights_equal", "equivalent")})); return 0 if doc["equivalent"] else 1
    if a.command == "merge":
        out = merge(a.root, a.sources); print(json.dumps({"units": list(out["units"]), "problems": out["problems"]}, indent=1)); return 0 if not out["problems"] else 1
    if a.command == "recost":
        print(json.dumps({k: v for k, v in recost(a.root).items() if k != "corrected_projection"} | {"corrected_total": recost(a.root)["corrected_projection"]["total_at_ceiling_seconds"]}, indent=1)); return 0
    if a.command == "scan":
        out = scan_stops(a.root); print(json.dumps({"units": len(out["units"]), "corrected": out["corrected"]})); return 0
    if a.command == "profile":
        print(json.dumps(profile(a.root, a.arm, batches=a.batches), indent=1)); return 0
    if a.command == "prepare":
        cmd_prepare(a)
        return 0
    if a.command == "pilot":
        return cmd_pilot(a)
    if a.command == "execute":
        cmd_execute(a)
        return 0
    close(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
