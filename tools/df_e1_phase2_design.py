#!/usr/bin/env python3
"""Phase 2 of the E1 diagnostic, SUCCESSOR design (v2, RP68) — sealed before anything runs.

The v1 draft (27 cells, docs/audits/evidence/d3_k5_20260917/RP65/PHASE2_DESIGN_SEALED.json) is
preserved. Musashi's review measured three things it stated wrongly, and this successor corrects
them at the code that will actually run (tools/df_e1_block.py), not in prose:

  1. the "clamped depth" long-window arm is NOT a null: with the W60 dilations at W=1440 the model
     reaches 67 raw samples (branch 5 + core 63 - 1), measured by gradient and perturbation. It is
     kept as `long_window_local_support_67` — EXTRA CONTEXT, declared — and the exact information
     null is `long_window_crop60`: the raw input cropped to its last 60 rows BEFORE the extractor
     (same weights, same padding, same rows as W60; paired weights and equal outputs are tested).
  2. validation "every epoch" changes the stopping opportunities with the volume tier (627/1257/2517
     updates per pass at 28/56/112 days). Validation now happens every 200 OBSERVED optimizer updates
     with patience 3 EVENTS, fixed across every arm and tier: 20 checkpoint opportunities each. A fit
     that reaches the ceiling is CENSORED wherever its best event fell.
  3. volume arms keep the COMMON train-only scaler of the 28-day baseline and ONE evaluation sigma;
     counts (unique support rows, train-only rows excluding validation support, labels, windows,
     exposures) come from row identities at prepare. A refit-scaler variant is a separate factor,
     NOT run in this phase.

Blocks, in the fixed priority the order sets: DEV_MATCHED (the modular continuity model against the
adapted literature GRU), then Q1 calendar, Q2 context, Q3 volume. Each block registers its own units
under governance; a baseline is reused across blocks only when its complete contract matches.

    python tools/df_e1_phase2_design.py --seal DESIGN.json [--source-run ROOT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "df_e1_phase2_design.v2"
SEEDS = (1, 2, 3)
DAY = 1440
V1_PRESERVED = "docs/audits/evidence/d3_k5_20260917/RP65/PHASE2_DESIGN_SEALED.json"


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reach(model, X: np.ndarray, probe_rows: list) -> dict:
    """Gradient reach on the full graph plus perturbation at declared rows — measured, never asserted."""
    tf = _module("df_mod_e0")._tf()
    xt = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        out = model(xt, training=False)
    g = np.abs(np.asarray(tape.gradient(out, xt))).max(axis=(0, 2))
    rows = np.flatnonzero(g > 1e-9)
    base = np.asarray(model.predict(X, verbose=0))
    pert = {}
    for k in probe_rows:
        Xk = X.copy()
        Xk[:, k, :] += 100.0
        pert[str(k)] = float(np.max(np.abs(np.asarray(model.predict(Xk, verbose=0))-base)))
    return {"reach_by_gradient": int(X.shape[1]-rows.min()) if rows.size else 0,
            "perturbation_max_abs_change_at_row": pert, "windows": int(X.shape[0])}


def capacity_and_reach(assignment: list, p: int, j: int) -> dict:
    """Parameters and measured reach of every arm this phase can run, by building the models."""
    K = _module("df_e1_block")
    P = _module("df_e1_pilot")
    rng = np.random.default_rng(0)
    out = {}
    X60 = rng.normal(size=(2, 60, 7)).astype(np.float32)
    X1440 = rng.normal(size=(2, 1440, 7)).astype(np.float32)
    for arm, spec in K.ARMS.items():
        extra = {"calendar": 4, "randomised_calendar": 4, "daily_lag": 1}.get(spec["features"], 0)
        asg = list(assignment) + ([max(assignment)+1]*4 if extra == 4 else [assignment[j]] if extra == 1 else [])
        model = K.build_model(K.arm_spec(arm), asg, p+extra, j, 1)
        entry = {"window": spec["window"], "channels": p+extra, "parameters": K.n_params(model), "family": spec["family"],
                 "dilations": spec.get("dilations") or (None if spec["family"] == "gru" else P.core_dilations(spec.get("crop") or spec["window"])),
                 "crop": spec.get("crop"), "role": spec.get("role", "ARM")}
        if extra == 0 and spec["window"] == 1440 and spec["family"] == "modular":
            entry["reach_measured"] = _reach(model, X1440, [1440-68, 1440-67, 1440-61, 1440-60, 1439])
        elif extra == 0 and spec["window"] == 60:
            entry["reach_measured"] = _reach(model, X60, [0, 59])
        out[arm] = entry
    return out


def daily_lag_availability(h: int, lag: int = DAY) -> dict:
    return {"channel": f"y(t + h - {lag})", "horizon": h, "lag": lag, "sample_index_relative_to_origin": h-lag,
            "available_at_decision": (h-lag) <= 0,
            "reading": f"the label is y(t+{h}); one day before it is row t+{h}-{lag} = t{h-lag:+d}, which is "
                       f"{'at or before' if (h-lag) <= 0 else 'AFTER'} the origin t",
            "delayed_observation_rule": "this archive declares no publication delay (UNKNOWN is not zero delay: it is "
                                        "undeclared); the wall-clock label is used and said so; the household task stays "
                                        "retrospective/offline and is no evidence of live availability"}


def seal(source_run: Path) -> dict:
    K = _module("df_e1_block")
    E = _module("df_mod_e0")
    source = Path(source_run)
    src = json.loads((source/"DESIGN.json").read_text())
    data_json = json.loads((source/"DATA.json").read_text())
    assignment = src["graph"]["assignment"]
    p, j = len(data_json["input_columns"]), int(data_json["target_channel"])
    caps = capacity_and_reach(assignment, p, j)
    dev = K.seal("DEV_MATCHED", source_run=source)
    blocks = {}
    for name in ("DEV_MATCHED", "Q1_CALENDAR", "Q2_CONTEXT", "Q3_VOLUME"):
        b = K.seal(name, source_run=source)
        blocks[name] = {"question": b["question"], "arms": [a["arm"] for a in b["arms"]], "cells": b["cells"],
                        "pilots": [c["cell_id"] for c in b["pilots"]], "rows": b["rows"], "block_design_sha256": b["design_sha256"],
                        "state": "SEALED_NOT_EXECUTED",
                        "reuse": ("modular_w60 of DEV_MATCHED is the baseline of Q1/Q2/Q3 when its complete contract (rows, "
                                  "inputs, scaler, recipe, cadence, seeds) matches; otherwise the baseline is run inside the block")}
    design = {
        "schema": SCHEMA, "purpose": "E1_DIAGNOSTIC_PHASE_2_SUCCESSOR", "phase": "DEVELOPMENT", "state": "SEALED_NOT_EXECUTED",
        "supersedes": {"v1": V1_PRESERVED, "cells_v1": 27, "why": "Musashi's post-Huber review, findings 3 and 5 (measured 67-sample reach; "
                                                                   "cadence in epochs; volume counts and refitted scalers)"},
        "source_run": {"root": str(source), "design_sha256": src["design_sha256"], "data_sha256": data_json["data_sha256"],
                       "panel_sha256": data_json["panel_sha256"], "evaluation_origins": int(data_json["enumerator"]["validation"]["admissible"])},
        "runner": "tools/df_e1_block.py (features, enumeration, scaling, cadence, governance, closure) — tested through the consumer in "
                  "tests/test_df_e1_block.py",
        "recipe": K.RECIPE, "pilot": K.PILOT, "limits": K.LIMITS,
        "held_constant": ["forecast task W60/h60 on the successor rows", "the COMMON train-only scaler (28 d, W60 windows) and one evaluation sigma",
                          "the common evaluation set derived at prepare", "the continuity recipe with validation in observed updates",
                          "three paired seeds", "no pretraining"],
        "capacity_and_reach_measured_at_seal": caps,
        "context_control": {
            "exact_information_null": "long_window_crop60: raw input cropped to its last 60 rows BEFORE the extractor; paired weights with "
                                      "modular_w60 (same seed -> same initial weights), equal outputs on the same rows, same left padding",
            "extra_context_arm": "long_window_local_support_67: the W60 dilations at W=1440 reach 67 raw samples (measured above); "
                                 "it is declared as extra context, never as a null",
            "long_short_x_shallow_deep": {"short_shallow": "modular_w60", "long_deep": "long_window_own_depth",
                                          "long_shallow_support67": "long_window_local_support_67", "short_deep": "short_window_deep_core",
                                          "reading": "the contrast stays identifiable: window and depth move separately; padding is equalised by the crop arm"},
            "daily_lag": daily_lag_availability(60)},
        "calendar": {"honest": "hour/weekday sin-cos from each row's own label (tools/df_e1_calendar.py); NAIVE_WALL_CLOCK, timezone UNKNOWN, "
                               "DST reported per slice",
                     "control": "randomised_calendar_control: the same four channels from the label plus a per-row random offset hashed from "
                                "(seed, panel row id) — deterministic, prefix-stable, same capacity; its finite-sample association with the true "
                                "clock is measured at prepare and reported, not declared zero",
                     "acceptance": "tests/test_df_e1_block.py: prefix stability of both, distinctness, the shift-minus-one leak fails"},
        "volume": {"tiers_days": [28, 56, 112], "evaluation": "FIXED: the same DEV validation origins, byte-identical across tiers",
                   "scaler": "COMMON (the 28 d baseline's); a refit-scaler variant is a SEPARATE factor, NOT_RUN in this phase",
                   "cadence": "validation every 200 observed updates, patience 3 events, 20 opportunities — identical across tiers",
                   "counts": "unique support rows, train-only rows (validation support excluded), labels, windows, exposures from row "
                             "identities at prepare (tests/test_df_e1_block.py::test_volume_tiers_grow_backwards...)",
                   "extension": "a second update tier (8 000) is a predeclared new budget tier for the SAME cells, declared before the first "
                                "tier's scores are read; never a claim of convergence or of an error improvement"},
        "training_adequacy": {"validation_cadence": "every 200 observed optimizer updates", "patience": 3, "patience_unit": "validation events",
                              "checkpoint_opportunities": 20, "restore_check": "restored predictions reproduce the best validation event",
                              "ceiling_policy": "a fit that reaches 4 000 updates is CENSORED wherever its best event fell",
                              "claims": "finite-budget comparisons only; no convergence claim from a ceiling"},
        "blocks": blocks, "execution_priority": ["DEV_MATCHED", "Q1_CALENDAR", "Q2_CONTEXT", "Q3_VOLUME"],
        "cells": [{"block": name, **c} for name, b in blocks.items() for c in b["cells"]],
        "benchmark_contract": dev["benchmark_contract"],
        "literature_comparator": {"module": "tools/df_gru_reference.py", "what": "GRU-MIMO family of Gasparin 2019 adapted to our one 60-min "
                                  "target and the modular arm's inputs under the continuity recipe; article facts, adaptations and unknowns kept apart",
                                  "baselines": ["persistence", "daily seasonal persistence", "train-only constant"],
                                  "published_values": "Table 5 stays in the source notes (RP66/GASPARIN_2019_REPRODUCTION_CONFIG.json); never in the comparison column"},
        "resource_assignment": {"blocks": "all arms of one seed on ONE host; workers only after their wrapper records parent+child CPU and peak memory",
                                "no_padding": "no cell is repeated to occupy a machine"},
        "reading_rules": ["three seeds on one task are development evidence", "unrun blocks stay NOT_EXECUTED, never 'no effect'",
                          "no cell is removed after its score is seen", "no adequacy margin, cell or epoch is changed from partial outcomes"],
    }
    design["design_sha256"] = E.sha_obj(design)
    return design


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", type=Path, required=True)
    ap.add_argument("--source-run", type=Path, default=Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser())
    a = ap.parse_args(argv)
    design = seal(a.source_run)
    _module("df_d3_campaign").write_once(a.seal, design)
    print(json.dumps({"design_sha256": design["design_sha256"], "cells": len(design["cells"]),
                      "capacity": {k: v["parameters"] for k, v in design["capacity_and_reach_measured_at_seal"].items()},
                      "reach": {k: v["reach_measured"]["reach_by_gradient"] for k, v in design["capacity_and_reach_measured_at_seal"].items() if "reach_measured" in v},
                      "state": design["state"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
