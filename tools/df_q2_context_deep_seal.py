#!/usr/bin/env python3
"""Seal the six-arm Q2_CONTEXT_DEEP block, with its budget override, BEFORE any cell of it has a score.

Why this file exists rather than a shell one-liner: the only two things this block changes against
Q2_CONTEXT_BOUNDED are the arm list (four arms -> the same four plus the two W1440 FULL-DEPTH arms) and the
recipe's update ceiling (4 000 -> 600 observed updates), plus the resource limits a 7.9-10.3 GiB resident
cell needs. Those overrides are science-relevant, so they are committed as code and their reason is written
beside them, not typed into a terminal and lost.

Everything else is `tools/df_e1_block.seal` unchanged: the same source run, panel, rows, pad, scaler,
common-evaluation rule, COMMON_INTERSECTION train population, loss, optimizer, learning rate, batch,
validation cadence (every 200 observed updates), patience (3 events), min_delta, restore_best, monitor,
metric, scale and the same three seeds.

The ceiling comes from the RETAINED cost pilots' own measured rates on this host and from nothing else; the
block's `informed_by` field carries those measurements verbatim. Its cost is that EVERY cell of EVERY arm is
CENSORED_BY_BUDGET, and the block's own `question` field says so, with the UNDERTRAINED_AT_CEILING rule that
refuses to read a contrast as an effect for an arm still improving at the ceiling.

    python tools/df_q2_context_deep_seal.py --out ROOT/DESIGN.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MAX_UPDATES = 600
BUDGET_DECLARATION = (
    "600 observed updates, not the 4 000 of Q2_CONTEXT_BOUNDED, chosen BEFORE any cell of this block had a score from the "
    "retained cost pilots' own measured rates on this host: the two W1440 full-depth arms cost 4.165 and 4.4427 CPU s per "
    "train update, i.e. 17 507 s and 18 406 s of CPU per cell at a 4 000-update ceiling and roughly 19 h of wall for their "
    "six cells, against about 2 630 s and 2 760 s per cell at 600. Consequence, declared here and not discovered later: "
    "with validation every 200 updates there are 3 checkpoint opportunities and patience 3 can never expire, so EVERY cell "
    "of EVERY arm reaches the ceiling and is CENSORED_BY_BUDGET wherever its best checkpoint fell. Nothing in this block "
    "claims convergence, and the UNDERTRAINED_AT_CEILING rule in the block's question field refuses to read a contrast as "
    "an effect for an arm still improving at the ceiling."
)
LIMITS = {
    "child_cpu_seconds": 5400,
    "child_wall_seconds": 7200,
    "parallel_children": 1,
    "campaign_cpu_seconds": 32400,
    "closure_reserve_seconds": 2000,
    "why": "one cell resident at a time (parallel_children 1) because a single W1440 full-depth cell holds 7.9-9.6 GiB; the "
           "per-child CPU and wall ceilings are 1.9x and 2.6x the projected cost of the most expensive cell at this update "
           "ceiling, so a slower host loses no cell to a resource kill; the campaign ceiling covers the 18 fits, the 6 cost "
           "pilots, the closure and its reserve with headroom",
}


def _block():
    spec = importlib.util.spec_from_file_location("df_e1_block", HERE / "df_e1_block.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["df_e1_block"] = mod
    spec.loader.exec_module(mod)
    return mod


def build() -> dict:
    B = _block()
    recipe = {**B.RECIPE, "max_updates": MAX_UPDATES,
              "checkpoint_opportunities": MAX_UPDATES // B.RECIPE["validate_every_updates"],
              "budget_declaration": BUDGET_DECLARATION}
    tier = ("TIER1 CADENCE AND PATIENCE, BUDGET-MATCHED CEILING: " + B.TIERS["TIER1"]["why"]
            + f"; ceiling {MAX_UPDATES} observed updates for every arm of the block, see recipe.budget_declaration")
    return B.seal("Q2_CONTEXT_DEEP", recipe=recipe, limits=LIMITS, tier=tier)


def crossing(design: dict) -> dict:
    """The 2x2 the block declares, DERIVED from the sealed arms and from nothing typed: each arm's raw input window, the
    dilation list its graph is actually built with, the depth that list gives, and the causal receptive field
    1 + 2*sum(dilations) that decides how many raw samples the arm can USE. This is what makes the confound visible
    before any number exists: an arm's usable context is min(window_after_crop, receptive_field), so 1440 raw rows are
    only information when depth is large enough to reach them."""
    B = _block()
    P = B._module("df_e1_pilot")
    cells = {}
    for a in design["arms"]:
        W = int(a["window"])
        eff_W = int(a["crop"]) if a.get("crop") else W
        dil = list(a["dilations"]) if a.get("dilations") else list(P.core_dilations(eff_W))
        reach = 1 + 2 * sum(dil)
        cells[a["arm"]] = {"raw_window": W, "crop": a.get("crop"), "window_after_crop": eff_W,
                           "dilations": dil, "depth_blocks": len(dil), "receptive_field_samples": reach,
                           "usable_context_samples": min(eff_W, reach),
                           "parameters": design["capacity"][a["arm"]]["parameters"],
                           "features": a["features"], "role": a.get("role"),
                           "train_days": a["train_days"]}
    def find(window, depth):
        return [k for k, v in cells.items() if v["raw_window"] == window and v["depth_blocks"] == depth and v["features"] == "base"
                and not v["crop"]]
    grid = {f"window_{w}__depth_{d}": find(w, d) for w in (60, 1440) for d in (5, 10)}
    return {"schema": "q2_context_deep_extension_seal.v1",
            "sealed_at_state": design["state"], "design_sha256": design["design_sha256"], "block": design["block"],
            "extends": {"block": "Q2_CONTEXT_BOUNDED",
                        "design_sha256": "47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea",
                        "what_is_added": ["long_window_local_support_67", "long_window_own_depth"],
                        "what_is_refitted": "all six arms, under one budget, so no arm is compared across budgets"},
            "train_population": design["train_population"], "train_population_why": design["train_population_why"],
            "seeds": design["seeds"], "recipe": design["recipe"], "limits": design["limits"],
            "common_evaluation_rule": design["common_evaluation_rule"], "scaler_rule": design["scaler_rule"],
            "question": design["question"], "informed_by": design["factorial"]["informed_by"],
            "reading_rules": design["reading_rules"],
            "arms_derived": cells, "crossing_window_x_depth": grid,
            "contrasts_declared_before_any_score": {
                "context_at_matched_depth_10": "long_window_own_depth - short_window_deep_core",
                "context_at_matched_depth_5": "long_window_local_support_67 - modular_w60",
                "depth_at_matched_context_60": "short_window_deep_core - modular_w60",
                "depth_at_matched_context_1440": "long_window_own_depth - long_window_local_support_67",
                "interaction": "(long_window_own_depth - short_window_deep_core) - (long_window_local_support_67 - modular_w60)",
                "exact_information_null": "long_window_crop60 - modular_w60, expected 0 by RP87 and measured, not assumed",
                "causal_channel": "daily_lag - modular_w60",
                "volume": "held fixed BY CONSTRUCTION: one COMMON_INTERSECTION train origin set for every arm",
                "pairing": "every contrast is taken WITHIN a seed and then reported per seed with its sign count; no interval "
                           "is claimed from n = 3 and a direction is never called an effect on n = 3"},
            "undertrained_at_ceiling_rule": {
                "statistic": "improvement in validation MAE_z over the last 200 observed updates, i.e. "
                             "val_mae(second-to-last event) - val_mae(last event), from each cell's own retained events",
                "threshold": 0.005,
                "verdict": "an arm whose improvement exceeds the threshold in 2 or more of its 3 seeds is "
                           "UNDERTRAINED_AT_CEILING, and no contrast involving it is read as a context or depth effect",
                "declared": "before any cell of this block was fitted"},
            "what_this_seal_does_not_settle": [
                "nothing at convergence: every cell reaches the 600-update ceiling and is CENSORED_BY_BUDGET",
                "no context effect beyond 67 raw samples at depth 5: the architecture family cannot reach further without "
                "depth, so that cell of the crossing does not exist and is not claimed to",
                "nothing verified: this host holds no data-gov service key, so no terminal is ever accepted and the "
                "authoritative verifier reports no model error at all for every unit",
                "no causal effect: a difference between two forecast errors is not a causal claim, and no retained-row "
                "error verifies one"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crossing-out", type=Path, help="write the derived crossing/extension seal record beside the design")
    ap.add_argument("--from-design", type=Path, help="read an already sealed design instead of building it (crossing only)")
    a = ap.parse_args(argv)
    B = _block()
    if a.from_design:
        design = json.loads(a.from_design.read_text())
        B.validate(design)
        if a.crossing_out:
            a.crossing_out.write_text(json.dumps(crossing(design), indent=1, sort_keys=True, default=str))
            print(json.dumps({"crossing_out": str(a.crossing_out), "design_sha256": design["design_sha256"],
                              "crossing": crossing(design)["crossing_window_x_depth"]}, indent=1))
        return 0
    design = build()
    B.validate(design)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(design, indent=1, sort_keys=True, default=str))
    if a.crossing_out:
        a.crossing_out.write_text(json.dumps(crossing(design), indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": str(a.out), "design_sha256": design["design_sha256"], "block": design["block"],
                      "state": design["state"], "arms": [x["arm"] for x in design["arms"]],
                      "cells": len(design["cells"]), "pilots": len(design["pilots"]),
                      "max_updates": design["recipe"]["max_updates"], "seeds": design["seeds"],
                      "train_population": design["train_population"], "limits": design["limits"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
