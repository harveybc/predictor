"""Lane C2: the closed interval calibration as an explicit, applicable rule (owner order MAINLINE_PARALLEL_CONTINUATION 4.C.1).

Rule `c2_interval_rule.v1` (DEVELOPMENT), for a population of (feature, horizon) cells whose rung-2 partial association is
estimated by the cross-fitted partially linear model of `c2_causal_dossier`:

* block length  L = max(tau_Y, tau_X, h + 6), tau = 1 + 2 * sum rho_k up to the first lag with rho_k < 0.05 (cap n / 20);
* interval      circular moving-block bootstrap of the score with block length L, B >= 200 (seeded); a cell REJECTS the null
                when |theta| > 1.96 * se_boot. The wider HAC (bandwidth L) is the accepted alternative;
* thresholds    over the cells of a population: the scrambled-label rejection rate must be <= 0.05 + 2*sqrt(0.05*0.95/n_cells);
                the noise-treatment rejection rate must satisfy the same bound; the future-shift template must fire in 100 %
                of cells; otherwise the battery is BATTERY_SUSPECT and NO interval of that population may be read;
* rung state    this rule never moves a rung: a calibrated interval is necessary, not sufficient, for identification.

`check(calibration_json)` applies the thresholds to a `c2_battery_calibration.v1` document and returns a verdict; the CLI prints it
and exits non-zero on BATTERY_SUSPECT. Runnable on the committed ETH 4h evidence:

    python tools/c2_interval_rule.py --calibration docs/audits/evidence/lane_c2_eth_20261001/calibration/BATTERY_CALIBRATION.json \
        --dossier-index docs/audits/evidence/lane_c2_eth_20261001/dossiers/DOSSIER_INDEX.json

Producing a calibration for a population (the command the rule prescribes):
  ETH 4h variant A:   python tools/c2_battery_calibration.py --view <ethusdt_4h_..._model_ready.csv> --manifest <manifest FROZEN_DEVELOPMENT> \
                          --split <SPLIT_eth4h_l24_h6_v1.json> --out <dir> --boot 200
  EURUSD 1h variant A: the same command with that population's view, manifest and split. NOT YET MEASURED here: `bind_population` is bound to
                          the ETH digests, so a EURUSD bind needs its own manifest/split digests passed to it (the rule states the thresholds; the
                          measured block lengths and rates for EURUSD do not exist until that run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULE = {
    "schema": "c2_interval_rule.v1", "label": "DEVELOPMENT",
    "block_length": "max(tau_Y, tau_X, h + 6); tau = 1 + 2 sum rho_k to first rho_k < 0.05, cap n/20",
    "interval": "circular moving-block bootstrap, block L, B >= 200, seeded; or HAC bandwidth L",
    "reject_if": "abs(theta) > 1.96 * se",
    "nominal": 0.05, "tolerance": "nominal + 2 * sqrt(0.05 * 0.95 / n_cells)",
    "thresholds": {"scrambled_label_rate_max": "tolerance", "noise_treatment_rate_max": "tolerance", "future_shift_fired_min": 1.0},
    "never": "moves a rung state; a calibrated interval is necessary, not sufficient, for identification",
    "measured": {"ETH_4h_variant_A": "calibration/BATTERY_CALIBRATION.json (498 cells)", "EURUSD_1h_variant_A": "NOT_MEASURED"},
}


def tolerance(n_cells: int, nominal: float = 0.05) -> float:
    return nominal + 2 * (nominal * (1 - nominal) / max(n_cells, 1)) ** 0.5


def check(calibration: dict, dossier_index: dict | None = None) -> dict:
    agg = calibration["aggregate"]
    n = int(agg["cells"])
    tol = tolerance(n)
    methods = {m: agg[f"scrambled_rate_{m}"] <= tol for m in ("hac_lag_h6", "hac_bandwidth_L", "block_bootstrap_L")}
    out = {"rule": RULE["schema"], "cells": n, "tolerance": tol, "scrambled_rate_by_method": {m: agg[f"scrambled_rate_{m}"] for m in methods},
           "calibrated_methods": [m for m, ok in methods.items() if ok], "block_length": {"min": agg["block_length_min"], "median": agg["block_length_median"], "max": agg["block_length_max"]}}
    problems = []
    if not methods["block_bootstrap_L"]:
        problems.append("SCRAMBLED_RATE_ABOVE_TOLERANCE_FOR_BLOCK_BOOTSTRAP")
    if dossier_index is not None:
        b = dossier_index["battery"]
        if b["future_shift_template_fired"] != b["cells"]:
            problems.append("FUTURE_SHIFT_TEMPLATE_DID_NOT_FIRE_IN_EVERY_CELL")
        if b["noise_null_rejected_5pct"] / b["cells"] > tol:
            problems.append("NOISE_TREATMENT_RATE_ABOVE_TOLERANCE")
    out["problems"] = problems
    out["verdict"] = "CONTROLS_FAIL_AS_REQUIRED" if not problems else "BATTERY_SUSPECT"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", required=True)
    p.add_argument("--dossier-index", default=None)
    p.add_argument("--emit-rule", default=None, help="write the rule document here")
    args = p.parse_args(argv)
    cal = json.loads(Path(args.calibration).read_text())
    idx = json.loads(Path(args.dossier_index).read_text()) if args.dossier_index else None
    res = check(cal, idx)
    print(json.dumps(res, indent=1))
    if args.emit_rule:
        Path(args.emit_rule).write_text(json.dumps(RULE, indent=1))
    return 0 if res["verdict"] == "CONTROLS_FAIL_AS_REQUIRED" else 1


if __name__ == "__main__":
    sys.exit(main())
