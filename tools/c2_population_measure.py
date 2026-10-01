"""Lane C2: the first measurement of a population under the calibrated interval rule, with the zero-return and
intercept-only controls (owner order 4.C.1; coordinator 2026-10-01). Populations: ETH 4h (default) or EURUSD 1h (`--population eurusd_1h`),
both bound by `c2_eth_population.PopulationSpec`; TRAIN rows only; calibrated block-bootstrap interval of `c2_interval_rule`.

Per horizon h (bars): zero-return naive, intercept-only (train mean), 24 h seasonal, all-features ridge, all held out within TRAIN
in five time blocks with embargo (same rows for every arm). Per (feature, h) cell: the partial association theta of Y_h with the
feature given the others, with the interval of three kinds (HAC h+6 before, HAC bandwidth L, block bootstrap L), and the controls
that must fail run S times per cell (scrambled label, noise treatment) plus the RL01 future-shift template. The rates over cells x S draws
feed `c2_interval_rule.check`, whose verdict says whether any interval of this population may be read.
Never moves a rung: the output carries FINDING_NOT_EFFECT_DEVELOPMENT.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import ETH_4H_SPEC, EURUSD_1H_SPEC, bind_population, blocked_splits  # noqa: E402
from c2_causal_dossier import crossfit_residuals, hac_variance, shift_control_detects_future, REDUNDANCY_CORR  # noqa: E402
from c2_battery_calibration import block_bootstrap_se, integrated_autocorr_length, scrambled_labels, aggregate  # noqa: E402
from c2_feature_contribution import run as contribution_run  # noqa: E402
from c2_interval_rule import check, tolerance  # noqa: E402

Z95 = 1.96


def _interval_row(treat, w, y, splits, L, h, rng, boot):
    r_x, r_y, _, _, ok = crossfit_residuals(treat, w, y, splits)
    rx, ry = r_x[ok], r_y[ok]
    denom = float(rx @ rx)
    theta = float(rx @ ry) / denom
    score = rx * (ry - theta * rx)
    m = len(rx)
    se6 = float(np.sqrt(max(hac_variance(score, h + 6) * m, 0.0)) / denom)
    seL = float(np.sqrt(max(hac_variance(score, L) * m, 0.0)) / denom)
    seB = block_bootstrap_se(score, denom, L, boot, rng)
    return {"theta": theta, "se_hac_lag_h6": se6, "se_hac_bandwidth_L": seL, "se_block_bootstrap_L": seB,
            "reject_hac_lag_h6": bool(abs(theta) > Z95 * se6), "reject_hac_bandwidth_L": bool(abs(theta) > Z95 * seL),
            "reject_block_bootstrap_L": bool(abs(theta) > Z95 * seB), "n": int(m)}


def measure_cell(pop, feature, h, *, scrambles=10, boot=100, seed=20261001):
    rng = np.random.default_rng(seed)
    y = pop.m07_target(h)
    xall = pop.feature_matrix_train()
    j = pop.features.index(feature)
    rows = pop.origins[np.isfinite(y[pop.origins])]
    xs = xall[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([np.corrcoef(xs[:, j], xs[:, k])[0, 1] if k != j else 1.0 for k in range(xs.shape[1])])
    w_idx = [k for k in range(len(pop.features)) if k != j and not abs(corr[k]) > REDUNDANCY_CORR]
    splits = blocked_splits(rows, h)
    x_t = xall[:, j]
    n = len(rows)
    L = int(max(integrated_autocorr_length(y[rows], n // 20), integrated_autocorr_length(x_t[rows], n // 20), h + 6))
    out = {"feature": feature, "horizon_bars": h, "n": int(n), "block_length": L, "redundant_with": [pop.features[k] for k in range(len(pop.features)) if k != j and k not in w_idx],
           "tau_y": integrated_autocorr_length(y[rows], n // 20), "tau_x": integrated_autocorr_length(x_t[rows], n // 20)}
    if not w_idx:   # every other feature is a near-copy of the treatment: nothing to adjust for, the cell is not estimable
        out["state"] = "TREATMENT_COLLINEAR_WITH_ALL_CONTROLS"
        return out
    w = xall[:, w_idx]
    brng = np.random.default_rng(seed + 1)
    out["real"] = _interval_row(x_t, w, y, splits, L, h, brng, boot)
    out["scrambled"] = [_interval_row(x_t, w, scrambled_labels(y, np.random.default_rng(seed + 10 + s)), splits, L, h, brng, boot) for s in range(scrambles)]
    out["noise"] = [_interval_row(np.random.default_rng(seed + 100 + s).standard_normal(len(x_t)), w, y, splits, L, h, brng, boot) for s in range(scrambles)]
    fired, causal_moved = shift_control_detects_future(x_t, h, [int(v) for v in np.quantile(rows, [0.3, 0.6, 0.9])])
    out["future_shift_template_fired"] = bool(fired and not causal_moved)
    out["state"] = "MEASURED"
    return out


def build_check_documents(cells, scrambles):
    ms = [c for c in cells if c["state"] == "MEASURED"]
    n_draws = len(ms) * scrambles
    agg = {"cells": n_draws, "nominal_rate": 0.05, "block_length_median": float(np.median([c["block_length"] for c in ms])),
           "block_length_min": int(min(c["block_length"] for c in ms)), "block_length_max": int(max(c["block_length"] for c in ms))}
    for method in ("hac_lag_h6", "hac_bandwidth_L", "block_bootstrap_L"):
        k = sum(1 for c in ms for r in c["scrambled"] if r[f"reject_{method}"])
        agg[f"scrambled_reject_{method}"], agg[f"scrambled_rate_{method}"] = k, k / n_draws
        kr = sum(1 for c in ms if c["real"][f"reject_{method}"])
        agg[f"real_reject_{method}"], agg[f"real_rate_{method}"] = kr, kr / len(ms)
    battery = {"cells": n_draws, "future_shift_template_fired": sum(1 for c in ms for _ in range(scrambles) if c["future_shift_template_fired"]),
               "noise_null_rejected_5pct": sum(1 for c in ms for r in c["noise"] if r["reject_block_bootstrap_L"])}
    return {"aggregate": agg}, {"battery": battery}


def references(pop, horizons, out_dir):
    table, summary = contribution_run(pop, horizons, Path(out_dir) / "references", protocols=("blocks5",), skip_hgb=True)
    ref = {}
    for key, r in summary["reference"].items():
        h = int(key.split("|h")[1])
        ref[h] = {arm: {"mae_z_mean": v["mae_z_mean"], "mae_log_return_mean": v["mae_log_return_mean"], "n_eval_total": v["n_eval_total"]} for arm, v in r.items()}
    return ref


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", choices=["eth_4h", "eurusd_1h"], default="eurusd_1h")
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", default="1,2,3,4,5,6")
    p.add_argument("--features", default=None)
    p.add_argument("--scrambles", type=int, default=10)
    p.add_argument("--boot", type=int, default=100)
    args = p.parse_args(argv)
    spec = {"eth_4h": ETH_4H_SPEC, "eurusd_1h": EURUSD_1H_SPEC}[args.population]
    pop = bind_population(args.view, args.manifest, args.split, spec=spec)
    hs = [int(v) for v in args.horizons.split(",")]
    feats = args.features.split(",") if args.features else list(pop.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    ref = references(pop, hs, out)
    cells = []
    for f in feats:
        for h in hs:
            c = measure_cell(pop, f, h, scrambles=args.scrambles, boot=args.boot)
            cells.append(c)
            if c["state"] == "MEASURED":
                print(f"{f} h{h} L={c['block_length']} real_boot_reject={int(c['real']['reject_block_bootstrap_L'])} theta={c['real']['theta']:+.4f}+-{c['real']['se_block_bootstrap_L']:.4f} "
                      f"scr_boot={sum(r['reject_block_bootstrap_L'] for r in c['scrambled'])}/{args.scrambles} scr_hac6={sum(r['reject_hac_lag_h6'] for r in c['scrambled'])}/{args.scrambles}", flush=True)
            else:
                print(f"{f} h{h} {c['state']}", flush=True)
    cal, idx = build_check_documents(cells, args.scrambles)
    verdict = check(cal, idx)
    doc = {"schema": "c2_population_measure.v1", "label": "FINDING_NOT_EFFECT_DEVELOPMENT", "population": spec.name, "bindings": pop.bindings,
           "horizons": hs, "scrambles_per_cell": args.scrambles, "bootstrap_resamples": args.boot, "references_blocks5": ref, "cells": cells,
           "calibration_aggregate": cal["aggregate"], "battery": idx["battery"], "interval_rule_verdict": verdict,
           "cpu_seconds": time.process_time() - t0, "peak_rss_bytes_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    (out / "POPULATION_MEASUREMENT.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({"verdict": verdict, "references_h1": ref.get(hs[0])}, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
