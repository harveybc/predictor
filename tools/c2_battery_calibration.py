"""Lane C2: calibration of the rung-2 interval (coordinator ruling 2026-10-01).

The first battery rejected 18 % of scrambled labels at a nominal 5 % with a Newey-West HAC of lag h + 6. Here every
(feature, horizon) cell gets, on the same cross-fitted residuals:

* the measured integrated autocorrelation length of the target Y_h and of the feature X_j on the TRAIN origins
  (tau = 1 + 2 * sum rho_k over the lags until rho_k first drops below 0.05, capped at n / 20);
* a block length L = max(tau_Y, tau_X, h + 6);
* three intervals for theta: HAC lag h + 6 (before), HAC bandwidth L (wider HAC), and a circular moving-block bootstrap
  of the score with block length L (B resamples, seeded);
* the three rejection decisions on the real label and on the scrambled label (same scrambling as the dossier run).

The aggregate rates (nominal 5 %, scrambled, real) before and after are the battery verdict; the per-cell numbers are
written into each dossier's `rung2.sensitivity` and `rung2.placebo` and into DOSSIER_INDEX.json by `--patch-dossiers`.
No rung state changes: every dossier stays NOT_IDENTIFIED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import bind_population, blocked_splits  # noqa: E402
from c2_causal_dossier import crossfit_residuals, hac_variance, REDUNDANCY_CORR  # noqa: E402

Z95 = 1.96


def integrated_autocorr_length(x: np.ndarray, max_lag: int, floor: float = 0.05) -> int:
    """tau = 1 + 2 sum_{k=1..K} rho_k, K = first lag with rho_k < floor (or max_lag); at least 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    v = float(x @ x)
    if v <= 0 or len(x) < 10:
        return 1
    tau = 1.0
    for k in range(1, max_lag + 1):
        rho = float(x[k:] @ x[:-k]) / v
        if rho < floor:
            break
        tau += 2 * rho
    return int(max(1, round(tau)))


def block_bootstrap_se(score: np.ndarray, denom: float, block: int, b: int, rng) -> float:
    """SE of theta_hat = sum(score)/denom under a circular moving-block bootstrap of the score series."""
    n = len(score)
    block = max(1, min(block, n // 2))
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(b, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, -1)[:, :n] % n
    sums = score[idx].sum(axis=1)
    return float(np.std(sums / denom, ddof=1))


def scrambled_labels(y: np.ndarray, rng) -> np.ndarray:
    """The dossier run's scrambling: block-permute the finite labels and circularly shift them."""
    y_scr = y.copy()
    fin = np.where(np.isfinite(y))[0]
    parts = np.array_split(fin, 5)
    order = rng.permutation(5)
    vals = np.concatenate([y[parts[i]] for i in order])
    vals = np.roll(vals, int(rng.integers(500, len(vals) - 500)))
    y_scr[fin] = vals
    return y_scr


def calibrate_cell(pop, feature, h, *, seed=20261001, boot=200):
    rng = np.random.default_rng(seed)        # same seed as the dossier run -> same scrambling
    y = pop.m07_target(h)
    xall = pop.feature_matrix_train()
    j = pop.features.index(feature)
    rows = pop.origins[np.isfinite(y[pop.origins])]
    xs = xall[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([np.corrcoef(xs[:, j], xs[:, k])[0, 1] if k != j else 1.0 for k in range(xs.shape[1])])
    w_idx = [k for k in range(len(pop.features)) if k != j and not abs(corr[k]) > REDUNDANCY_CORR]
    splits = blocked_splits(rows, h)
    x_t, w = xall[:, j], xall[:, w_idx]
    n = len(rows)
    tau_y = integrated_autocorr_length(y[rows], max_lag=n // 20)
    tau_x = integrated_autocorr_length(x_t[rows], max_lag=n // 20)
    L = int(max(tau_y, tau_x, h + 6))
    out = {"feature": feature, "horizon_bars": h, "n": int(n), "tau_y": tau_y, "tau_x": tau_x, "block_length": L,
           "hac_lag_before": h + 6, "bootstrap_resamples": boot}
    brng = np.random.default_rng(seed + 1)
    for label, y_used in (("real", y), ("scrambled", scrambled_labels(y, rng))):
        r_x, r_y, _, _, ok = crossfit_residuals(x_t, w, y_used, splits)
        rx, ry = r_x[ok], r_y[ok]
        denom = float(rx @ rx)
        theta = float(rx @ ry) / denom
        score = rx * (ry - theta * rx)
        m = len(rx)
        se_hac6 = float(np.sqrt(max(hac_variance(score, h + 6) * m, 0.0)) / denom)
        se_hacL = float(np.sqrt(max(hac_variance(score, L) * m, 0.0)) / denom)
        se_boot = block_bootstrap_se(score, denom, L, boot, brng)
        out[label] = {"theta": theta, "se_hac_lag_h6": se_hac6, "se_hac_bandwidth_L": se_hacL, "se_block_bootstrap_L": se_boot,
                      "reject_hac_lag_h6": bool(abs(theta) > Z95 * se_hac6),
                      "reject_hac_bandwidth_L": bool(abs(theta) > Z95 * se_hacL),
                      "reject_block_bootstrap_L": bool(abs(theta) > Z95 * se_boot)}
    return out


def aggregate(cells):
    n = len(cells)
    agg = {"cells": n, "nominal_rate": 0.05, "block_length_median": float(np.median([c["block_length"] for c in cells])),
           "block_length_min": int(min(c["block_length"] for c in cells)), "block_length_max": int(max(c["block_length"] for c in cells)),
           "tau_y_median": float(np.median([c["tau_y"] for c in cells])), "tau_x_median": float(np.median([c["tau_x"] for c in cells]))}
    for method in ("hac_lag_h6", "hac_bandwidth_L", "block_bootstrap_L"):
        for label in ("scrambled", "real"):
            k = sum(1 for c in cells if c[label][f"reject_{method}"])
            agg[f"{label}_reject_{method}"] = k
            agg[f"{label}_rate_{method}"] = k / n if n else float("nan")
    tol = 0.05 + 2 * (0.05 * 0.95 / max(n, 1)) ** 0.5     # two binomial sd above nominal
    agg["scrambled_tolerance_rate"] = tol
    agg["calibrated_methods"] = [m for m in ("hac_lag_h6", "hac_bandwidth_L", "block_bootstrap_L") if agg[f"scrambled_rate_{m}"] <= tol]
    agg["battery_verdict"] = "CONTROLS_FAIL_AS_REQUIRED" if agg["calibrated_methods"] else "BATTERY_SUSPECT"
    return agg


def patch_dossiers(dossier_dir: Path, cells, agg):
    by = {(c["feature"], c["horizon_bars"]): c for c in cells}
    index_path = dossier_dir / "DOSSIER_INDEX.json"
    idx = json.loads(index_path.read_text())
    patched = 0
    for d in idx["dossiers"]:
        c = by.get((d["feature"], d["horizon_bars"]))
        if not c:
            continue
        path = dossier_dir / d["file"]
        doc = json.loads(path.read_text())
        sens = doc["rung2"]["sensitivity"]
        sens.update({"calibration_block_length": c["block_length"], "calibration_tau_y": c["tau_y"], "calibration_tau_x": c["tau_x"],
                     "theta_se_hac_bandwidth_L": c["real"]["se_hac_bandwidth_L"], "theta_se_block_bootstrap_L": c["real"]["se_block_bootstrap_L"],
                     "scrambled_label_t_block_bootstrap": (c["scrambled"]["theta"] / c["scrambled"]["se_block_bootstrap_L"]) if c["scrambled"]["se_block_bootstrap_L"] > 0 else 0.0,
                     "calibration_battery_verdict": agg["battery_verdict"]})
        for t in doc["rung2"]["placebo"]["tests"]:
            if t["name"] == "scrambled_label":
                rej = c["scrambled"]["reject_block_bootstrap_L"]
                t["verdict"] = "FAILED_AS_REQUIRED" if not rej else "NOT_FAILED_AT_NOMINAL_5PCT_BLOCK_BOOTSTRAP"
        doc["rung2"]["placebo"]["state"] = "PASSED" if all(t["verdict"] == "FAILED_AS_REQUIRED" for t in doc["rung2"]["placebo"]["tests"]) else "FAILED"
        doc["limitations"].append(f"interval calibration (coordinator ruling 2026-10-01): block length {c['block_length']} (tau_y {c['tau_y']}, tau_x {c['tau_x']}); "
                                  f"scrambled-label rejection rates over 498 cells: HAC lag h+6 {agg['scrambled_rate_hac_lag_h6']:.3f}, HAC bandwidth L {agg['scrambled_rate_hac_bandwidth_L']:.3f}, "
                                  f"block bootstrap L {agg['scrambled_rate_block_bootstrap_L']:.3f} (nominal 0.05); real-label rates {agg['real_rate_hac_lag_h6']:.3f} / {agg['real_rate_hac_bandwidth_L']:.3f} / {agg['real_rate_block_bootstrap_L']:.3f}")
        path.write_text(json.dumps(doc, indent=1, sort_keys=True))
        d["calibration"] = {"block_length": c["block_length"], "real_reject_block_bootstrap_L": c["real"]["reject_block_bootstrap_L"],
                            "scrambled_reject_block_bootstrap_L": c["scrambled"]["reject_block_bootstrap_L"]}
        patched += 1
    idx["battery_calibration"] = agg
    index_path.write_text(json.dumps(idx, indent=1, sort_keys=True))
    return patched


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features", default=None, help="comma list; default all")
    p.add_argument("--horizons", default="1,2,3,4,5,6")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--patch-dossiers", default=None, help="dossier directory to patch in place")
    args = p.parse_args(argv)
    pop = bind_population(args.view, args.manifest, args.split)
    feats = args.features.split(",") if args.features else list(pop.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    cells = []
    for f in feats:
        for h in [int(v) for v in args.horizons.split(",")]:
            c = calibrate_cell(pop, f, h, boot=args.boot)
            cells.append(c)
            print(f"{f} h{h} L={c['block_length']} (tau_y {c['tau_y']}, tau_x {c['tau_x']}) scr reject hac6/hacL/boot="
                  f"{int(c['scrambled']['reject_hac_lag_h6'])}{int(c['scrambled']['reject_hac_bandwidth_L'])}{int(c['scrambled']['reject_block_bootstrap_L'])} "
                  f"real={int(c['real']['reject_hac_lag_h6'])}{int(c['real']['reject_hac_bandwidth_L'])}{int(c['real']['reject_block_bootstrap_L'])}", flush=True)
    agg = aggregate(cells)
    agg["cpu_seconds"] = time.process_time() - t0
    agg["label"] = "DEVELOPMENT"
    doc = {"schema": "c2_battery_calibration.v1", "bindings": pop.bindings, "aggregate": agg, "cells": cells}
    (out / "BATTERY_CALIBRATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(agg, indent=1))
    if args.patch_dossiers:
        print("patched", patch_dossiers(Path(args.patch_dossiers), cells, agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
