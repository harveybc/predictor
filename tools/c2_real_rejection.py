"""Lane C2 follow-up findings (coordinator 2026-10-01), DEVELOPMENT, TRAIN rows only, calibrated interval.

(1) WHERE the real-label rejection (0.16 against nominal 0.05) lives, and whether it survives conditioning on the zero-return
naive's own past prediction errors (a persistent-regime artefact?). Per cell (feature, h) the partial association theta of
Y_h with X_j given the other features W is estimated by the cross-fitted partially linear model of c2_causal_dossier, and
its interval is the circular block bootstrap with block length L = max(tau_Y, tau_X, h + 6) of c2_battery_calibration.
Variants of the same cell:
  base         : Y_h on X_j given W (the dossier model);
  scaled       : Y_h divided by a trailing scale of the naive's own errors, s_t = mean |log_return_1| over the last 60 bars
                 (rows <= t only), i.e. the regime-standardized label;
  regime_ctrl  : base plus W extended by s_t and by the signed h-bar return realized over (t - h, t] (the persistence
                 naive's own error history, rows <= t only);
  both         : scaled label and the extended W.
Per-block theta (sign stability across the five time blocks) is reported for the base variant.
A rejection that disappears under `scaled`/`regime_ctrl` is labelled REGIME_ARTEFACT_CANDIDATE; one that survives is
SURVIVES_REGIME_CONDITIONING. Both are FINDINGS about association under a declared model, never an effect.

(2) The seasonal reference. s_h(t) is the h-bar return realized one 24 h period earlier (lag 6 bars, or the first multiple
of 6 that is >= h, so it is observed at t): exactly the lane-D seasonal reference and this repository's naive_seasonal_24h.
The test is the partial association of Y_h with s_h(t) beyond all 83 features (calibrated block interval, per block signs),
beside the same test for the persistence reference (the h-bar return over (t - h, t]).
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
from c2_eth_population import bind_population, blocked_splits  # noqa: E402
from c2_causal_dossier import crossfit_residuals, REDUNDANCY_CORR  # noqa: E402
from c2_battery_calibration import block_bootstrap_se, integrated_autocorr_length  # noqa: E402

Z95 = 1.96
REGIME_WINDOW = 60


def trailing_scale(log_return_1: np.ndarray, window: int = REGIME_WINDOW) -> np.ndarray:
    """s_t = mean |r| over rows (t - window, t]: uses rows <= t only (the naive's own past errors at h = 1)."""
    a = np.abs(np.nan_to_num(log_return_1))
    c = np.concatenate([[0.0], np.cumsum(a)])
    s = np.full(len(a), np.nan)
    idx = np.arange(window - 1, len(a))
    s[idx] = (c[idx + 1] - c[idx + 1 - window]) / window
    return s


def past_return(log_close: np.ndarray, h: int, lag: int = 0) -> np.ndarray:
    """log(CLOSE[t - lag] / CLOSE[t - lag - h]): the h-bar return that ENDED at t - lag; observed at t for lag >= 0."""
    out = np.full(len(log_close), np.nan)
    t = np.arange(lag + h, len(log_close))
    out[t] = log_close[t - lag] - log_close[t - lag - h]
    return out


def seasonal_return(log_close: np.ndarray, h: int) -> np.ndarray:
    """The h-bar return one 24 h period (6 bars, or the first multiple >= h) before the label window, observed at t."""
    lag = 6 * int(np.ceil(h / 6))
    out = np.full(len(log_close), np.nan)
    t = np.arange(lag, len(log_close) - 0)
    # the lane-D / naive_seasonal_24h reference: the h-bar return starting at t - lag, i.e. log C[t-lag+h] - log C[t-lag]
    ok = t - lag + h <= t
    tt = t[ok]
    out[tt] = log_close[tt - lag + h] - log_close[tt - lag]
    return out


def _fit(treat, w, y, splits, L, boot, rng):
    r_x, r_y, _, _, ok = crossfit_residuals(treat, w, y, splits)
    rx, ry = r_x[ok], r_y[ok]
    denom = float(rx @ rx)
    theta = float(rx @ ry) / denom
    score = rx * (ry - theta * rx)
    se = block_bootstrap_se(score, denom, L, boot, rng)
    return {"theta": theta, "se": se, "t": theta / se if se > 0 else 0.0, "reject": bool(abs(theta) > Z95 * se), "n": int(ok.sum())}, ok, r_x, r_y


def per_block_theta(r_x, r_y, splits):
    out = []
    for sp in splits:
        ev = sp["eval"]
        m = np.isfinite(r_x[ev]) & np.isfinite(r_y[ev])
        rx, ry = r_x[ev][m], r_y[ev][m]
        out.append(float(rx @ ry / (rx @ rx)) if len(rx) > 10 and rx @ rx > 0 else float("nan"))
    return out


def regime_cell(pop, feature, h, *, seed=20261001, boot=200):
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
    x_t, w = xall[:, j], xall[:, w_idx]
    n = len(rows)
    L = int(max(integrated_autocorr_length(y[rows], n // 20), integrated_autocorr_length(x_t[rows], n // 20), h + 6))
    lr1 = pop.frame["log_return_1"].to_numpy(dtype=np.float64)[: pop.train_end]
    s_t = trailing_scale(lr1)
    logc = np.log(pop.frame["CLOSE"].to_numpy(dtype=np.float64)[: pop.train_end])
    pr = past_return(logc, h)
    ctrl = np.column_stack([s_t, pr])
    # rows without the trailing scale or the past return are not scored in the conditioned variants; keep the same
    # rows in every variant of the cell so the comparison is paired
    ok_rows = np.isfinite(s_t) & np.isfinite(pr)
    keep = np.zeros(len(y), bool)
    keep[rows] = True
    keep &= ok_rows
    sub = {"fit": None}
    splits_k = []
    for sp in splits:
        splits_k.append({**sp, "fit": sp["fit"][keep[sp["fit"]]], "eval": sp["eval"][keep[sp["eval"]]]})
    y_scaled = y / s_t
    w_ctrl = np.column_stack([w, ctrl])
    out = {"feature": feature, "horizon_bars": h, "block_length": L, "rows": int(keep.sum())}
    base, _, r_x, r_y = _fit(x_t, w, y, splits_k, L, boot, rng)
    out["base"] = base
    out["blocks_theta_base"] = per_block_theta(r_x, r_y, splits_k)
    th = [b for b in out["blocks_theta_base"] if np.isfinite(b)]
    out["sign_agreement_base"] = float(max(sum(b > 0 for b in th), sum(b < 0 for b in th)) / len(th)) if th else float("nan")
    out["scaled"], *_ = _fit(x_t, w, y_scaled, splits_k, L, boot, rng)
    out["regime_ctrl"], *_ = _fit(x_t, w_ctrl, y, splits_k, L, boot, rng)
    out["both"], *_ = _fit(x_t, w_ctrl, y_scaled, splits_k, L, boot, rng)
    if base["reject"]:
        out["finding"] = ("SURVIVES_REGIME_CONDITIONING" if (out["scaled"]["reject"] or out["regime_ctrl"]["reject"]) and out["both"]["reject"]
                          else "REGIME_ARTEFACT_CANDIDATE")
    else:
        out["finding"] = "NOT_REJECTED_AT_BASE"
    return out


def seasonal_cell(pop, h, *, seed=20261001, boot=200):
    """Partial association of Y_h with the seasonal and persistence references beyond ALL 83 features."""
    rng = np.random.default_rng(seed)
    y = pop.m07_target(h)
    xall = pop.feature_matrix_train()
    logc = np.log(pop.frame["CLOSE"].to_numpy(dtype=np.float64)[: pop.train_end])
    refs = {"seasonal_24h": seasonal_return(logc, h), "persistence_hbar": past_return(logc, h)}
    rows = pop.origins[np.isfinite(y[pop.origins])]
    splits = blocked_splits(rows, h)
    n = len(rows)
    out = {"horizon_bars": h}
    for name, ref in refs.items():
        z = (ref - np.nanmean(ref[rows])) / np.nanstd(ref[rows])
        keep = np.zeros(len(y), bool)
        keep[rows] = True
        keep &= np.isfinite(z)
        sp_k = [{**sp, "fit": sp["fit"][keep[sp["fit"]]], "eval": sp["eval"][keep[sp["eval"]]]} for sp in splits]
        L = int(max(integrated_autocorr_length(y[rows], n // 20), integrated_autocorr_length(z[rows], n // 20), h + 6))
        res, _, r_x, r_y = _fit(np.nan_to_num(z), xall, y, sp_k, L, boot, rng)
        bt = per_block_theta(r_x, r_y, sp_k)
        th = [b for b in bt if np.isfinite(b)]
        res.update({"block_length": L, "blocks_theta": bt,
                    "sign_agreement": float(max(sum(b > 0 for b in th), sum(b < 0 for b in th)) / len(th)) if th else float("nan"),
                    "theta_log_return_per_sd": float(pop.sigma * res["theta"]),
                    "corr_with_label": float(np.corrcoef(z[keep], np.nan_to_num(y[keep]))[0, 1])})
        out[name] = res
    return out


def summarize_regime(cells):
    def rate(key):
        return sum(1 for c in cells if c[key]["reject"]) / len(cells)
    by_h, by_f = {}, {}
    for c in cells:
        by_h.setdefault(c["horizon_bars"], []).append(c)
        by_f.setdefault(c["feature"], []).append(c)
    agg = {"cells": len(cells), "rate_base": rate("base"), "rate_scaled": rate("scaled"), "rate_regime_ctrl": rate("regime_ctrl"), "rate_both": rate("both"),
           "nominal": 0.05,
           "by_horizon": {h: {"base": sum(c["base"]["reject"] for c in v) / len(v), "both": sum(c["both"]["reject"] for c in v) / len(v)} for h, v in sorted(by_h.items())},
           "findings": {k: sum(1 for c in cells if c["finding"] == k) for k in ("SURVIVES_REGIME_CONDITIONING", "REGIME_ARTEFACT_CANDIDATE", "NOT_REJECTED_AT_BASE")}}
    feat = sorted(by_f, key=lambda f: -sum(c["base"]["reject"] for c in by_f[f]))
    agg["features_by_base_rejections"] = [(f, int(sum(c["base"]["reject"] for c in by_f[f])), int(sum(c["both"]["reject"] for c in by_f[f]))) for f in feat[:25]]
    agg["rejecting_cells_sign_stable_4of5"] = sum(1 for c in cells if c["base"]["reject"] and c["sign_agreement_base"] >= 0.8)
    agg["rejecting_cells_total"] = sum(1 for c in cells if c["base"]["reject"])
    return agg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features", default=None)
    p.add_argument("--horizons", default="1,2,3,4,5,6")
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--only", default="both", choices=["regime", "seasonal", "both"])
    args = p.parse_args(argv)
    pop = bind_population(args.view, args.manifest, args.split)
    feats = args.features.split(",") if args.features else list(pop.features)
    hs = [int(v) for v in args.horizons.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    result = {"schema": "c2_real_rejection.v1", "label": "FINDING_NOT_EFFECT_DEVELOPMENT", "bindings": pop.bindings}
    if args.only in ("regime", "both"):
        cells = []
        for f in feats:
            for h in hs:
                c = regime_cell(pop, f, h, boot=args.boot)
                cells.append(c)
                print(f"{f} h{h} base={int(c['base']['reject'])} scaled={int(c['scaled']['reject'])} ctrl={int(c['regime_ctrl']['reject'])} both={int(c['both']['reject'])} {c['finding']}", flush=True)
        result["regime"] = {"aggregate": summarize_regime(cells), "cells": cells}
    if args.only in ("seasonal", "both"):
        seas = []
        for h in hs:
            s = seasonal_cell(pop, h, boot=args.boot)
            seas.append(s)
            print(f"h{h} seasonal t={s['seasonal_24h']['t']:.2f} reject={s['seasonal_24h']['reject']} signs={s['seasonal_24h']['sign_agreement']:.1f} | persistence t={s['persistence_hbar']['t']:.2f} reject={s['persistence_hbar']['reject']}", flush=True)
        result["seasonal"] = seas
    result["cpu_seconds"] = time.process_time() - t0
    result["peak_rss_bytes_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    (out / "REAL_REJECTION_FINDINGS.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    if "regime" in result:
        print(json.dumps(result["regime"]["aggregate"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
