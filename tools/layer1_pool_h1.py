#!/usr/bin/env python3
"""Layer-1 feature pool for EURUSD H1 on the admitted grid, with per-feature DATA-DRIVEN preprocessing (owner rule
2026-10-22: a transform only when a measured metric justifies it; everything documented per feature).

Groups assembled (all on the same UTC bar-start grid, decision at bar start + 1 h):
  inventory   the owners' 30 technical/calendar columns (incl. dow_sin) and raw OHLC-derived rows (v4 pool, admitted)
  price_pool  41 multi-scale price features (px.*), calendar_known (cal.*)
  candle      body / wicks / range in PIPS relative to the close: O-C, H-C, C-L, H-L (new; the predictor-era candle)
  fractal     12 causal fractal columns (w48/96/200)
  cross_asset 75 (r3): 6 G10 pairs, USD/EUR strength, BTC, daily yields/VIX/indices/commodities at release time,
              CFTC EUR, PCA factors and Kalman slopes fitted on 2015-2019
  event       6 fix/cut clock features (Tokyo / London fix, NY cut; DST per zone)
  activity    P(barrier hit) from the confirmed activity model (TRAIN-fitted)
  ssa         causal trailing-window SSA (L=120, K=24, rank 3) trend/cycle components of the close, in pips,
              computed from closes <= t only (the two-sided SSA is forbidden)

Per-feature diagnostics on TRAIN 2020-2022 only: ADF p, KPSS p, skew, kurtosis, min/max, share of zeros, and the
declared decision: NONE / DIFF (first difference, non-stationary level) / ASINH (strongly skewed, only if it halves
|skew|) — bounded indicators are never transformed. Scaling (TRAIN-only mean/std for every feature) is applied by
the model loader, not here; the store keeps the raw and the transformed value side by side (``feature`` and
``feature@diff`` / ``feature@asinh``). Future-perturbation check on the price-derived groups: every cell whose
decision time is <= the cut must be bit-identical after replacing the bars after the cut by noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PIP = 1e-4
BOUNDED = {"RSI": (0, 100), "Stochastic_K": (0, 100), "Stochastic_D": (0, 100), "WilliamsR": (-100, 0), "BB_position": None,
           "hod_sin": (-1, 1), "hod_cos": (-1, 1), "dow_sin": (-1, 1), "dow_cos": (-1, 1)}
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- new groups
def candle_pips(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    return pd.DataFrame({"cdl.body_pips": (c - o) / PIP, "cdl.upper_wick_pips": (h - np.maximum(o, c)) / PIP,
                         "cdl.lower_wick_pips": (np.minimum(o, c) - l) / PIP, "cdl.range_pips": (h - l) / PIP,
                         "cdl.close_minus_low_pips": (c - l) / PIP, "cdl.high_minus_close_pips": (h - c) / PIP}, index=df.index)


def causal_ssa(close: np.ndarray, L: int = 120, K: int = 24, rank: int = 3) -> np.ndarray:
    """Trailing-window SSA: for each t, embed close[t-L+1..t] (K lags), take the first ``rank`` singular triplets and
    reconstruct the LAST point of the window (diagonal averaging restricted to the last anti-diagonal). Uses only
    closes <= t. Returns (n, rank) components in price units; NaN during the warm-up."""
    n = len(close)
    out = np.full((n, rank), np.nan)
    for t in range(L - 1, n):
        w = close[t - L + 1:t + 1]
        m = w.mean()
        x = w - m
        H = np.lib.stride_tricks.sliding_window_view(x, K).T          # K x (L-K+1)
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        for r in range(min(rank, len(s))):
            comp = s[r] * np.outer(U[:, r], Vt[r])
            # last point of the window = last anti-diagonal element H[K-1, L-K]
            out[t, r] = comp[K - 1, L - K] + (m if r == 0 else 0.0)
    return out


# ----------------------------------------------------------------------------- diagnostics and decisions
def diagnostics(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    d = {"n": int(len(x))}
    if len(x) < 100:
        return d | {"adf_p": None, "kpss_p": None, "skew": None, "kurtosis": None}
    from scipy.stats import kurtosis, skew
    d.update(min=float(x.min()), max=float(x.max()), mean=float(x.mean()), std=float(x.std()), skew=float(skew(x)),
             kurtosis=float(kurtosis(x)), zero_share=float(np.mean(x == 0)),
             median_pos=float(np.median(x[x > 0])) if (x > 0).any() else None)
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        xs = x[-20000:] if len(x) > 20000 else x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d["adf_p"] = float(adfuller(xs, maxlag=24, autolag=None)[1])
            d["kpss_p"] = float(kpss(xs, regression="c", nlags=24)[1])
    except Exception as exc:
        d["adf_p"], d["kpss_p"], d["stationarity_error"] = None, None, str(exc)[:80]
    return d


def decide(name: str, x: np.ndarray, d: dict, train=None) -> tuple[str, str, np.ndarray | None]:
    """(decision, reason, transformed values or None). ``d`` = diagnostics of the TRAIN rows; ``train`` = the TRAIN mask
    over ``x`` (every candidate's skew is measured on TRAIN only; without a mask the whole series is used, legacy).

    Skew rule (rev 2026-10-22, scale-aware): for a non-negative variable with |skew| > 2 on TRAIN, the first candidate
    that at least halves |skew| on TRAIN is applied, in this declared order:
      ASINH   asinh(x)            the original rule; kept wherever it already works (e.g. candle geometry in pips)
      LOG     log(x)              when x > 0 on every TRAIN row (scale-free); if a later row is <= 0 the candidate is
                                  rejected (domain check only, recorded) and the next one is tried
      ASINHS  asinh(x / s)        s = TRAIN median of the positive values (frozen constant); defined for zeros
    The original rule applied asinh in native units only, which is the identity for magnitudes ~1e-3 (ranges,
    realized volatilities, spectral energies): those variables stayed skewed although a transform halves their skew.
    Bounded variables, calendar encodings and regime probabilities / entropies are never transformed."""
    if name in BOUNDED or name.startswith(("cal.", "ev.", "hod_", "dow_", "rg.")):
        return "NONE", "bounded, calendar encoding or regime probability/entropy", None
    if d.get("adf_p") is not None and d["adf_p"] > 0.05 and d.get("kpss_p") is not None and d["kpss_p"] < 0.05:
        return "DIFF", f"non-stationary level (ADF p={d['adf_p']:.3f}, KPSS p={d['kpss_p']:.3f})", np.r_[np.nan, np.diff(x)]
    sk = d.get("skew")
    if sk is not None and abs(sk) > 2 and d.get("min") is not None and d["min"] >= 0:
        from scipy.stats import skew
        tr = np.ones(len(x), bool) if train is None else np.asarray(train, bool)
        notes = []
        cands = [("ASINH", lambda v: np.arcsinh(v), "asinh(x)")]
        if d["min"] > 0:
            cands.append(("LOG", lambda v: np.log(v), "log(x)"))
        s_ = d.get("median_pos")
        if s_:
            cands.append(("ASINHS", lambda v, s_=s_: np.arcsinh(v / s_), f"asinh(x / {s_:.6g}) with the TRAIN median of positive values"))
        for dec, fn, desc in cands:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = fn(x)
            src_ok = np.isfinite(x)
            bad_out = int((src_ok & ~np.isfinite(y)).sum())
            if bad_out:
                notes.append(f"{dec} rejected: {bad_out} finite rows outside its domain")
                continue
            ys = y[tr & np.isfinite(y)]
            sk2 = float(skew(ys))
            if abs(sk2) < 0.5 * abs(sk):
                if dec == "ASINHS":
                    d["transform_scale"] = s_
                return dec, f"positive and skewed (TRAIN skew {sk:.2f} -> {sk2:.2f} with {desc})" + (f"; {'; '.join(notes)}" if notes else ""), y
            notes.append(f"{dec} {sk:.2f}->{sk2:.2f} does not halve it")
        return "NONE", f"skewed (TRAIN skew {sk:.2f}) but no candidate halves it: {'; '.join(notes)}", None
    return "NONE", "stationary, not strongly skewed", None


# ----------------------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v4-eval-input", required=True, help="v4 snapshot eval_input/features_all.parquet (admitted grid, 93 features + OHLC)")
    ap.add_argument("--cross-asset", required=True, help="cross-asset r3 store parquet")
    ap.add_argument("--activity", required=True, help="P_HIT.parquet")
    ap.add_argument("--admitted-sha256", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tools import timing_event_clock as ec
    base = pd.read_parquet(a.v4_eval_input)
    ts = pd.DatetimeIndex(pd.to_datetime(base["timestamp"])).tz_convert("UTC")
    base.index = ts
    if (ts >= TEST_START).any():
        raise SystemExit("TEST_ROWS_FORBIDDEN")
    feats = base.drop(columns=["timestamp"])
    groups = {c: ("raw_price" if c in ("OPEN", "HIGH", "LOW", "CLOSE") else "price_pool" if c.startswith("px.") else
                  "calendar" if c.startswith(("cal.", "hod_", "dow_")) else "fractal" if "_w" in c and any(k in c for k in ("hurst", "variogram", "box_")) else "inventory")
              for c in feats.columns}
    cdl = candle_pips(base)
    groups.update({c: "candle" for c in cdl.columns})
    ca = pd.read_parquet(a.cross_asset)
    ca.index = pd.DatetimeIndex(pd.to_datetime(ca.pop("timestamp"))).tz_convert("UTC")
    ca = ca.reindex(ts)
    groups.update({c: "cross_asset" for c in ca.columns})
    evc = ec.event_clock(ts)
    groups.update({c: "event" for c in evc.columns})
    act = pd.read_parquet(a.activity)
    act.index = pd.DatetimeIndex(pd.to_datetime(act.pop("timestamp"))).tz_convert("UTC")
    act = act.reindex(ts).rename(columns={"p_hit": "act.p_hit"})
    groups["act.p_hit"] = "activity"
    close = base["CLOSE"].to_numpy(float)
    S = causal_ssa(close)
    ssa = pd.DataFrame({"ssa.trend_minus_close_pips": (S[:, 0] - close) / PIP, "ssa.c2_pips": S[:, 1] / PIP, "ssa.c3_pips": S[:, 2] / PIP}, index=ts)
    groups.update({c: "ssa" for c in ssa.columns})
    X = pd.concat([feats, cdl, ca, evc, act, ssa], axis=1)
    # diagnostics + decisions on TRAIN only
    train = (ts >= pd.Timestamp("2020-01-01", tz="UTC")) & (ts < pd.Timestamp("2023-01-01", tz="UTC"))
    table, extra = [], {}
    for c in X.columns:
        x = X[c].to_numpy(float)
        d = diagnostics(x[train])
        dec, why, y = decide(c, x, d, train=np.asarray(train))
        table.append({"feature": c, "group": groups[c], **d, "decision": dec, "reason": why})
        if y is not None:
            extra[f"{c}@{dec.lower()}"] = y
    if extra:
        X = pd.concat([X, pd.DataFrame(extra, index=ts)], axis=1)
    # future perturbation of the price-derived new groups (candle, ssa): rebuild from perturbed closes
    cut = int(len(ts) * 0.6)
    rng = np.random.default_rng(1)
    pert = base.copy()
    for k in ("OPEN", "HIGH", "LOW", "CLOSE"):
        v = pert[k].to_numpy(float).copy()
        v[cut:] = v[cut:] * np.exp(rng.normal(0, 0.003, len(v) - cut))
        pert[k] = v
    cdl2 = candle_pips(pert)
    S2 = causal_ssa(pert["CLOSE"].to_numpy(float))
    ssa2 = (S2[:, 0] - pert["CLOSE"].to_numpy(float)) / PIP
    pert_ok = bool(np.allclose(cdl.to_numpy()[:cut], cdl2.to_numpy()[:cut], equal_nan=True) and
                   np.allclose(ssa["ssa.trend_minus_close_pips"].to_numpy()[:cut], ssa2[:cut], equal_nan=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = X.reset_index().rename(columns={"index": "timestamp"})
    store.to_parquet(out / "layer1_pool_eurusd_h1.parquet", index=False)
    diag = pd.DataFrame(table)
    diag.to_csv(out / "FEATURE_DIAGNOSTICS.csv", index=False)
    man = {"schema": "layer1_pool_h1.v1", "source_sha256": a.admitted_sha256, "rows": int(len(store)), "span": [str(ts[0]), str(ts[-1])],
           "features": list(X.columns), "groups": groups | {k: groups[k.split("@")[0]] for k in extra}, "decisions": {r["feature"]: r["decision"] for r in table},
           "decision_counts": diag.decision.value_counts().to_dict(), "group_counts": pd.Series(groups).value_counts().to_dict(),
           "future_perturbation_candle_ssa": {"cut_row": cut, "passed": pert_ok}, "inputs": {"v4_eval_input": sha(a.v4_eval_input), "cross_asset": sha(a.cross_asset), "activity": sha(a.activity)},
           "file_sha256": sha(out / "layer1_pool_eurusd_h1.parquet"), "test_read": False,
           "scaling_note": "no scaling in the store; the model loader standardizes every feature with TRAIN-only mean/std"}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    L = ["# Layer-1 pool: per-feature diagnostics (TRAIN 2020-2022) and preprocessing decisions", "",
         f"Features {len(X.columns)} ({man['group_counts']}); decisions {man['decision_counts']}; candle/SSA future-perturbation passed = {pert_ok}.", "",
         "| feature | group | ADF p | KPSS p | skew | kurt | decision | reason |", "|---|---|---|---|---|---|---|---|"]
    for r in table:
        f = lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"  # noqa: E731
        L.append(f"| {r['feature']} | {r['group']} | {f(r.get('adf_p'))} | {f(r.get('kpss_p'))} | {f(r.get('skew'))} | {f(r.get('kurtosis'))} | {r['decision']} | {r['reason']} |")
    (out / "FEATURE_DIAGNOSTICS.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"features": len(X.columns), "groups": man["group_counts"], "decisions": man["decision_counts"], "perturbation": pert_ok}))


if __name__ == "__main__":
    main()
