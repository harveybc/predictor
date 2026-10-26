#!/usr/bin/env python3
"""Activity producer P(barrier hit), rebuilt from RETAINED inputs with an explicit fit scope (Musashi's final joint
audit A1, 2026-10-23: supervised derived features need out-of-fold training predictions and label availability).

The original `act.p_hit` (heuristic-strategy tools/expert_guided_risk_d.py activity, 2026-10-20) was an HGB fitted on
TRAIN 2020-2022 labels and applied to every bar: inside the feature-selection folds (2020-2023) its values were
IN-SAMPLE predictions of the very barrier labels the selection scored. Its exact input (the data-screen
`screen_input.parquet`) was not retained: the only retained eligibility mask gives 14 773 eligible TRAIN rows against
the original 14 823, so the original column is NOT reproducible bit for bit and is retired.

This producer keeps its declared definition and changes only the fit scope:
  label     data_screen.prepare: symmetric barriers m = 2 sigma around open(t+1), horizon 72 bars, forced flat 12 h,
            causal sigma of the pinned ZigZag module; 1 = a barrier touched before the timeout; business-eligible rows
            of the retained v4 eligibility mask; a label is usable only once resolved (resolution bar recorded)
  features  the 14 activity features CONFIRMED by the data-phase screen (r3 DECISION.json), each at its lag
  model     HistGradientBoosting depth 3, min leaf 200, 300 iterations, lr 0.05, random_state 0
  scope     fit rows = the scope's allowed rows whose label RESOLVES inside an allowed span (never inside an
            excluded span, never after the scope's end); rows of the fit span receive OUT-OF-FOLD predictions from a
            purged contiguous K-fold cross-fit inside the allowed rows; every other row receives the prediction of
            the model fitted on all allowed rows
Residual (declared, not repaired here): the 14-feature set itself was chosen by the data-phase screen on TRAIN
2020-2022 with confirmation on 2023 S1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HGB = dict(max_depth=3, min_samples_leaf=200, max_iter=300, learning_rate=0.05, random_state=0)
EMBARGO = 96


def load_inputs(snapshot: Path, admitted: Path, short_horizon: Path, pool_parquet: Path, decision: Path):
    """(bars, X, hit, resolution_bar) on the admitted grid, from retained, pinned inputs."""
    from tools import timing_selection as ts
    from tools import timing_pivot_targets as tp
    from tools import timing_profit_targets as tpf
    from tools import timing_event_clock as ec
    from tools.data_screen import HORIZON, M_SIGMA
    cfg = json.loads((snapshot / "PREPARED.json").read_text())["config"]
    src = cfg["source"]
    src["path"], src["admission"]["path"] = str(admitted / "eurusd.parquet"), str(admitted / "ADMISSION.json")
    src["closures"]["path"] = str(admitted / "CLOSURE_INTERVALS.json")
    src["bar_source_module"] = str(short_horizon / "bar_source.py")
    cfg["zigzag"]["module"] = str(short_horizon / "zigzag.py")
    bars, closures, _ = ts.load_bars(cfg)                        # pins verified by the loader
    zz = tp.load_zigzag_module(cfg["zigzag"]["module"], cfg["zigzag"]["sha256"])
    sigma = np.asarray(zz.causal_sigma(bars), float)
    flat = tpf.flat_bar_positions(bars.index, closures, 12.0)
    outc, res = tpf.barrier_outcomes(bars.OPEN, bars.HIGH, bars.LOW, bars.CLOSE, sigma, flat, horizon_bars=HORIZON, m=M_SIGMA)
    el = pd.read_parquet(snapshot / "ELIGIBILITY.parquet")
    if len(el) != len(bars):
        raise SystemExit("ELIGIBILITY_GRID_MISMATCH")
    business = el["business_reason"].to_numpy() == "ELIGIBLE"
    res = np.asarray(res, int)
    hit = np.where(business & (np.asarray(outc) >= 0) & (res >= 0), (np.asarray(outc) >= 1).astype(float), np.nan)
    pool = pd.read_parquet(pool_parquet)
    pool.index = pd.DatetimeIndex(pd.to_datetime(pool.pop("timestamp"), utc=True))
    pool = pool.reindex(bars.index)
    evc = ec.event_clock(bars.index)
    for c in evc.columns:
        if c not in pool.columns:
            pool[c] = evc[c].to_numpy()
    feats = sorted({(d["feature"], int(d["lag"])) for d in json.loads(decision.read_text())["activity_confirmed"]})
    X = pd.DataFrame({f"{f}@{L}": pool[f].shift(L).to_numpy(float) for f, L in feats}, index=bars.index)
    return bars, X, hit, res, [f"{f}@{L}" for f, L in feats]


def _fit(X, y, rows):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(**HGB)
    m.fit(X[rows], y[rows].astype(int))
    return m


def max_lag(feats) -> int:
    return max(int(f.rsplit("@", 1)[1]) for f in feats)


def usable(allowed: np.ndarray, hit: np.ndarray, res: np.ndarray, lookback: int = 1) -> np.ndarray:
    """Rows usable for fitting: allowed, labelled, every bar of the INPUT window [t - lookback + 1, t] allowed (a
    lagged input from a held-out bar would carry it into the fit) and the label RESOLVES without crossing a
    non-allowed bar (a label resolved in an excluded span or beyond the scope carries held-out information)."""
    n = len(hit)
    ok = allowed & np.isfinite(hit) & (res >= 0) & (res < n)
    bad = np.concatenate([[0], np.cumsum(~allowed)])
    idx = np.flatnonzero(ok)
    lo = np.maximum(idx - lookback + 1, 0)
    ok[idx] = ((bad[res[idx] + 1] - bad[lo]) == 0) & (idx - lookback + 1 >= 0)
    return ok


def scope_predictions(X: pd.DataFrame, hit: np.ndarray, res: np.ndarray, allowed: np.ndarray, k: int = 5, embargo: int = EMBARGO,
                      lookback: int = 1, mode: str = "purged_kfold"):
    """(p, report): out-of-fold predictions on the allowed bars (purged contiguous k-fold over the usable rows,
    embargo on both sides), full-model predictions elsewhere. Nothing outside ``allowed`` is ever fitted on.

    ``mode="expanding"`` (audit A1-12, lane F7 2026-10-26): FORWARD-only cross-fitting. The allowed bars are cut into
    ``k`` contiguous blocks in time; block i is predicted by a model fitted on the usable rows strictly BEFORE the
    block minus the embargo, so no training row of any prediction lies after it. The first block has no past and
    receives NaN (counted), never an in-sample value. Bars after the allowed span still get the full model, which
    was fitted only on earlier (allowed) rows."""
    if mode not in ("purged_kfold", "expanding"):
        raise ValueError(f"ACTIVITY_MODE_UNKNOWN:{mode}")
    Xv = X.to_numpy(float)
    rows = np.flatnonzero(usable(allowed, hit, res, lookback))
    full = _fit(Xv, hit, rows)
    p = full.predict_proba(Xv)[:, 1]
    alw = np.flatnonzero(allowed)
    edges = np.linspace(0, len(alw), k + 1).astype(int)
    if mode == "expanding":
        nan_rows = 0
        for i in range(k):
            lo_b, hi_b = alw[edges[i]], alw[min(edges[i + 1], len(alw)) - 1]
            sel = np.zeros(len(hit), bool)
            sel[lo_b:hi_b + 1] = True
            sel &= allowed
            past = np.zeros(len(hit), bool)
            past[:max(lo_b - embargo, 0)] = True
            tr = np.flatnonzero(usable(allowed & past, hit, res, lookback))
            if i == 0 or len(tr) < 2 or len(np.unique(hit[tr])) < 2:
                p[sel] = np.nan
                nan_rows += int(sel.sum())
                continue
            m = _fit(Xv, hit, tr)
            p[sel] = m.predict_proba(Xv[sel])[:, 1]
        return p, {"fit_rows_full": int(len(rows)), "mode": "expanding", "blocks": k, "embargo_bars": embargo,
                   "input_lookback_bars": lookback, "allowed_bars": int(allowed.sum()), "nan_first_block_rows": nan_rows}
    for i in range(k):
        lo_b, hi_b = alw[edges[i]], alw[min(edges[i + 1], len(alw)) - 1]
        held = np.zeros(len(hit), bool)
        held[lo_b:hi_b + 1] = True
        guard = np.zeros(len(hit), bool)
        guard[max(lo_b - embargo, 0):min(hi_b + embargo + 1, len(hit))] = True
        tr = np.flatnonzero(usable(allowed & ~guard, hit, res, lookback))
        m = _fit(Xv, hit, tr)
        sel = held & allowed
        p[sel] = m.predict_proba(Xv[sel])[:, 1]
    return p, {"fit_rows_full": int(len(rows)), "mode": "purged_kfold", "oof_folds": k, "embargo_bars": embargo, "input_lookback_bars": lookback,
               "allowed_bars": int(allowed.sum())}
