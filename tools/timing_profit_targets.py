"""Profit-direction targets for feature selection (family ``timing_profit``, semantics ``timing_targets.profit.v1``).

Owner decision 2026-10-20 (improve the signal before the architecture): select inputs for the label that
actually worked in the expert-guided lane, "would an order opened now reach its take-profit before its
stop-loss?", instead of the pivot-in-k label.  Same contract as ``timing_pivot_targets`` (per-row label
support ledger, censoring, purges at cuts, finality replays) so the pinned driver, the shared eligibility
mask, the engines and the evaluator consume it through the same API; the pivot family is untouched.

Order rule (the lane's detector rule, declared): decision after the completed bar t, executable fill at
OPEN(t+1); sigma = the causal 24 h volatility known at the decision (``zz.causal_sigma``, bars <= t);
TP = fill +/- m_tp * sigma, SL = fill -/+ m_sl * sigma.  From bar t+1 on, the first touch decides: a bar
whose LOW <= SL (long) counts as a loss even if its HIGH >= TP (SL-first, conservative); the order is also
closed at the session's forced flat (close - preclose_hours, from the admitted closures) or after
``horizon_bars`` observed bars, whichever comes first, at that bar's CLOSE; the P&L in sigma units net of
``cost_sigma`` decides.  Both directions are evaluated at every origin.

Targets:
  T1 PRIMARY  ``timing_profit_h{H}``: 1 the LONG order wins and the SHORT does not; 2 the SHORT wins and the
              LONG does not; 0 otherwise (both lose, or both win = ambiguous, never a direction).
              Indicators ``timing_profit_long_h{H}`` / ``timing_profit_short_h{H}``: one-vs-rest.
  T2          ``timing_long_order_wins_h{H}``: 1 if the long order wins (regardless of the short); comparator.
  T3          ``timing_best_order_pnl_sigma_h{H}``: max over directions of the order's net P&L in sigma.
Support end = the later of the two resolution bars; a row whose orders are not resolved inside the loaded
data is CENSORED (never 0); a row whose support reaches a declared cut is PURGED before that cut.
Labels are hindsight (they read the future path): labels only, never inputs.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from tools.timing_pivot_targets import (  # noqa: F401  (re-exported for the driver)
    BAR, PIP, STATUS, TEST_START, TimingError, load_admitted_bars, load_closures, load_lake_bars,
    load_short_horizon_module, load_zigzag_module, sha256_file, _cut_positions, _status, _utc,
)

FAMILY = "timing_profit"
SEMANTICS = "timing_targets.profit.v1"
DEFAULT_SPEC = {"horizon_bars": 72, "m_tp": 2.0, "m_sl": 2.0, "cost_sigma": 0.1, "preclose_hours": 12.0}
TARGET_LEDGER = {"timing_profit_h{k}": "t1", "timing_profit_long_h{k}": "t1", "timing_profit_short_h{k}": "t1",
                 "timing_long_order_wins_h{k}": "t2", "timing_best_order_pnl_sigma_h{k}": "t3"}


def primary(k: int) -> str:
    return f"timing_profit_h{int(k)}"


def indicators(k: int) -> list[str]:
    return [f"timing_profit_long_h{int(k)}", f"timing_profit_short_h{int(k)}"]


def comparators(k: int) -> list[str]:
    return [f"timing_long_order_wins_h{int(k)}", f"timing_best_order_pnl_sigma_h{int(k)}"]


def spec_of(cfg_target: dict | None) -> dict:
    s = dict(DEFAULT_SPEC)
    s.update({k: v for k, v in (cfg_target or {}).items() if k in DEFAULT_SPEC})
    return s


# ----------------------------------------------------------------------------- calendar: forced-flat bar per origin
def flat_bar_positions(index: pd.DatetimeIndex, closures, preclose_hours: float) -> np.ndarray:
    """For each bar t, the position of the LAST bar whose close (start + 1 h) is <= the session's forced flat
    (session close - preclose_hours); n-1 when no closure follows inside the data.  Uses only the calendar."""
    from tools import timing_eligibility as EL
    cl = closures if isinstance(closures, pd.DataFrame) else EL.parse_closures(closures)
    _, _, close_ns, _ = EL.sessions(index, cl)
    v = index.asi8
    ff = close_ns - int(preclose_hours * 3.6e12)
    n = len(v)
    out = np.empty(n, np.int64)
    for t in range(n):
        if ff[t] > v[-1] + BAR.value:
            out[t] = n                                                        # no forced flat inside the data (sentinel)
            continue
        last = int(np.searchsorted(v + BAR.value, ff[t], side="right")) - 1   # last bar complete at or before ff
        out[t] = max(last, t)
    return out


# ----------------------------------------------------------------------------- pure label function (replayable)
def _resolve(d, t, open_, high, low, close, sigma, flat_bar, H, m_tp, m_sl, cost):
    """(win, resolution_bar, pnl_sigma) of the direction-d order opened at open(t+1); resolution_bar = n if unresolved."""
    n = len(close)
    if t + 1 >= n:
        return 0, n, np.nan
    s = sigma[t + 1]
    if not np.isfinite(s) or s <= 0:
        return 0, -1, np.nan                                              # no causal sigma: not labellable
    fill = open_[t + 1]
    tp = fill + d * m_tp * s
    sl = fill - d * m_sl * s
    end = min(t + H, int(flat_bar[t]))
    if end < t + 1:
        return 0, -1, np.nan                                              # forced flat before the fill: no executable order
    k = t + 1
    while k <= end and k < n:
        if (low[k] <= sl) if d > 0 else (high[k] >= sl):
            return 0, k, -m_sl - cost
        if (high[k] >= tp) if d > 0 else (low[k] <= tp):
            return 1, k, m_tp - cost
        k += 1
    if end >= n:
        return 0, n, np.nan                                               # unresolved inside the data
    pnl = d * (close[end] - fill) / s - cost
    return int(pnl > 0), end, pnl


def label_arrays(high, low, close, zz=None, params=None, *, k_bars: int, open_=None, sigma=None, flat_bar=None,
                 spec: dict | None = None, **_) -> dict:
    """T1/T2/T3 values and support ends on the given arrays (no cuts).  ``open_``, ``sigma`` (causal, known at
    t+1) and ``flat_bar`` must be given (see ``arrays_from_bars``); a pure function of its arrays."""
    high, low, close, open_, sigma = (np.asarray(x, float) for x in (high, low, close, open_, sigma))
    flat_bar = np.asarray(flat_bar, np.int64)
    sp = spec_of(spec)
    H, m_tp, m_sl, cost = int(k_bars), float(sp["m_tp"]), float(sp["m_sl"]), float(sp["cost_sigma"])
    n = len(close)
    t1 = np.zeros(n)
    s1 = np.full(n, -1, np.int64)
    t2 = np.zeros(n)
    t3 = np.full(n, np.nan)
    winl = np.zeros(n, bool)
    wins = np.zeros(n, bool)
    both = np.zeros(n, bool)
    for t in range(n):
        wl, rl, pl = _resolve(1, t, open_, high, low, close, sigma, flat_bar, H, m_tp, m_sl, cost)
        ws, rs, ps = _resolve(-1, t, open_, high, low, close, sigma, flat_bar, H, m_tp, m_sl, cost)
        if rl < 0 or rs < 0:
            s1[t] = -1                                                    # censored: no causal sigma
            continue
        s1[t] = max(rl, rs)
        if s1[t] >= n:
            continue                                                      # censored: unresolved inside the data
        winl[t], wins[t] = bool(wl), bool(ws)
        both[t] = bool(wl and ws)
        t1[t] = 1.0 if (wl and not ws) else (2.0 if (ws and not wl) else 0.0)
        t2[t] = float(wl)
        t3[t] = max(pl, ps)
    return {"n": n, "t1": t1, "s1": s1, "t2": t2, "s2": s1.copy(), "t3": t3, "s3": s1.copy(), "has_leg": s1 >= 0,
            "both_win": both, "long_wins": winl, "short_wins": wins, "flat_bar": flat_bar, "spec": sp,
            "open": open_, "sigma": sigma}


def arrays_from_bars(bars: pd.DataFrame, zz, closures, *, k_bars: int, spec: dict | None = None) -> dict:
    """Derive the open / causal sigma / forced-flat arrays from the bars and the admitted closures, then label."""
    sp = spec_of(spec)
    sigma = np.asarray(zz.causal_sigma(bars), float)
    flat = flat_bar_positions(bars.index, closures, float(sp["preclose_hours"])) if closures is not None \
        else np.full(len(bars), len(bars), np.int64)                      # no calendar: never resolve by forced flat
    return label_arrays(bars.HIGH.to_numpy(float), bars.LOW.to_numpy(float), bars.CLOSE.to_numpy(float), zz, None,
                        k_bars=k_bars, open_=bars.OPEN.to_numpy(float), sigma=sigma, flat_bar=flat, spec=sp)


# ----------------------------------------------------------------------------- frame + ledger with cuts
def build_targets(bars: pd.DataFrame, zz, params, *, k_bars: int, cuts, arrays: dict | None = None,
                  with_ledger: bool = False, closures=None, spec: dict | None = None, **_):
    n = len(bars)
    k = int(k_bars)
    a = arrays or arrays_from_bars(bars, zz, closures, k_bars=k, spec=spec)
    ts = bars.index
    cut_pos = _cut_positions(ts, cuts)
    t1, st1 = _status(a["t1"], a["s1"], n, cut_pos)
    t2, st2 = _status(a["t2"], a["s2"], n, cut_pos)
    t3, st3 = _status(a["t3"], a["s3"], n, cut_pos, has_value=a["has_leg"])
    long_ = np.where(np.isfinite(t1), (t1 == 1).astype(float), np.nan)
    short = np.where(np.isfinite(t1), (t1 == 2).astype(float), np.nan)

    def _time(pos):
        pos = np.asarray(pos)
        out = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
        ok = (pos >= 0) & (pos < n)
        out[ok] = ts[pos[ok]]
        return pd.DatetimeIndex(out)

    frame = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64), "timestamp": ts,
                          primary(k): t1, indicators(k)[0]: long_, indicators(k)[1]: short,
                          comparators(k)[0]: t2, comparators(k)[1]: t3})
    raw1 = a["t1"]
    ledger = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64), "origin_time": ts, "origin_available_time": ts + BAR,
        "event_time": _time(a["s1"]),
        "event_kind": np.where(raw1 == 1, "LONG_WINS", np.where(raw1 == 2, "SHORT_WINS", "NONE")),
        "both_kinds_in_window": a["both_win"], "window_gap_hours": np.nan,
        "window_crosses_gap": np.zeros(n, bool),                          # orders close at the forced flat
        "label_group_id": a["s1"], "expert_trade_id": np.full(n, -1, np.int64),
        "forced_flat_bar": a["flat_bar"],
        "t1_support_end": _time(a["s1"]), "t1_label_available_time": _time(a["s1"]) + BAR, "t1_status": st1,
        "t2_support_end": _time(a["s2"]), "t2_label_available_time": _time(a["s2"]) + BAR, "t2_status": st2,
        "t3_support_end": _time(a["s3"]), "t3_label_available_time": _time(a["s3"]) + BAR, "t3_status": st3,
    })
    frame.attrs.update(pivots=0, confirmed=0, unconfirmed=0, cut_positions=cut_pos, spec=a["spec"],
                       long_wins=int(a["long_wins"].sum()), short_wins=int(a["short_wins"].sum()), both_win=int(a["both_win"].sum()))
    return (frame, ledger) if with_ledger else frame


def manifest_targets(k_bars: int) -> list[dict]:
    k = int(k_bars)
    return [
        {"target_id": primary(k), "column": primary(k), "family": FAMILY, "head": "classification", "horizon_hours": k,
         "role": "PRIMARY_PROFITABLE_DIRECTION"},
        {"target_id": indicators(k)[0], "column": indicators(k)[0], "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "PRIMARY_INDICATOR"},
        {"target_id": indicators(k)[1], "column": indicators(k)[1], "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "PRIMARY_INDICATOR"},
        {"target_id": comparators(k)[0], "column": comparators(k)[0], "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "DIRECTION_INDEPENDENT_COMPARATOR"},
        {"target_id": comparators(k)[1], "column": comparators(k)[1], "family": FAMILY, "head": "regression",
         "horizon_hours": k, "role": "BEST_ORDER_NET_PNL_SIGMA"},
    ]


def status_table(led: pd.DataFrame, periods: dict) -> dict:
    ts = pd.DatetimeIndex(led["origin_time"])
    out = {}
    for name, (a, b) in periods.items():
        m = (ts >= _utc(a)) & (ts < _utc(b))
        out[name] = {"origins": int(m.sum())}
        for tgt in ("t1", "t2", "t3"):
            vc = led.loc[m, f"{tgt}_status"].value_counts()
            out[name][tgt] = {s: int(vc.get(s, 0)) for s in STATUS}
            if sum(out[name][tgt].values()) != int(m.sum()):
                raise TimingError("ORIGINS_UNACCOUNTED")
        out[name]["t1_window_crosses_gap"] = 0
        out[name]["t1_both_kinds_in_window"] = int(led.loc[m, "both_kinds_in_window"].sum())
    return out


def target_identity(*, params=None, k_bars, zigzag_sha256, lake_sha256, spec: dict | None = None, **_) -> str:
    body = {"semantics": SEMANTICS, "horizon_bars": int(k_bars), "zigzag_sigma_module": zigzag_sha256, "lake": lake_sha256,
            "spec": spec_of(spec)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def targets_document(frame: pd.DataFrame, *, params=None, k_bars: int, lake_sha256: str, zigzag_sha256: str,
                     spec: dict | None = None, **_) -> dict:
    k = int(k_bars)
    y = frame[primary(k)]
    return {"label": "HINDSIGHT_PROFIT_DIRECTION_TARGETS_DEVELOPMENT_ONLY", "family": FAMILY, "semantics": SEMANTICS,
            "spec": spec_of(spec), "k_bars": k, "k_unit": "OBSERVED_MARKET_BARS_HORIZON",
            "order_rule": "fill open(t+1); TP/SL = m_tp/m_sl x causal sigma(t+1); SL-first in a bar; closed at the forced flat or the horizon at that CLOSE; net of cost_sigma",
            "bars": int(len(frame)), "pivots": 0, "confirmed_pivots": 0, "unconfirmed_pivots": 0,
            "long_wins": frame.attrs.get("long_wins"), "short_wins": frame.attrs.get("short_wins"), "both_win": frame.attrs.get("both_win"),
            "class_counts_labelled": {str(c): int((y == c).sum()) for c in (0, 1, 2)}, "not_labelled_rows": int(y.isna().sum()),
            "lake_sha256": lake_sha256, "zigzag_module_sha256": zigzag_sha256, "test_read": False,
            "span": [str(frame.timestamp.iloc[0]), str(frame.timestamp.iloc[-1])],
            "identity": target_identity(k_bars=k, zigzag_sha256=zigzag_sha256, lake_sha256=lake_sha256, spec=spec),
            "note": "labels read the future price path (hindsight); labels only, never inputs"}


# ----------------------------------------------------------------------------- finality evidence (A02/A03)
def finality_check(bars: pd.DataFrame, zz, params, *, k_bars: int, cut_positions, seed: int = 0, full: dict | None = None,
                   closures=None, spec: dict | None = None, **_) -> dict:
    """Truncation and future-perturbation replays: every row t < m whose support end is < m must be identical.
    The replays recompute the causal sigma from the replayed closes and keep the calendar (forced-flat bars)."""
    high, low, close, open_ = (bars[c].to_numpy(float) for c in ("HIGH", "LOW", "CLOSE", "OPEN"))
    n = len(high)
    full = full or arrays_from_bars(bars, zz, closures, k_bars=k_bars, spec=spec)
    rng = np.random.default_rng(seed)
    out = []
    for m in cut_positions:
        rec = {"cut_position": int(m), "cut_time": str(bars.index[m])}
        for mode in ("truncation", "future_perturbation"):
            if mode == "truncation":
                h2, l2, c2, o2 = high[:m], low[:m], close[:m], open_[:m]
                flat2 = np.where(full["flat_bar"][:m] >= m, m, full["flat_bar"][:m])   # beyond the truncation = unknown
            else:
                base = close[m - 1]
                c_new = base * np.exp(rng.normal(0, 8e-4, n - m).cumsum())
                spread = np.abs(rng.normal(0, 4e-4, n - m)) * base
                o_new = np.r_[base, c_new[:-1]]
                h2, l2, c2, o2 = np.r_[high[:m], c_new + spread], np.r_[low[:m], c_new - spread], np.r_[close[:m], c_new], np.r_[open_[:m], o_new]
                flat2 = full["flat_bar"]
            bars2 = pd.DataFrame({"OPEN": o2, "HIGH": h2, "LOW": l2, "CLOSE": c2}, index=bars.index[:len(c2)])
            sig2 = np.asarray(zz.causal_sigma(bars2), float)
            rep = label_arrays(h2, l2, c2, k_bars=k_bars, open_=o2, sigma=sig2, flat_bar=flat2, spec=full["spec"])
            res = {}
            for tg in ("t1", "t2", "t3"):
                s = full[f"s{tg[1]}"][:m]
                final = (s >= 0) & (s < m)
                fv, rv = full[tg][:m], rep[tg][:m]
                same = (fv == rv) | (np.isnan(fv) & np.isnan(rv))
                res[tg] = {"rows_declared_final": int(final.sum()), "changed_among_final": int((~same & final).sum()),
                           "changed_among_not_final": int((~same & ~final).sum())}
            rec[mode] = res
        out.append(rec)
    ok = all(rec[mode][tg]["changed_among_final"] == 0 for rec in out for mode in ("truncation", "future_perturbation") for tg in rec[mode])
    return {"rule": SEMANTICS, "cuts": out, "all_final_rows_unchanged": bool(ok)}


# ----------------------------------------------------------------------------- D5: split label (2026-10-21, additive)
OUTCOME_TIMEOUT, OUTCOME_UPPER, OUTCOME_LOWER, OUTCOME_BOTH, OUTCOME_CENSORED = 0, 1, 2, 3, -1


def barrier_outcomes(open_, high, low, close, sigma, flat_bar, *, horizon_bars: int, m: float = 2.0):
    """Symmetric barriers around the executable fill open(t+1): upper = fill + m*sigma(t+1), lower = fill - m*sigma.
    Per origin t: (outcome, resolution_bar). outcome 1 = upper touched first, 2 = lower first, 3 = both in the same
    bar (order unknown inside the bar), 0 = neither before min(t + H, forced flat) (timeout), -1 = not labellable
    (no causal sigma, fill after the forced flat, or unresolved inside the data). Labels for D5:
      barrier_hit       = outcome in {1, 2, 3}  vs  0                (volatility / activity part)
      upper_first|hit   = 1 if outcome == 1, 0 if outcome == 2, NaN otherwise   (direction part)"""
    open_, high, low, close, sigma = (np.asarray(x, float) for x in (open_, high, low, close, sigma))
    flat_bar = np.asarray(flat_bar, np.int64)
    n = len(close)
    out = np.full(n, OUTCOME_CENSORED, np.int64)
    res = np.full(n, -1, np.int64)
    for t in range(n - 1):
        s = sigma[t + 1]
        end = min(t + int(horizon_bars), int(flat_bar[t]))
        if not np.isfinite(s) or s <= 0 or end < t + 1:
            continue
        up, lo = open_[t + 1] + m * s, open_[t + 1] - m * s
        k = t + 1
        while k <= end and k < n:
            hu, hl = high[k] >= up, low[k] <= lo
            if hu or hl:
                out[t], res[t] = (OUTCOME_BOTH if (hu and hl) else (OUTCOME_UPPER if hu else OUTCOME_LOWER)), k
                break
            k += 1
        else:
            if end < n:
                out[t], res[t] = OUTCOME_TIMEOUT, end
    return out, res
