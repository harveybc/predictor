"""Timing targets for feature selection, derived from the sealed ZigZag hindsight expert, with a per-row
label-support ledger (semantics ``timing_targets.v2``; plan audit 2026-10-19 F01/F05).

The expert lives in heuristic-strategy ``app/short_horizon/zigzag.py`` (branch
satoshi/short-horizon-strategy-20261018, @b9b4f9c); it is loaded BY PATH with a pinned sha256 and never
copied here.  Its pivots are the final repainted ZigZag: hindsight ground truth of the FUTURE, allowed as
LABELS only.  Nothing in this module is a model input.  This is expert-demonstration / hindsight-guided
learning, not human-preference RLHF.

Clock and grid.  ``t`` is the integer position of an OBSERVED market bar on the lake grid (vendor bar-start
label, clock as labelled); the bar is complete at label + 1 h, which is the origin's availability.
``k`` counts OBSERVED market bars, not elapsed hours: a window may span a closure; the ledger flags it
(``window_crosses_gap``, ``window_gap_hours``).  Decisions are taken after the completed bar t; the pinned
strategy adapter fills at the next executable quote (never at the pivot's high/low).

Targets (family ``timing_pivot``):

  T1 PRIMARY, ANTICIPATORY  ``timing_pivot_k{k}``: the EARLIEST final pivot inside the window
       (t, t+k] (t excluded, t+k included) decides the class: 1 valley, 2 peak; 0 NONE if the window holds
       no final pivot.  Pivots alternate and occupy distinct bars, so a tie cannot occur (refused if it
       did); a second, opposite pivot in the same window is flagged (``both_kinds_in_window``) and does
       not change the class (earliest-event rule).  Indicators ``timing_valley_k{k}`` /
       ``timing_peak_k{k}`` are the same rows one-vs-rest.  Every hourly origin is kept (no thinning).
  T2 REACTIVE COMPARATOR    ``timing_action_confirmed``: +1 / -1 at the bar on which the expert's
       confirmation rule locks a final valley / peak, 0 elsewhere.  Reported separately; never pooled
       with T1.
  T3 STATE-CONDITIONAL      ``timing_expert_leg_remaining_pips``: for an agent HOLDING the expert's
       current leg direction at t (long after a valley, short after a peak), the remaining GROSS distance
       in pips from close(t) to the next final pivot level.  Not a trade P&L: no costs, no SL/TP, no
       entry price; it is not full-trade profit repeated at every row.

Confirmation definition (configurable, part of the target identity): ``confirmation_bars`` (expert v1: the
next FINAL pivot + Backstep, replayed) or ``confirmation_bars_fast`` (expert v2, abbfaa6: the first opposite
candidate + Backstep, brute-force verified by its owner).  Every support end below uses the configured one.

Label support (F01).  ``support_end`` is the last bar whose value can change the label:
  T1: confirmation bar of the first final pivot with index > t+k (it locks every pivot up to it);
  T2: max(t, confirmation bar of the first final pivot with index > t);
  T3: confirmation bar of the next pivot (index > t) of the current leg.
These rules are not inferred from a median lag: ``finality_check`` recomputes the labels on truncated
and on future-perturbed series and verifies that every row declared final is unchanged.
A row whose certifying pivot is never confirmed by the expert's rule, or confirmed beyond the loaded
data, is CENSORED (NaN), never NONE or zero.  A row whose support reaches a declared cut is PURGED for
the period before that cut.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
FAMILY = "timing_pivot"
SEMANTICS = "timing_targets.v2"
PIP = 1e-4
BAR = pd.Timedelta(hours=1)
STATUS = ("LABELLED", "CENSORED_UNCONFIRMED", "CENSORED_END_OF_DATA", "PURGED_SUPPORT_CROSSES_CUT", "NO_LEG")


class TimingError(ValueError):
    """A typed refusal; the message starts with its code."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_zigzag_module(path, expected_sha256: str):
    """Load the strategy's zigzag module by path; refuse when its bytes differ from the pin."""
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise TimingError(f"ZIGZAG_MODULE_DIGEST: {path.name} is {actual[:12]}, pinned {expected_sha256[:12]}")
    spec = importlib.util.spec_from_file_location(f"timing_zigzag_expert_{actual[:8]}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        # the real module imports its siblings (.blocks, .calendar, .execution): load it as a package member
        pkg_dir = path.parent
        name = f"short_horizon_expert_pkg_{actual[:8]}"
        pkg_spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py",
                                                          submodule_search_locations=[str(pkg_dir)])
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[name] = pkg
        pkg_spec.loader.exec_module(pkg)
        spec = importlib.util.spec_from_file_location(f"{name}.zigzag", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.zigzag"] = mod
        spec.loader.exec_module(mod)
    return mod


def load_lake_bars(path, start, end_exclusive) -> pd.DataFrame:
    """Lake OHLC parquet ``[start, end)`` with the TEST boundary enforced at read time (no row created)."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end_exclusive, tz="UTC")
    if end > TEST_START:
        raise TimingError(f"TEST_ROWS_FORBIDDEN: window end {end} reaches {TEST_START}")
    raw = pd.read_parquet(Path(path))
    idx = pd.DatetimeIndex(raw["datetime"])
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    keep = (idx >= start) & (idx < end)
    frame = raw.loc[keep, ["open", "high", "low", "close"]].astype(float)
    frame.columns = ["OPEN", "HIGH", "LOW", "CLOSE"]
    frame.index = idx[keep].rename("DATE_TIME")
    if len(frame) and frame.index.max() >= TEST_START:
        raise TimingError("TEST_ROWS_FORBIDDEN: TEST rows present after filtering")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise TimingError("MALFORMED_BARS: labels must be unique and increasing")
    if not np.isfinite(frame.to_numpy()).all():
        raise TimingError("MALFORMED_BARS: non-finite OHLC")
    return frame


# ----------------------------------------------------------------------------- core label algebra
def _pivots(zz, high, low, params):
    piv = zz.hindsight_pivots(np.asarray(high, float), np.asarray(low, float), params)
    idx = np.array([i for i, _, _ in piv], np.int64)
    kind = np.array([kd for _, kd, _ in piv], np.int64)
    lvl = np.array([lv for _, _, lv in piv], float)
    if len(idx) and np.any(np.diff(idx) <= 0):
        raise TimingError("PIVOT_TIE: final pivots must occupy strictly increasing bars")
    return piv, idx, kind, lvl


def confirm(zz, high, low, piv, params, *, confirmation: str = "confirmation_bars", max_conf_lag: int = 400):
    """The expert's confirmation bars under the configured definition (refused when the module lacks it)."""
    fn = getattr(zz, confirmation, None)
    if fn is None:
        raise TimingError(f"CONFIRMATION_DEFINITION_ABSENT: the expert module has no {confirmation}")
    if confirmation == "confirmation_bars":
        return fn(high, low, piv, params, max_lag=max_conf_lag)
    return fn(high, low, piv, params)


def label_arrays(high, low, close, zz, params, *, k_bars: int, conf=None, max_conf_lag: int = 400,
                 confirmation: str = "confirmation_bars") -> dict:
    """T1/T2/T3 values and their support ends on the given bars (no cuts applied).  A pure function of the
    arrays: used for the full series and for every truncated / perturbed replay in ``finality_check``."""
    high, low, close = (np.asarray(x, float) for x in (high, low, close))
    n = len(high)
    k = int(k_bars)
    piv, idx, kind, lvl = _pivots(zz, high, low, params)
    P = len(idx)
    if conf is None:
        conf = confirm(zz, high, low, piv, params, confirmation=confirmation, max_conf_lag=max_conf_lag) if P else []
    conf = np.asarray(conf, np.int64).reshape(-1)
    t = np.arange(n)
    if P == 0:
        z = np.zeros(n)
        neg = np.full(n, -1, np.int64)
        return {"n": n, "pivot_bar": idx, "pivot_kind": kind, "pivot_level": lvl, "conf": conf, "t1": z.copy(),
                "s1": neg, "event_pivot": neg, "both_kinds": np.zeros(n, bool), "certifying_pivot": neg, "t2": z.copy(),
                "s2": neg, "t3": np.full(n, np.nan), "s3": neg, "has_leg": np.zeros(n, bool), "leg": neg}
    cl = lambda a: np.clip(a, 0, P - 1)  # noqa: E731
    e = np.searchsorted(idx, t, side="right")                      # first pivot with index > t
    has_event = (e < P) & (idx[cl(e)] <= t + k)
    both = has_event & (e + 1 < P) & (idx[cl(e + 1)] <= t + k)
    t1 = np.where(has_event, np.where(kind[cl(e)] == zz.VALLEY, 1.0, 2.0), 0.0)
    q = np.searchsorted(idx, t + k, side="right")                  # certifying pivot: first index > t+k
    s1 = np.where(q < P, conf[cl(q)], -1)
    t2 = np.zeros(n)
    for j in range(P):
        c = conf[j]
        if 0 <= c < n:
            t2[c] = 1.0 if kind[j] == zz.VALLEY else -1.0
    s2c = np.where(e < P, conf[cl(e)], -1)
    s2 = np.where(s2c >= 0, np.maximum(t, s2c), -1)
    j = e - 1
    has_leg = (j >= 0) & (e < P)
    d = np.where(kind[cl(j)] == zz.VALLEY, 1.0, -1.0)
    t3 = np.where(has_leg, d * (lvl[cl(e)] - close) / PIP, np.nan)
    s3 = np.where(has_leg, conf[cl(e)], -1)
    return {"n": n, "pivot_bar": idx, "pivot_kind": kind, "pivot_level": lvl, "conf": conf,
            "t1": t1, "s1": s1.astype(np.int64), "event_pivot": np.where(has_event, e, -1), "both_kinds": both,
            "certifying_pivot": np.where(q < P, q, -1), "t2": t2, "s2": s2.astype(np.int64),
            "t3": t3, "s3": s3.astype(np.int64), "has_leg": has_leg, "leg": np.where(has_leg, j, -1)}


def _status(values, support, n, cut_pos, has_value=None):
    """Per-row status and the NaN-masked values: censored (unconfirmed / beyond data), purged at cuts."""
    v = np.asarray(values, float).copy()
    st = np.full(n, "LABELLED", object)
    if has_value is not None:
        st[~np.asarray(has_value, bool)] = "NO_LEG"
    st[(support < 0) & (st == "LABELLED")] = "CENSORED_UNCONFIRMED"
    st[(support >= n) & (st == "LABELLED")] = "CENSORED_END_OF_DATA"
    t = np.arange(n)
    for c in cut_pos:
        st[(t < c) & (support >= c) & (st == "LABELLED")] = "PURGED_SUPPORT_CROSSES_CUT"
    v[st != "LABELLED"] = np.nan
    return v, st


def _cut_positions(ts, cuts):
    out = []
    for cut in cuts:
        cut = pd.Timestamp(cut)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
        c = int(ts.searchsorted(cut))
        if 0 < c < len(ts):
            out.append(c)
    return sorted(set(out))


def build_targets(bars: pd.DataFrame, zz, params, *, k_bars: int, cuts, max_conf_lag: int = 400,
                  arrays: dict | None = None, with_ledger: bool = False, confirmation: str = "confirmation_bars"):
    """All timing targets on the bar grid of ``bars``; with ``with_ledger`` returns (frame, ledger)."""
    high, low, close = (bars[c].to_numpy(float) for c in ("HIGH", "LOW", "CLOSE"))
    n = len(bars)
    k = int(k_bars)
    a = arrays or label_arrays(high, low, close, zz, params, k_bars=k, max_conf_lag=max_conf_lag, confirmation=confirmation)
    ts = bars.index
    cut_pos = _cut_positions(ts, cuts)
    t1, st1 = _status(a["t1"], a["s1"], n, cut_pos)
    t2, st2 = _status(a["t2"], a["s2"], n, cut_pos)
    t3, st3 = _status(a["t3"], a["s3"], n, cut_pos, has_value=a["has_leg"])
    valley = np.where(np.isfinite(t1), (t1 == 1).astype(float), np.nan)
    peak = np.where(np.isfinite(t1), (t1 == 2).astype(float), np.nan)
    gap_after = np.r_[np.diff(ts.asi8) / 3.6e12, np.nan]                   # hours from bar i to bar i+1
    win_gap = np.full(n, np.nan)
    for i in range(n - k):
        win_gap[i] = float(np.max(gap_after[i:i + k]))

    def _time(pos):
        pos = np.asarray(pos)
        out = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
        ok = (pos >= 0) & (pos < n)
        out[ok] = ts[pos[ok]]
        return pd.DatetimeIndex(out)

    ev = a["event_pivot"]
    ev_bar = np.where(ev >= 0, a["pivot_bar"][np.clip(ev, 0, max(len(a["pivot_bar"]) - 1, 0))] if len(a["pivot_bar"]) else -1, -1)
    raw1 = a["t1"]
    frame = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64), "timestamp": ts,
        f"timing_pivot_k{k}": t1, f"timing_valley_k{k}": valley, f"timing_peak_k{k}": peak,
        "timing_action_confirmed": t2, "timing_expert_leg_remaining_pips": t3,
    })
    ledger = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64),
        "origin_time": ts, "origin_available_time": ts + BAR,
        "event_time": _time(ev_bar),
        "event_kind": np.where(raw1 == 1, "VALLEY", np.where(raw1 == 2, "PEAK", "NONE")),
        "both_kinds_in_window": a["both_kinds"], "window_gap_hours": win_gap,
        "window_crosses_gap": np.nan_to_num(win_gap, nan=0.0) > 1.0,
        "label_group_id": a["certifying_pivot"], "expert_trade_id": a["leg"],
        "t1_support_end": _time(a["s1"]), "t1_label_available_time": _time(a["s1"]) + BAR, "t1_status": st1,
        "t2_support_end": _time(a["s2"]), "t2_label_available_time": _time(a["s2"]) + BAR, "t2_status": st2,
        "t3_support_end": _time(a["s3"]), "t3_label_available_time": _time(a["s3"]) + BAR, "t3_status": st3,
    })
    frame.attrs.update(pivots=int(len(a["pivot_bar"])), confirmed=int((a["conf"] >= 0).sum()),
                       unconfirmed=int((a["conf"] < 0).sum()), cut_positions=cut_pos)
    return (frame, ledger) if with_ledger else frame


def manifest_targets(k_bars: int) -> list[dict]:
    k = int(k_bars)
    return [
        {"target_id": f"timing_pivot_k{k}", "column": f"timing_pivot_k{k}", "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "PRIMARY_ANTICIPATORY"},
        {"target_id": f"timing_valley_k{k}", "column": f"timing_valley_k{k}", "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "PRIMARY_INDICATOR"},
        {"target_id": f"timing_peak_k{k}", "column": f"timing_peak_k{k}", "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "PRIMARY_INDICATOR"},
        {"target_id": "timing_action_confirmed", "column": "timing_action_confirmed", "family": FAMILY, "head": "classification",
         "horizon_hours": k, "role": "REACTIVE_COMPARATOR"},
        {"target_id": "timing_expert_leg_remaining_pips", "column": "timing_expert_leg_remaining_pips", "family": FAMILY,
         "head": "regression", "horizon_hours": k, "role": "STATE_CONDITIONAL_HOLDING_EXPERT_LEG"},
    ]


def status_table(led: pd.DataFrame, periods: dict) -> dict:
    """Counts of every hourly origin by period x target x status (all origins accounted for)."""
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
        out[name]["t1_window_crosses_gap"] = int(led.loc[m, "window_crosses_gap"].sum())
        out[name]["t1_both_kinds_in_window"] = int(led.loc[m, "both_kinds_in_window"].sum())
    return out


def _utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def target_identity(*, params, k_bars, zigzag_sha256, lake_sha256, confirmation: str = "confirmation_bars") -> str:
    body = {"semantics": SEMANTICS, "k": int(k_bars), "zigzag": zigzag_sha256, "lake": lake_sha256,
            "params": [int(params.depth), float(params.deviation_pips), int(params.backstep)]}
    if confirmation != "confirmation_bars":           # v2 identities stay byte-identical for the v1 definition
        body["confirmation"] = confirmation
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def targets_document(frame: pd.DataFrame, *, params, k_bars: int, lake_sha256: str, zigzag_sha256: str,
                     confirmation: str = "confirmation_bars") -> dict:
    k = int(k_bars)
    y = frame[f"timing_pivot_k{k}"]
    return {"label": "HINDSIGHT_ZIGZAG_TIMING_TARGETS_DEVELOPMENT_ONLY", "family": FAMILY, "semantics": SEMANTICS,
            "params": {"depth": params.depth, "deviation_pips": params.deviation_pips, "backstep": params.backstep},
            "k_bars": k, "k_unit": "OBSERVED_MARKET_BARS", "window": "(t, t+k] earliest event", "bars": int(len(frame)),
            "pivots": frame.attrs.get("pivots"), "confirmed_pivots": frame.attrs.get("confirmed"),
            "unconfirmed_pivots": frame.attrs.get("unconfirmed"),
            "class_counts_labelled": {str(c): int((y == c).sum()) for c in (0, 1, 2)}, "not_labelled_rows": int(y.isna().sum()),
            "lake_sha256": lake_sha256, "zigzag_module_sha256": zigzag_sha256, "test_read": False,
            "span": [str(frame.timestamp.iloc[0]), str(frame.timestamp.iloc[-1])],
            "confirmation": confirmation,
            "identity": target_identity(params=params, k_bars=k, zigzag_sha256=zigzag_sha256, lake_sha256=lake_sha256,
                                        confirmation=confirmation),
            "note": "labels are the final repainted ZigZag (hindsight); labels only, never inputs"}


# ----------------------------------------------------------------------------- finality evidence (A02/A03)
def finality_check(bars: pd.DataFrame, zz, params, *, k_bars: int, cut_positions, seed: int = 0,
                   full: dict | None = None, check_t2: bool = True, confirmation: str = "confirmation_bars") -> dict:
    """For each cut m: (a) TRUNCATION, recompute every target on bars[:m]; (b) FUTURE PERTURBATION, replace
    bars[m:] by a seeded random walk and recompute.  Every row t < m whose declared support end is < m must be
    identical in both replays; rows whose support reaches m may change (permitted, never used across m)."""
    high, low, close = (bars[c].to_numpy(float) for c in ("HIGH", "LOW", "CLOSE"))
    n = len(high)
    full = full or label_arrays(high, low, close, zz, params, k_bars=k_bars, confirmation=confirmation)
    rng = np.random.default_rng(seed)
    out = []
    for m in cut_positions:
        rec = {"cut_position": int(m), "cut_time": str(bars.index[m])}
        for mode in ("truncation", "future_perturbation"):
            if mode == "truncation":
                h2, l2, c2 = high[:m], low[:m], close[:m]
            else:
                base = close[m - 1]
                c_new = base * np.exp(rng.normal(0, 8e-4, n - m).cumsum())
                spread = np.abs(rng.normal(0, 4e-4, n - m)) * base
                h2, l2, c2 = np.r_[high[:m], c_new + spread], np.r_[low[:m], c_new - spread], np.r_[close[:m], c_new]
            conf = None if check_t2 else [-1] * len(zz.hindsight_pivots(h2, l2, params))
            rep = label_arrays(h2, l2, c2, zz, params, k_bars=k_bars, conf=conf, confirmation=confirmation)
            res = {}
            for tg in ("t1", "t2", "t3"):
                if tg in ("t2", "t3") and not check_t2:
                    continue
                s = full[f"s{tg[1]}"][:m]
                final = (s >= 0) & (s < m)
                fv, rv = full[tg][:m], rep[tg][:m]
                same = (fv == rv) | (np.isnan(fv) & np.isnan(rv))
                res[tg] = {"rows_declared_final": int(final.sum()), "changed_among_final": int((~same & final).sum()),
                           "changed_among_not_final": int((~same & ~final).sum())}
            rec[mode] = res
        out.append(rec)
    ok = all(rec[mode][tg]["changed_among_final"] == 0 for rec in out for mode in ("truncation", "future_perturbation")
             for tg in rec[mode])
    return {"rule": SEMANTICS, "confirmation": confirmation, "cuts": out, "all_final_rows_unchanged": bool(ok)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake", required=True)
    ap.add_argument("--lake-sha256", required=True)
    ap.add_argument("--zigzag-module", required=True)
    ap.add_argument("--zigzag-sha256", required=True)
    ap.add_argument("--start", default="2019-12-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--deviation", type=float, default=36.3)
    ap.add_argument("--backstep", type=int, default=2)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cut", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if sha256_file(a.lake) != a.lake_sha256:
        raise TimingError("LAKE_DIGEST: lake file differs from the pin")
    zz = load_zigzag_module(a.zigzag_module, a.zigzag_sha256)
    bars = load_lake_bars(a.lake, a.start, a.end)
    p = zz.Params(depth=a.depth, deviation_pips=a.deviation, backstep=a.backstep)
    frame, ledger = build_targets(bars, zz, p, k_bars=a.k, cuts=a.cut, with_ledger=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "timing_targets.parquet", index=False)
    ledger.to_parquet(out / "LABEL_LEDGER.parquet", index=False)
    doc = targets_document(frame, params=p, k_bars=a.k, lake_sha256=a.lake_sha256, zigzag_sha256=a.zigzag_sha256)
    doc["targets_sha256"] = sha256_file(out / "timing_targets.parquet")
    doc["ledger_sha256"] = sha256_file(out / "LABEL_LEDGER.parquet")
    (out / "TARGETS.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
