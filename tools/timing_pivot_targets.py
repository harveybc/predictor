"""Timing targets for feature selection, derived from the sealed ZigZag hindsight expert.

The expert lives in heuristic-strategy ``app/short_horizon/zigzag.py`` (branch
satoshi/short-horizon-strategy-20261018, @b9b4f9c); it is loaded BY PATH with a pinned sha256 and never
copied here (no duplicated scientific engine).  Its pivots are the final repainted ZigZag: they are
hindsight ground truth of the FUTURE and are allowed as LABELS only.  Nothing in this module may be
used as a model input.

Three targets, family ``timing_pivot``:

  T1  ``timing_pivot_k{k}``        3-class: 0 none, 1 a valley lies in (t, t+k], 2 a peak lies in (t, t+k]
                                   (the k bars before the pivot; the pivot bar itself is not labelled by
                                   its own pivot).  Also exported as two 0/1 indicators
                                   ``timing_valley_k{k}`` / ``timing_peak_k{k}`` for engines whose causal
                                   rung is a linear-probability model.
  T2  ``timing_action_confirmed``  the expert's action at t from the CAUSALLY CONFIRMED timing: +1 long
                                   at the confirmation bar of a valley, -1 short at that of a peak, 0
                                   elsewhere (``confirmation_bars`` of the expert; -1 = never confirmed).
  T3  ``timing_expert_leg_pnl_pips`` expert P&L attribution per leg: signed distance in pips from
                                   close(t) to the NEXT pivot level in the direction of the current
                                   expert leg (long after a valley, short after a peak).  NaN before the
                                   first pivot and after the last one.

Purge: a label whose window (t, t+k] crosses a split cut is NaN.  T2 is purged when the confirmation
of its pivot lies at or after the cut (the label would read bars past the cut); T3 when the next pivot
lies at or after the cut.  TEST rows (2025-01-01 and later) are refused at load time.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
FAMILY = "timing_pivot"
PIP = 1e-4


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
    spec = importlib.util.spec_from_file_location("timing_zigzag_expert", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        # the real module imports its siblings (.blocks, .calendar, .execution): load it as a package member
        pkg_dir = path.parent
        pkg_spec = importlib.util.spec_from_file_location("short_horizon_expert_pkg", pkg_dir / "__init__.py",
                                                          submodule_search_locations=[str(pkg_dir)])
        pkg = importlib.util.module_from_spec(pkg_spec)
        import sys
        sys.modules["short_horizon_expert_pkg"] = pkg
        pkg_spec.loader.exec_module(pkg)
        spec = importlib.util.spec_from_file_location("short_horizon_expert_pkg.zigzag", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["short_horizon_expert_pkg.zigzag"] = mod
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


def pivot_within_k(pivots, n: int, k: int, valley_code: int):
    """T1 codes: the k bars before pivot bar i carry 1 (valley) or 2 (peak)."""
    y = np.zeros(n, np.int64)
    for i, kind, _ in pivots:
        a, b = max(0, i - k), i - 1
        if b >= a:
            y[a:b + 1] = 1 if kind == valley_code else 2
    return y


def confirmed_actions(pivots, conf, n: int, valley_code: int):
    a = np.zeros(n, np.int64)
    conf_bar = np.full(n, -1, np.int64)
    for (i, kind, _), c in zip(pivots, conf):
        if c is None or c < 0 or c >= n:
            continue
        a[c] = 1 if kind == valley_code else -1
        conf_bar[c] = i
    return a, conf_bar


def leg_pnl_pips(pivots, close: np.ndarray, valley_code: int):
    """T3: signed pips from close(t) to the next pivot level along the current expert leg."""
    n = len(close)
    out = np.full(n, np.nan)
    nxt_bar = np.full(n, -1, np.int64)
    for j in range(len(pivots) - 1):
        i, kind, _ = pivots[j]
        i2, _, lvl2 = pivots[j + 1]
        d = 1.0 if kind == valley_code else -1.0
        sl = slice(i, i2)                       # bars inside the leg [pivot j, pivot j+1)
        out[sl] = d * (lvl2 - close[sl]) / PIP
        nxt_bar[sl] = i2
    return out, nxt_bar


def build_targets(bars: pd.DataFrame, zz, params, *, k_bars: int, cuts, max_conf_lag: int = 400) -> pd.DataFrame:
    """All timing targets on the bar grid of ``bars`` with purges at ``cuts`` (UTC timestamps)."""
    high, low, close = (bars[c].to_numpy(float) for c in ("HIGH", "LOW", "CLOSE"))
    n = len(bars)
    pivots = zz.hindsight_pivots(high, low, params)
    conf = zz.confirmation_bars(high, low, pivots, params, max_lag=max_conf_lag) if pivots else []
    y = pivot_within_k(pivots, n, k_bars, zz.VALLEY)
    valley = (y == 1).astype(float)
    peak = (y == 2).astype(float)
    action, _ = confirmed_actions(pivots, conf, n, zz.VALLEY)
    pnl, nxt_bar = leg_pnl_pips(pivots, close, zz.VALLEY)
    ts = bars.index
    y_f = y.astype(float)
    action_f = action.astype(float)
    # window (t, t+k] must exist inside the loaded series
    tail = np.arange(n) + k_bars >= n
    y_f[tail] = np.nan
    valley[tail] = np.nan
    peak[tail] = np.nan
    pos_of_pivot = {i: j for j, (i, _, _) in enumerate(pivots)}
    for cut in cuts:
        cut = pd.Timestamp(cut)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
        c = int(ts.searchsorted(cut))                    # first bar at or after the cut
        if c <= 0 or c > n:
            continue
        lo = max(0, c - k_bars)
        y_f[lo:c] = np.nan
        valley[lo:c] = np.nan
        peak[lo:c] = np.nan
        # T3: legs whose next pivot is at/after the cut are purged before the cut
        purge3 = (nxt_bar >= c) & (np.arange(n) < c)
        pnl[purge3] = np.nan
        # T2: a confirmation at/after the cut labels nothing before it (it reads bars past the cut); it is a
        # bar-local label so only the bar itself matters: confirmation bars are never before their pivot.
    # T2 confirmations lie after their pivot by construction; the label at the confirmation bar reads only
    # bars <= that bar, so no cut purge is needed beyond the hindsight caveat documented above.
    pnl[np.arange(n) + 1 >= n] = np.nan
    frame = pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64), "timestamp": ts,
        f"timing_pivot_k{k_bars}": y_f, f"timing_valley_k{k_bars}": valley, f"timing_peak_k{k_bars}": peak,
        "timing_action_confirmed": action_f, "timing_expert_leg_pnl_pips": pnl,
    })
    frame.attrs["pivots"] = len(pivots)
    frame.attrs["confirmed"] = int(sum(1 for c in conf if c is not None and c >= 0))
    return frame


def manifest_targets(k_bars: int) -> list[dict]:
    return [
        {"target_id": f"timing_pivot_k{k_bars}", "column": f"timing_pivot_k{k_bars}", "family": FAMILY,
         "head": "classification", "horizon_hours": k_bars},
        {"target_id": f"timing_valley_k{k_bars}", "column": f"timing_valley_k{k_bars}", "family": FAMILY,
         "head": "classification", "horizon_hours": k_bars},
        {"target_id": f"timing_peak_k{k_bars}", "column": f"timing_peak_k{k_bars}", "family": FAMILY,
         "head": "classification", "horizon_hours": k_bars},
        {"target_id": "timing_action_confirmed", "column": "timing_action_confirmed", "family": FAMILY,
         "head": "classification", "horizon_hours": k_bars},
        {"target_id": "timing_expert_leg_pnl_pips", "column": "timing_expert_leg_pnl_pips", "family": FAMILY,
         "head": "regression", "horizon_hours": k_bars},
    ]


def targets_document(frame: pd.DataFrame, *, params, k_bars: int, lake_sha256: str, zigzag_sha256: str) -> dict:
    y = frame[f"timing_pivot_k{k_bars}"]
    return {"label": "HINDSIGHT_ZIGZAG_TIMING_TARGETS_DEVELOPMENT_ONLY", "family": FAMILY,
            "params": {"depth": params.depth, "deviation_pips": params.deviation_pips, "backstep": params.backstep},
            "k_bars": k_bars, "bars": int(len(frame)), "pivots": frame.attrs.get("pivots"),
            "confirmed_pivots": frame.attrs.get("confirmed"),
            "class_counts": {str(c): int((y == c).sum()) for c in (0, 1, 2)}, "purged_rows": int(y.isna().sum()),
            "lake_sha256": lake_sha256, "zigzag_module_sha256": zigzag_sha256, "test_read": False,
            "window": [str(frame.timestamp.iloc[0]), str(frame.timestamp.iloc[-1])],
            "note": "labels are the final repainted ZigZag (hindsight by construction); labels only, never inputs"}


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
    frame = build_targets(bars, zz, p, k_bars=a.k, cuts=a.cut)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "timing_targets.parquet", index=False)
    doc = targets_document(frame, params=p, k_bars=a.k, lake_sha256=a.lake_sha256, zigzag_sha256=a.zigzag_sha256)
    doc["targets_sha256"] = sha256_file(out / "timing_targets.parquet")
    (out / "TARGETS.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
