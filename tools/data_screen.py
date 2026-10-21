#!/usr/bin/env python3
"""D-screen: lag x stratum screen of candidate features against the split labels (D5), TRAIN only, then a
confirmation on INNER_A (2023 S1) of the survivors. Literature-guided (docs/timing_selection/LITERATURE_DATA_PHASE_
2026_10_21.md): lags in observed bars {0,1,2,4,8,24,48,120}; a small PRE-REGISTERED strata grid; a blocked
circular-shift null; Benjamini-Hochberg over the WHOLE grid (labels x strata x features x lags).

  prepare  (selection interpreter, CPU)  admitted bars + closures, split labels, pool, event clock, strata -> input
  screen   (torch, GPU when present)     statistics, null, BH, INNER_A confirmation -> results + summary

Labels (symmetric barriers m = 2 sigma around the executable fill open(t+1), horizon 72 bars, forced flat):
  hit   1 if a barrier is touched before the timeout, 0 if timeout                (volatility / activity part)
  up    1 if the upper barrier is touched first, 0 if the lower, among hits only  (direction part; same-bar ties dropped)
Rows: business-eligible (shared mask), labellable, and the label's support ends inside its period (purge at cuts).

Strata (17, fixed before any read): all; volatility tercile (px.rv_24, TRAIN terciles) x session of the decision
hour UTC (asia 0-7, europe 7-13, us 13-21, late 21-24) = 12; 2 h before / after the London 16:00 fix; 2 h before /
after the Tokyo 09:55 fix.

Statistic per (label, stratum, feature, lag): difference of the mean normal score of the feature (rank-Gauss inside
the stratum, NaN -> 0) between label 1 and label 0 rows (a standardized mean difference). Null: the label series
circularly shifted against the features by K offsets in [168, n - 168] TRAIN rows that break the daily and weekly
phase (keeps both serial structures; see phase_breaking_offsets),
restricted to the same stratum; z against the null mean / sd; two-sided normal p (the null's sd is estimated from the
shifts; declared approximation that allows p-values below 1/K); BH at q over every test. A survivor is CONFIRMED when
the same statistic on INNER_A has the same sign and its own shift-null z >= 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LAGS = (0, 1, 2, 4, 8, 24, 48, 120)
HORIZON, M_SIGMA = 72, 2.0
SESSIONS = {"asia": (0, 7), "europe": (7, 13), "us": (13, 21), "late": (21, 24)}
PERIODS = {"TRAIN": ("2020-01-01", "2023-01-01"), "INNER_A": ("2023-01-01", "2023-07-01")}


def _utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ----------------------------------------------------------------------------- prepare
def prepare(a):
    from tools import timing_selection as ts
    from tools import timing_pivot_targets as tp
    from tools import timing_profit_targets as tpf
    from tools import timing_event_clock as ec
    cfg = json.loads(Path(a.config).read_text())
    run = Path(a.run)
    bars, closures, ident = ts.load_bars(cfg)
    zz = tp.load_zigzag_module(cfg["zigzag"]["module"], cfg["zigzag"]["sha256"])
    sigma = np.asarray(zz.causal_sigma(bars), float)
    flat = tpf.flat_bar_positions(bars.index, closures, 12.0)
    outc, res = tpf.barrier_outcomes(bars.OPEN, bars.HIGH, bars.LOW, bars.CLOSE, sigma, flat, horizon_bars=HORIZON, m=M_SIGMA)
    pool = pd.read_parquet(run / "data" / "pool_all.parquet")
    if len(pool) != len(bars):
        raise SystemExit("POOL_GRID_MISMATCH")
    pool.index = bars.index
    if a.extra_store:
        man = json.loads((Path(a.extra_store).parent / "MANIFEST.json").read_text())
        if tp.sha256_file(a.extra_store) != man["file_sha256"] or man["source_sha256"] != cfg["source"]["sha256"]:
            raise SystemExit("EXTRA_STORE_IDENTITY")
        if not all(c["cells_changed_at_or_before"] == 0 for c in man["future_perturbation"]):
            raise SystemExit("EXTRA_STORE_PERTURBATION")
        ex = pd.read_parquet(a.extra_store)
        ex.index = pd.DatetimeIndex(pd.to_datetime(ex.pop("timestamp"))).tz_convert("UTC")
        ex = ex.reindex(bars.index)
        pool = pool.drop(columns=[c for c in ex.columns if c in pool.columns])     # the newer store replaces
        pool = pd.concat([pool, ex], axis=1)
    el = pd.read_parquet(run / "ELIGIBILITY.parquet")
    if len(el) != len(bars):
        raise SystemExit("ELIGIBILITY_GRID_MISMATCH")
    business = (el["business_reason"].to_numpy() == "ELIGIBLE")
    evc = ec.event_clock(bars.index)
    period = np.full(len(bars), "", object)
    end_pos = {}
    for name, (s, e) in PERIODS.items():
        m = (bars.index >= _utc(s)) & (bars.index < _utc(e))
        period[m] = name
        end_pos[name] = int(np.searchsorted(bars.index.asi8, _utc(e).value))
    support_ok = np.array([(res[i] >= 0) and (period[i] == "" or res[i] < end_pos[period[i]]) for i in range(len(bars))])
    ok = business & (outc >= 0) & support_ok
    hit = np.where(ok, (outc >= 1).astype(float), np.nan)
    up = np.where(ok & np.isin(outc, (1, 2)), (outc == 1).astype(float), np.nan)
    hour = (bars.index + pd.Timedelta(hours=1)).hour
    sess = np.full(len(bars), "", object)
    for k, (h0, h1) in SESSIONS.items():
        sess[(hour >= h0) & (hour < h1)] = k
    df = pd.concat([pd.DataFrame({"timestamp": bars.index, "period": period, "label_hit": hit, "label_up": up,
                                  "session": sess}, index=bars.index), pool, evc], axis=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(out / "screen_input.parquet", index=False)
    meta = {"source": ident, "labels": {"hit": "barrier touched before timeout", "up": "upper first | hit"},
            "horizon_bars": HORIZON, "m_sigma": M_SIGMA, "features": list(pool.columns) + list(evc.columns),
            "event_clock": ec.event_meta(evc.columns), "counts": {
                p: {"rows": int((period == p).sum()), "hit_rows": int(np.isfinite(hit[period == p]).sum()),
                    "hit_rate": float(np.nanmean(hit[period == p])), "up_rows": int(np.isfinite(up[period == p]).sum()),
                    "up_rate": float(np.nanmean(up[period == p]))} for p in PERIODS},
            "outcome_counts_all": {int(k): int(v) for k, v in zip(*np.unique(outc, return_counts=True))}}
    (out / "SCREEN_INPUT.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta["counts"]))


# ----------------------------------------------------------------------------- screen
def strata_masks(df: pd.DataFrame, train: np.ndarray) -> dict:
    rv = df["px.rv_24"].to_numpy(float)
    q1, q2 = np.nanquantile(rv[train], [1 / 3, 2 / 3])
    terc = np.where(np.isnan(rv), -1, np.where(rv <= q1, 0, np.where(rv <= q2, 1, 2)))
    m = {"all": np.ones(len(df), bool)}
    for t, tn in enumerate(("lowvol", "midvol", "highvol")):
        for s in SESSIONS:
            m[f"{tn}_{s}"] = (terc == t) & (df["session"].to_numpy() == s)
    lt, ls = df["ev.london_fix.hours_to"].to_numpy(float), df["ev.london_fix.hours_since"].to_numpy(float)
    tt, tsn = df["ev.tokyo_fix.hours_to"].to_numpy(float), df["ev.tokyo_fix.hours_since"].to_numpy(float)
    m["pre_london_fix_2h"], m["post_london_fix_2h"] = lt <= 2, ls <= 2
    m["pre_tokyo_fix_2h"], m["post_tokyo_fix_2h"] = tt <= 2, tsn <= 2
    return m, {"rv24_train_terciles": [float(q1), float(q2)]}


def normal_scores(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata
    out = np.zeros_like(x, dtype=np.float32)
    for j in range(x.shape[1]):
        v = x[:, j]
        ok = np.isfinite(v)
        if ok.sum() > 10 and np.nanstd(v[ok]) > 0:
            r = rankdata(v[ok])
            out[ok, j] = norm.ppf((r - 0.5) / ok.sum())
    return out


def phase_breaking_offsets(rng, n: int, K: int) -> np.ndarray:
    """Circular-shift offsets in [168, n - 168] rows that do NOT realign the daily or the weekly phase: offset mod 24
    in [4, 20] and offset mod 120 (5 trading days of observed bars) in [12, 108]. Offsets near a multiple of a day or a
    week would keep clock-driven association in the null and blind the test to calendar / event-clock features
    (correction declared 2026-10-21 after the first run, before any re-read)."""
    out = []
    while len(out) < K:
        o = int(rng.integers(168, n - 168))
        if 4 <= o % 24 <= 20 and 12 <= o % 120 <= 108:
            out.append(o)
    return np.asarray(out)


def screen(a):
    import torch
    from scipy.stats import norm
    inp = Path(a.inp)
    df = pd.read_parquet(inp / "screen_input.parquet")
    meta = json.loads((inp / "SCREEN_INPUT.json").read_text())
    feats = meta["features"]
    period = df["period"].to_numpy()
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = df[feats].to_numpy(float)
    lagged = {L: np.vstack([np.full((L, X.shape[1]), np.nan), X[:-L]]) if L else X for L in LAGS}
    rows = []
    t0 = time.time()
    for pname, K in (("TRAIN", a.k), ("INNER_A", a.k_confirm)):
        sel = np.flatnonzero(period == pname)
        n = len(sel)
        offs = phase_breaking_offsets(rng, n, K)
        masks, terc = strata_masks(df.iloc[sel].reset_index(drop=True), np.ones(n, bool)) if pname == "TRAIN" else (None, None)
        if pname == "TRAIN":
            train_masks, train_terc = masks, terc
        else:
            # INNER_A strata use the TRAIN terciles (no 2023 information in the definitions)
            sub = df.iloc[sel].reset_index(drop=True)
            rv = sub["px.rv_24"].to_numpy(float)
            q1, q2 = train_terc["rv24_train_terciles"]
            tercv = np.where(np.isnan(rv), -1, np.where(rv <= q1, 0, np.where(rv <= q2, 1, 2)))
            masks = {"all": np.ones(n, bool)}
            for t, tn in enumerate(("lowvol", "midvol", "highvol")):
                for s in SESSIONS:
                    masks[f"{tn}_{s}"] = (tercv == t) & (sub["session"].to_numpy() == s)
            masks["pre_london_fix_2h"] = sub["ev.london_fix.hours_to"].to_numpy(float) <= 2
            masks["post_london_fix_2h"] = sub["ev.london_fix.hours_since"].to_numpy(float) <= 2
            masks["pre_tokyo_fix_2h"] = sub["ev.tokyo_fix.hours_to"].to_numpy(float) <= 2
            masks["post_tokyo_fix_2h"] = sub["ev.tokyo_fix.hours_since"].to_numpy(float) <= 2
        Ylab = {}
        for lab in ("label_hit", "label_up"):
            y = df[lab].to_numpy(float)[sel]
            idx = (np.arange(n)[:, None] + np.r_[0, offs][None, :]) % n
            Ylab[lab] = (y, y[idx])                                                 # n x (K+1), column 0 = observed
        for sname, sm in masks.items():
            rws = np.flatnonzero(sm)
            if len(rws) < 300:
                continue
            Xs = np.concatenate([normal_scores(lagged[L][sel][rws]) for L in LAGS], axis=1)       # nS x (F*L)
            XT = torch.as_tensor(Xs.T.copy(), device=dev)
            cover = np.concatenate([np.isfinite(lagged[L][sel][rws]).mean(0) for L in LAGS])
            for lab, (y, Y) in Ylab.items():
                Ys = Y[rws]
                A1 = torch.as_tensor((Ys == 1).astype(np.float32), device=dev)
                A0 = torch.as_tensor((Ys == 0).astype(np.float32), device=dev)
                n1, n0 = A1.sum(0).clamp(min=1), A0.sum(0).clamp(min=1)
                stat = ((XT @ A1) / n1 - (XT @ A0) / n0).cpu().numpy()                 # (F*L) x (K+1)
                obs, null = stat[:, 0], stat[:, 1:]
                mu, sd = null.mean(1), null.std(1)
                z = np.where(sd > 0, (obs - mu) / np.where(sd > 0, sd, 1), 0.0)
                p = 2 * norm.sf(np.abs(z))
                for j in range(len(obs)):
                    L, f = LAGS[j // len(feats)], feats[j % len(feats)]
                    rows.append((pname, lab[6:], sname, f, L, float(obs[j]), float(z[j]), float(p[j]), float(cover[j]),
                                 int(len(rws)), int(np.isfinite(y[rws]).sum())))
            print(json.dumps({"period": pname, "stratum": sname, "s": round(time.time() - t0, 1)}), flush=True)
    R = pd.DataFrame(rows, columns=["period", "label", "stratum", "feature", "lag", "smd", "z", "p", "coverage", "rows", "labelled"])
    tr = R[(R.period == "TRAIN") & (R.coverage >= 0.5)].copy()
    tr = tr.sort_values("p").reset_index(drop=True)
    m = len(tr)
    tr["bh_threshold"] = a.q * (np.arange(1, m + 1)) / m
    passed = tr["p"] <= tr["bh_threshold"]
    kmax = int(np.flatnonzero(passed)[-1]) + 1 if passed.any() else 0
    tr["pass_bh"] = np.arange(m) < kmax
    ia = R[R.period == "INNER_A"].set_index(["label", "stratum", "feature", "lag"])
    conf = []
    for _, r in tr[tr.pass_bh].iterrows():
        key = (r.label, r.stratum, r.feature, r.lag)
        if key in ia.index:
            v = ia.loc[key]
            conf.append(bool(np.sign(v.smd) == np.sign(r.smd) and abs(v.z) >= 2 and np.sign(v.z) == np.sign(r.z)))
        else:
            conf.append(False)
    tr["confirmed_inner_a"] = pd.Series([None] * len(tr), dtype=object)
    tr.loc[tr.pass_bh, "confirmed_inner_a"] = pd.Series(conf, index=tr.index[tr.pass_bh], dtype=object)
    R.to_parquet(out / "SCREEN_ALL.parquet", index=False)
    tr.to_parquet(out / "SCREEN_TRAIN_BH.parquet", index=False)
    fam = lambda f: f.split(".")[0] if "." in f else "inventory"  # noqa: E731
    surv = tr[tr.pass_bh].copy()
    surv["family"] = surv.feature.map(fam)
    summary = {"tests_train": int(m), "q": a.q, "pass_bh": int(tr.pass_bh.sum()),
               "confirmed_inner_a": int(surv.get("confirmed_inner_a", pd.Series(dtype=bool)).fillna(False).sum()),
               "by_label": surv.groupby("label").size().to_dict(),
               "by_family_label": surv.groupby(["label", "family"]).size().rename("n").reset_index().to_dict("records"),
               "strata": list(train_masks), "lags": list(LAGS), "k_null": a.k, "k_null_confirm": a.k_confirm, "device": dev,
               "rv24_train_terciles": train_terc["rv24_train_terciles"], "input_counts": meta["counts"]}
    (out / "SCREEN_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str))
    L = ["# D-screen (lag x stratum, split labels), TRAIN 2020-2022 with INNER_A confirmation", "",
         f"Tests (coverage >= 0.5): {m}; BH q = {a.q}: **{int(tr.pass_bh.sum())} pass**; confirmed on INNER_A: "
         f"**{summary['confirmed_inner_a']}**.", "", "Input counts: " + json.dumps(meta["counts"]), "",
         "## Survivors (top 60 by p)", "", "| label | stratum | feature | lag | SMD | z | p | confirmed 2023 S1 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in surv.head(60).iterrows():
        L.append(f"| {r.label} | {r.stratum} | {r.feature} | {r.lag} | {r.smd:+.3f} | {r.z:+.2f} | {r.p:.1e} | {r.get('confirmed_inner_a')} |")
    (out / "SCREEN_SUMMARY.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: summary[k] for k in ("tests_train", "pass_bh", "confirmed_inner_a", "by_label")}))


# ----------------------------------------------------------------------------- stability + 2023 confirmation
YEARS = {"Y2020": ("2020-01-01", "2021-01-01"), "Y2021": ("2021-01-01", "2022-01-01"), "Y2022": ("2022-01-01", "2023-01-01")}


def _stat_z_block(x_block, y_block, mask, offs):
    """SMD of the stratum's normal scores between y == 1 and y == 0 rows, z against the block's label circularly
    shifted over the WHOLE block (phase-breaking offsets) and then restricted to the same stratum rows."""
    rows = np.flatnonzero(mask)
    if len(rows) < 50:
        return np.nan, 0.0, 0
    xs = normal_scores(x_block[rows][:, None])[:, 0]
    n = len(y_block)
    idx = (rows[:, None] + np.r_[0, offs][None, :]) % n
    Y = y_block[idx]
    A1, A0 = (Y == 1).astype(float), (Y == 0).astype(float)
    st = (xs @ A1) / np.maximum(A1.sum(0), 1) - (xs @ A0) / np.maximum(A0.sum(0), 1)
    sd = st[1:].std()
    nlab = int(np.isfinite(y_block[rows]).sum())
    return float(st[0]), (float((st[0] - st[1:].mean()) / sd) if sd > 0 else 0.0), nlab


def stability(a):
    """BH survivors of ``screen`` must (1) keep the sign with |z| >= ``min_year_z`` in EACH TRAIN year (block
    stability, Meinshausen & Buehlmann style with years as blocks), then (2) confirm on the whole of 2023 (joint
    development) in the same stratum, or in the coarser parent stratum when the stratum has fewer than ``min_rows``
    labelled rows: same sign and z >= 2. Each year / 2023 gets its own phase-breaking shift null."""
    inp, out = Path(a.inp), Path(a.out)
    df = pd.read_parquet(inp / "screen_input.parquet")
    tr = pd.read_parquet(Path(a.screen) / "SCREEN_TRAIN_BH.parquet")
    surv = tr[tr.pass_bh].copy()
    ts = pd.DatetimeIndex(df["timestamp"])
    rng = np.random.default_rng(a.seed)
    train_sel = np.flatnonzero((ts >= _utc("2020-01-01")) & (ts < _utc("2023-01-01")))
    _, terc = strata_masks(df.iloc[train_sel].reset_index(drop=True), np.ones(len(train_sel), bool))
    q1, q2 = terc["rv24_train_terciles"]

    def masks_for(sub):
        rv = sub["px.rv_24"].to_numpy(float)
        tv = np.where(np.isnan(rv), -1, np.where(rv <= q1, 0, np.where(rv <= q2, 1, 2)))
        m = {"all": np.ones(len(sub), bool)}
        for t, tn in enumerate(("lowvol", "midvol", "highvol")):
            m[tn] = tv == t
            for s_ in SESSIONS:
                m[f"{tn}_{s_}"] = (tv == t) & (sub["session"].to_numpy() == s_)
        for s_ in SESSIONS:
            m[s_] = sub["session"].to_numpy() == s_
        m["pre_london_fix_2h"] = sub["ev.london_fix.hours_to"].to_numpy(float) <= 2
        m["post_london_fix_2h"] = sub["ev.london_fix.hours_since"].to_numpy(float) <= 2
        m["pre_tokyo_fix_2h"] = sub["ev.tokyo_fix.hours_to"].to_numpy(float) <= 2
        m["post_tokyo_fix_2h"] = sub["ev.tokyo_fix.hours_since"].to_numpy(float) <= 2
        return m

    def parent(sname):
        if sname.count("_") == 1 and sname.split("_")[0].endswith("vol"):
            return sname.split("_")[0]                      # e.g. lowvol_us -> lowvol
        return "all"

    blocks = dict(YEARS)
    blocks["Y2023"] = ("2023-01-01", "2024-01-01")
    cache = {}
    for b, (s0, s1) in blocks.items():
        sel = np.flatnonzero((ts >= _utc(s0)) & (ts < _utc(s1)))
        sub = df.iloc[sel].reset_index(drop=True)
        cache[b] = (sel, sub, masks_for(sub), phase_breaking_offsets(rng, len(sel), a.k))
    X = {}
    recs = []
    for _, r in surv.iterrows():
        rec = {k: r[k] for k in ("label", "stratum", "feature", "lag", "smd", "z", "p")}
        col = df[r.feature].to_numpy(float)
        xl = np.r_[np.full(int(r.lag), np.nan), col[:-int(r.lag)]] if r.lag else col
        y_all = df[f"label_{r.label}"].to_numpy(float)
        ok_years = True
        for b in YEARS:
            sel, sub, m, offs = cache[b]
            smd, z, nl = _stat_z_block(xl[sel], y_all[sel], m[r.stratum], offs)
            rec[f"{b}_smd"], rec[f"{b}_z"], rec[f"{b}_n"] = smd, z, nl
            ok_years &= bool(nl >= 100 and np.isfinite(smd) and np.sign(smd) == np.sign(r.smd) and abs(z) >= a.min_year_z
                             and np.sign(z) == np.sign(r.z))
        rec["stable_all_years"] = ok_years
        sel, sub, m, offs = cache["Y2023"]
        st = r.stratum
        if np.isfinite(y_all[sel][m[st]]).sum() < a.min_rows:
            st = parent(st)
        smd, z, nl = _stat_z_block(xl[sel], y_all[sel], m[st], offs)
        rec.update(confirm_stratum=st, Y2023_smd=smd, Y2023_z=z, Y2023_n=nl,
                   confirmed_2023=bool(ok_years and nl >= a.min_rows and np.isfinite(smd) and np.sign(smd) == np.sign(r.smd)
                                       and np.sign(z) == np.sign(r.z) and abs(z) >= 2))
        recs.append(rec)
    R = pd.DataFrame(recs)
    out.mkdir(parents=True, exist_ok=True)
    R.to_parquet(out / "STABILITY.parquet", index=False)
    summ = {"survivors": int(len(R)), "stable_all_years": int(R.stable_all_years.sum()) if len(R) else 0,
            "confirmed_2023": int(R.confirmed_2023.sum()) if len(R) else 0,
            "by_label": R.groupby("label")[["stable_all_years", "confirmed_2023"]].sum().astype(int).to_dict() if len(R) else {},
            "rule": stability.__doc__, "min_year_z": a.min_year_z, "min_rows": a.min_rows, "k": a.k}
    (out / "STABILITY_SUMMARY.json").write_text(json.dumps(summ, indent=1, default=str))
    keep = R[R.stable_all_years].sort_values("p") if len(R) else R
    L = ["# D-screen stability (years as blocks) and 2023 confirmation", "",
         f"Survivors {summ['survivors']}; stable in 2020, 2021 and 2022: **{summ['stable_all_years']}**; confirmed on 2023: "
         f"**{summ['confirmed_2023']}**.", "",
         "| label | stratum | feature | lag | SMD | z2020 | z2021 | z2022 | 2023 stratum | z2023 | confirmed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in keep.iterrows():
        L.append(f"| {r.label} | {r.stratum} | {r.feature} | {r.lag} | {r.smd:+.3f} | {r.Y2020_z:+.2f} | {r.Y2021_z:+.2f} | {r.Y2022_z:+.2f} | "
                 f"{r.confirm_stratum} | {r.Y2023_z:+.2f} | {r.confirmed_2023} |")
    (out / "STABILITY_SUMMARY.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: summ[k] for k in ("survivors", "stable_all_years", "confirmed_2023", "by_label")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="verb", required=True)
    p1 = sp.add_parser("prepare")
    p1.add_argument("--config", required=True)
    p1.add_argument("--run", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--extra-store", default="", help="cross-asset store parquet (MANIFEST.json beside it); replaces same-name columns")
    p2 = sp.add_parser("screen")
    p2.add_argument("--inp", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--k", type=int, default=1000)
    p2.add_argument("--k-confirm", type=int, default=500)
    p2.add_argument("--q", type=float, default=0.1)
    p2.add_argument("--seed", type=int, default=20261021)
    p2.add_argument("--cpu", action="store_true")
    p3 = sp.add_parser("stability")
    p3.add_argument("--inp", required=True)
    p3.add_argument("--screen", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--k", type=int, default=1000)
    p3.add_argument("--min-year-z", type=float, default=1.0)
    p3.add_argument("--min-rows", type=int, default=150)
    p3.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    {"prepare": prepare, "screen": screen, "stability": stability}[a.verb](a)


if __name__ == "__main__":
    main()
