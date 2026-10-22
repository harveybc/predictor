#!/usr/bin/env python3
"""Machine-made FEATURE_CATALOG for the EURUSD H1 candidate pool: one row per candidate column, generated from the
producers' manifests/diagnostics and the real store files (never typed by hand).

Stores read (all on the admitted UTC bar-start grid, decision at bar start + 1 h, rows >= test_start_closed
refused): the layer-1 pool (tools/layer1_pool_h1.py), the layer-2 pool (tools/layer2_pool_h1.py; the default fit set
and, when present, the fit-end-2023-01-01 set for the RL development arm) and the cross-asset store: r4 when its
directory exists, otherwise the r3 columns already packaged in layer-1 are catalogued with generation_status
REPAIR_IN_PROGRESS (HistData lake missing 25-40 % of bars Feb-Jul 2023; recovery from the independent H1 source in
progress). Agreed families that cannot be produced yet are emitted as PENDING rows with the concrete repair in
params_json.

Per column, from the data: period_start/end (first/last finite bar), monthly finite coverage, warm-up (leading NaN
rows), missing share on TRAIN (2020-01-01 .. 2023-01-01) and on 2023, outlier share on TRAIN = share of finite TRAIN
values beyond 6 robust sigmas (median +- 6 x 1.4826 x MAD; NaN when MAD = 0). From the manifests: parent, family,
component, algorithm, proxy_of, params, unit, fit period and fitted statistics, the preprocessing decision and the
metric that justified it. producer_sha256 is the sha256 of the producer file as it exists at generation time.
``selection_decision`` and ``consumed_by`` are left PENDING for the selection lane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.layer1_pool_h1 import decide, diagnostics  # noqa: E402

COLUMNS = ["feature_id", "store", "column", "source", "parent", "family", "component", "algorithm", "proxy_of", "params_json", "clock", "unit",
           "availability_offset_hours", "period_start", "period_end", "coverage_monthly_json", "warmup_bars", "missing_share_train",
           "missing_share_2023", "outlier_share_train", "fitted_stats_json", "fit_period", "producer_path", "producer_sha256",
           "generation_status", "preprocessing_decision", "preprocessing_metric_json", "selection_decision", "consumed_by"]
CLOCK = "UTC bar start; decision at bar start + 1 h (value uses bars <= t only)"
TRAIN = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))
Y2023 = (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
R3_REPAIR = "HistData lake missing 25-40 % of bars Feb-Jul 2023; recovery from the independent H1 source in progress"
R3_SUPERSEDED = "r4 store exists: these layer-1 values still come from the r3 (HistData-gap) build; the selection lane should consume r4:<column> instead"

# layer-1 column knowledge (source / parent / algorithm / unit by group or prefix); everything else comes from the files
L1_GROUP = {
    "raw_price": ("admitted EURUSD H1 Bid OHLC (v4 snapshot eval_input)", "Bid OHLC", "raw price level", "price (EURUSD)"),
    "inventory": ("v4 snapshot eval_input features_all.parquet (owners' inventory technicals)", "Bid OHLC", "inventory technical indicator (v4 recipe, name = indicator)", "indicator units (see name)"),
    "price_pool": ("v4 snapshot eval_input features_all.parquet (px.* multi-scale price pool)", "Bid OHLC", "multi-scale price feature over k trailing bars (px.<stat>_<k>)", "log return / log range / fraction / hours (by name)"),
    "calendar": ("timestamp (calendar encoding)", "timestamp", "sin/cos clock or session indicator", "dimensionless [-1, 1] or {0, 1}"),
    "fractal": ("feature-eng causal_fractal plugin store (trailing windows w48/w96/w200 of closes)", "BidClose", "causal fractal estimator on a trailing window (R/S Hurst, variogram Hurst, Hurst spectrum width, box-counting dimension)", "dimensionless"),
    "candle": ("admitted Bid OHLC", "Bid OHLC", "candle geometry in pips relative to the close", "pips"),
    "event": ("fix/cut clock rules (tools/timing_event_clock.py; DST per zone)", "timestamp", "hours since / to the Tokyo fix, London fix, NY cut", "hours"),
    "activity": ("activity store P_HIT.parquet (TRAIN-fitted P(barrier hit) model)", "layer-1 inputs listed in the activity store", "logistic P(barrier hit) from the confirmed activity model", "probability [0, 1]"),
    "ssa": ("admitted BidClose", "BidClose", "trailing-window SSA (L=120, K=24, rank 3), last anti-diagonal only", "pips"),
    "cross_asset": ("cross-asset r3 store (feature-eng cross_asset_h1)", "see r3 manifest", "see r3 manifest", "see r3 manifest"),
}
PENDING_ROWS = [
    {"feature_id": "pending:ev.calendar.consensus_surprise", "column": "ev.calendar.*", "family": "event_calendar", "component": "consensus / actual / surprise per release",
     "params_json": {"repair": "FXMacroData history has no consensus; Investing.com export is research-only; candidate: Trading Economics/FXStreet acquisition — owner decision"},
     "source": "economic calendar with point-in-time consensus (not available)"},
    {"feature_id": "pending:wv.usd_strength", "column": "wv.usd.* / wv.eur.*", "family": "strength", "produced_as": "strength", "component": "A4/D1..D4 of fx.usd_strength / fx.eur_strength returns",
     "params_json": {"repair": "needs the cross-asset r4 store: r3 fx.usd_strength.ret_1 / fx.eur_strength.ret_1 have 6.1 % / 5.7 % non-finite rows concentrated in the Feb-Jul 2023 lake gap, so a 256-bar trailing window would be NaN across most of 2023; rebuild on r4 when it exists"},
     "source": "cross-asset r4 (in progress)"},
    {"feature_id": "pending:hb.usd_strength", "column": "hb.usd.* / hb.eur.*", "family": "strength", "produced_as": "strength", "component": "amplitude / phase / instantaneous frequency of the strength returns",
     "params_json": {"repair": "same as pending:wv.usd_strength (128-bar window; waits for r4)"}, "source": "cross-asset r4 (in progress)"},
]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jd(o) -> str:
    return json.dumps(o, default=str, separators=(",", ":"), sort_keys=True)


def load_store(d: Path, parquet: str, manifest: str, diagnostics: str | None):
    man = json.loads((d / manifest).read_text())
    df = pd.read_parquet(d / parquet)
    ts = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"))).tz_convert("UTC")
    df.index = ts
    if (ts >= pd.Timestamp(man.get("test_start_closed", "2025-01-01"), tz="UTC")).any():
        raise SystemExit(f"TEST_ROWS_FORBIDDEN in {d / parquet}")
    diag = pd.read_csv(d / diagnostics).set_index("feature") if diagnostics and (d / diagnostics).exists() else None
    return df, man, diag


def column_stats(s: pd.Series) -> dict:
    x = s.to_numpy(float)
    ts = s.index
    fin = np.isfinite(x)
    first = int(np.argmax(fin)) if fin.any() else len(x)
    tr = (ts >= TRAIN[0]) & (ts < TRAIN[1])
    y23 = (ts >= Y2023[0]) & (ts < Y2023[1])
    xt = x[tr & fin]
    if len(xt):
        med = float(np.median(xt))
        mad = float(np.median(np.abs(xt - med))) * 1.4826
        outl = float(np.mean(np.abs(xt - med) > 6 * mad)) if mad > 0 else float("nan")
    else:
        med, mad, outl = float("nan"), float("nan"), float("nan")
    cov = pd.Series(fin, index=ts).groupby(ts.strftime("%Y-%m")).mean().round(4)
    return {"period_start": str(ts[fin][0]) if fin.any() else "", "period_end": str(ts[fin][-1]) if fin.any() else "",
            "coverage_monthly_json": jd(cov.to_dict()), "warmup_bars": first,
            "missing_share_train": float(np.mean(~fin[tr])) if tr.any() else float("nan"),
            "missing_share_2023": float(np.mean(~fin[y23])) if y23.any() else float("nan"),
            "outlier_share_train": outl, "_robust": {"median": med, "robust_sigma": mad, "n_train": int(len(xt))}}


def diag_fields(diag: pd.DataFrame | None, col: str) -> tuple[str, dict, dict]:
    base = col.split("@")[0]
    if diag is None or base not in diag.index:
        return "NONE", {}, {}
    r = diag.loc[base]
    met = {k: (None if pd.isna(r.get(k)) else float(r.get(k))) for k in ("adf_p", "kpss_p", "skew", "kurtosis")}
    met["reason"] = r.get("reason")
    stats = {k: (None if pd.isna(r.get(k)) else float(r.get(k))) for k in ("mean", "std", "min", "max", "zero_share") if k in r}
    dec = str(r["decision"])
    if "@" in col:
        dec = f"DERIVED_{col.split('@')[1].upper()}_OF_{base}"
    return dec, met, stats


def rows_layer1(d: Path, producer: Path, cross_asset_repair: bool, r3_meta: dict, r4_present: bool = False):
    df, man, diag = load_store(d, "layer1_pool_eurusd_h1.parquet", "MANIFEST.json", "FEATURE_DIAGNOSTICS.csv")
    psha = sha(producer)
    out = []
    for c in df.columns:
        base = c.split("@")[0]
        grp = man["groups"].get(c, man["groups"].get(base, "inventory"))
        src, parent, algo, unit = L1_GROUP.get(grp, L1_GROUP["inventory"])
        params = {"group": grp, "layer1_source_sha256": man.get("inputs", {}), "fit_period_note": None}
        fit_period = "NONE (no fitted parameter)"
        status = "GENERATED"
        if grp == "cross_asset":
            m = r3_meta.get(base, {})
            parent, algo, unit = m.get("family", "cross_asset"), m.get("transform", algo), m.get("unit", unit)
            params |= {"availability_rule": m.get("availability_rule"), "r3_family": m.get("family")}
            if base.startswith("fac."):
                fit_period = "2015-01-01 .. 2019-12-31 (PCA loadings and Kalman q, pre-grid)"
            if cross_asset_repair:
                status, params["repair"] = "REPAIR_IN_PROGRESS", R3_REPAIR
            elif r4_present:
                status, params["repair"] = "SUPERSEDED_BY_R4", R3_SUPERSEDED
        elif grp == "activity":
            fit_period = "TRAIN (activity model; see activity store ACTIVITY.json)"
        elif grp == "ssa":
            params |= {"L": 120, "K": 24, "rank": 3}
        if "@" in c:
            t = c.split("@")[1]
            parent, algo = base, {"diff": "first difference of the base column (layer-1 decision DIFF)", "asinh": "asinh of the base column (layer-1 decision ASINH)"}[t]
        dec, met, stats = diag_fields(diag, c)
        st = column_stats(df[c])
        out.append({"feature_id": f"l1:{c}", "store": "layer1_pool", "column": c, "source": src, "parent": parent, "family": grp,
                    "component": c if "@" not in c else f"{c.split('@')[1]} companion", "algorithm": algo, "proxy_of": None, "params_json": jd(params),
                    "clock": CLOCK, "unit": unit, "availability_offset_hours": 1.0, **{k: v for k, v in st.items() if not k.startswith("_")},
                    "fitted_stats_json": jd({"train_diagnostics": stats, "robust": st["_robust"]}), "fit_period": fit_period,
                    "producer_path": "tools/layer1_pool_h1.py", "producer_sha256": psha, "generation_status": status,
                    "preprocessing_decision": dec, "preprocessing_metric_json": jd(met), "selection_decision": "PENDING", "consumed_by": "PENDING"})
    return out, man


def rows_layer2(d: Path, producer: Path, suffix: str, store_name: str):
    df, man, diag = load_store(d, f"layer2_pool_eurusd_h1{suffix}.parquet", f"MANIFEST{suffix}.json", f"FEATURE_DIAGNOSTICS{suffix}.csv")
    psha = sha(producer)
    libs = {p.name: sha(p) for p in sorted((ROOT / "tools/lib").glob("*.py"))}
    fp = man["fit_period"]
    out = []
    for c in df.columns:
        base = c.split("@")[0]
        cm = man["columns"][c]
        fam = cm["family"]
        fm = man["family_meta"][fam]
        learned = fam in fp["applies_to"]
        params = {k: v for k, v in cm.items() if k not in ("parent", "component", "unit", "algorithm", "proxy_of", "family")}
        params |= {"config": fm.get("config", fm.get("params", {}).get("config")), "lib_sha256": libs, "note": fm.get("note"), "edge_effect": fm.get("edge_effect"),
                   "library": fm.get("library"), "parents": man["parents"]}
        fitted = {}
        if learned:
            p = fm["params"]
            if fam == "kalman":
                fitted = {k: p[k] for k in ("q_level", "q_slope", "r", "negloglik", "converged", "boundary_hits", "boundary_note") if k in p}
            elif fam in ("gmm", "hmm"):
                fitted = {k: p[k] for k in ("state_rv24_mean_raw", "weights", "means", "variances", "transmat", "startprob", "converged", "loglik", "lower_bound", "mu", "sd", "fit_rows") if k in p}
            elif fam == "fracdiff":
                fitted = {"d": p["d"], "window": p["window"], "adf_table": p["adf_table"], "rule": p["rule"]}
        parent, algo = cm["parent"], cm["algorithm"]
        comp = cm["component"]
        if "@" in c:
            t = c.split("@")[1]
            parent, algo, comp = base, {"diff": "first difference of the base column (layer-1 decision rule DIFF)", "asinh": "asinh of the base column (layer-1 decision rule ASINH)"}[t], f"{t} companion"
        dec, met, stats = diag_fields(diag, c)
        st = column_stats(df[c])
        pert = (man.get("future_perturbation") or {}).get(fam)
        out.append({"feature_id": f"{store_name}:{c}", "store": store_name, "column": c, "source": f"admitted EURUSD H1 Bid OHLC (sha256 {man['source_sha256'][:12]}...)",
                    "parent": parent, "family": fam, "component": comp, "algorithm": algo, "proxy_of": None, "params_json": jd(params), "clock": CLOCK, "unit": cm["unit"],
                    "availability_offset_hours": 1.0, **{k: v for k, v in st.items() if not k.startswith("_")},
                    "fitted_stats_json": jd({"fitted": fitted, "train_diagnostics": stats, "robust": st["_robust"], "future_perturbation": pert}),
                    "fit_period": f"{fp['start'][:10]} .. {fp['end_exclusive'][:10]} (exclusive)" if learned else "NONE (no fitted parameter)",
                    "producer_path": "tools/layer2_pool_h1.py", "producer_sha256": psha, "generation_status": "GENERATED",
                    "preprocessing_decision": dec, "preprocessing_metric_json": jd(met), "selection_decision": "PENDING", "consumed_by": "PENDING"})
    return out, man


def rows_cross_asset_r4(d: Path, producer: Path | None):
    man = json.loads((d / "MANIFEST.json").read_text())
    df = pd.read_parquet(d / man["file"])
    ts = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"))).tz_convert("UTC")
    df.index = ts
    if (ts >= pd.Timestamp(man.get("test_start_closed", "2025-01-01"), tz="UTC")).any():
        raise SystemExit("TEST_ROWS_FORBIDDEN in cross-asset r4")
    meta = {f["feature_id"]: f for f in man.get("features", [])}
    psha = sha(producer) if producer and Path(producer).exists() else "UNKNOWN (producer file not reachable at catalog time)"
    out = []
    train = np.asarray((ts >= TRAIN[0]) & (ts < TRAIN[1]))
    for c in df.columns:
        m = meta.get(c, {})
        st = column_stats(df[c])
        x = df[c].to_numpy(float)
        dg = diagnostics(x[train])
        dec, why, _ = decide(c, x, dg)
        met = {k: dg.get(k) for k in ("adf_p", "kpss_p", "skew", "kurtosis")} | {"reason": why, "note": "computed by the catalog on TRAIN with the layer-1 rules; the transformed companion is not materialised in the r4 store"}
        out.append({"feature_id": f"r4:{c}", "store": "cross_asset_r4", "column": c, "source": "cross-asset r4 store (feature-eng cross_asset_h1)", "parent": m.get("family", "cross_asset"),
                    "family": "cross_asset", "component": c, "algorithm": m.get("transform", "see r4 manifest"), "proxy_of": None,
                    "params_json": jd({"availability_rule": m.get("availability_rule"), "revision": man.get("revision"), "source_pins": man.get("source_pins")}), "clock": CLOCK,
                    "unit": m.get("unit", ""), "availability_offset_hours": 1.0, **{k: v for k, v in st.items() if not k.startswith("_")},
                    "fitted_stats_json": jd({"robust": st["_robust"], "train_diagnostics": {k: dg.get(k) for k in ("mean", "std", "min", "max", "zero_share")}, "factor_info": man.get("factor_info") if c.startswith("fac.") else None}),
                    "fit_period": "2015-01-01 .. 2019-12-31 (pre-grid)" if c.startswith("fac.") else "NONE (no fitted parameter)",
                    "producer_path": "feature-eng:app/plugins/cross_asset_h1.py", "producer_sha256": psha, "generation_status": "GENERATED",
                    "preprocessing_decision": dec, "preprocessing_metric_json": jd(met),
                    "selection_decision": "PENDING", "consumed_by": "PENDING"})
    return out, man


def pending_rows(extra: list[dict], produced_families: set[str]) -> list[dict]:
    """Agreed-but-unproduced families; a PENDING row is dropped once the family exists in a store."""
    out = []
    for p in PENDING_ROWS + extra:
        if p.get("produced_as") and p["produced_as"] in produced_families:
            continue
        row = {k: "" for k in COLUMNS}
        row |= {"store": "none", "parent": "", "algorithm": "", "proxy_of": None, "clock": CLOCK, "unit": "", "availability_offset_hours": float("nan"),
                "coverage_monthly_json": jd({}), "warmup_bars": -1, "missing_share_train": float("nan"), "missing_share_2023": float("nan"), "outlier_share_train": float("nan"),
                "fitted_stats_json": jd({}), "fit_period": "", "producer_path": "", "producer_sha256": "", "generation_status": "PENDING",
                "preprocessing_decision": "PENDING", "preprocessing_metric_json": jd({}), "selection_decision": "PENDING", "consumed_by": "PENDING"}
        row |= {k: (jd(v) if k == "params_json" else v) for k, v in p.items() if k != "produced_as"}
        out.append(row)
    return out


def summary_md(cat: pd.DataFrame, stores: dict) -> str:
    L = ["# FEATURE_CATALOG summary (EURUSD H1 candidate pool)", "", f"Rows: {len(cat)}. Stores: " + "; ".join(f"{k} = {v}" for k, v in stores.items()) + ".", "",
         "## Counts per family x generation_status", ""]
    pv = cat.pivot_table(index="family", columns="generation_status", values="feature_id", aggfunc="count", fill_value=0)
    pv["total"] = pv.sum(axis=1)
    L += ["| family | " + " | ".join(pv.columns) + " |", "|---|" + "---|" * len(pv.columns)]
    L += [f"| {f} | " + " | ".join(str(int(v)) for v in r) + " |" for f, r in pv.iterrows()]
    L += ["", "| status | rows |", "|---|---|"] + [f"| {s} | {n} |" for s, n in cat.generation_status.value_counts().items()]
    L += ["", "## Preprocessing decisions (generated rows)", "", "| decision | rows |", "|---|---|"]
    L += [f"| {s} | {n} |" for s, n in cat[cat.generation_status == "GENERATED"].preprocessing_decision.value_counts().items()]
    L += ["", "## PENDING / REPAIR rows", ""]
    for _, r in cat[cat.generation_status != "GENERATED"].drop_duplicates(["family", "generation_status"]).iterrows():
        L.append(f"- `{r['family']}` ({r['generation_status']}): {json.loads(r['params_json']).get('repair', '')}")
    L += ["", "## Fitted parameters (layer-2 learned families)", ""]
    for _, r in cat[cat.store.str.startswith("l2") & cat.fit_period.str.startswith("20")].drop_duplicates(["store", "family"]).iterrows():
        f = json.loads(r["fitted_stats_json"]).get("fitted", {})
        if r["family"] == "fracdiff":
            f = {"d": f.get("d"), "window": f.get("window"), "adf": [(x["d"], round(x["adf_stat"], 3), f"{x['adf_p']:.2e}") for x in f.get("adf_table", [])]}
        elif r["family"] in ("gmm", "hmm"):
            f = {k: f.get(k) for k in ("state_rv24_mean_raw", "weights", "transmat", "converged")}
        L.append(f"- `{r['store']}` / `{r['family']}` (fit {r['fit_period']}): {jd(f)}")
    L += ["", "Outlier share = share of finite TRAIN (2020-2022) values beyond median +- 6 x 1.4826 x MAD (NaN when MAD = 0). "
          "selection_decision and consumed_by are PENDING (selection lane). Rows >= 2025-01-01 (TEST) are never read.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer1", required=True)
    ap.add_argument("--layer2", required=True)
    ap.add_argument("--layer2-suffixes", default=",_fit_end_2023-01-01", help="comma list of layer-2 file suffixes to catalog (missing ones skipped)")
    ap.add_argument("--cross-asset-r4", default="/var/tmp/features-candidates/eurusd_h1_cross_asset_r4")
    ap.add_argument("--cross-asset-r3", default="/var/tmp/features-candidates/eurusd_h1_cross_asset_r3")
    ap.add_argument("--cross-asset-producer", default=None, help="path of feature-eng app/plugins/cross_asset_h1.py for its sha256 (optional)")
    ap.add_argument("--pending-json", default=None, help="extra PENDING rows (list of dicts)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r3_meta = {}
    r3m = Path(a.cross_asset_r3) / "MANIFEST.json"
    if r3m.exists():
        r3_meta = {f["feature_id"]: f for f in json.loads(r3m.read_text()).get("features", [])}
    r4 = Path(a.cross_asset_r4)
    r4_present = (r4 / "MANIFEST.json").exists()
    rows, stores = [], {}
    l1, man1 = rows_layer1(Path(a.layer1), ROOT / "tools/layer1_pool_h1.py", cross_asset_repair=not r4_present, r3_meta=r3_meta, r4_present=r4_present)
    rows += l1
    stores["layer1_pool"] = f"{len(l1)} columns, file sha256 {man1['file_sha256'][:12]}..."
    for suf in [s for s in a.layer2_suffixes.split(",")]:
        d = Path(a.layer2)
        if not (d / f"layer2_pool_eurusd_h1{suf}.parquet").exists():
            continue
        name = "l2" if not suf else "l2" + suf.replace("_fit_end_", "_fit").replace("-", "")
        l2, man2 = rows_layer2(d, ROOT / "tools/layer2_pool_h1.py", suf, name)
        rows += l2
        stores[name] = f"{len(l2)} columns, fit end {man2['fit_period']['end_exclusive'][:10]}, file sha256 {man2['file_sha256'][:12]}..."
    extra = []
    if r4_present:
        r4rows, man4 = rows_cross_asset_r4(r4, Path(a.cross_asset_producer) if a.cross_asset_producer else None)
        rows += r4rows
        stores["cross_asset_r4"] = f"{len(r4rows)} columns"
    else:
        stores["cross_asset_r4"] = "ABSENT (r3 columns catalogued from layer-1 as REPAIR_IN_PROGRESS)"
    if a.pending_json:
        extra = json.loads(Path(a.pending_json).read_text())
    rows += pending_rows(extra, {r["family"] for r in rows})
    cat = pd.DataFrame(rows)[COLUMNS]
    if cat.feature_id.duplicated().any():
        raise SystemExit("DUPLICATE_FEATURE_ID")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out / "FEATURE_CATALOG.csv", index=False)
    recs = cat.to_dict(orient="records")
    for r in recs:
        for k in ("params_json", "coverage_monthly_json", "fitted_stats_json", "preprocessing_metric_json"):
            r[k] = json.loads(r[k]) if r[k] else {}
    (out / "FEATURE_CATALOG.json").write_text(json.dumps({"schema": "feature_catalog_h1.v1", "columns": COLUMNS, "stores": stores, "rows": len(recs),
                                                           "test_read": False, "producer": "tools/feature_catalog_h1.py", "producer_sha256": sha(Path(__file__)),
                                                           "features": recs}, indent=1, default=str))
    (out / "CATALOG_SUMMARY.md").write_text(summary_md(cat, stores))
    print(json.dumps({"rows": len(cat), "status": cat.generation_status.value_counts().to_dict(), "stores": stores}, default=str))


if __name__ == "__main__":
    main()
