#!/usr/bin/env python3
"""Fold-local learned producers and data-dependent decisions (Musashi's final joint audit A1, 2026-10-23).

Defect: the layer-2 learned families (Kalman, GMM, HMM, fractional differencing) were fitted on 2020-2023, the
preprocessing decisions on 2020-2022 and the activity model on 2020-2022 labels, while the feature selection scored
purged folds INSIDE 2020-2023 and the consumers' inner block is 2023 S2. A held-out row could therefore shape the
very feature values it was scored on.

Correction: one SCOPE per evaluation partition. A scope names its allowed fit bars (a window minus excluded spans);
inside a scope every learned producer is FITTED on the allowed bars only and APPLIED causally to every bar (history
is kept, held-out rows are never fitted on); every preprocessing decision (ADF/KPSS/skew rule, the ASINHS median) is
re-decided on the allowed bars of the diagnostics window; the activity model is cross-fitted (tools/activity_producer).
For each scope the tool writes the columns whose values DIFFER from the bound global stores (pool names, decided
companions), with the decisions, parameters and the per-column changes, and a perturbation self-check: perturbing
the bars of an excluded span and REBUILDING (fitting included) must leave every fitted parameter and every value
before the span unchanged (the frozen-apply property of tools/layer2_pool_h1.perturbation_check is a different test).

  build   --scopes SCOPES.json --out DIR      one parquet + one report per scope, SCOPES_REPORT.json
  check   --scopes SCOPES.json --scope NAME   the rebuild-with-fit perturbation witness on real data
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
from tools import layer2_pool_h1 as L2  # noqa: E402
from tools.layer1_pool_h1 import decide, diagnostics  # noqa: E402

H = pd.Timedelta(hours=1)
# input lookback (bars) of each learned family's FIT rows: a fit row must have its whole input window on allowed bars,
# otherwise held-out prices reach the fit through the window (the rebuild-with-fit tests catch exactly this)
LOOKBACK = {"kalman": 1, "gmm": 25, "hmm": 25, "fracdiff": 256}
DIAG_GUARD = 512        # decision diagnostics: rows whose longest pool window (<= 480 bars) is on allowed bars


def shrink(allowed: np.ndarray, lookback: int) -> np.ndarray:
    """rows t with allowed[t - lookback + 1 .. t] all True."""
    if lookback <= 1:
        return allowed.copy()
    bad = np.concatenate([[0], np.cumsum(~allowed)])
    t = np.arange(len(allowed))
    lo = np.maximum(t - lookback + 1, 0)
    return allowed & ((bad[t + 1] - bad[lo]) == 0) & (t - lookback + 1 >= 0)


def exact_parents(raw: pd.DataFrame) -> pd.DataFrame:
    """layer-2 parents with rv_24 summed window by window: pandas' running rolling sum carries floating-point error
    beyond its 24-bar window, so a held-out bar would still reach later fit rows at the 1e-18 level."""
    P = L2.parents(raw)
    r2 = P["logret_1"].to_numpy(float) ** 2
    rv = np.full(len(r2), np.nan)
    if len(r2) >= 24:
        w = np.lib.stride_tricks.sliding_window_view(r2, 24)
        s = w.sum(axis=1)
        s[~np.isfinite(w).all(axis=1)] = np.nan
        rv[23:] = np.sqrt(s)
    P["rv_24"] = rv
    return P


def build_scoped(P: pd.DataFrame, allowed: np.ndarray):
    """Learned families fitted on their own window-complete allowed rows, applied to every bar."""
    frames, meta, params, masks = [], {}, {}, {}
    for fam in L2.LEARNED:
        m = shrink(allowed, LOOKBACK[fam])
        X, me, pa = L2.build(P, (fam,), m)
        frames.append(X)
        meta.update(me)
        params.update(pa)
        masks[fam] = int(m.sum())
    return pd.concat(frames, axis=1), meta, params, masks


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def allowed_mask(ts: pd.DatetimeIndex, scope: dict) -> np.ndarray:
    """Bars of the scope's fit window (availability = bar start + 1 h strictly before the window end) minus every
    excluded span (bar starts inside [a, b], inclusive)."""
    a, b = (utc(x) for x in scope["fit_window"])
    m = np.asarray((ts >= a) & (ts + H < b))
    for lo, hi in scope.get("exclude", []):
        m &= ~np.asarray((ts >= utc(lo)) & (ts <= utc(hi)))
    return m


def load_parents(admitted: Path, admitted_sha: str, cross_asset: Path, cross_sha: str, test_start="2025-01-01"):
    if sha(admitted) != admitted_sha:
        raise SystemExit("ADMITTED_SHA256_MISMATCH")
    if sha(cross_asset) != cross_sha:
        raise SystemExit("CROSS_ASSET_SHA256_MISMATCH")
    t0 = utc(test_start)
    raw = pd.read_parquet(admitted, columns=["timestamp", "BidOpen", "BidHigh", "BidLow", "BidClose"])
    raw["timestamp"] = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"])).tz_convert("UTC")
    raw = raw[raw["timestamp"] < t0].reset_index(drop=True)
    ca = pd.read_parquet(cross_asset, columns=["timestamp", *L2.STRENGTH_INPUTS.values()])
    ca.index = pd.DatetimeIndex(pd.to_datetime(ca.pop("timestamp"))).tz_convert("UTC")
    ca = ca.reindex(raw["timestamp"])
    for c in L2.STRENGTH_INPUTS.values():
        raw[c] = ca[c].to_numpy(float)
    return raw, exact_parents(raw)


def store_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"), utc=True))
    return df


def decided_name(c: str, dec: str) -> str:
    return c if dec == "NONE" else f"{c}@{dec.lower()}"


def redecide(raw_cols: dict, global_dec: dict, diag: np.ndarray) -> tuple[dict, dict]:
    """{pool name: values} for every column whose decision or frozen constant changes on this scope's diagnostics
    rows, and the per-column decision record. A decision flip keeps the GLOBAL pool name (the slot) and records the
    scope decision; values are the scope recipe applied to the raw column."""
    out, rec = {}, {}
    for c, x in raw_cols.items():
        g = global_dec[c]
        d = diagnostics(x[diag])
        dec, why, y = decide(c, x, d, train=diag)
        name = decided_name(c, g)
        changed = dec != g or (dec == "ASINHS")
        if changed:
            out[name] = x if y is None else y
        rec[c] = {"global": g, "scope": dec, "slot": name, "flip": dec != g,
                  **({"asinhs_scale": d.get("transform_scale")} if dec == "ASINHS" else {}), "reason": why}
    return out, rec


class Builder:
    def __init__(self, a):
        self.a = a
        self.raw, self.P = load_parents(Path(a.admitted), a.admitted_sha256, Path(a.cross_asset), a.cross_asset_sha256)
        self.ts = self.P.index
        self.l1 = store_frame(Path(a.layer1)).reindex(self.ts)
        self.l2 = store_frame(Path(a.layer2)).reindex(self.ts)
        self.dec1 = {c: (v if isinstance(v, str) else v["decision"]) for c, v in json.loads((Path(a.layer1).parent / "MANIFEST.json").read_text())["decisions"].items()}
        self.dec2 = {c: (v if isinstance(v, str) else v["decision"]) for c, v in json.loads((Path(a.layer2).parent / "MANIFEST.json").read_text())["decisions"].items()}
        self.pool = store_frame(Path(a.pool)).reindex(self.ts)
        self.act = None

    def activity(self):
        if self.act is None:
            from tools import activity_producer as AP
            bars, X, hit, res, feats = AP.load_inputs(Path(self.a.snapshot), Path(self.a.admitted).parent, Path(self.a.short_horizon),
                                                      Path(self.a.pool), Path(self.a.activity_decision))
            if not pd.DatetimeIndex(bars.index).tz_convert("UTC").equals(self.ts):
                raise SystemExit("ACTIVITY_GRID_MISMATCH")
            self.act = (X, hit, res, feats)
        return self.act

    def scope(self, sc: dict, P=None, raw_override=None):
        """(columns {pool name: values}, report) for one scope."""
        P = self.P if P is None else P
        allowed = allowed_mask(self.ts, sc)
        dw = sc.get("diag_window", ["2020-01-01", "2023-01-01"])
        in_dw = np.asarray((self.ts >= utc(dw[0])) & (self.ts < utc(dw[1])))     # layer-1 / layer-2 TRAIN rule
        diag = in_dw & (allowed if not sc.get("exclude") else shrink(allowed, DIAG_GUARD) | (allowed & ~shrink(np.ones(len(allowed), bool), DIAG_GUARD)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, meta, params, fmasks = build_scoped(P, allowed)
        cols, rep = {}, {"allowed_bars": int(allowed.sum()), "diag_bars": int(diag.sum()), "fit_bars_by_family": fmasks,
                         "params": jsonable({f: params.get(f) for f in L2.LEARNED})}
        # learned layer-2 columns: decide on the scope's diagnostics rows, slot = the global pool name
        learned_raw = {c: X[c].to_numpy(float) for c in X.columns}
        lr_dec = {}
        for c, x in learned_raw.items():
            gname = c
            g = self.dec2.get(c)
            if g is None and c.startswith("fd."):                         # fracdiff column named by its d
                gcand = [k for k in self.dec2 if k.startswith("fd.")]
                g, gname = self.dec2[gcand[0]], gcand[0]
            d = diagnostics(x[diag])
            dec, why, y = decide(c, x, d, train=diag)
            slot = decided_name(gname, g)
            cols[slot] = x if y is None else y
            lr_dec[c] = {"global": g, "scope": dec, "slot": slot, "flip": dec != g, "reason": why}
        rep["learned_decisions"] = lr_dec
        # fixed layer-1 / layer-2 columns: only a changed decision or a changed ASINHS constant changes values
        raw1 = {c: self.l1[c].to_numpy(float) for c in self.dec1 if c in self.l1.columns}
        raw2 = {c: self.l2[c].to_numpy(float) for c in self.dec2 if c in self.l2.columns and not c.startswith(("kf.", "rg.", "fd."))}
        if raw_override:
            raw1.update({k: v for k, v in raw_override.items() if k in raw1})
        c1, r1 = redecide(raw1, self.dec1, diag)
        c2, r2 = redecide(raw2, self.dec2, diag)
        cols.update(c1)
        cols.update(c2)
        rep["fixed_decisions_changed"] = {c: r for c, r in {**r1, **r2}.items() if r["flip"] or r["scope"] == "ASINHS"}
        # activity: cross-fitted on the allowed bars
        if not self.a.skip_activity:
            from tools import activity_producer as AP
            Xa, hit, res, feats = self.activity()
            p, arep = AP.scope_predictions(Xa, hit, res, allowed, lookback=AP.max_lag(feats) + 1,
                                           mode=getattr(self.a, "activity_mode", "purged_kfold"))
            cols["act.p_hit"] = p
            rep["activity"] = arep | {"features": feats}
        # keep only the slots of the bound pool whose values differ from it
        changed = {}
        per = {}
        for name, v in cols.items():
            if name not in self.pool.columns:
                per[name] = {"status": "NOT_IN_POOL"}
                continue
            g = self.pool[name].to_numpy(float)
            fin_same = bool(np.array_equal(np.isfinite(v), np.isfinite(g)))
            same = bool(np.array_equal(v, g, equal_nan=True))
            both = np.isfinite(v) & np.isfinite(g)
            per[name] = {"identical": same, "finite_mask_identical": fin_same,
                         "max_abs_diff": float(np.max(np.abs(v[both] - g[both]))) if both.any() else None,
                         "cells_changed": int(np.sum(~np.isclose(v, g, rtol=0, atol=0, equal_nan=True)))}
            if not same:
                changed[name] = np.asarray(v, float)
        rep["columns"] = per
        rep["changed"] = sorted(changed)
        return changed, rep


def cmd_build(a):
    spec = json.loads(Path(a.scopes).read_text())
    b = Builder(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for sc in spec["scopes"]:
        if a.only and sc["name"] not in a.only.split(","):
            continue
        cols, rep = b.scope(sc)
        f = out / f"SCOPE_{sc['name']}.parquet"
        pd.DataFrame({"timestamp": b.ts, **cols}).to_parquet(f, index=False)
        rep |= {"scope": sc, "file": f.name, "sha256": sha(f)}
        (out / f"SCOPE_{sc['name']}.json").write_text(json.dumps(jsonable(rep), indent=1))
        reports[sc["name"]] = {"file": f.name, "sha256": rep["sha256"], "changed": rep["changed"]}
        print(json.dumps({"scope": sc["name"], "allowed": rep["allowed_bars"], "changed": len(rep["changed"])}), flush=True)
    (out / "SCOPES_REPORT.json").write_text(json.dumps({"scopes_file": Path(a.scopes).name, "scopes_sha256": sha(a.scopes),
                                                        "inputs": {k: sha(getattr(a, k)) for k in ("layer1", "layer2", "pool", "admitted", "cross_asset")},
                                                        "scopes": reports}, indent=1))


def cmd_check(a):
    """Rebuild-with-fit witness: perturb the raw bars of the scope's first excluded span (or of the bars after its
    fit window when it has none), REBUILD the learned families with fitting, and compare parameters and every value
    strictly before the perturbed span with the unperturbed rebuild."""
    spec = json.loads(Path(a.scopes).read_text())
    sc = next(s for s in spec["scopes"] if s["name"] == a.scope)
    b = Builder(a)
    ts = b.ts
    if sc.get("exclude"):
        lo, hi = (utc(x) for x in sc["exclude"][0])
        span = np.asarray((ts >= lo) & (ts <= hi))
    else:                                   # the held-out block after the fit window, up to the selection end
        span = ~allowed_mask(ts, sc) & np.asarray((ts >= utc(sc["fit_window"][0])) & (ts < utc("2024-01-01")))
        lo, hi = ts[span][0], ts[span][-1]
    rng = np.random.default_rng(11)
    raw2 = b.raw.copy()
    for k in ("BidOpen", "BidHigh", "BidLow", "BidClose"):
        v = raw2[k].to_numpy(float).copy()
        v[span] *= np.exp(rng.normal(0, 0.003, span.sum()))
        raw2[k] = v
    raw2["BidHigh"] = np.maximum.reduce([raw2[k].to_numpy(float) for k in ("BidOpen", "BidHigh", "BidLow", "BidClose")])
    raw2["BidLow"] = np.minimum.reduce([raw2[k].to_numpy(float) for k in ("BidOpen", "BidHigh", "BidLow", "BidClose")])
    P2 = exact_parents(raw2)
    allowed = allowed_mask(ts, sc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X0, _, p0, _ = build_scoped(b.P, allowed)
        X1, _, p1, _ = build_scoped(P2, allowed)
    first = int(np.argmax(span))
    same_params = json.dumps(jsonable(p0), sort_keys=True) == json.dumps(jsonable(p1), sort_keys=True)
    before = slice(0, first)
    same_before = {c: bool(np.array_equal(X0[c].to_numpy()[before], X1[c].to_numpy()[before], equal_nan=True)) for c in X0.columns}
    changed_inside = {c: int(np.sum(~np.isclose(X0[c].to_numpy()[span], X1[c].to_numpy()[span], equal_nan=True))) for c in X0.columns}
    # the contaminated baseline: the GLOBAL fit window (2020-2023, spans included) on the same perturbation
    g = {"fit_window": ["2020-01-01", "2024-01-01"], "exclude": []}
    ga = allowed_mask(ts, g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G0, _, _ = L2.build(b.P, L2.LEARNED, ga)
        G1, _, _ = L2.build(P2, L2.LEARNED, ga)
    glob_changed = int(sum(np.sum(~np.isclose(G0[c].to_numpy()[before], G1[c].to_numpy()[before], equal_nan=True)) for c in G0.columns))
    rep = {"scope": sc["name"], "perturbed_bars": int(span.sum()), "span": [str(lo), str(hi)],
           "scope_fit": {"parameters_identical": bool(same_params), "values_before_span_identical": same_before,
                         "all_before_identical": bool(all(same_before.values())), "cells_changed_inside_span": changed_inside},
           "global_fit_baseline": {"cells_changed_before_span": glob_changed,
                                   "note": "the contaminated fit (2020-2023 including the span) moves earlier values"},
           "passed": bool(same_params and all(same_before.values()) and glob_changed > 0 and sum(changed_inside.values()) > 0)}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("scope", "perturbed_bars", "passed")} | {"global_cells_changed_before": glob_changed}))
    return 0 if rep["passed"] else 2


def add_inputs(p):
    for k in ("admitted", "admitted-sha256", "cross-asset", "cross-asset-sha256", "layer1", "layer2", "pool", "snapshot",
              "short-horizon", "activity-decision"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--scopes", required=True)
    p.add_argument("--skip-activity", action="store_true")
    p.add_argument("--activity-mode", default="purged_kfold", choices=("purged_kfold", "expanding"),
                   help="act.p_hit cross-fit: purged k-fold (A1) or FORWARD-only expanding blocks (A1-12)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); add_inputs(b); b.add_argument("--out", required=True); b.add_argument("--only", default="")
    c = sub.add_parser("check"); add_inputs(c); c.add_argument("--scope", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.exit({"build": cmd_build, "check": cmd_check}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
