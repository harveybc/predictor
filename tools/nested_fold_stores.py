#!/usr/bin/env python3
"""Fold-local consumer stores for the nested walk-forward folds (lane F7, 2026-10-26).

One store per fold, built from the consolidated pool's RAW columns of a fixed feature list (core19). None of those
columns is a learned-producer output, so the only fold-local statistics are the TRANSFORM decisions (``decide`` /
``diagnostics`` of layer 1, re-run on the fold's TRAIN rows) and the hygiene rules below; the scaler is the
consumer's and is fitted on the fold's TRAIN by the recipe, never here.

Hygiene (audit A1-3 / A1-4 / A1-6, decided per fold on its TRAIN rows only):
  STD_ZERO_COLUMN     a column with zero TRAIN variance is refused (never mapped to a constant input)
  dedup               pairs with TRAIN R^2 >= DEDUP_R2 keep the FIRST column in list order; the pair is recorded
  robust spike        a non-negative column with TRAIN kurtosis > SPIKE_KURTOSIS gets LOG1P_SHIFT: log1p(x - min_TRAIN)
                      (monotone, bounded, keeps the weekend/holiday distinction; the shift is a TRAIN constant)

Slot names stay the global pool names (``HIGH@diff`` ...): a decision flip keeps the slot and is recorded, as the
consolidated tool does. Every row of the pool (< 2025-01-01) is written so later blocks can be scored; no statistic
reads a row at or after the fold's TRAIN end (``check`` proves it by perturbing those rows)."""
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

H = pd.Timedelta(hours=1)
DIAG_GUARD = 512          # as fold_local_producers: decision diagnostics skip the pool's longest warm-up
DEDUP_R2 = 0.99
SPIKE_KURTOSIS = 20.0
SCHEMA = "nested_fold_store.v1"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def raw_name(slot: str) -> str:
    return slot.split("@")[0]


def train_mask(ts: pd.DatetimeIndex, window) -> np.ndarray:
    """Bars whose availability (bar start + 1 h) lies strictly inside the fold's TRAIN window."""
    a, b = (utc(x) for x in window)
    return np.asarray((ts >= a) & (ts + H < b))


def dedup_pairs(X: pd.DataFrame, order: list, r2: float = DEDUP_R2) -> tuple[list, list]:
    """Columns to keep (first in ``order`` wins) and the recorded dropped pairs."""
    C = X[order].corr().to_numpy()
    keep, dropped = [], []
    for j, c in enumerate(order):
        dup = next((order[i] for i in range(j) if order[i] in keep and np.isfinite(C[i, j]) and C[i, j] ** 2 >= r2), None)
        if dup is None:
            keep.append(c)
        else:
            dropped.append({"dropped": c, "kept": dup, "train_r2": float(C[order.index(dup), j] ** 2)})
    return keep, dropped


def fold_columns(pool: pd.DataFrame, slots: list, global_dec: dict, train: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Transformed columns for one fold and the per-column record. ``train`` = TRAIN mask over the pool rows."""
    diag = train.copy()
    diag[:DIAG_GUARD] = False
    out, rec = {}, {}
    for slot in slots:
        c = raw_name(slot)
        x = pool[c].to_numpy(float)
        g = global_dec.get(c, "NONE")
        if not np.isfinite(x[train]).any() or float(np.nanstd(x[train])) == 0.0:
            raise SystemExit(f"STD_ZERO_COLUMN:{c}")
        d = diagnostics(x[diag])
        dec, why, y = decide(c, x, d, train=diag)
        v = x if y is None else y
        extra = {}
        if c == "RAW_ELAPSED_HOURS" or (dec == "NONE" and d.get("kurtosis") is not None and d["kurtosis"] > SPIKE_KURTOSIS
                                        and d.get("min") is not None and d["min"] >= 0):
            shift = float(np.nanmin(x[diag]))
            with np.errstate(invalid="ignore"):
                v = np.log1p(x - shift)          # a later row below the TRAIN minimum becomes NaN (counted), never a new recipe
            dec, why, extra = "LOG1P_SHIFT", f"TRAIN kurtosis {d.get('kurtosis', float('nan')):.1f} > {SPIKE_KURTOSIS:g}: log1p(x - {shift:g})", {"shift": shift}
        out[slot] = v
        rec[c] = {"slot": slot, "global": g, "scope": dec, "flip": dec != g and not (dec == "LOG1P_SHIFT" and g == "NONE"),
                  "reason": why, "train_kurtosis": d.get("kurtosis"), **extra,
                  **({"asinhs_scale": d.get("median_pos")} if dec == "ASINHS" else {})}
    return pd.DataFrame(out, index=pool.index), rec


def build_fold(pool: pd.DataFrame, man0: dict, slots: list, fold: str, fit_window, out_dir: Path, decision=None, embargo=None) -> dict:
    ts = pd.DatetimeIndex(pd.to_datetime(pool["timestamp"], utc=True))
    train = train_mask(ts, fit_window)
    X, rec = fold_columns(pool, slots, man0["decisions"], train)
    keep, dropped = dedup_pairs(X[train], list(X.columns))
    X = X[keep]
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "full_pool_eurusd_h1.parquet"
    df = pd.concat([pool[["timestamp"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    df.to_parquet(f, index=False)
    man = {"schema": SCHEMA, "fold": fold, "fit_window": [str(fit_window[0]), str(fit_window[1])],
           "decision": decision, "embargo": embargo, "source_sha256": man0["source_sha256"],
           "parent_pool": {"file": man0.get("file"), "sha256": man0["file_sha256"]},
           "rows": int(len(df)), "train_rows": int(train.sum()), "columns": list(df.columns), "features": keep,
           "feature_list_requested": slots, "decisions": rec,
           "hygiene": {"std_zero_refused": True, "dedup_r2": DEDUP_R2, "dedup_dropped": dropped, "spike_kurtosis": SPIKE_KURTOSIS},
           "learned_producers": [], "scaler": "consumer-side, fitted on this fold's TRAIN by the recipe",
           "test_read": False, "rule": "no statistic reads a row with availability >= TRAIN end; proven by check",
           "file": f.name, "file_sha256": sha(f)}
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def check_fold(pool: pd.DataFrame, man0: dict, out_dir: Path, seed: int = 20261026) -> dict:
    """Witness: perturb every raw row at or after the TRAIN end, rebuild, compare decisions/constants and the TRAIN rows."""
    man = json.loads((out_dir / "MANIFEST.json").read_text())
    built = pd.read_parquet(out_dir / man["file"])
    if sha(out_dir / man["file"]) != man["file_sha256"]:
        raise SystemExit("FOLD_STORE_IDENTITY")
    ts = pd.DatetimeIndex(pd.to_datetime(pool["timestamp"], utc=True))
    train = train_mask(ts, man["fit_window"])
    rng = np.random.default_rng(seed)
    P = pool.copy()
    for slot in man["feature_list_requested"]:
        c = raw_name(slot)
        v = P[c].to_numpy(float).copy()
        v[~train] = v[~train] * (1 + rng.normal(0, 0.5, (~train).sum())) + rng.normal(0, 1, (~train).sum())
        P[c] = v
    X, rec = fold_columns(P, man["feature_list_requested"], man0["decisions"], train)
    keep, dropped = dedup_pairs(X[train], list(X.columns))
    same_dec = all(rec[c]["scope"] == man["decisions"][c]["scope"] and rec[c].get("shift") == man["decisions"][c].get("shift")
                   and rec[c].get("asinhs_scale") == man["decisions"][c].get("asinhs_scale") for c in rec)
    same_keep = keep == man["features"]
    diffs = {}
    for c in man["features"]:
        a, b = built[c].to_numpy(float)[train], X[c].to_numpy(float)[train]
        diffs[c] = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
    ok = same_dec and same_keep and max(diffs.values()) == 0.0
    doc = {"fold": man["fold"], "status": "PASS" if ok else "FAIL", "decisions_identical": same_dec, "kept_identical": same_keep,
           "max_abs_diff_on_train_rows": diffs, "perturbed_rows": int((~train).sum()), "seed": seed}
    (out_dir / "WITNESS.json").write_text(json.dumps(doc, indent=1))
    return doc


def hygiene(X: pd.DataFrame, train: np.ndarray, order: list) -> tuple[list, dict]:
    """Full-list hygiene on the fold's TRAIN rows: std == 0 columns DROPPED (recorded, audit A1-3: never a constant
    input), then R^2 >= DEDUP_R2 pairs keep the first column in ``order``. Returns (kept, record)."""
    std_zero = [c for c in order if not np.isfinite(X[c].to_numpy(float)[train]).any() or float(np.nanstd(X[c].to_numpy(float)[train])) == 0.0]
    rest = [c for c in order if c not in std_zero]
    keep, dropped = dedup_pairs(X.loc[train, rest], rest)
    return keep, {"std_zero_dropped": std_zero, "dedup_r2": DEDUP_R2, "dedup_dropped": dropped, "spike_kurtosis": SPIKE_KURTOSIS}


def elapsed_transform(x: np.ndarray, diag: np.ndarray) -> tuple[np.ndarray, dict]:
    d = diagnostics(x[diag])
    shift = float(np.nanmin(x[diag]))
    with np.errstate(invalid="ignore"):
        v = np.log1p(x - shift)
    return v, {"scope": "LOG1P_SHIFT", "shift": shift, "train_kurtosis": d.get("kurtosis"),
               "reason": f"TRAIN kurtosis {d.get('kurtosis', float('nan')):.1f}: log1p(x - {shift:g}) (declared robust transform)"}


def build_full(pool: pd.DataFrame, man0: dict, columns: list, fold: str, fit_window, scope_parquet: Path, scope_report: Path,
               out_dir: Path, decision=None, embargo=None) -> dict:
    """Fold-local store for a FULL column list (recipe v2's 121): the consolidated pool's slots with every column the
    fold-local producer run changed (SCOPE_<fold>.parquet from tools/fold_local_producers.py: learned families refit on
    the fold's TRAIN, fixed columns re-decided there, act.p_hit cross-fitted forward-only) substituted, then hygiene."""
    ts = pd.DatetimeIndex(pd.to_datetime(pool["timestamp"], utc=True))
    train = train_mask(ts, fit_window)
    diag = train.copy()
    diag[:DIAG_GUARD] = False
    rep = json.loads(scope_report.read_text())
    if rep.get("sha256") != sha(scope_parquet) or rep["scope"]["name"] != fold or list(rep["scope"]["fit_window"]) != [str(fit_window[0]), str(fit_window[1])]:
        raise SystemExit("SCOPE_IDENTITY")
    sc = pd.read_parquet(scope_parquet)
    if not pd.DatetimeIndex(pd.to_datetime(sc["timestamp"], utc=True)).equals(ts):
        raise SystemExit("SCOPE_GRID_MISSING_ROWS")
    changed = [c for c in sc.columns if c != "timestamp"]
    missing = [c for c in columns if c not in pool.columns]
    if missing:
        raise SystemExit(f"COLUMNS_NOT_IN_POOL:{missing}")
    X = pool[columns].copy()
    X.index = pool.index
    subst = {}
    for c in changed:
        if c in X.columns:
            g, v = X[c].to_numpy(float), sc[c].to_numpy(float)
            both = np.isfinite(g) & np.isfinite(v)
            subst[c] = {"max_abs_change": float(np.max(np.abs(g[both] - v[both]))) if both.any() else 0.0,
                        "support_rows_changed": int(np.sum(np.isfinite(g) != np.isfinite(v)))}
            X[c] = v
    extra = {}
    if "RAW_ELAPSED_HOURS" in X.columns:
        v, extra["RAW_ELAPSED_HOURS"] = elapsed_transform(X["RAW_ELAPSED_HOURS"].to_numpy(float), diag)
        X["RAW_ELAPSED_HOURS"] = v
    keep, hyg = hygiene(X, train, columns)
    X = X[keep]
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "full_pool_eurusd_h1.parquet"
    df = pd.concat([pool[["timestamp"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    df.to_parquet(f, index=False)
    learned = sorted(c for c in changed if c.startswith(("kf.", "rg.", "fd.", "act.")))
    man = {"schema": "nested_fold_store_full.v1", "fold": fold, "fit_window": [str(fit_window[0]), str(fit_window[1])],
           "decision": decision, "embargo": embargo, "source_sha256": man0["source_sha256"],
           "parent_pool": {"file": man0.get("file"), "sha256": man0["file_sha256"]},
           "scope": {"file": scope_parquet.name, "sha256": rep["sha256"], "report_sha256": sha(scope_report), "changed_columns": changed,
                     "activity": rep.get("activity"), "learned_decisions_flipped": [c for c, r in rep.get("learned_decisions", {}).items() if r.get("flip")],
                     "fixed_decisions_changed": sorted(rep.get("fixed_decisions_changed", {}))},
           "substituted": subst, "learned_producers_refit": learned,
           "rows": int(len(df)), "train_rows": int(train.sum()), "columns": list(df.columns), "features": keep,
           "columns_requested": columns, "hygiene": hyg, "transforms": extra,
           "scaler": "consumer-side, fitted on this fold's TRAIN by the recipe", "test_read": False,
           "rule": "learned producers and decisions fitted on the fold's TRAIN only (fold_local_producers check + check-full witnesses)",
           "file": f.name, "file_sha256": sha(f)}
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def check_full(pool: pd.DataFrame, man0: dict, out_dir: Path, seed: int = 20261026) -> dict:
    """Witness for the assembled store's FIXED columns and hygiene: perturb every pool row at/after the TRAIN end,
    re-run the hygiene/transform decisions, compare constants, kept set and TRAIN-row values (the learned columns'
    witness is tools/fold_local_producers.py check, recorded beside this one)."""
    man = json.loads((out_dir / "MANIFEST.json").read_text())
    built = pd.read_parquet(out_dir / man["file"])
    if sha(out_dir / man["file"]) != man["file_sha256"]:
        raise SystemExit("FOLD_STORE_IDENTITY")
    ts = pd.DatetimeIndex(pd.to_datetime(pool["timestamp"], utc=True))
    train = train_mask(ts, man["fit_window"])
    diag = train.copy()
    diag[:DIAG_GUARD] = False
    rng = np.random.default_rng(seed)
    B = built.copy()
    cols = man["features"]
    for c in cols:
        v = B[c].to_numpy(float).copy()
        v[~train] = v[~train] * (1 + rng.normal(0, 0.5, (~train).sum())) + rng.normal(0, 1, (~train).sum())
        B[c] = v
    keep, hyg = hygiene(B, train, cols)
    same_keep = keep == cols and hyg["std_zero_dropped"] == [] and hyg["dedup_dropped"] == []
    same_elapsed = True
    if "RAW_ELAPSED_HOURS" in man["transforms"]:
        raw = pool["RAW_ELAPSED_HOURS"].to_numpy(float).copy()
        raw[~train] = raw[~train] + rng.integers(0, 200, (~train).sum())
        _, e = elapsed_transform(raw, diag)
        same_elapsed = e["shift"] == man["transforms"]["RAW_ELAPSED_HOURS"]["shift"]
    diffs = {c: float(np.nanmax(np.abs(built[c].to_numpy(float)[train] - B[c].to_numpy(float)[train]))) for c in cols}
    ok = same_keep and same_elapsed and max(diffs.values()) == 0.0
    doc = {"fold": man["fold"], "status": "PASS" if ok else "FAIL", "kept_identical": same_keep, "elapsed_constant_identical": same_elapsed,
           "max_abs_diff_on_train_rows": max(diffs.values()), "perturbed_rows": int((~train).sum()), "seed": seed,
           "learned_columns_witness": "tools/fold_local_producers.py check (FLP_CHECK.json beside this file)"}
    (out_dir / "WITNESS.json").write_text(json.dumps(doc, indent=1))
    return doc


def load_pool(pool_path: Path):
    man0 = json.loads((pool_path.parent / "MANIFEST.json").read_text())
    if sha(pool_path) != man0["file_sha256"]:
        raise SystemExit("POOL_IDENTITY")
    pool = pd.read_parquet(pool_path)
    ts = pd.to_datetime(pool["timestamp"], utc=True)
    if (ts >= utc("2025-01-01")).any():
        raise SystemExit("TEST_ROWS_IN_POOL")
    return pool, man0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build", "check", "build-full", "check-full"))
    ap.add_argument("--pool", required=True)
    ap.add_argument("--folds", required=True, help="JSON {fold: {producer_fit_window: [a, b], decision: ..., embargo: ...}}")
    ap.add_argument("--features", required=True, help="FEATURE_LIST json with 'features', or a recipe json (tensor.columns[*].store_column)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--scopes-out", default=None, help="build-full: directory with SCOPE_<fold>.parquet/.json from fold_local_producers")
    a = ap.parse_args(argv)
    pool, man0 = load_pool(Path(a.pool))
    folds = json.loads(Path(a.folds).read_text())
    folds = folds.get("folds", folds)
    fl = json.loads(Path(a.features).read_text())
    slots = fl["features"] if "features" in fl else [c["store_column"] for c in fl["tensor"]["columns"]]
    names = [a.fold] if a.fold else list(folds)
    for name in names:
        fd = folds[name]
        out_dir = Path(a.out) / name
        if a.cmd == "build-full":
            so = Path(a.scopes_out)
            man = build_full(pool, man0, slots, name, fd["producer_fit_window"], so / f"SCOPE_{name}.parquet", so / f"SCOPE_{name}.json",
                             out_dir, fd.get("decision"), fd.get("embargo"))
            print(json.dumps({"fold": name, "sha256": man["file_sha256"], "features": len(man["features"]), "learned_refit": man["learned_producers_refit"],
                              "std_zero_dropped": man["hygiene"]["std_zero_dropped"], "dedup_dropped": [d["dropped"] for d in man["hygiene"]["dedup_dropped"]]}))
        elif a.cmd == "check-full":
            print(json.dumps(check_full(pool, man0, out_dir)))
        elif a.cmd == "build":
            man = build_fold(pool, man0, slots, name, fd["producer_fit_window"], out_dir, fd.get("decision"), fd.get("embargo"))
            print(json.dumps({"fold": name, "sha256": man["file_sha256"], "features": len(man["features"]), "dropped": man["hygiene"]["dedup_dropped"],
                              "flips": [c for c, r in man["decisions"].items() if r["flip"]]}))
        else:
            print(json.dumps(check_fold(pool, man0, out_dir)))


if __name__ == "__main__":
    main()
