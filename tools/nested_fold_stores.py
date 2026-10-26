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
    ap.add_argument("cmd", choices=("build", "check"))
    ap.add_argument("--pool", required=True)
    ap.add_argument("--folds", required=True, help="JSON {fold: {producer_fit_window: [a, b], decision: ..., embargo: ...}}")
    ap.add_argument("--features", required=True, help="FEATURE_LIST json with 'features'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default=None)
    a = ap.parse_args(argv)
    pool, man0 = load_pool(Path(a.pool))
    folds = json.loads(Path(a.folds).read_text())
    folds = folds.get("folds", folds)
    slots = json.loads(Path(a.features).read_text())["features"]
    names = [a.fold] if a.fold else list(folds)
    for name in names:
        fd = folds[name]
        out_dir = Path(a.out) / name
        if a.cmd == "build":
            man = build_fold(pool, man0, slots, name, fd["producer_fit_window"], out_dir, fd.get("decision"), fd.get("embargo"))
            print(json.dumps({"fold": name, "sha256": man["file_sha256"], "features": len(man["features"]), "dropped": man["hygiene"]["dedup_dropped"],
                              "flips": [c for c, r in man["decisions"].items() if r["flip"]]}))
        else:
            print(json.dumps(check_fold(pool, man0, out_dir)))


if __name__ == "__main__":
    main()
