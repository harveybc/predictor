"""Lane C2: the causal-ordering (leak) probe for the 83 ETH 4h features.

Template: lane G's RL01 `causal_timestamp_probe` (agent-multi `rl_temporal/lake_binding.py`): perturb the rows at or
after a step, recompute the observation, and require that it does not move; perturb the rows before it and require
that it does. The features of the view are precomputed, so the probe replays their producer -- the verbatim copy of
financial-data `_scripts/workers/stage22_trading_features_worker.py` @ `19fe375a` (sha `7495a0d9...`, named by
FEATURE_DAG.v3) -- on the view's own OPEN/HIGH/LOW/CLOSE/VOLUME, and decides per column:

1. identity        : does the recomputed column reproduce the stored column after a burn-in (the view starts with its
                     warm-up rows already dropped, so recursive EMAs need rows to converge)?  If not, the stored
                     column is `PRODUCER_MISMATCH` and its causality is NOT verified by recomputation.
2. future rows     : with every raw row strictly after the probe step moved by +1000 (and volume x3), does the
                     recomputed feature at rows <= step change?  Required: no.
3. past rows       : with the rows inside the lookback moved, does it change?  Required: yes for a lookback feature.

Statistical flags on every stored column, independent of recomputation: the Spearman correlation of the feature at
t with the next-bar raw log return and with the 6-bar return; a magnitude above 0.2 is SUSPECT (an in-sample 4 h
crypto feature with that much next-bar correlation is a leak until proven otherwise).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_eth_population import bind_population, sha256_file  # noqa: E402

VENDOR = Path(__file__).resolve().parent / "c2_vendor" / "stage22_trading_features_worker.py"
VENDOR_SHA256 = "7495a0d974a7eb086eef0c11927b1c536946746ee3544ad82f6c256d1c6ee050"
BURN_IN = 1500        # rows before which recursive/long-window columns are not compared
SUSPECT_RHO = 0.2


def load_producer():
    actual = sha256_file(VENDOR)
    if actual != VENDOR_SHA256:
        raise RuntimeError(f"VENDORED_PRODUCER_SHA_MISMATCH {actual}")
    spec = importlib.util.spec_from_file_location("stage22_vendor", VENDOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_frame(view: pd.DataFrame, upto: int) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": view["DATE_TIME"].iloc[:upto].to_numpy(),
                         "open": view["OPEN"].iloc[:upto].to_numpy(dtype=float),
                         "high": view["HIGH"].iloc[:upto].to_numpy(dtype=float),
                         "low": view["LOW"].iloc[:upto].to_numpy(dtype=float),
                         "close": view["CLOSE"].iloc[:upto].to_numpy(dtype=float),
                         "volume": view["VOLUME"].iloc[:upto].to_numpy(dtype=float)})


def recompute(producer, raw: pd.DataFrame, features) -> pd.DataFrame:
    tech = producer.compute_technical(raw)
    stat = producer.compute_statistical(raw)
    out = pd.DataFrame(index=raw.index)
    for f in features:
        if f == "statistical__log_return_1":
            out[f] = stat["log_return_1"].to_numpy()
        elif f in tech.columns:
            out[f] = tech[f].to_numpy()
        elif f in stat.columns:
            out[f] = stat[f].to_numpy()
        else:
            out[f] = np.nan
    return out


def identity_check(stored: np.ndarray, recomputed: np.ndarray, name: str, burn_in=BURN_IN) -> dict:
    s, r = stored[burn_in:].astype(np.float64), recomputed[burn_in:].astype(np.float64)
    if name == "obv":  # a cumulative sum: identical up to the unknown start offset; compare first differences
        s, r = np.diff(s), np.diff(r)
    ok = np.isfinite(s) & np.isfinite(r)
    if ok.sum() < 100:
        return {"state": "NOT_COMPARABLE_TOO_FEW_FINITE", "n": int(ok.sum())}
    scale = float(np.std(s[ok])) or 1.0
    diff = np.abs(s[ok] - r[ok])
    rel = float(diff.max() / scale)
    rho = float(np.corrcoef(s[ok], r[ok])[0, 1]) if np.std(r[ok]) > 0 else float("nan")
    # float32 storage of the view: ~1e-7 relative; EMA start-offset decay; allow 1e-3 of the column's own dispersion
    state = "IDENTICAL_WITHIN_TOL" if rel < 1e-3 else "PRODUCER_MISMATCH"
    return {"state": state, "max_abs_diff_over_std": rel, "corr": rho, "n": int(ok.sum()), "nonfinite_stored": int((~np.isfinite(s)).sum())}


def perturbation_probe(producer, view, features, steps, lookback=300) -> dict:
    """RL01 template: future rows must not move the feature at rows <= step; past rows must (for lookback columns)."""
    upto = max(steps) + lookback + 10
    raw = raw_frame(view, upto)
    base = recompute(producer, raw, features)
    verdicts = {f: {"future_rows_influence": False, "past_rows_influence": False, "steps": []} for f in features}
    for step in steps:
        fut = raw.copy()
        for c in ("open", "high", "low", "close"):
            fut.loc[fut.index > step, c] += 1000.0
        fut.loc[fut.index > step, "volume"] *= 3.0
        rec_f = recompute(producer, fut, features)
        past = raw.copy()
        lo = max(0, step - lookback)
        for c in ("open", "high", "low", "close"):
            past.loc[(past.index >= lo) & (past.index <= step), c] += 1000.0
        past.loc[(past.index >= lo) & (past.index <= step), "volume"] *= 3.0
        rec_p = recompute(producer, past, features)
        window = slice(lo, step + 1)
        for f in features:
            b = base[f].to_numpy()[window]
            a_f = rec_f[f].to_numpy()[window]
            a_p = rec_p[f].to_numpy()[window]
            moved_future = not np.allclose(np.nan_to_num(b), np.nan_to_num(a_f), rtol=1e-9, atol=1e-9)
            moved_past = not np.allclose(np.nan_to_num(b), np.nan_to_num(a_p), rtol=1e-9, atol=1e-9)
            verdicts[f]["future_rows_influence"] |= moved_future
            verdicts[f]["past_rows_influence"] |= moved_past
            verdicts[f]["steps"].append({"step": int(step), "future": moved_future, "past": moved_past})
    return verdicts


def statistical_flags(pop, features) -> dict:
    y1, y6 = pop.raw_log_return(1), pop.raw_log_return(6)
    x = pop.feature_matrix_train()
    out = {}
    for j, f in enumerate(features):
        flags = {}
        for name, y in (("rho_next_bar_return", y1), ("rho_6_bar_return", y6)):
            ok = np.isfinite(y) & np.isfinite(x[:, j])
            flags[name] = float(spearmanr(x[ok, j], y[ok])[0]) if ok.sum() > 100 else float("nan")
        flags["suspect"] = bool(abs(flags["rho_next_bar_return"]) > SUSPECT_RHO)
        out[f] = flags
    return out


def run(pop, out_dir, steps=(3000, 6000, 9000, 12000, 13600), features=None) -> dict:
    t0 = time.process_time()
    features = list(features or pop.features)
    producer = load_producer()
    raw = raw_frame(pop.frame, pop.train_end)
    recomputed = recompute(producer, raw, features)
    identity = {f: identity_check(pop.frame[f].to_numpy()[: pop.train_end], recomputed[f].to_numpy(), f) for f in features}
    perturb = perturbation_probe(producer, pop.frame, features, [s for s in steps if s < pop.train_end])
    flags = statistical_flags(pop, features)
    verdict = {}
    for f in features:
        if perturb[f]["future_rows_influence"]:
            v = "FUTURE_ROWS_INFLUENCE_PRODUCER"
        elif identity[f]["state"] != "IDENTICAL_WITHIN_TOL":
            v = "CAUSALITY_NOT_VERIFIED_PRODUCER_MISMATCH"
        else:
            v = "CAUSAL_BY_RECOMPUTATION"
        if flags[f]["suspect"]:
            v += "+SUSPECT_NEXT_BAR_CORRELATION"
        verdict[f] = v
    doc = {"schema": "c2_leak_probe.v1", "label": "DEVELOPMENT", "template": "agent-multi rl_temporal.lake_binding.causal_timestamp_probe (RL01)",
           "producer": {"vendored": str(VENDOR.relative_to(VENDOR.parents[2])), "sha256": VENDOR_SHA256,
                        "source": "financial-data _scripts/workers/stage22_trading_features_worker.py @ 19fe375a"},
           "population": pop.bindings["view"], "split": pop.bindings["split"], "rows_recomputed": [0, pop.train_end],
           "burn_in_rows": BURN_IN, "probe_steps": list(int(s) for s in steps), "identity": identity, "perturbation": perturb,
           "statistical_flags": flags, "verdict": verdict,
           "counts": {k: sum(1 for v in verdict.values() if v.startswith(k)) for k in
                      ("CAUSAL_BY_RECOMPUTATION", "CAUSALITY_NOT_VERIFIED_PRODUCER_MISMATCH", "FUTURE_ROWS_INFLUENCE_PRODUCER")},
           "suspect_count": int(sum(1 for v in flags.values() if v["suspect"])),
           "cpu_seconds": time.process_time() - t0}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leak_probe.json").write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    pop = bind_population(args.view, args.manifest, args.split)
    doc = run(pop, args.out)
    print(json.dumps({"counts": doc["counts"], "suspect": doc["suspect_count"], "cpu_seconds": doc["cpu_seconds"]}))
    for f, v in doc["verdict"].items():
        if not v.startswith("CAUSAL_BY_RECOMPUTATION") or "SUSPECT" in v:
            print(f, v, doc["identity"][f].get("max_abs_diff_over_std"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
