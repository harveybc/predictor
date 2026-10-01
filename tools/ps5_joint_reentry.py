"""PS5, the M01 half of FS16: joint re-entry of a feature pair with REFIT of the temporal model.

Lane B's PS2 screen (feature-eng 1b22c64) marks pairs whose joint association beats each member's
(SYNERGY_PAIR a*b on target T, per inner fold). A screen is not a model, so PS5 refits the modular
temporal forecaster from scratch, with the SAME seeds, for four arms on one inner fold:

    base, base+a, base+b, base+a+b

and the pair RE-ENTERS only if base+a+b is the strict minimum of the four arms' mean validation MAE
(no tolerance; full precision). Only TRAIN rows are read: the fold's encoder-train rows fit the per-feature
scaler and the model (early stop on the purged last 15 % of the fold train), the fold validation rows score
it, and every origin whose label row t+h falls outside its segment is dropped. The paired naive is the zero
return on identical rows. Every number is DEVELOPMENT evidence, not selection on outer validation or test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

PURGE = 60


def _segments(fold, n, window, horizon_bars):
    te = fold["train"][1]
    vs, ve = fold["val"]
    if not (0 <= fold["train"][0] < te <= vs < ve <= n) or vs - te < max(0, window - 1):
        raise ValueError("fold segments must be ordered, purged and inside the data (train < val)")
    train = np.arange(fold["train"][0] + window - 1, te - horizon_bars)
    cut = train[int(0.85 * len(train))]
    enc, enc_val = train[train < cut], train[train >= cut + PURGE]
    val = np.arange(vs + window - 1, ve - horizon_bars)
    return enc, enc_val, val, cut


def _windows(cols, names, origins, rows_fit, window):
    zs = []
    for name in names:
        v = np.asarray(cols[name], dtype="float64")
        fit = v[rows_fit]
        fit = fit[np.isfinite(fit)]
        mu, sd = fit.mean(), fit.std()
        zs.append(((v - mu) / (sd if sd > 1e-12 else 1.0)).astype("float32"))
    z = np.stack(zs, axis=-1)
    return np.stack([z[t - window + 1:t + 1] for t in origins])


def _finite_windows(cols, names, origins, window):
    """Origins whose whole input window is finite in every arm's columns (warm-up rows are dropped)."""
    bad = np.zeros(len(next(iter(cols.values()))), dtype=bool)
    for name in names:
        bad |= ~np.isfinite(np.asarray(cols[name], dtype="float64"))
    csum = np.concatenate([[0], np.cumsum(bad)])
    return origins[(csum[origins + 1] - csum[origins + 1 - window]) == 0]


def _fit(names, cols, target, fold, seed, settings, window, sample_hours, core_params, output_steps,
         output_channels, horizon_bars, all_names=None):
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    n = len(target)
    enc, enc_val, val, cut = _segments(fold, n, window, horizon_bars)
    rows_fit = np.arange(fold["train"][0], cut)
    y = np.asarray(target, dtype="float32")
    union = list(all_names or names)
    keep = lambda o: _finite_windows(cols, union, o[np.isfinite(y[o])], window)
    enc, enc_val, val = keep(enc), keep(enc_val), keep(val)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    cfg = mt.default_config(list(names))
    cfg.update(sample_hours=sample_hours, window=window, branch_steps=window, horizons=[horizon_bars],
               output_steps=output_steps, output_channels=output_channels)
    cfg["core"]["params"] = dict(core_params or {})
    bundle = mt.build_modular(cfg)
    x, vx, sx = (_windows(cols, names, o, rows_fit, window) for o in (enc, enc_val, val))
    r = fit_with_early_stopping(bundle.forecast_model, x, y[enc][:, None, None], vx, y[enc_val][:, None, None],
                                dict(settings, seed=int(seed)))
    pred = np.asarray(bundle.forecast_model(sx, training=False))[:, 0, 0]
    truth = y[val]
    return {"val_mae": float(np.mean(np.abs(pred - truth))), "naive_mae": float(np.mean(np.abs(truth))),
            "updates": int(r["observed_updates"]), "epochs": int(r["epochs_completed"]),
            "stop_reason": r["stop_reason"], "val_rows": int(len(val)),
            "val_rows_sha256": hashlib.sha256(val.astype("int64").tobytes()).hexdigest()}


def decide(arms, pair):
    a, b = pair
    joint = arms[f"base+{a}+{b}"]["mean_val_mae"]
    others = [arms[k]["mean_val_mae"] for k in ("base", f"base+{a}", f"base+{b}")]
    return "RE_ENTERS" if joint < min(others) else "DOES_NOT_RE_ENTER"


def evaluate_pair(cols, target, fold, *, base, pair, seeds, settings, window=24, sample_hours=4,
                  core_params=None, output_steps=6, output_channels=8, horizon_bars=1, fit_fn=None):
    fit_fn = fit_fn or _fit
    a, b = pair
    variants = {"base": list(base), f"base+{a}": [*base, a], f"base+{b}": [*base, b],
                f"base+{a}+{b}": [*base, a, b]}
    arms, naive = {}, None
    for label, names in variants.items():
        per = {}
        for seed in seeds:
            per[str(seed)] = fit_fn(names, {k: cols[k] for k in variants[f"base+{a}+{b}"]}, target, fold, seed,
                                    settings, window, sample_hours, core_params, output_steps, output_channels,
                                    horizon_bars, variants[f"base+{a}+{b}"])
        shas = {r["val_rows_sha256"] for r in per.values()}
        if len(shas) != 1:
            raise ValueError("arms were scored on different rows")
        naive = naive or {"kind": "zero_return", "mae": next(iter(per.values()))["naive_mae"],
                          "rows_sha256": shas.pop()}
        arms[label] = {"features": names, "per_seed": per,
                       "mean_val_mae": float(np.mean([r["val_mae"] for r in per.values()]))}
    if len({arm["per_seed"][str(seeds[0])]["val_rows_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("arms were scored on different rows")
    return {"schema": "ps5.joint_reentry.v1", "label": "DEVELOPMENT", "fold": fold["name"], "pair": list(pair),
            "base": list(base), "seeds": list(seeds), "arms": arms, "naive": naive,
            "decision": decide(arms, pair), "rule": "strict minimum of base+a+b over base, base+a, base+b"}


# ------------------------------------------------------------------------------- ETH runner
MANIFEST_FILE_SHA = "fdff0c85fc376cd6930cede4981a64b076022bd701c6a045339689e3ab892d4c"
WORKLIST_SHA = "397b67d6f40eaf324c0aa720291f67560c55cf0b216bc53cb9c4886b1ae66175"
DATA_SHA = "1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f"
FOLDS = {"inner_1": {"name": "inner_1", "train": (0, 7474), "val": (7534, 9589)},
         "inner_2": {"name": "inner_2", "train": (0, 9529), "val": (9589, 11644)},
         "inner_3": {"name": "inner_3", "train": (0, 11584), "val": (11644, 13699)}}
RUN_SETTINGS = {"max_epochs": 30, "patience": 5, "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
                "loss": "mae", "min_delta": 0.0, "max_updates": 5000}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(manifest_path, worklist_path):
    """Every (fold, pair, target) SYNERGY_PAIR of lane B's work list; base = the fold's PRIORITY and
    REPRESENTATIVE features minus the pair, in manifest order; every name must be in the frozen manifest."""
    import csv
    import re
    if _sha(manifest_path) != MANIFEST_FILE_SHA or _sha(worklist_path) != WORKLIST_SHA:
        raise SystemExit("manifest or work list bytes are not the frozen ones")
    manifest = json.loads(Path(manifest_path).read_text())
    order = manifest["features"]
    rows = list(csv.DictReader(open(worklist_path, newline="")))
    plan, seen = [], set()
    for r in rows:
        for m in re.finditer(r"SYNERGY_PAIR (\S+?)\*(\S+?) on (Y_l|Y_s)@(\d+)h", r["reasons"]):
            a, b = sorted(m.group(1, 2))
            key = (r["fold"], a, b, int(m.group(4)))
            if key in seen:
                continue
            seen.add(key)
            core = {x["feature"] for x in rows if x["fold"] == r["fold"]
                    and x["tier"] in ("PRIORITY", "REPRESENTATIVE")} - {a, b}
            if not {a, b} | core <= set(order):
                raise SystemExit(f"feature outside the frozen manifest in {key}")
            plan.append({"fold": r["fold"], "pair": [a, b], "target_hours": int(m.group(4)),
                         "target": f"{m.group(3)}@{m.group(4)}h", "base": [f for f in order if f in core]})
    return sorted(plan, key=lambda p: (p["fold"], p["pair"], p["target_hours"]))


def _eth(data_path, train_rows=13699):
    import pandas as pd
    frame = pd.read_csv(data_path)
    stamps = pd.to_datetime(frame["DATE_TIME"])
    train = frame[stamps < pd.Timestamp("2024-01-01")].reset_index(drop=True)
    if len(train) != train_rows:
        raise SystemExit(f"TRAIN has {len(train)} rows, expected {train_rows}")
    ts = ((pd.to_datetime(train["DATE_TIME"]) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy("int64")
    return train, ts


def _target(train, ts, hours):
    """Elapsed-time label: log CLOSE(t+h)/CLOSE(t), NaN when no bar sits exactly h hours later."""
    price = train["CLOSE"].to_numpy("float64")
    j = np.searchsorted(ts, ts + hours * 3600)
    ok = j < len(ts)
    ok[ok] = ts[j[ok]] == ts[ok] + hours * 3600
    y = np.full(len(ts), np.nan)
    y[ok] = np.log(price[j[ok]] / price[ok])
    return y


def _child(args):
    out = _fit(*args)
    out["child_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return out


def spawned_fit(*args):
    """One fresh process per fit (PS3R-MEM-01: Keras state accumulates across fits in one process)."""
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(_child, (args,))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--worklist", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", type=int, default=None, help="run only this plan index (the measured pilot)")
    p.add_argument("--max-epochs", type=int, default=RUN_SETTINGS["max_epochs"])
    a = p.parse_args()
    if _sha(a.data) != DATA_SHA:
        raise SystemExit("data bytes are not the manifest's resource")
    plan = build_plan(a.manifest, a.worklist)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "PLAN.json").write_text(json.dumps(plan, indent=1) + "\n")
    train, ts = _eth(a.data)
    cols = {c: train[c].to_numpy("float64") for c in train.columns if c != "DATE_TIME"}
    settings = dict(RUN_SETTINGS, max_epochs=a.max_epochs)
    for i, item in enumerate(plan):
        if a.only is not None and i != a.only:
            continue
        path = out / f"ps5_{i:03d}_{item['fold']}_{item['pair'][0]}__{item['pair'][1]}_{item['target_hours']}h.json"
        if path.is_file():
            continue
        started = time.monotonic()
        result = evaluate_pair(cols, _target(train, ts, item["target_hours"]), FOLDS[item["fold"]],
                               base=item["base"], pair=tuple(item["pair"]), seeds=(2021, 2022), settings=settings,
                               window=24, sample_hours=4, horizon_bars=item["target_hours"] // 4,
                               fit_fn=spawned_fit)
        result.update(target=item["target"], seconds=time.monotonic() - started, settings=settings,
                      peak_child_rss_bytes=max(r["child_peak_rss_bytes"] for arm in result["arms"].values()
                                               for r in arm["per_seed"].values()),
                      identity={"data_sha256": DATA_SHA, "manifest_file_sha256": MANIFEST_FILE_SHA,
                                "worklist_sha256": WORKLIST_SHA, "split": "M07 SPLIT 116a5b64, TRAIN [0,13699)",
                                "device_class": "cpu"})
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
        print(json.dumps({"i": i, "fold": item["fold"], "pair": item["pair"], "target": item["target"],
                          "decision": result["decision"], "seconds": round(result["seconds"], 1),
                          "peak_child_rss_gib": round(result["peak_child_rss_bytes"] / 2**30, 3)}), flush=True)


if __name__ == "__main__":
    main()
