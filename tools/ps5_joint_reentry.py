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
        mu, sd = v[rows_fit].mean(), v[rows_fit].std()
        zs.append(((v - mu) / (sd if sd > 1e-12 else 1.0)).astype("float32"))
    z = np.stack(zs, axis=-1)
    return np.stack([z[t - window + 1:t + 1] for t in origins])


def _fit(names, cols, target, fold, seed, settings, window, sample_hours, core_params, output_steps,
         output_channels, horizon_bars):
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    n = len(target)
    enc, enc_val, val, cut = _segments(fold, n, window, horizon_bars)
    rows_fit = np.arange(fold["train"][0], cut)
    y = np.asarray(target, dtype="float32")
    keep = lambda o: o[np.isfinite(y[o])]
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
                  core_params=None, output_steps=6, output_channels=8, horizon_bars=1):
    a, b = pair
    variants = {"base": list(base), f"base+{a}": [*base, a], f"base+{b}": [*base, b],
                f"base+{a}+{b}": [*base, a, b]}
    arms, naive = {}, None
    for label, names in variants.items():
        per = {}
        for seed in seeds:
            per[str(seed)] = _fit(names, cols, target, fold, seed, settings, window, sample_hours, core_params,
                                  output_steps, output_channels, horizon_bars)
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
def _eth(data_path, max_rows=13699):
    import pandas as pd
    frame = pd.read_csv(data_path)
    stamps = pd.to_datetime(frame["DATE_TIME"])
    train = frame[stamps < pd.Timestamp("2024-01-01")].reset_index(drop=True)
    if len(train) != max_rows:
        raise SystemExit(f"TRAIN has {len(train)} rows, expected {max_rows}")
    ts = ((pd.to_datetime(train["DATE_TIME"]) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy("int64")
    return train, ts


def _target(train, ts, hours):
    price = train["CLOSE"].to_numpy("float64")
    j = np.searchsorted(ts, ts + hours * 3600)
    ok = j < len(ts)
    ok[ok] = ts[j[ok]] == ts[ok] + hours * 3600
    y = np.full(len(ts), np.nan)
    y[ok] = np.log(price[j[ok]] / price[ok])
    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--plan", required=True, help="JSON list of {fold, base, pair, target_hours}")
    p.add_argument("--out", required=True)
    p.add_argument("--only", type=int, default=None, help="run only this plan index (pilot)")
    a = p.parse_args()
    folds = {"inner_1": {"name": "inner_1", "train": (0, 7474), "val": (7534, 9589)},
             "inner_2": {"name": "inner_2", "train": (0, 9529), "val": (9589, 11644)},
             "inner_3": {"name": "inner_3", "train": (0, 11584), "val": (11644, 13699)}}
    plan = json.loads(Path(a.plan).read_text())
    train, ts = _eth(a.data)
    cols = {c: train[c].to_numpy("float64") for c in train.columns if c not in ("DATE_TIME",)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"max_epochs": 30, "patience": 5, "batch_size": 64, "learning_rate": 1e-3,
                "weight_decay": 1e-4, "loss": "mae", "min_delta": 0.0, "max_updates": 5000}
    for i, item in enumerate(plan):
        if a.only is not None and i != a.only:
            continue
        path = out / f"ps5_{i:03d}_{item['fold']}_{'_'.join(item['pair'])}.json"
        if path.is_file():
            continue
        started = time.monotonic()
        target = _target(train, ts, item["target_hours"])
        result = evaluate_pair(cols, target, folds[item["fold"]], base=item["base"], pair=tuple(item["pair"]),
                               seeds=(2021, 2022), settings=settings, window=24, sample_hours=4,
                               horizon_bars=item["target_hours"] // 4)
        result.update(target=f"Y@{item['target_hours']}h", seconds=time.monotonic() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest())
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)
        print(json.dumps({"i": i, "decision": result["decision"], "seconds": round(result["seconds"], 1)}),
              flush=True)


if __name__ == "__main__":
    main()
