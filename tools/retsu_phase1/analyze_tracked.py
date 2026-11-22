#!/usr/bin/env python
"""Panel of the nine champions whose Flipout posterior is a Keras weight.

Does not call evaluate. Epoch MAE stays in epochs.jsonl.
One posterior-mean forward per split, plus the distance of each
posterior tensor from a freshly built init with the same seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_head as th

PHASE = Path("/var/tmp/retsu-phase1")
OUT = PHASE / "analysis_tracked.json"
TABLE = PHASE / "seasonal_predictions_best.npz"


def corr(a: np.ndarray, b: np.ndarray):
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def column_stats(pred, target, origin, horizons):
    rows = []
    for i, horizon in enumerate(horizons):
        p = pred[:, i]
        y = target[:, i]
        o = origin
        ss_tot = float(np.sum(np.square(y - np.mean(y))))
        ss_res = float(np.sum(np.square(y - p)))
        rows.append({
            "horizon": int(horizon),
            "rows": int(p.size),
            "mae": float(np.mean(np.abs(p - y))),
            "naive_mae": float(np.mean(np.abs(o - y))),
            "r2": None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot),
            "pred_std": float(np.std(p)),
            "y_std": float(np.std(y)),
            "corr_pred_y": corr(p, y),
            "corr_pred_origin": corr(p, o),
        })
    return rows


def predict_price(model, use_mean, z, mean, std, scale):
    use_mean.assign(True)
    pred = model.predict(z / scale, batch_size=th.BATCH, verbose=0)
    if not isinstance(pred, (list, tuple)):
        pred = [pred]
    cols = [np.asarray(item).reshape(-1) for item in pred]
    out = np.column_stack(cols)
    return out * std.reshape(1, -1) + mean.reshape(1, -1)


def one(job_dir: Path, data: dict) -> dict:
    hold = json.loads((job_dir / "holdout.json").read_text())
    horizons = [int(h) for h in hold["train_mae_per_horizon"]]
    task = hold["task"]
    mean = np.asarray(hold["target_mean"], dtype=np.float64)
    std = np.asarray(hold["target_std"], dtype=np.float64)
    ok = data["ok_short"] if task == "short" else data["ok_long"]
    y_all = data["y_short"] if task == "short" else data["y_long"]
    pool = np.flatnonzero(data["pool"] & ok)[: int(hold["max_steps"])]
    z_train = data["z"][pool]
    fan_in = float(z_train.shape[1])
    z_scale = float(hold.get("embedding_scale") or np.sqrt(np.mean(np.square(z_train.astype(np.float64))) * fan_in))
    panel_train = pool[-256:] if pool.size >= 256 else pool
    monitor = np.flatnonzero(data["monitor"] & ok)
    holdout = np.flatnonzero(data["holdout"] & ok)
    tf.keras.backend.clear_session()
    kl = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight")
    use_mean = tf.Variable(True, trainable=False, dtype=tf.bool, name="use_posterior_mean")
    model = th.build(horizons, kl, use_mean)
    model(tf.zeros((1, 208), dtype=tf.float32))
    init = {variable.name: variable.numpy().copy() for variable in model.trainable_weights}
    model.load_weights(job_dir / "champion.weights.h5")
    weight_rows = []
    for variable in model.trainable_weights:
        if "flipout" not in variable.name and not variable.name.startswith("bias"):
            continue
        value = variable.numpy()
        weight_rows.append({
            "name": variable.name,
            "max_abs_from_init": float(np.max(np.abs(value - init[variable.name]))),
            "rms": float(np.sqrt(np.mean(np.square(value)))),
            "finite": bool(np.isfinite(value).all()),
        })
    flipout = []
    for layer in model.layers:
        if "flip" not in layer.name:
            continue
        dist = layer.kernel_posterior.distribution
        loc = np.asarray(dist.loc.numpy(), dtype=np.float64).reshape(-1)
        scale = np.asarray(dist.scale.numpy(), dtype=np.float64).reshape(-1)
        flipout.append({
            "layer": layer.name,
            "loc_type": type(dist.loc).__name__,
            "scale_type": type(dist.scale).__name__,
            "loc_std": float(np.std(loc)),
            "loc_l2": float(np.linalg.norm(loc)),
            "scale_mean": float(np.mean(scale)),
            "scale_min": float(np.min(scale)),
            "scale_max": float(np.max(scale)),
            "finite": bool(np.isfinite(loc).all() and np.isfinite(scale).all()),
        })
    origin = data["seasonal_origin"]
    splits = {}
    preds = {}
    for name, idx in (("train_panel_256", panel_train), ("monitor", monitor), ("holdout_2024", holdout)):
        pred = predict_price(model, use_mean, data["z"][idx], mean, std, z_scale)
        preds[name] = (idx, pred)
        splits[name] = column_stats(pred, y_all[idx], origin[idx], horizons)
    logged = float(hold["holdout_2024_mae"])
    reproduced = float(np.mean([row["mae"] for row in splits["holdout_2024"]]))
    return {
        "job": job_dir.name,
        "task": task,
        "flipout_tracked": any("flipout" in row["name"] and "loc" in row["name"] for row in weight_rows),
        "weights": weight_rows,
        "flipout": flipout,
        "logged_holdout_2024_mae": logged,
        "reproduced_holdout_2024_mae": reproduced,
        "holdout_abs_gap": abs(reproduced - logged),
        "logged_best_val_mae": hold.get("best_val_mae"),
        "naive_mae_monitor": hold.get("naive_mae_monitor"),
        "splits": splits,
        "_idx": {name: preds[name][0] for name in preds},
        "_pred": {name: preds[name][1] for name in preds},
        "_horizons": horizons,
        "_y": y_all,
    }


def main() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    raw = np.load(PHASE / "dataset.npz")
    data = {key: raw[key] for key in raw.files}
    packed = []
    for path in sorted((PHASE / "jobs").iterdir()):
        if (path / "champion.weights.h5").exists() and (path / "holdout.json").exists():
            packed.append(one(path, data))
            item = packed[-1]
            print(
                item["job"],
                "gap", item["holdout_abs_gap"],
                "tracked", item["flipout_tracked"],
                "scale", round(item["flipout"][0]["scale_mean"], 4),
                "loc_std", round(item["flipout"][0]["loc_std"], 4),
                flush=True,
            )
    # Seasonal table for the best 2024 head of each family. Not a price.
    chosen = {}
    for task in ("short", "long"):
        family = [item for item in packed if item["task"] == task]
        chosen[task] = min(family, key=lambda item: item["logged_holdout_2024_mae"])["job"]
    arrays = {"ts": data["ts"].astype(np.int64)}
    for item in packed:
        if item["job"] not in chosen.values():
            continue
        for split in ("monitor", "holdout_2024"):
            idx = item["_idx"][split]
            pred = item["_pred"][split]
            prefix = f"{item['job']}_{split}"
            arrays[prefix + "_y"] = item["_y"][idx]
            arrays[prefix + "_pred"] = pred.astype(np.float32)
            arrays[prefix + "_origin"] = data["seasonal_origin"][idx]
            arrays[prefix + "_ts"] = data["ts"][idx].astype(np.int64)
        arrays[item["job"] + "_horizons"] = np.asarray(item["_horizons"], dtype=np.int32)
    np.savez_compressed(TABLE, **arrays)
    public = []
    for item in packed:
        public.append({key: value for key, value in item.items() if not key.startswith("_")})
    payload = {
        "source": "posterior_mean_forward_not_evaluate",
        "mae_unit": "seasonal_price",
        "table": str(TABLE),
        "table_jobs": chosen,
        "table_note": "Columns are seasonal component values, not prices and not Prediction_h_*.",
        "jobs": public,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
