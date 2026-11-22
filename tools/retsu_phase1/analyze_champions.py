#!/usr/bin/env python
"""Read the nine saved champions. Do not call evaluate.

The logged epoch MAE stays the value already written in epochs.jsonl.
This script builds the same head, loads champion.weights.h5, and does one
posterior-mean forward on the train panel, the monitor, and 2024.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_head as th

PHASE = Path("/var/tmp/retsu-phase1")
OUT = PHASE / "analysis_flipout_untracked.json"


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def column_stats(pred: np.ndarray, target: np.ndarray, origin: np.ndarray, horizons: list[int]) -> list[dict]:
    rows = []
    for i, horizon in enumerate(horizons):
        p = pred[:, i]
        y = target[:, i]
        o = origin
        ss_res = float(np.sum(np.square(y - p)))
        ss_tot = float(np.sum(np.square(y - np.mean(y))))
        rows.append({
            "horizon": int(horizon),
            "rows": int(p.size),
            "mae": float(np.mean(np.abs(p - y))),
            "naive_mae": float(np.mean(np.abs(o - y))),
            "r2": None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot),
            "pred_std": float(np.std(p)),
            "y_std": float(np.std(y)),
            "origin_std": float(np.std(o)),
            "corr_pred_y": corr(p, y),
            "corr_pred_origin": corr(p, o),
            "corr_origin_y": corr(o, y),
        })
    return rows


def flipout_tensors(model: tf.keras.Model) -> list[dict]:
    found = []
    for layer in model.layers:
        if "flip" not in layer.name:
            continue
        dist = layer.kernel_posterior.distribution
        loc = np.asarray(dist.loc.numpy(), dtype=np.float64).reshape(-1)
        scale = np.asarray(dist.scale.numpy(), dtype=np.float64).reshape(-1)
        found.append({
            "layer": layer.name,
            "loc_type": type(dist.loc).__name__,
            "scale_type": type(dist.scale).__name__,
            "loc_l2": float(np.linalg.norm(loc)),
            "loc_std": float(np.std(loc)),
            "loc_mean": float(np.mean(loc)),
            "scale_mean": float(np.mean(scale)),
            "scale_min": float(np.min(scale)),
            "scale_max": float(np.max(scale)),
            "finite": bool(np.isfinite(loc).all() and np.isfinite(scale).all()),
        })
    return found


def bias_tensors(model: tf.keras.Model) -> list[dict]:
    rows = []
    for layer in model.layers:
        if not layer.name.startswith("bias"):
            continue
        kernel = np.asarray(layer.kernel.numpy(), dtype=np.float64).reshape(-1)
        bias = np.asarray(layer.bias.numpy(), dtype=np.float64).reshape(-1)
        rows.append({
            "layer": layer.name,
            "kernel_l2": float(np.linalg.norm(kernel)),
            "kernel_rms": float(np.sqrt(np.mean(np.square(kernel)))),
            "bias": float(bias[0]),
            "finite": bool(np.isfinite(kernel).all() and np.isfinite(bias).all()),
        })
    return rows


def predict_price(model, use_mean, z, mean, std, scale) -> np.ndarray:
    use_mean.assign(True)
    pred = model.predict(z / scale, batch_size=th.BATCH, verbose=0)
    if isinstance(pred, (list, tuple)):
        cols = [np.asarray(item).reshape(-1) for item in pred]
    else:
        cols = [np.asarray(pred).reshape(-1)]
    out = np.column_stack(cols)
    return out * std.reshape(1, -1) + mean.reshape(1, -1)


def one(job_dir: Path, data: dict) -> dict:
    hold = json.loads((job_dir / "holdout.json").read_text())
    horizons = [int(h) for h in hold["train_mae_per_horizon"]]
    # horizons are stored as string keys; keep file order from the job record.
    if "horizons" in hold:
        horizons = [int(h) for h in hold["horizons"]]
    else:
        horizons = [int(h) for h in hold["train_mae_per_horizon"].keys()]
    task = hold["task"]
    mean = np.asarray(hold["target_mean"], dtype=np.float64)
    std = np.asarray(hold["target_std"], dtype=np.float64)
    ok = data["ok_short"] if task == "short" else data["ok_long"]
    y_all = data["y_short"] if task == "short" else data["y_long"]
    pool = np.flatnonzero(data["pool"] & ok)[: int(hold["max_steps"])]
    # Same scale the trainer applied and did not write into the epoch record.
    z_train = data["z"][pool]
    fan_in = float(z_train.shape[1])
    z_scale = float(np.sqrt(np.mean(np.square(z_train.astype(np.float64))) * fan_in))
    if z_scale <= 1e-6:
        z_scale = 1.0
    panel_train = pool[-256:] if pool.size >= 256 else pool
    monitor = np.flatnonzero(data["monitor"] & ok)
    holdout = np.flatnonzero(data["holdout"] & ok)
    tf.keras.backend.clear_session()
    kl = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight")
    use_mean = tf.Variable(True, trainable=False, dtype=tf.bool, name="use_posterior_mean")
    model = th.build(horizons, kl, use_mean)
    model(tf.zeros((1, 208), dtype=tf.float32))
    names = [w.name for w in model.trainable_weights]
    model.load_weights(job_dir / "champion.weights.h5")
    origin = data["seasonal_origin"]
    splits = {
        "train_panel_256": panel_train,
        "monitor": monitor,
        "holdout_2024": holdout,
    }
    forward = {}
    for name, idx in splits.items():
        pred = predict_price(model, use_mean, data["z"][idx], mean, std, z_scale)
        forward[name] = column_stats(pred, y_all[idx], origin[idx], horizons)
    logged = hold.get("holdout_2024_mae")
    reproduced = float(np.mean([row["mae"] for row in forward["holdout_2024"]]))
    return {
        "job": job_dir.name,
        "tracked_weight_names": names,
        "flipout_in_keras_weights": any("flip" in name for name in names),
        "flipout": flipout_tensors(model),
        "bias": bias_tensors(model),
        "logged_holdout_2024_mae": logged,
        "reproduced_holdout_2024_mae": reproduced,
        "holdout_abs_gap": None if logged is None else abs(reproduced - float(logged)),
        "splits": forward,
    }


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    raw = np.load(PHASE / "dataset.npz")
    data = {key: raw[key] for key in raw.files}
    jobs = []
    for path in sorted((PHASE / "jobs").iterdir()):
        if (path / "champion.weights.h5").exists() and (path / "holdout.json").exists():
            jobs.append(one(path, data))
            print(path.name, "gap", jobs[-1]["holdout_abs_gap"], "tracked", jobs[-1]["flipout_in_keras_weights"], flush=True)
    payload = {
        "source": "posterior_mean_forward_not_evaluate",
        "mae_unit": "seasonal_price",
        "finding": "DenseFlipout loc/scale were created with tf.Variable outside add_variable. They are not in champion.weights.h5 and are not trainable_weights. A rebuild recreates the seeded init. The saved champion is the deterministic bias Dense plus that frozen posterior mean.",
        "jobs": jobs,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
