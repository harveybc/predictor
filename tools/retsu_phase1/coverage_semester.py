#!/usr/bin/env python
"""Posterior coverage and the 2024 semester split. CPU. No evaluate().

Coverage uses 24 Flipout samples. The semester MAE uses the posterior mean,
the same forward the holdout record used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

import train_head as th

PHASE = Path("/var/tmp/retsu-phase1")
FOLLOW = PHASE / "followup"
SAMPLES = 24
H1_END = np.datetime64("2024-07-01T00:00:00").astype("datetime64[ns]").astype(np.int64)


def corr(a: np.ndarray, b: np.ndarray):
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(pred: np.ndarray, target: np.ndarray, origin: np.ndarray, horizons: list[int]) -> list[dict]:
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
            "corr_pred_y": corr(p, y),
        })
    return rows


def price_from_z(pred_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if pred_z.ndim == 1:
        pred_z = pred_z.reshape(-1, 1)
    return pred_z * std.reshape(1, -1) + mean.reshape(1, -1)


def predict_mean(model, use_mean, z, mean, std) -> np.ndarray:
    use_mean.assign(True)
    pred = model.predict(z, batch_size=th.BATCH, verbose=0)
    if not isinstance(pred, (list, tuple)):
        pred = [pred]
    cols = np.column_stack([np.asarray(item).reshape(-1) for item in pred])
    return price_from_z(cols, mean, std)


def sample_prices(model, use_mean, z, mean, std) -> np.ndarray:
    """Return [samples, rows, horizons] in seasonal price units."""
    use_mean.assign(False)
    draws = []
    for _ in range(SAMPLES):
        pred = model.predict(z, batch_size=th.BATCH, verbose=0)
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        cols = np.column_stack([np.asarray(item).reshape(-1) for item in pred])
        draws.append(price_from_z(cols, mean, std))
    return np.stack(draws, axis=0)


def coverage(draws: np.ndarray, target: np.ndarray) -> list[dict]:
    low = np.percentile(draws, 5, axis=0)
    high = np.percentile(draws, 95, axis=0)
    rows = []
    for i in range(target.shape[1]):
        inside = (target[:, i] >= low[:, i]) & (target[:, i] <= high[:, i])
        rows.append({
            "horizon": i,
            "nominal": 0.90,
            "empirical_coverage": float(np.mean(inside)),
            "mean_width": float(np.mean(high[:, i] - low[:, i])),
            "sample_std": float(np.mean(np.std(draws[:, :, i], axis=0))),
            "y_std": float(np.std(target[:, i])),
        })
    return rows


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
    monitor = np.flatnonzero(data["monitor"] & ok)
    holdout = np.flatnonzero(data["holdout"] & ok)
    tf.keras.backend.clear_session()
    kl = tf.Variable(1e-3, trainable=False, dtype=tf.float32, name="kl_weight")
    use_mean = tf.Variable(True, trainable=False, dtype=tf.bool, name="use_posterior_mean")
    model = th.build(horizons, kl, use_mean)
    model(tf.zeros((1, 208), dtype=tf.float32))
    model.load_weights(job_dir / "champion.weights.h5")
    origin = data["seasonal_origin"]
    ts = data["ts"]
    result = {"job": job_dir.name, "task": task, "samples": SAMPLES, "splits": {}}
    for name, idx in (("monitor", monitor), ("holdout_2024", holdout)):
        z = data["z"][idx] / z_scale
        pred = predict_mean(model, use_mean, z, mean, std)
        draws = sample_prices(model, use_mean, z, mean, std)
        covered = coverage(draws, y_all[idx])
        for row, horizon in zip(covered, horizons):
            row["horizon"] = int(horizon)
        block = {
            "mae": metrics(pred, y_all[idx], origin[idx], horizons),
            "coverage_90": covered,
        }
        if name == "holdout_2024":
            h1 = ts[idx] < H1_END
            h2 = ~h1
            block["semester"] = {
                "2024H1": metrics(pred[h1], y_all[idx][h1], origin[idx][h1], horizons),
                "2024H2": metrics(pred[h2], y_all[idx][h2], origin[idx][h2], horizons),
            }
        result["splits"][name] = block
        print(job_dir.name, name, "rows", int(idx.size), flush=True)
    return result


def main() -> None:
    FOLLOW.mkdir(parents=True, exist_ok=True)
    raw = np.load(PHASE / "dataset.npz")
    data = {key: raw[key] for key in raw.files}
    jobs = []
    for path in sorted((PHASE / "jobs").iterdir()):
        if (path / "champion.weights.h5").exists() and (path / "holdout.json").exists():
            jobs.append(one(path, data))
    payload = {
        "device": "cpu",
        "interval": "central 90 percent of 24 Flipout samples",
        "semester_cut": "2024-07-01",
        "jobs": jobs,
    }
    (FOLLOW / "coverage_semester.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("COVERAGE_OK", len(jobs), flush=True)


if __name__ == "__main__":
    main()
