#!/usr/bin/env python
"""One posterior-mean predict of the change champions, checked against the saved 2024 MAE.

Writes price columns for the account comparison. The price is the origin CLOSE
plus the predicted change. The naive column is the origin CLOSE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE = Path("/var/tmp/retsu-phase1")
sys.path.insert(0, str(PHASE))
import train_head  # noqa: E402

PAIRS = (
    ("seed7", Path("/var/tmp/retsu-phase1-dclose-seed7"), "short_24896", "long_8946"),
    ("pretrained", Path("/var/tmp/retsu-phase1-dclose-pretrained"), "short_24896", "long_8946"),
)
ATOL = 1e-5
OUT = PHASE / "followup" / "account"


def close_lookup() -> dict[int, float]:
    series = np.load(PHASE / "stl_series.npz")
    avail = series["avail_ns"].astype(np.int64)
    close = series["close"].astype(np.float64)
    return {int(ts): float(value) for ts, value in zip(avail, close)}


def predict_job(root: Path, name: str, data: dict, ok_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    record = json.loads((root / "jobs" / name / "holdout.json").read_text())
    horizons = [int(h) for h in record["holdout_2024_mae_per_horizon"]]
    mean = np.asarray(record["target_mean"], np.float64)
    std = np.asarray(record["target_std"], np.float64)
    scale = float(record["embedding_scale"])
    rows = np.flatnonzero(data["holdout"] & data[ok_key])
    tf.keras.backend.clear_session()
    kl_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight")
    use_mean = tf.Variable(True, trainable=False, dtype=tf.bool, name="use_posterior_mean")
    model = train_head.build(horizons, kl_var, use_mean)
    model.load_weights(str(root / "jobs" / name / "champion.weights.h5"))
    use_mean.assign(True)
    raw = model.predict(data["z"][rows] / np.float32(scale), batch_size=256, verbose=0)
    if len(horizons) == 1:
        raw = [raw]
    change = np.column_stack([
        raw[i].reshape(-1).astype(np.float64) * std[i] + mean[i] for i in range(len(horizons))
    ])
    saved = record["holdout_2024_mae_per_horizon"]
    y = data[y_key][rows]
    for i, horizon in enumerate(horizons):
        mae = float(np.mean(np.abs(change[:, i] - y[:, i])))
        gap = abs(mae - float(saved[str(horizon)]))
        if gap > ATOL:
            raise RuntimeError(f"MAE_MISMATCH {name} h{horizon} {mae} {saved[str(horizon)]} {gap}")
    return data["ts"][rows].astype(np.int64), change


def frame(ts: np.ndarray, hourly: np.ndarray, daily: np.ndarray) -> pd.DataFrame:
    index = pd.to_datetime(ts, utc=True)
    out = pd.DataFrame(hourly, index=index, columns=[f"h{i}" for i in range(1, hourly.shape[1] + 1)])
    for i in range(daily.shape[1]):
        out[f"d{i + 1}"] = daily[:, i]
    out.index.name = "DATE_TIME"
    return out.sort_index()


def main() -> None:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = close_lookup()
    data = {key: np.load(PHASE / "dataset_dclose_seed7.npz")[key] for key in np.load(PHASE / "dataset_dclose_seed7.npz").files}
    # z differs by encoder. Reload per pair.
    for name, root, short_job, long_job in PAIRS:
        blob = np.load(PHASE / ("dataset_dclose_seed7.npz" if name == "seed7" else "dataset_dclose_pretrained.npz"))
        data = {key: blob[key] for key in blob.files}
        short_ts, short_change = predict_job(root, short_job, data, "ok_short", "y_short")
        long_ts, long_change = predict_job(root, long_job, data, "ok_long", "y_long")
        short_map = {int(ts): row for ts, row in zip(short_ts, short_change)}
        keep_ts = []
        hourly = []
        daily = []
        origin = []
        for ts, row in zip(long_ts, long_change):
            if int(ts) not in short_map:
                continue
            keep_ts.append(int(ts))
            hourly.append(short_map[int(ts)])
            daily.append(row)
            origin.append(lookup[int(ts)])
        origin_a = np.asarray(origin, np.float64)
        hourly_a = np.asarray(hourly, np.float64) + origin_a[:, None]
        daily_a = np.asarray(daily, np.float64) + origin_a[:, None]
        ts_a = np.asarray(keep_ts, np.int64)
        model = frame(ts_a, hourly_a, daily_a)
        naive = frame(ts_a, np.repeat(origin_a[:, None], hourly_a.shape[1], axis=1), np.repeat(origin_a[:, None], daily_a.shape[1], axis=1))
        model.to_csv(OUT / f"{name}_model.csv")
        naive.to_csv(OUT / f"{name}_naive.csv")
        move = np.max(np.abs(np.concatenate([
            hourly_a - origin_a[:, None],
            daily_a - origin_a[:, None],
        ], axis=1)), axis=0)
        print(name, "rows", len(model), "max_abs_move", move.tolist())
    print("ACCOUNT_PREDICT_OK")


if __name__ == "__main__":
    main()
