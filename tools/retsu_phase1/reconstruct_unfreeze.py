#!/usr/bin/env python
"""One posterior predict of the unfreeze champions, then price reconstruction.

Checks the saved seasonal MAE. Scores trend(origin) + seasonal forecast
against future CLOSE. Does not write Prediction columns.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

RUNTIME = Path("/var/tmp/sac399-series/runtime")
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, "/var/tmp/retsu-phase1")

from app.structural_refit import load_bundle  # noqa: E402
import train_unfreeze as tu  # noqa: E402

PHASE = Path("/var/tmp/retsu-phase1")
HOUR = 3600 * 10**9
BUNDLE = Path("/var/tmp/sac399-series/model/61/bundle")
JOBS = (
    ("frozen", "short"),
    ("frozen", "long"),
    ("tuned", "short"),
    ("tuned", "long"),
)
GATES = {6: 0.0014153621159493923, 24: 0.0033699586056172848}


def predict(mode: str, task: str, data: dict, meta: dict, windows: np.ndarray):
    root = Path(f"/var/tmp/retsu-phase1-unfreeze-{mode}/jobs/{mode}_{task}")
    record = json.loads((root / "holdout.json").read_text())
    horizons = [int(h) for h in record["holdout_2024_mae_per_horizon"]]
    mean = np.asarray(record["target_mean"], np.float64)
    std = np.asarray(record["target_std"], np.float64)
    scale = float(record["embedding_scale"])
    ok = data["ok_short"] if task == "short" else data["ok_long"]
    y = data["y_short"] if task == "short" else data["y_long"]
    hold = np.flatnonzero(data["holdout"] & ok)
    model, _ = load_bundle(BUNDLE, "cuda")
    chosen = tu.encoder_parameters(model)
    blob = torch.load(root / "champion.pt", map_location="cpu", weights_only=False)
    saved = dict(blob["encoder"])
    with torch.no_grad():
        for name, param in chosen:
            param.copy_(saved[name].to(param.device))
    head = nn.Linear(208, len(horizons)).to("cuda")
    head.load_state_dict(blob["head"])
    head.eval()
    pred_z = tu.run_split(model, model.loss.actor_network_params, head, scale, windows, y, hold, True)
    price, per = tu.price_mae(pred_z, y[hold], mean, std)
    for i, horizon in enumerate(horizons):
        gap = abs(per[i] - float(record["holdout_2024_mae_per_horizon"][str(horizon)]))
        if gap > 1e-5:
            raise RuntimeError(f"MAE_MISMATCH {mode} {task} {horizon} {gap}")
    seasonal = pred_z * std + mean
    return data["ts"][hold].astype(np.int64), y[hold].astype(np.float64), seasonal, horizons, record


def score(ts, y, pred, horizons, series_index, close, trend) -> list[dict]:
    rows = []
    for k, horizon in enumerate(horizons):
        label = ts + int(horizon) * HOUR
        ok = np.array([int(a) in series_index and int(b) in series_index for a, b in zip(ts, label)])
        io = np.array([series_index[int(a)] for a in ts[ok]])
        il = np.array([series_index[int(b)] for b in label[ok]])
        future = close[il]
        recon = trend[io] + pred[ok, k]
        oracle = trend[io] + y[ok, k]
        persist = close[io]
        def mae(hat):
            return float(np.mean(np.abs(hat - future)))
        recon_mae = mae(recon)
        persist_mae = mae(persist)
        rows.append({
            "horizon": int(horizon),
            "rows": int(ok.sum()),
            "seasonal_mae": float(np.mean(np.abs(pred[ok, k] - y[ok, k]))),
            "recon_mae": recon_mae,
            "oracle_mae": mae(oracle),
            "close_persistence_mae": persist_mae,
            "beats_close_persistence": recon_mae < persist_mae,
            "beats_published_gate": int(horizon) in GATES and recon_mae < GATES[int(horizon)],
        })
    return rows


def main() -> None:
    saved = np.load(PHASE / "dataset.npz")
    data = {key: saved[key] for key in saved.files}
    meta = json.loads((PHASE / "dataset_meta.json").read_text())
    windows = np.load(PHASE / "windows.npz")["windows"]
    series = np.load(PHASE / "stl_series.npz")
    avail = series["avail_ns"].astype(np.int64)
    close = series["close"].astype(np.float64)
    trend = series["trend"].astype(np.float64)
    order = np.argsort(avail, kind="mergesort")
    avail, close, trend = avail[order], close[order], trend[order]
    index = {int(ts): i for i, ts in enumerate(avail)}
    report = {"reconstruction": "trend_at_origin + seasonal_forecast", "gates": {"h6": GATES[6], "h24": GATES[24]}, "jobs": {}}
    beats = []
    for mode, task in JOBS:
        ts, y, pred, horizons, record = predict(mode, task, data, meta, windows)
        rows = score(ts, y, pred, horizons, index, close, trend)
        report["jobs"][f"{mode}_{task}"] = {
            "epoch": record["epoch"],
            "best_epoch": record["best_epoch"],
            "seasonal_holdout_mae": record["holdout_2024_mae"],
            "rows": rows,
        }
        for row in rows:
            if row["beats_published_gate"]:
                beats.append({"job": f"{mode}_{task}", **row})
            print(mode, task, "h", row["horizon"], "recon", round(row["recon_mae"], 6), "persist", round(row["close_persistence_mae"], 6), "oracle", round(row["oracle_mae"], 6))
        del ts
        torch.cuda.empty_cache()
    report["beats_published_gate"] = beats
    report["prediction_columns_written"] = False
    path = PHASE / "followup" / "unfreeze_results.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("beats", len(beats))
    print("UNFREEZE_RECON_OK")


if __name__ == "__main__":
    main()
