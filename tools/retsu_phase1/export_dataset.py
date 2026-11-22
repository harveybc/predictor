#!/usr/bin/env python
"""Causal CLOSE seasonal target plus frozen 61-encoder embeddings.

Runs on Dragon after the CPU identity test. Does not touch predictor's
plugin loader. STL parameters are chosen on train rows only.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RUNTIME = Path("/var/tmp/sac399-series/runtime")
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))

from app.causal_stl_inputs import choose_parameters, fit_history  # noqa: E402
from app.structural_refit import load_bundle  # noqa: E402
from tools.specialist_exposure403 import PARENT, ROOT, assemble, environment  # noqa: E402

OUT = Path("/var/tmp/retsu-phase1")
HOUR = 3600 * 10**9
STRENGTH_MIN = 0.30
ACF_MAX = 0.10
MONITOR_START = pd.Timestamp("2023-07-01", tz="UTC").value
HOLDOUT_START = pd.Timestamp("2024-01-01", tz="UTC").value
SHORT = (1, 2, 3, 4, 5, 6)
LONG = (24, 48, 72, 96, 120, 144)
WORKERS = 16

_CLOSE: np.ndarray | None = None
_CFG: dict | None = None


def status(payload: dict) -> None:
    payload = dict(payload, source="export_dataset")
    payload["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    tmp = OUT / "STATUS.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(OUT / "STATUS.json")


def _init_worker(close: np.ndarray, config: dict) -> None:
    global _CLOSE, _CFG
    _CLOSE = close
    _CFG = config


def _endpoint(t: int) -> tuple[float, float, float]:
    assert _CLOSE is not None and _CFG is not None
    hist = int(_CFG["history"])
    window = _CLOSE[t - hist + 1 : t + 1]
    if window.shape[0] != hist or not np.isfinite(window).all():
        return (np.nan, np.nan, np.nan)
    fit = fit_history(window, _CFG)
    return (float(fit.seasonal[-1]), float(fit.trend[-1]), float(fit.resid[-1]))


def acf_at(x: np.ndarray, lag: int) -> float:
    finite = np.isfinite(x)
    # Correlation on the contiguous finite train run, not a second model pass.
    if finite.sum() <= lag + 10:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 10 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def strict_table(market, risk, lo: str, hi: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return input-row ids, available timestamps and engine-bar timestamps."""
    env = environment(market, risk, 6, [lo, hi], 0.99)
    engine = env.engine
    engine.reset()
    bars = np.array(
        [b for b in range(engine.first, engine.last) if env.valid[b] and engine.entry_gate(b)[0]],
        dtype=np.int64,
    )
    rows = env.rows[bars]
    seg = engine.seg_of[bars]
    slots = market.inputs.slot_ts_ns[rows]
    strict = ((slots >= engine.op[seg, None]) & (slots < engine.ff[seg, None])).all(axis=1)
    bars = bars[strict]
    rows = rows[strict]
    avail = market.inputs.row_ts_ns[rows].astype(np.int64)
    # Map every engine bar's available time to its segment bounds for label checks.
    return rows, avail, bars


def segment_bounds(market, risk, lo: str, hi: str) -> dict[int, tuple[int, int]]:
    env = environment(market, risk, 6, [lo, hi], 0.99)
    engine = env.engine
    engine.reset()
    bounds: dict[int, tuple[int, int]] = {}
    for b in range(engine.first, engine.last):
        if not env.valid[b]:
            continue
        stamp = int(engine.v[b]) + HOUR
        seg = int(engine.seg_of[b])
        bounds[stamp] = (int(engine.op[seg]), int(engine.ff[seg]))
    return bounds


def encode(market, rows: np.ndarray, net, device: str) -> np.ndarray:
    windows = market.inputs.windows
    out = np.empty((len(rows), 208), np.float32)
    encoder = net.actor_net.encoder.eval()
    params = net.loss.actor_network_params
    torch.set_num_threads(8)
    with params.to_module(net.actor), torch.inference_mode():
        for start in range(0, len(rows), 2048):
            ix = rows[start : start + 2048]
            batch = torch.from_numpy(np.ascontiguousarray(windows[ix])).to(device)
            z = encoder({"features": batch})
            if z.shape[1] != 208 or not torch.isfinite(z).all():
                raise RuntimeError(f"ENCODER_OUTPUT {tuple(z.shape)}")
            out[start : start + len(ix)] = z.float().cpu().numpy()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "identity.log").read_text()
    if "IDENTITY_OK" not in log:
        status({"state": "failed", "stage": "identity", "reason": "IDENTITY_LOG_MISSING"})
        raise SystemExit(1)
    status({"state": "running", "stage": "assemble", "gpu": "idle_until_embeddings"})
    bundle = ROOT / "model/61/bundle"
    joined, real = assemble(PARENT, bundle, ROOT / "history397-61-r2")
    risk = json.loads((PARENT / "RISK.json").read_text())

    bars = pd.concat([joined.bars, real.bars])
    bars = bars[~bars.index.duplicated(keep="first")].sort_index()
    if not bars.index.is_monotonic_increasing:
        raise RuntimeError("CLOSE_INDEX_NOT_SORTED")
    close = bars["CLOSE"].to_numpy(np.float64)
    open_ns = bars.index.asi8
    avail_ns = open_ns + HOUR
    if not np.isfinite(close).all():
        raise RuntimeError("CLOSE_NONFINITE")
    train_end = int(np.searchsorted(avail_ns, HOLDOUT_START))
    status({
        "state": "running",
        "stage": "stl_select",
        "bars": int(len(close)),
        "train_bars": train_end,
        "gpu": "idle_cpu_stl_grid",
    })
    grid = choose_parameters(close, train_end)
    eligible = [
        row for row in grid["scores"]
        if row["seasonal_strength"] >= STRENGTH_MIN
        and abs(row["residual_period_correlation"]) <= ACF_MAX
    ]
    (OUT / "stl_grid.json").write_text(json.dumps(grid, indent=2) + "\n")
    if not eligible:
        best_any = min(grid["scores"], key=lambda row: abs(row["residual_period_correlation"]))
        status({
            "state": "failed",
            "stage": "stl_select",
            "reason": "STL_CRITERION_UNMET",
            "strength_min": STRENGTH_MIN,
            "acf_max": ACF_MAX,
            "closest": {
                "parameters": best_any["parameters"],
                "seasonal_strength": best_any["seasonal_strength"],
                "residual_period_correlation": best_any["residual_period_correlation"],
            },
        })
        raise SystemExit(2)
    selected = min(
        eligible,
        key=lambda row: (
            abs(row["residual_period_correlation"]),
            -row["seasonal_strength"],
            row["score"],
        ),
    )
    config = dict(selected["parameters"])
    (OUT / "stl_selected.json").write_text(json.dumps(selected, indent=2) + "\n")

    status({
        "state": "running",
        "stage": "stl_stream",
        "stl_parameters": config,
        "gpu": "idle_cpu_stl_stream",
        "workers": WORKERS,
    })
    indices = list(range(int(config["history"]) - 1, len(close)))
    with Pool(WORKERS, initializer=_init_worker, initargs=(close, config)) as pool:
        endpoints = pool.map(_endpoint, indices, chunksize=64)
    seasonal = np.full(len(close), np.nan, np.float64)
    trend = np.full(len(close), np.nan, np.float64)
    resid = np.full(len(close), np.nan, np.float64)
    for t, (s, tr, r) in zip(indices, endpoints):
        seasonal[t] = s
        trend[t] = tr
        resid[t] = r
    train_resid = resid[:train_end]
    full_acf = acf_at(train_resid, int(config["period"]))
    if not np.isfinite(full_acf) or abs(full_acf) > ACF_MAX:
        status({
            "state": "failed",
            "stage": "stl_stream",
            "reason": "FULL_TRAIN_RESIDUAL_STILL_PERIODIC",
            "full_train_residual_acf": full_acf,
            "acf_max": ACF_MAX,
            "parameters": config,
        })
        raise SystemExit(4)

    (OUT / "stl_full_acf.json").write_text(json.dumps({
        "full_train_residual_acf": full_acf,
        "period": int(config["period"]),
        "acf_max": ACF_MAX,
    }) + "\n")
    status({
        "state": "running",
        "stage": "embeddings",
        "full_train_residual_acf": full_acf,
        "gpu": "idle_cpu_encoder_matches_identity",
    })
    rows_j, avail_j, _ = strict_table(joined, risk, "2016-01-01", "2024-01-01")
    rows_r, avail_r, _ = strict_table(real, risk, "2024-01-01", "2025-01-01")
    cpu_net, _ = load_bundle(bundle, "cpu")
    probe, _ = load_bundle(bundle, "cuda")
    gpu_probe = encode(joined, rows_j[:4], probe, "cuda")
    cpu_probe = encode(joined, rows_j[:4], cpu_net, "cpu")
    gap = float(np.max(np.abs(gpu_probe - cpu_probe)))
    del probe
    torch.cuda.empty_cache()
    # Saved embeddings stay on CPU. A 4090 conv differs by about 1e-3 from the
    # identity device; that is recorded and is not the tensor used for training.
    if gap > 1e-2:
        status({"state": "failed", "stage": "embeddings", "reason": "CPU_GPU_ENCODER_GAP", "max_abs": gap})
        raise SystemExit(3)
    z_j = encode(joined, rows_j, cpu_net, "cpu")
    z_r = encode(real, rows_r, cpu_net, "cpu")
    del cpu_net

    by_avail = {int(ts): i for i, ts in enumerate(avail_ns)}
    bounds_j = segment_bounds(joined, risk, "2016-01-01", "2024-01-01")
    bounds_r = segment_bounds(real, risk, "2024-01-01", "2025-01-01")
    bounds = {**bounds_j, **bounds_r}

    def labels_for(avail: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
        out = np.full((len(avail), len(horizons)), np.nan, np.float32)
        for i, ts in enumerate(avail):
            origin = bounds.get(int(ts))
            if origin is None:
                continue
            op, ff = origin
            for k, h in enumerate(horizons):
                label_ts = int(ts) + h * HOUR
                if not (op <= label_ts < ff):
                    continue
                if bounds.get(label_ts) != origin:
                    continue
                j = by_avail.get(label_ts)
                if j is None or not np.isfinite(seasonal[j]):
                    continue
                out[i, k] = np.float32(seasonal[j])
        return out

    y_short_j = labels_for(avail_j, SHORT)
    y_long_j = labels_for(avail_j, LONG)
    y_short_r = labels_for(avail_r, SHORT)
    y_long_r = labels_for(avail_r, LONG)
    avail = np.concatenate([avail_j, avail_r])
    z = np.concatenate([z_j, z_r])
    y_short = np.concatenate([y_short_j, y_short_r])
    y_long = np.concatenate([y_long_j, y_long_r])
    order = np.argsort(avail, kind="mergesort")
    avail, z, y_short, y_long = avail[order], z[order], y_short[order], y_long[order]
    seasonal_origin = np.array(
        [seasonal[by_avail[int(ts)]] if int(ts) in by_avail else np.nan for ts in avail],
        np.float32,
    )
    pool = avail < MONITOR_START
    monitor = (avail >= MONITOR_START) & (avail < HOLDOUT_START)
    holdout = avail >= HOLDOUT_START
    ok_short = np.isfinite(y_short).all(axis=1)
    # Longest prefix of the long grid that still leaves a monitor and a train pool.
    long_keep = 0
    for k in range(1, y_long.shape[1] + 1):
        mask = np.isfinite(y_long[:, :k]).all(axis=1)
        if int((mask & pool).sum()) >= 1575 and int((mask & monitor).sum()) >= 200:
            long_keep = k
        else:
            break
    horizons_long = list(LONG[:long_keep])
    y_long = y_long[:, :long_keep] if long_keep else np.zeros((len(avail), 0), np.float32)
    ok_long = np.isfinite(y_long).all(axis=1) if long_keep else np.zeros(len(avail), dtype=bool)

    np.savez_compressed(
        OUT / "dataset.npz",
        ts=avail.astype(np.int64),
        z=z.astype(np.float32),
        y_short=y_short.astype(np.float32),
        y_long=y_long.astype(np.float32),
        ok_short=ok_short,
        ok_long=ok_long,
        pool=pool,
        monitor=monitor,
        holdout=holdout,
        seasonal_origin=seasonal_origin,
    )
    meta = {
        "horizons_short": list(SHORT),
        "horizons_long": horizons_long,
        "horizons_long_requested": list(LONG),
        "stl": config,
        "origin_residual_acf": selected["residual_period_correlation"],
        "full_train_residual_acf": full_acf,
        "seasonal_strength_origins": selected["seasonal_strength"],
        "rows": int(len(avail)),
        "pool_short": int((ok_short & pool).sum()),
        "monitor_short": int((ok_short & monitor).sum()),
        "holdout_short": int((ok_short & holdout).sum()),
        "pool_long": int((ok_long & pool).sum()),
        "monitor_long": int((ok_long & monitor).sum()),
        "holdout_long": int((ok_long & holdout).sum()),
        "encoder_cpu_gpu_max_abs": gap,
        "encoder_device": "cpu",
        "criterion": {"strength_min": STRENGTH_MIN, "acf_max": ACF_MAX, "fit": "TRAIN available_at < 2024-01-01"},
    }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez_compressed(
        OUT / "stl_series.npz",
        avail_ns=avail_ns.astype(np.int64),
        close=close.astype(np.float32),
        seasonal=seasonal.astype(np.float32),
        trend=trend.astype(np.float32),
        resid=resid.astype(np.float32),
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sl = slice(train_end - 24 * 20, train_end - 24 * 10)
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        times = pd.to_datetime(avail_ns[sl], utc=True)
        for ax, series, title in zip(
            axes,
            (close[sl], trend[sl], seasonal[sl], resid[sl]),
            ("CLOSE", "trend endpoint", "seasonal endpoint", "residual endpoint"),
        ):
            ax.plot(times, series, lw=0.8)
            ax.set_ylabel(title)
        fig.tight_layout()
        fig.savefig(OUT / "stl_train_slice.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # figure is diagnostic, the arrays are the record
        (OUT / "stl_plot_error.txt").write_text(repr(exc) + "\n")
    status({"state": "running", "stage": "dataset_ready", "meta": meta, "gpu": "free_for_training"})
    print("DATASET_OK", json.dumps(meta), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        status({"state": "failed", "stage": "export", "reason": "UNCAUGHT", "error": repr(exc)})
        raise
