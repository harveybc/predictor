#!/usr/bin/env python
"""Candidate evaluation worker.

Runs a single (gen,candidate) evaluation in a fresh Python process to avoid
in-process TensorFlow / CUDA / allocator cache growth across GA candidates.

This module is invoked by `optimizer_plugins/default_optimizer.py` via:
  python -m optimizer_plugins.candidate_worker --input <json> --output <json>

Input JSON schema:
{
  "gen": int,
  "cand": int,
  "config": { ... },
  "hyper": { ... }
}

Output JSON schema:
{
  "ok": bool,
  "fitness": float,
  "naive_mae": float|null,
  "error": str|null
}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    # candidate_worker.py -> optimizer_plugins/ -> <repo_root>
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(p: str | None) -> str | None:
    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((_repo_root() / pp).resolve())
    except Exception:
        return str(p)


def _read_proc_status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    return int(parts[1])
    except Exception:
        return None
    return None


def _read_gpu_mem_bytes() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return (None, None)
        info = tf.config.experimental.get_memory_info("GPU:0")  # type: ignore[attr-defined]
        cur = int(info.get("current")) if isinstance(info, dict) and info.get("current") is not None else None
        peak = int(info.get("peak")) if isinstance(info, dict) and info.get("peak") is not None else None
        return (cur, peak)
    except Exception:
        return (None, None)


def _append_optimizer_resource_row(config: dict, stage: str, gen: int | None, cand: int | None, extra: dict | None = None) -> None:
    log_path = _resolve_repo_path(config.get("optimizer_resource_log_file"))
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("ts,stage,generation,candidate,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,extra\n")

        ts = time.time()
        rss = _read_proc_status_kb("VmRSS")
        hwm = _read_proc_status_kb("VmHWM")
        gpu_cur, gpu_peak = _read_gpu_mem_bytes() if bool(config.get("memory_log_gpu", True)) else (None, None)
        extra_json = ""
        if extra is not None:
            try:
                extra_json = json.dumps(extra, separators=(",", ":"))
            except Exception:
                extra_json = ""

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ts:.3f},{stage},{gen if gen is not None else ''},{cand if cand is not None else ''},"
                f"{rss if rss is not None else ''},{hwm if hwm is not None else ''},"
                f"{gpu_cur if gpu_cur is not None else ''},{gpu_peak if gpu_peak is not None else ''},"
                f"{extra_json}\n"
            )
            f.flush()
    except Exception:
        # Never fail the worker due to logging.
        return


def evaluate_candidate(*, config: dict, hyper: dict, gen: int, cand: int) -> tuple[float, float | None]:
    """Run preprocessing, build model, train, and compute fitness + naive_mae.

    Fitness is the denormalized MAE for the max horizon (pipeline parity).
    """

    # Resolve log paths to repo root for consistency.
    for k in ("memory_log_file", "optimizer_resource_log_file", "batch_memory_log_file"):
        if config.get(k):
            config[k] = _resolve_repo_path(config.get(k))

    # Tag for per-epoch/batch logs.
    config.setdefault("memory_log_tag", f"ga_gen{int(gen)}_cand{int(cand)}")

    # Safety: avoid post-fit uncertainty during GA eval.
    config.setdefault("disable_postfit_uncertainty", True)
    config.setdefault("mc_samples", 1)
    config.setdefault("predict_batch_size", config.get("batch_size", 32))

    from app.plugin_loader import load_plugin

    # Instantiate plugins inside the worker process.
    target_plugin_name = config.get("target_plugin", "default_target")
    target_class, _ = load_plugin("target.plugins", target_plugin_name)
    target_plugin = target_class()
    target_plugin.set_params(**config)

    preprocessor_name = config.get("preprocessor_plugin", "default_preprocessor")
    preprocessor_class, _ = load_plugin("preprocessor.plugins", preprocessor_name)
    preprocessor_plugin = preprocessor_class()
    preprocessor_plugin.set_params(**config)

    predictor_name = config.get("plugin") or config.get("predictor_plugin")
    predictor_class, _ = load_plugin("predictor.plugins", predictor_name)
    predictor_plugin = predictor_class(config)
    predictor_plugin.set_params(**config)

    _append_optimizer_resource_row(config, "before_preprocess", gen, cand)
    datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
    if isinstance(datasets, tuple):
        datasets = datasets[0]
    _append_optimizer_resource_row(config, "after_preprocess", gen, cand)

    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]

    def _ensure_2d_targets(y):
        if isinstance(y, dict):
            out = {}
            for k, v in y.items():
                arr = __import__("numpy").asarray(v)
                out[k] = arr.reshape(-1, 1).astype("float32")
            return out
        return y

    y_train = _ensure_2d_targets(y_train)
    y_val = _ensure_2d_targets(y_val)

    window_size = config.get("window_size")
    _append_optimizer_resource_row(config, "before_build_model", gen, cand)
    if predictor_name in ["lstm", "cnn", "transformer", "ann", "mimo"]:
        input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)
        predictor_plugin.build_model(input_shape=input_shape, x_train=x_train, config=config)
    else:
        predictor_plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)
    _append_optimizer_resource_row(config, "after_build_model", gen, cand)

    _append_optimizer_resource_row(config, "before_fit", gen, cand, extra={"params": hyper})
    history, _, _, val_preds, _ = predictor_plugin.train(
        x_train,
        y_train,
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 32),
        threshold_error=config.get("threshold_error", 0.001),
        x_val=x_val,
        y_val=y_val,
        config=config,
    )
    _append_optimizer_resource_row(config, "after_fit", gen, cand)

    # --- Pipeline parity metrics (max horizon) ---
    import numpy as np
    from pipeline_plugins.stl_norm import denormalize, denormalize_returns

    predicted_horizons = config.get("predicted_horizons", [1])
    max_horizon = max(predicted_horizons)
    max_h_idx = predicted_horizons.index(max_horizon)

    val_preds_h = val_preds[max_h_idx].flatten()

    if isinstance(y_val, dict):
        y_true_max_h = y_val[f"output_horizon_{max_horizon}"].flatten()
    elif isinstance(y_val, list):
        y_true_max_h = y_val[max_h_idx].flatten()
    else:
        y_true_max_h = y_val.flatten()

    baseline_val = datasets.get("baseline_val")

    n = min(len(val_preds_h), len(y_true_max_h))
    if baseline_val is not None:
        n = min(n, len(baseline_val))

    val_preds_h = val_preds_h[:n]
    y_true_max_h = y_true_max_h[:n]

    val_target_price = denormalize(y_true_max_h, config)
    # Model MAE (pipeline does denormalize_returns(pred - target))
    fitness = float(np.mean(np.abs(denormalize_returns(val_preds_h - y_true_max_h, config))))

    naive_mae = None
    if baseline_val is not None:
        baseline_val_h = baseline_val[:n].flatten()
        naive_mae = float(np.mean(np.abs(denormalize(baseline_val_h, config) - val_target_price)))

    # Cleanup best-effort.
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    return fitness, naive_mae


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Configure TF allocator before importing TF via plugins.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    payload = json.load(open(args.input, "r", encoding="utf-8"))
    gen = int(payload.get("gen", 0))
    cand = int(payload.get("cand", 0))
    config = payload.get("config", {})
    hyper = payload.get("hyper", {})

    # Apply hyperparams into config (this must match parent behavior).
    config = dict(config)
    config.update(hyper)

    out = {"ok": False, "fitness": float("inf"), "naive_mae": None, "error": None}
    try:
        fitness, naive_mae = evaluate_candidate(config=config, hyper=hyper, gen=gen, cand=cand)
        out.update({"ok": True, "fitness": float(fitness), "naive_mae": naive_mae})
    except Exception as e:
        out.update({"ok": False, "fitness": float("inf"), "naive_mae": None, "error": str(e)})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f)

    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
