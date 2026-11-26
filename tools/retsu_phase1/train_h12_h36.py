#!/usr/bin/env python3
"""12h and 36h seasonal increments. Reads H12_H36_PLAN.md and does not search.

Each horizon keeps its own same-segment rows. Both models train before either
one predicts 2024. The saved causal STL series is not refit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/var/tmp/retsu-phase1")
import train_phase2_step4 as step4  # noqa: E402
import export_dataset as exp  # noqa: E402

step3 = step4.step3


def load_plan(path: Path) -> dict:
    text = path.read_text()
    marker = "```h12-h36\n"
    start = text.index(marker) + len(marker)
    plan = json.loads(text[start : text.index("\n```", start)])
    if plan.get("schema") != "retsu_h12_h36.v1":
        raise SystemExit("PLAN_SCHEMA")
    if plan.get("horizons") != [12, 36]:
        raise SystemExit("PLAN_HORIZONS")
    if plan.get("scope") != "full_encoder" or plan.get("head") != "linear":
        raise SystemExit("PLAN_MODEL")
    rates = {int(key): float(value) for key, value in plan["encoder_lr"].items()}
    if abs(rates[12] - 3e-6) > 1e-15 or abs(rates[36] - 3e-5) > 1e-15:
        raise SystemExit("PLAN_ENCODER_LR")
    plan["encoder_lr"] = rates
    return plan


def labels_for(ts: np.ndarray, seasonal: np.ndarray, by_avail: dict, bounds: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    increment = np.full(len(ts), np.nan, np.float64)
    admitted = np.zeros(len(ts), dtype=bool)
    step = horizon * exp.HOUR
    for i, origin in enumerate(ts):
        segment = bounds.get(int(origin))
        if segment is None:
            continue
        op, ff = segment
        label_ts = int(origin) + step
        if not (op <= label_ts < ff) or bounds.get(label_ts) != segment:
            continue
        origin_i = by_avail.get(int(origin))
        label_i = by_avail.get(label_ts)
        if origin_i is None or label_i is None:
            continue
        left = seasonal[origin_i]
        right = seasonal[label_i]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        increment[i] = right - left
        admitted[i] = True
    return increment.astype(np.float32), admitted


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUIRED")
    plan = load_plan(Path(sys.argv[sys.argv.index("--plan") + 1]))
    out = Path(plan["out_dir"]).resolve()
    for finished in plan["finished_dirs"]:
        finished_path = Path(finished).resolve()
        if out == finished_path or str(out).startswith(str(finished_path) + "/"):
            raise SystemExit("REFUSING_FINISHED_DIR")
    if (out / "H12_H36_DONE").exists():
        print("H12_H36_DONE", flush=True)
        return
    phase = Path(plan["phase2_dir"])
    recipe = json.loads((phase / "RECIPE.json").read_text())
    if recipe["identity"] != plan["recipe_identity"] or recipe["list"]["n"] != 164:
        raise SystemExit("RECIPE_IDENTITY")
    data = np.load(phase / "dataset.npz")
    base = np.load(plan["base_dataset"])
    loaded = np.load(phase / "windows.npz")
    if not np.array_equal(data["ts"], base["ts"]) or not np.array_equal(loaded["ts"], data["ts"]):
        raise SystemExit("WINDOWS_TS")
    windows = loaded["windows"]
    if windows.shape[1] != 13 or windows.shape[2] != 164:
        raise SystemExit(f"WINDOWS_SHAPE {windows.shape}")
    nonzero = np.any(windows != 0, axis=(1, 2))
    if not np.all(nonzero[data["ok_short"]]):
        raise SystemExit("ZERO_WINDOW_ON_SHORT")
    series = np.load(plan["stl_series"])
    avail = series["avail_ns"].astype(np.int64)
    seasonal = series["seasonal"].astype(np.float64)
    by_avail = {int(stamp): i for i, stamp in enumerate(avail)}
    ts = data["ts"].astype(np.int64)
    plain = np.full(len(ts), np.nan, np.float64)
    for i, origin in enumerate(ts):
        left_i = by_avail.get(int(origin))
        right_i = by_avail.get(int(origin) + exp.HOUR)
        if left_i is None or right_i is None:
            continue
        left = seasonal[left_i]
        right = seasonal[right_i]
        if np.isfinite(left) and np.isfinite(right):
            plain[i] = right - left
    ok = data["ok_short"]
    gap = float(np.nanmax(np.abs(plain[ok] - data["y_short"][ok, 0].astype(np.float64))))
    if not np.isfinite(gap) or gap > 1e-6:
        raise SystemExit(f"SERIES_MISMATCH {gap}")
    bundle_dir = exp.ROOT / "model/61/bundle"
    joined, real = exp.assemble(exp.PARENT, bundle_dir, exp.ROOT / "history397-61-r2")
    risk = json.loads((exp.PARENT / "RISK.json").read_text())
    bounds = {
        **exp.segment_bounds(joined, risk, "2016-01-01", "2024-01-01"),
        **exp.segment_bounds(real, risk, "2024-01-01", "2025-01-01"),
    }
    effective = json.loads(step3.MODEL_CONFIG.read_text())
    out.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for horizon in plan["horizons"]:
        increment, admitted = labels_for(ts, seasonal, by_avail, bounds, horizon)
        mask = admitted & nonzero
        parts = {
            "fit": np.flatnonzero(data["pool"] & mask),
            "monitor": np.flatnonzero(data["monitor"] & mask),
            "hold": np.flatnonzero(data["holdout"] & mask),
        }
        ids = {key: set(value.tolist()) for key, value in parts.items()}
        for left, right in (("fit", "monitor"), ("fit", "hold"), ("monitor", "hold")):
            if ids[left] & ids[right]:
                raise SystemExit(f"SPLIT_OVERLAP {horizon} {left} {right}")
        counts = {key: int(len(value)) for key, value in parts.items()}
        declared = {
            "fit": int(plan["alone_pool"][str(horizon)]),
            "monitor": int(plan["alone_monitor"][str(horizon)]),
            "hold": int(plan["alone_holdout"][str(horizon)]),
        }
        keep = float(plan["row_keep_fraction"])
        for key in ("fit", "monitor", "hold"):
            if counts[key] > declared[key] or counts[key] < declared[key] * keep:
                raise SystemExit(f"ROW_DRIFT {horizon} {key} {counts[key]} {declared[key]}")
        if counts["fit"] < int(plan["min_pool"]) or counts["monitor"] < int(plan["min_monitor"]):
            raise SystemExit(f"ROWS_{horizon} {counts}")
        prepared[horizon] = {"y": increment[:, None], "parts": parts, "counts": counts}
        print(f"ROWS h{horizon} fit {counts['fit']} monitor {counts['monitor']} hold {counts['hold']}", flush=True)
    (out / "DECLARED.json").write_text(json.dumps({
        "schema": plan["schema"],
        "recipe": recipe["identity"],
        "series_h1_max_abs": gap,
        "holdout_2024": "not predicted until both champions exist",
        "rows": {str(horizon): item["counts"] for horizon, item in prepared.items()},
    }, indent=2) + "\n")
    print("H12_H36_DECLARED", flush=True)
    for horizon in plan["horizons"]:
        job_dir = out / f"h{horizon}"
        item = prepared[horizon]
        if not step3.job_done(job_dir):
            model = step3.fresh_model(recipe, effective)
            trainable = step4.encoder_trainable(model, "full_encoder")
            head = step4.make_head("linear", model.actor_net.encoder.features_dim, [horizon], float(horizon))
            step4.train_cell(
                model, head, trainable, windows, item["y"], item["parts"]["fit"], item["parts"]["monitor"],
                plan, job_dir, f"h{horizon}", [horizon], float(plan["encoder_lr"][horizon]),
            )
            del model, head
            torch.cuda.empty_cache()
    for horizon in plan["horizons"]:
        hold_path = out / f"h{horizon}" / "holdout_2024.json"
        if hold_path.exists():
            print(f"HOLDOUT_2024 h{horizon} already written, not predicted again", flush=True)
            continue
        item = prepared[horizon]
        bundle = torch.load(out / f"h{horizon}" / "champion.pt", map_location="cpu", weights_only=False)
        model = step3.fresh_model(recipe, effective)
        head = step4.make_head("linear", model.actor_net.encoder.features_dim, [horizon], float(horizon))
        head.load_state_dict(bundle["head"])
        step4.restore_encoder(model, bundle["encoder"])
        record = bundle["record"]
        mean = np.asarray(record["target_mean"], dtype=np.float64)
        std = np.asarray(record["target_std"], dtype=np.float64)
        idx = item["parts"]["hold"]
        measured = step4.predict_holdout(
            model, model.loss.actor_network_params, head, float(record["embedding_scale"]),
            windows, idx, item["y"][idx], mean, std, int(plan["batch"]),
        )
        payload = {
            "horizon": horizon,
            "definition": "MAE of the seasonal increment in price units. The naive is zero increment, seasonal-level persistence.",
            "split": "2024 holdout, one predict after both champions were restored",
            "rows": measured["rows"],
            "mae": measured["mae"],
            "naive": measured["naive"][0],
            "champion_epoch": record["best_epoch"],
            "validation_h2_2023_mae": record["champion_val_mae"],
            "validation_rows": record["validation_rows"],
        }
        hold_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"HOLDOUT_2024 h{horizon} mae {measured['mae']:.8f} rows {measured['rows']}", flush=True)
        del model, head
        torch.cuda.empty_cache()
    (out / "H12_H36_DONE").write_text("one 2024 predict per horizon\n")
    print("H12_H36_DONE", flush=True)


if __name__ == "__main__":
    main()
