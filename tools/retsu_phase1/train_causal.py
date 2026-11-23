#!/usr/bin/env python
"""Seasonal-increment head on the causal rebuild.

Fresh seed-7 temporal encoder. The pretrained 61-column model.pt is not
loaded. Frozen means the encoder is not updated. Tuned means it is.

Declared before the run: minimize monitor MAE of the seasonal increment,
min_delta 1e-8, AdamW lr 3e-4, weight decay 0.004, ReduceLR grace 20,
patience 20, factor 0.2, floor 4.8e-7, EarlyStopping patience 80 from epoch
100, max 2000. Champion restore keeps the reduced learning rate. 2024 is one
predict after that restore. No Prediction columns.
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

from app.structural_refit import build_model  # noqa: E402

PHASE = Path(os.environ.get("RETSU_PHASE", "/var/tmp/retsu-phase1-causal"))
MODEL_CONFIG = Path("/var/tmp/sac399-series/model/61/bundle/MODEL_CONFIG.json")
FORBIDDEN = {"recipe_b8fa85625fe025a7", "recipe_b5fb7bcdfa1dbec8"}
LR = 3e-4
LR_FLOOR = 4.8e-7
LR_FACTOR = 0.2
LR_PATIENCE = 20
LR_GRACE = 20
ES_PATIENCE = 80
ES_START = 100
MIN_DELTA = 1e-8
MAX_EPOCHS = 2000
BATCH = 256
WD = 0.004


def encoder_parameters(model):
    chosen = [(name, param) for name, param in model.loss.named_parameters() if "encoder" in name and "actor" in name]
    if not chosen:
        names = [name for name, _ in model.loss.named_parameters()]
        raise RuntimeError("NO_ENCODER_PARAMS " + " ".join(names[:40]))
    return chosen


def price_mae(pred_z: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[float, list[float]]:
    per = []
    for i in range(y.shape[1]):
        hat = pred_z[:, i] * std[i] + mean[i]
        per.append(float(np.mean(np.abs(hat - y[:, i]))))
    return float(np.mean(per)), per


def batches(n: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n, BATCH):
        yield order[start : start + BATCH]


def embed(model, params, windows: torch.Tensor, freeze: bool) -> torch.Tensor:
    encoder = model.actor_net.encoder
    if freeze:
        encoder.eval()
        with torch.no_grad(), params.to_module(model.actor):
            z = encoder({"features": windows})
        return z.detach()
    encoder.train()
    with params.to_module(model.actor):
        return encoder({"features": windows})


def run_split(model, params, head, scale, windows, idx) -> np.ndarray:
    head.eval()
    out = np.empty((len(idx), head.out_features), np.float32)
    with torch.no_grad():
        for start in range(0, len(idx), BATCH):
            take = idx[start : start + BATCH]
            x = torch.from_numpy(windows[take]).to("cuda")
            z = embed(model, params, x, freeze=True)
            pred = head(z / scale)
            out[start : start + len(take)] = pred.float().cpu().numpy()
    return out


def tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_cpu(item) for item in value]
    return value


def tree_cuda(value):
    if torch.is_tensor(value):
        return value.to("cuda")
    if isinstance(value, dict):
        return {key: tree_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_cuda(item) for item in value]
    return value


def write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATUS.json").write_text(json.dumps(payload, indent=2) + "\n")


def fresh_model(recipe, effective):
    model = build_model(recipe, effective, seed=7, device="cuda")
    width = int(model.actor_net.encoder.features_dim)
    if width != 13 * 16:
        raise SystemExit(f"FEATURES_DIM {width}")
    return model


def pool_scale(model, params, windows, pool_idx) -> float:
    squares = 0.0
    count = 0
    width = None
    with torch.no_grad():
        for start in range(0, len(pool_idx), BATCH):
            take = pool_idx[start : start + BATCH]
            x = torch.from_numpy(windows[take]).to("cuda")
            z = embed(model, params, x, freeze=True).float()
            squares += float(torch.sum(z * z))
            count += int(z.numel())
            width = int(z.shape[1])
    if width != 13 * 16 or count == 0:
        raise SystemExit("SCALE")
    return float(np.sqrt(squares / count * width))


def one_job(model, mode: str, task: str, data: dict, protocol: dict, windows: np.ndarray) -> None:
    freeze = mode == "frozen"
    horizons = protocol["horizons_short"] if task == "short" else protocol["horizons_long"]
    ok = data["ok_short"] if task == "short" else data["ok_long"]
    y_all = data["y_short"] if task == "short" else data["y_long"]
    pool = np.flatnonzero(data["pool"] & ok)
    monitor = np.flatnonzero(data["monitor"] & ok)
    hold = np.flatnonzero(data["holdout"] & ok)
    if len(pool) < 1575 or len(monitor) < 200 or len(hold) < 200:
        raise SystemExit(f"ROWS {task} {len(pool)} {len(monitor)} {len(hold)}")
    y_train = y_all[pool]
    mean = y_train.mean(axis=0).astype(np.float64)
    std = np.where(y_train.std(axis=0) < 1e-8, 1.0, y_train.std(axis=0)).astype(np.float64)
    y_train_z = ((y_train - mean) / std).astype(np.float32)
    params = model.loss.actor_network_params
    scale = pool_scale(model, params, windows, pool)
    head = nn.Linear(model.actor_net.encoder.features_dim, len(horizons)).to("cuda")
    torch.manual_seed(123)
    nn.init.normal_(head.weight, std=0.05)
    nn.init.zeros_(head.bias)
    chosen = encoder_parameters(model)
    for _, param in model.loss.named_parameters():
        param.requires_grad_(False)
    if not freeze:
        for _, param in chosen:
            param.requires_grad_(True)
    groups = [{"params": head.parameters(), "weight_decay": WD}]
    if not freeze:
        groups.append({"params": [param for _, param in chosen], "weight_decay": WD})
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=WD)
    out = Path(os.environ["RETSU_OUT"])
    job = f"{mode}_{task}"
    log_path = out / "jobs" / job / "epochs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    best = np.inf
    best_epoch = 0
    wait = 0
    lr_wait = 0
    reductions = 0
    best_head = None
    best_enc = None
    best_opt = None
    best_train = None
    best_val_per = None
    lr = LR
    rng = np.random.default_rng(20261123)
    y_monitor = y_all[monitor]
    train_price = None
    val_price = None
    val_per = None
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        if not freeze:
            model.actor_net.encoder.train()
        total = 0.0
        seen = 0
        for take in batches(len(pool), rng):
            x = torch.from_numpy(windows[pool[take]]).to("cuda")
            target = torch.from_numpy(y_train_z[take]).to("cuda")
            opt.zero_grad(set_to_none=True)
            pred = head(embed(model, params, x, freeze) / scale)
            loss = torch.mean(torch.abs(pred - target))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(take)
            seen += len(take)
        train_z = total / seen
        pred_m = run_split(model, params, head, scale, windows, monitor)
        val_price, val_per = price_mae(pred_m, y_monitor, mean, std)
        # Pooled z-L1 times the mean train std. Not a per-horizon train MAE.
        train_price = float(train_z * float(np.mean(std)))
        improved = val_price < best - MIN_DELTA
        if improved:
            best = val_price
            best_epoch = epoch
            wait = 0
            lr_wait = 0
            best_train = train_price
            best_val_per = list(val_per)
            best_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            best_enc = [(name, param.detach().cpu().clone()) for name, param in chosen]
            best_opt = tree_cpu(opt.state_dict())
        else:
            if epoch >= ES_START:
                wait += 1
            if epoch >= LR_GRACE:
                lr_wait += 1
            if epoch >= LR_GRACE and lr_wait >= LR_PATIENCE and lr * LR_FACTOR >= LR_FLOOR - 1e-15:
                lr = max(lr * LR_FACTOR, LR_FLOOR)
                reductions += 1
                lr_wait = 0
                for group in opt.param_groups:
                    group["lr"] = lr
        for group in opt.param_groups:
            group["lr"] = lr
        record = {
            "epoch": epoch,
            "train_mae": train_price,
            "train_mae_definition": "pooled z-space L1 times mean train-target std; not per horizon",
            "val_mae": val_price,
            "val_mae_definition": "mean across horizons of MAE of the seasonal increment on H2 2023",
            "val_mae_per_horizon": {str(h): val_per[i] for i, h in enumerate(horizons)},
            "lr": lr,
            "early_stopping": {
                "wait": wait,
                "patience": ES_PATIENCE,
                "started": epoch >= ES_START,
                "best_epoch": best_epoch,
                "min_delta": MIN_DELTA,
            },
            "reduce_lr": {
                "wait": lr_wait,
                "patience": LR_PATIENCE,
                "grace": LR_GRACE,
                "reductions": reductions,
                "factor": LR_FACTOR,
                "floor": LR_FLOOR,
            },
            "job": job,
            "mode": mode,
            "task": task,
            "rows": {"train": int(len(pool)), "validation_h2_2023": int(len(monitor))},
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        write_status(out, record | {"state": "running", "stage": "train"})
        print(
            f"{job} epoch {epoch} train {train_price:.6f} val {val_price:.6f} lr {lr:.3e} es {wait}",
            flush=True,
        )
        if epoch >= ES_START and wait >= ES_PATIENCE:
            break
    if best_head is None or best_val_per is None:
        raise RuntimeError("NO_CHAMPION")
    head.load_state_dict(best_head)
    head.to("cuda")
    with torch.no_grad():
        for name, param in chosen:
            saved = dict(best_enc)[name]
            param.copy_(saved.to(param.device))
    opt.load_state_dict(tree_cuda(best_opt))
    for group in opt.param_groups:
        group["lr"] = lr
    pred_h = run_split(model, params, head, scale, windows, hold)
    hold_price, hold_per = price_mae(pred_h, y_all[hold], mean, std)
    record = {
        "state": "job_done",
        "stage": "train",
        "job": job,
        "mode": mode,
        "task": task,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "train_mae": train_price,
        "train_mae_definition": "pooled z-space L1 times mean train-target std at the last epoch; not per horizon",
        "champion_train_mae": best_train,
        "champion_train_mae_definition": "pooled z-space L1 times mean train-target std at the champion epoch; not per horizon",
        "val_mae": val_price,
        "best_val_mae": best,
        "champion_val_mae": best,
        "champion_val_mae_per_horizon": {str(h): best_val_per[i] for i, h in enumerate(horizons)},
        "val_mae_split": "H2 2023 monitor, champion epoch",
        "lr": lr,
        "early_stopping": {"wait": wait, "patience": ES_PATIENCE, "best_epoch": best_epoch},
        "reduce_lr": {"reductions": reductions, "wait": lr_wait, "patience": LR_PATIENCE},
        "holdout_2024_mae": hold_price,
        "holdout_2024_mae_definition": "MAE of the seasonal increment, one predict after champion restore",
        "holdout_2024_mae_per_horizon": {str(h): hold_per[i] for i, h in enumerate(horizons)},
        "holdout_rows": int(len(hold)),
        "monitor_rows": int(len(monitor)),
        "train_rows": int(len(pool)),
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "embedding_scale": scale,
        "freeze": freeze,
        "encoder": "seed-7 fresh build; model.pt not loaded",
        "recipe_identity": protocol["identity"],
    }
    torch.save({"head": best_head, "encoder": best_enc, "lr": lr, "record": record}, log_path.parent / "champion.pt")
    (log_path.parent / "holdout.json").write_text(json.dumps(record, indent=2) + "\n")
    write_status(out, record)
    print("JOB_DONE", job, "hold", round(hold_price, 6), flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUIRED")
    mode = os.environ["RETSU_MODE"]
    if mode not in ("frozen", "tuned"):
        raise SystemExit("RETSU_MODE")
    torch.backends.cudnn.benchmark = False
    protocol = json.loads((PHASE / "protocol.json").read_text())
    if protocol.get("model_mae_2024") != "not computed yet; one predict after champion restore":
        raise SystemExit("PROTOCOL_ALREADY_HAS_TEST_MAE")
    recipe = json.loads((PHASE / "RECIPE.json").read_text())
    if recipe["identity"] != protocol["identity"] or recipe["identity"] in FORBIDDEN:
        raise SystemExit("RECIPE_IDENTITY")
    if recipe["list"]["n"] <= 61:
        raise SystemExit("RECIPE_STILL_61")
    effective = json.loads(MODEL_CONFIG.read_text())
    dataset_path = Path(os.environ.get("RETSU_DATASET", str(PHASE / "dataset.npz")))
    saved = np.load(dataset_path)
    data = {key: saved[key] for key in saved.files}
    windows_path = Path(os.environ.get("RETSU_WINDOWS", str(PHASE / "windows.npz")))
    loaded = np.load(windows_path)
    if not np.array_equal(loaded["ts"], data["ts"]):
        raise SystemExit("WINDOWS_TS")
    frame = loaded["windows"]
    if frame.ndim != 3 or frame.shape[1] != 13 or frame.shape[2] != recipe["list"]["n"]:
        raise SystemExit(f"WINDOWS_SHAPE {frame.shape}")
    declared = {
        "state": "declared",
        "mode": mode,
        "recipe_identity": recipe["identity"],
        "n_columns": recipe["list"]["n"],
        "protocol": protocol["champion_protocol"],
        "naive": protocol["naive"],
        "model_mae_2024": "not computed yet",
    }
    write_status(Path(os.environ["RETSU_OUT"]), declared)
    (Path(os.environ["RETSU_OUT"]) / "DECLARED.json").write_text(json.dumps(declared, indent=2) + "\n")
    for task in ("short", "long"):
        model = fresh_model(recipe, effective)
        one_job(model, mode, task, data, protocol, frame)
        del model
        torch.cuda.empty_cache()
    print("CAUSAL_QUEUE_OK", mode, flush=True)


if __name__ == "__main__":
    main()
