#!/usr/bin/env python3
"""Phase-2 step 4. Eight cells per family, then one 2024 predict for the winner.

The grid is the one in MINI_WORK_PLAN.md: encoder learning rate 3e-5 or 3e-6,
last core block or the full actor encoder, linear head or the quadratic
trajectory head. Early stopping uses H2 2023. A cell replaces the step-3
refit only when its champion validation MAE is lower by more than min_delta.
2024 is predicted once per family, after that choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_phase2_step3 as step3  # noqa: E402

MODEL_CONFIG = Path("/var/tmp/sac399-series/model/61/bundle/MODEL_CONFIG.json")


class TrajectoryHead(nn.Module):
    def __init__(self, width: int, horizons: list[int], h_ref: float):
        super().__init__()
        self.map = nn.Linear(width, 3)
        self.out_features = len(horizons)
        ratios = torch.tensor(horizons, dtype=torch.float32) / float(h_ref)
        basis = torch.stack([torch.ones_like(ratios), ratios, ratios * ratios], dim=1)
        self.register_buffer("basis", basis)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.map(z) @ self.basis.T


def cells(plan: dict) -> list[dict]:
    rates = [float(value) for value in plan["encoder_lrs"]]
    if len(rates) != 2 or abs(rates[0] - 3e-5) > 1e-12 or abs(rates[1] - 3e-6) > 1e-12:
        raise SystemExit("PLAN_GRID")
    if int(plan["trajectory_coefficients"]) != 3:
        raise SystemExit("PLAN_HEAD")
    grid = []
    for rate in rates:
        for scope in ("last_block", "full_encoder"):
            for head in ("linear", "trajectory"):
                grid.append({"encoder_lr": rate, "scope": scope, "head": head, "name": f"lr{rate:.0e}_{scope}_{head}"})
    if len(grid) != 8:
        raise SystemExit("PLAN_GRID_COUNT")
    return grid


def encoder_trainable(model, scope: str) -> list:
    chosen = [(name, param) for name, param in model.loss.named_parameters() if "encoder" in name and "actor" in name]
    if not chosen:
        raise SystemExit("NO_ENCODER_PARAMS")
    if scope == "full_encoder":
        return chosen
    if scope != "last_block":
        raise SystemExit(f"SCOPE {scope}")
    last = [item for item in chosen if ".encoder.core.1." in item[0]]
    if len(last) != 4:
        raise SystemExit(f"LAST_BLOCK {len(last)}")
    return last


def make_head(kind: str, width: int, horizons: list[int], h_ref: float) -> nn.Module:
    torch.manual_seed(123)
    if kind == "linear":
        head = nn.Linear(width, len(horizons))
    elif kind == "trajectory":
        head = TrajectoryHead(width, horizons, h_ref)
    else:
        raise SystemExit(f"HEAD {kind}")
    linear = head if kind == "linear" else head.map
    nn.init.normal_(linear.weight, std=0.05)
    nn.init.zeros_(linear.bias)
    return head.to("cuda")


def embed_train(model, params, windows: torch.Tensor) -> torch.Tensor:
    encoder = model.actor_net.encoder
    encoder.train()
    with params.to_module(model.actor):
        return encoder({"features": windows})


def train_cell(model, head, trainable, windows, y_all, fit_idx, es_idx, plan, out: Path, job: str, horizons, encoder_lr: float, val_split_name: str = "H2 2023 monitor") -> dict:
    batch = int(plan["batch"])
    head_lr = float(plan["head_lr"])
    wd = float(plan["weight_decay"])
    min_delta = float(plan["min_delta"])
    es_patience = int(plan["es_patience"])
    es_start = int(plan["es_start_epoch"])
    lr_grace = int(plan["reduce_lr_grace"])
    lr_patience = int(plan["reduce_lr_patience"])
    lr_factor = float(plan["reduce_lr_factor"])
    lr_floor = float(plan["lr_floor"])
    max_epochs = int(plan["max_epochs"])
    y_fit = y_all[fit_idx]
    mean = y_fit.mean(axis=0).astype(np.float64)
    std = np.where(y_fit.std(axis=0) < 1e-8, 1.0, y_fit.std(axis=0)).astype(np.float64)
    y_fit_z = ((y_fit - mean) / std).astype(np.float32)
    params = model.loss.actor_network_params
    scale = step3.pool_scale(model, params, windows, fit_idx)
    for _, param in model.loss.named_parameters():
        param.requires_grad_(False)
    for _, param in trainable:
        param.requires_grad_(True)
    # Group 0 is the head. Group 1 is the unfrozen encoder. AdamW rejects a
    # custom "name" key, and the step-3 loop that wrote one LR into every
    # group must not be copied here.
    opt = torch.optim.AdamW(
        [
            {"params": list(head.parameters()), "lr": head_lr, "weight_decay": wd},
            {"params": [param for _, param in trainable], "lr": encoder_lr, "weight_decay": wd},
        ],
        lr=head_lr,
        weight_decay=wd,
    )
    if len(opt.param_groups) != 2:
        raise SystemExit("OPT_GROUPS")
    opt.param_groups[0]["lr"] = head_lr
    opt.param_groups[1]["lr"] = encoder_lr
    if abs(opt.param_groups[0]["lr"] - head_lr) > 1e-18 or abs(opt.param_groups[1]["lr"] - encoder_lr) > 1e-18:
        raise SystemExit("LR_GROUPS")
    log_path = out / "epochs.jsonl"
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
    best_train = None
    best_val_per = None
    rng = np.random.default_rng(20261123)
    y_es = y_all[es_idx]
    current_head_lr = head_lr
    current_enc_lr = encoder_lr
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(fit_idx))
        for start in range(0, len(fit_idx), batch):
            take = order[start : start + batch]
            x = torch.from_numpy(windows[fit_idx[take]]).to("cuda")
            target = torch.from_numpy(y_fit_z[take]).to("cuda")
            opt.zero_grad(set_to_none=True)
            pred = head(embed_train(model, params, x) / scale)
            loss = torch.mean(torch.abs(pred - target))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(take)
            seen += len(take)
        train_price = float((total / seen) * float(np.mean(std)))
        pred_m = step3.predict(model, params, head, scale, windows, es_idx, batch)
        val_price, val_per = step3.price_mae(pred_m, y_es, mean, std)
        if val_price < best - min_delta:
            best = val_price
            best_epoch = epoch
            wait = 0
            lr_wait = 0
            best_train = train_price
            best_val_per = list(val_per)
            best_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            best_enc = [(name, param.detach().cpu().clone()) for name, param in trainable]
        else:
            if epoch >= es_start:
                wait += 1
            if epoch >= lr_grace:
                lr_wait += 1
            if epoch >= lr_grace and lr_wait >= lr_patience:
                next_head = max(current_head_lr * lr_factor, lr_floor)
                next_enc = max(current_enc_lr * lr_factor, lr_floor)
                if next_head < current_head_lr - 1e-18 or next_enc < current_enc_lr - 1e-18:
                    current_head_lr = next_head
                    current_enc_lr = next_enc
                    reductions += 1
                    lr_wait = 0
                    opt.param_groups[0]["lr"] = current_head_lr
                    opt.param_groups[1]["lr"] = current_enc_lr
        rows = {"train": int(len(fit_idx)), "validation": int(len(es_idx))}
        if val_split_name == "H2 2023 monitor":
            rows["validation_h2_2023"] = int(len(es_idx))
        record = {
            "epoch": epoch,
            "train_mae": train_price,
            "train_mae_definition": "pooled z-space L1 times mean train-target std; not per horizon",
            "val_mae": val_price,
            "val_mae_definition": "mean across horizons of MAE of the seasonal increment on " + val_split_name,
            "val_mae_per_horizon": {str(h): val_per[i] for i, h in enumerate(horizons)},
            "lr": current_head_lr,
            "encoder_lr": current_enc_lr,
            "early_stopping": {"wait": wait, "patience": es_patience, "started": epoch >= es_start, "best_epoch": best_epoch, "min_delta": min_delta},
            "reduce_lr": {"wait": lr_wait, "patience": lr_patience, "grace": lr_grace, "reductions": reductions, "factor": lr_factor, "floor": lr_floor},
            "job": job,
            "rows": rows,
            "validation_split": val_split_name,
            "holdout_2024": "not predicted",
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        step3.write_status(out, record | {"state": "running"})
        print(
            f"{job} epoch {epoch} train {train_price:.6f} val {val_price:.6f} lr {current_head_lr:.3e} enc_lr {current_enc_lr:.3e} es {wait}/{es_patience} split {val_split_name}",
            flush=True,
        )
        if epoch >= es_start and wait >= es_patience:
            break
    if best_head is None or best_val_per is None:
        raise RuntimeError("NO_CHAMPION")
    done = {
        "state": "job_done",
        "job": job,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "champion_train_mae": best_train,
        "champion_val_mae": best,
        "champion_val_mae_per_horizon": {str(h): best_val_per[i] for i, h in enumerate(horizons)},
        "val_mae_split": val_split_name,
        "lr": current_head_lr,
        "encoder_lr": current_enc_lr,
        "early_stopping": {"wait": wait, "patience": es_patience, "best_epoch": best_epoch},
        "reduce_lr": {"reductions": reductions, "wait": lr_wait, "patience": lr_patience},
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "embedding_scale": scale,
        "train_rows": int(len(fit_idx)),
        "validation_rows": int(len(es_idx)),
        "unfrozen_parameters": int(sum(param.numel() for _, param in trainable)),
        "holdout_2024": "not predicted",
    }
    torch.save({"head": best_head, "encoder": best_enc, "record": done}, out / "champion.pt")
    step3.write_status(out, done)
    print(f"JOB_DONE {job} champion_epoch {best_epoch} val {best:.6f}", flush=True)
    return done


def restore_encoder(model, saved) -> None:
    bank = dict(saved)
    if not bank:
        raise SystemExit("ENCODER_RESTORE_EMPTY")
    found = set()
    with torch.no_grad():
        for name, param in model.loss.named_parameters():
            if name in bank:
                param.copy_(bank[name].to(param.device))
                found.add(name)
    missing = [name for name in bank if name not in found]
    if missing:
        raise SystemExit(f"ENCODER_RESTORE {missing[0]}")


def choose_winner(step3_val: float, ranked: list[dict], min_delta: float) -> dict:
    contenders = [row for row in ranked if step3_val - row["val"] > min_delta]
    if not contenders:
        return {"source": "step3_refit", "val": step3_val, "reason": "no cell beat the step-3 refit by more than min_delta"}
    contenders.sort(key=lambda row: row["val"])
    best_val = contenders[0]["val"]
    tied = [row for row in contenders if abs(row["val"] - best_val) <= min_delta]
    tied.sort(key=lambda row: (row["unfrozen"], row["encoder_lr"], 0 if row["head"] == "linear" else 1, row["name"]))
    winner = tied[0]
    winner["source"] = "step4_cell"
    winner["reason"] = "lowest H2 2023 champion MAE among cells that beat the step-3 refit by more than min_delta"
    return winner


def predict_holdout(model, params, head, scale, windows, idx, y, mean, std, batch: int) -> dict:
    pred = step3.predict(model, params, head, scale, windows, idx, batch)
    mae, per = step3.price_mae(pred, y, mean, std)
    naive = [float(np.mean(np.abs(y[:, i]))) for i in range(y.shape[1])]
    return {"mae": mae, "per": per, "naive": naive, "rows": int(len(idx))}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUIRED")
    plan = step3.load_plan(Path(sys.argv[sys.argv.index("--plan") + 1]))
    phase = Path(plan["out_dir"]).resolve()
    finished = Path(plan["finished_dir"]).resolve()
    if phase == finished:
        raise SystemExit("REFUSING_FINISHED_DIR")
    if (phase / "step4" / "STEP4_DONE").exists():
        print("STEP4_DONE", flush=True)
        return
    step3_root = phase / "step3"
    if not (step3_root / "STEP3_REFIT_OK").exists():
        raise SystemExit("STEP3_NOT_DONE")
    grid = cells(plan)
    data = {key: array for key, array in np.load(phase / "dataset.npz").items()}
    loaded = np.load(phase / "windows.npz")
    if not np.array_equal(loaded["ts"], data["ts"]):
        raise SystemExit("WINDOWS_TS")
    full_windows = loaded["windows"]
    names = json.loads((phase / "MANIFEST.json").read_text())["columns"]
    effective = json.loads(MODEL_CONFIG.read_text())
    out = phase / "step4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "DECLARED.json").write_text(json.dumps({
        "state": "declared",
        "step": 4,
        "cells": [cell["name"] for cell in grid],
        "holdout_2024": "one predict per family after the winner is chosen",
    }, indent=2) + "\n")
    print("STEP4_DECLARED", flush=True)
    families = {
        "short": (plan["short_horizons"], "ok_short", data["y_short"], float(plan["short_h_ref"])),
        "long": (plan["long_horizons"], "ok_long", data["y_long"], float(plan["long_h_ref"])),
    }
    for task, (horizons, ok_name, y_all, h_ref) in families.items():
        parts = step3.cuts(data["ts"].astype(np.int64), data, data[ok_name], plan)
        recipe = json.loads((step3_root / "mask" / f"RECIPE_{task}.json").read_text())
        selected = recipe["list"]["columns"]
        index = {name: i for i, name in enumerate(names)}
        windows = np.ascontiguousarray(full_windows[:, :, [index[name] for name in selected]])
        baseline = json.loads((step3_root / "refit" / task / "STATUS.json").read_text())
        step3_val = float(baseline["champion_val_mae"])
        ranked = []
        for cell in grid:
            job_dir = out / task / cell["name"]
            status_path = job_dir / "STATUS.json"
            if status_path.exists() and (job_dir / "champion.pt").exists() and json.loads(status_path.read_text()).get("state") == "job_done":
                done = json.loads(status_path.read_text())
            else:
                model = step3.fresh_model(recipe, effective)
                trainable = encoder_trainable(model, cell["scope"])
                head = make_head(cell["head"], model.actor_net.encoder.features_dim, horizons, h_ref)
                done = train_cell(
                    model, head, trainable, windows, y_all, parts["refit"], parts["monitor"],
                    plan, job_dir, f"{task}_{cell['name']}", horizons, float(cell["encoder_lr"]),
                )
                done["scope"] = cell["scope"]
                done["head_kind"] = cell["head"]
                done["declared_encoder_lr"] = cell["encoder_lr"]
                step3.write_status(job_dir, done)
                del model, head
                torch.cuda.empty_cache()
            ranked.append({
                "name": cell["name"],
                "val": float(done["champion_val_mae"]),
                "unfrozen": int(done["unfrozen_parameters"]),
                "encoder_lr": float(cell["encoder_lr"]),
                "head": cell["head"],
                "scope": cell["scope"],
                "best_epoch": done["best_epoch"],
                "per_horizon": done["champion_val_mae_per_horizon"],
            })
            print(f"CELL {task} {cell['name']} val {ranked[-1]['val']:.8f}", flush=True)
        winner = choose_winner(step3_val, ranked, float(plan["min_delta"]))
        (out / task / "GRID.json").write_text(json.dumps({"step3_val": step3_val, "cells": ranked, "winner": winner}, indent=2) + "\n")
        hold_path = out / task / "holdout_2024.json"
        if hold_path.exists():
            print(f"HOLDOUT_2024 {task} already written, not predicted again", flush=True)
            continue
        model = step3.fresh_model(recipe, effective)
        if winner["source"] == "step3_refit":
            bundle = torch.load(step3_root / "refit" / task / "champion.pt", map_location="cpu", weights_only=False)
            head = nn.Linear(model.actor_net.encoder.features_dim, len(horizons)).to("cuda")
            head.load_state_dict(bundle["head"])
            saved_enc = bundle["encoder"]
            record = bundle["record"]
        else:
            bundle = torch.load(out / task / winner["name"] / "champion.pt", map_location="cpu", weights_only=False)
            head = make_head(winner["head"], model.actor_net.encoder.features_dim, horizons, h_ref)
            head.load_state_dict(bundle["head"])
            saved_enc = bundle["encoder"]
            record = bundle["record"]
        if winner["source"] == "step4_cell" and winner["scope"] == "last_block" and len(saved_enc) != 4:
            raise SystemExit("LAST_BLOCK_RESTORE")
        restore_encoder(model, saved_enc)
        mean = np.asarray(record["target_mean"], dtype=np.float64)
        std = np.asarray(record["target_std"], dtype=np.float64)
        if len(parts["hold"]) < 200:
            raise SystemExit(f"ROWS_hold {task} {len(parts['hold'])}")
        measured = predict_holdout(
            model, model.loss.actor_network_params, head, float(record["embedding_scale"]),
            windows, parts["hold"], y_all[parts["hold"]], mean, std, int(plan["batch"]),
        )
        payload = {
            "family": task,
            "winner": winner,
            "definition": "MAE of the seasonal increment in price units. The naive is zero increment, seasonal-level persistence.",
            "split": "2024 holdout, one predict after champion restore",
            "rows": measured["rows"],
            "mae": measured["mae"],
            "mae_per_horizon": {str(h): measured["per"][i] for i, h in enumerate(horizons)},
            "naive_per_horizon": {str(h): measured["naive"][i] for i, h in enumerate(horizons)},
            "horizons": horizons,
        }
        hold_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"HOLDOUT_2024 {task} {winner['source']} mae {measured['mae']:.8f} rows {measured['rows']}", flush=True)
        del model, head
        torch.cuda.empty_cache()
    (out / "STEP4_DONE").write_text("one 2024 predict per family\n")
    print("STEP4_DONE", flush=True)


if __name__ == "__main__":
    main()
