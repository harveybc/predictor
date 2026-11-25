#!/usr/bin/env python3
"""Phase-2 step 3. Reads MINI_WORK_PLAN.md and does not look at 2024.

Preparatory seed-7 DoubleTCN, encoder frozen, linear head, fit before
2022-07-01, early stopping on the inner cut through 2023-01-01. Then
TwoPassGreedy on blocks. Score is the mean over horizons of
(naive MAE - model MAE) on [2023-01-01, 2023-07-01). Deletion may accept
an exact tie. Reinsertion must improve. The mask zeros channels.

After the columns are sealed, one refit per family on the full train pool,
early stopping on H2 2023. That refit does not predict 2024.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

RUNTIME = Path("/var/tmp/sac399-series/runtime")
SELECTOR = Path("/var/tmp/retsu-phase1-causal-p2/vendor")
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(SELECTOR))

from app.rl_subset import seal_recipe, subset_recipe  # noqa: E402
from app.structural_refit import build_model  # noqa: E402
from feature_selector.search import TwoPassGreedy  # noqa: E402

MODEL_CONFIG = Path("/var/tmp/sac399-series/model/61/bundle/MODEL_CONFIG.json")


def load_plan(path: Path) -> dict:
    text = path.read_text()
    marker = "```phase2-plan\n"
    start = text.index(marker) + len(marker)
    plan = json.loads(text[start : text.index("\n```", start)])
    if plan.get("schema") != "retsu_phase2_mini_plan.v1":
        raise SystemExit("PLAN_SCHEMA")
    if not plan.get("remove_on_tie_deletion_only"):
        raise SystemExit("PLAN_TIE")
    return plan


def ns(stamp: str) -> int:
    return int(pd.Timestamp(stamp).value)


def write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATUS.json").write_text(json.dumps(payload, indent=2) + "\n")


def encoder_parameters(model):
    chosen = [(name, param) for name, param in model.loss.named_parameters() if "encoder" in name and "actor" in name]
    if not chosen:
        raise RuntimeError("NO_ENCODER_PARAMS")
    return chosen


def fresh_model(recipe, effective):
    model = build_model(recipe, effective, seed=7, device="cuda")
    width = int(model.actor_net.encoder.features_dim)
    if width != 13 * 16:
        raise SystemExit(f"FEATURES_DIM {width}")
    return model


def embed(model, params, windows: torch.Tensor) -> torch.Tensor:
    encoder = model.actor_net.encoder
    encoder.eval()
    with torch.no_grad(), params.to_module(model.actor):
        return encoder({"features": windows}).detach()


def price_mae(pred_z: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[float, list[float]]:
    per = []
    for i in range(y.shape[1]):
        hat = pred_z[:, i] * std[i] + mean[i]
        per.append(float(np.mean(np.abs(hat - y[:, i]))))
    return float(np.mean(per)), per


def pool_scale(model, params, windows, pool_idx) -> float:
    squares = 0.0
    count = 0
    width = None
    with torch.no_grad():
        for start in range(0, len(pool_idx), 256):
            take = pool_idx[start : start + 256]
            z = embed(model, params, torch.from_numpy(windows[take]).to("cuda")).float()
            squares += float(torch.sum(z * z))
            count += int(z.numel())
            width = int(z.shape[1])
    if width != 13 * 16 or count == 0:
        raise SystemExit("SCALE")
    return float(np.sqrt(squares / count * width))


def predict(model, params, head, scale, windows, idx, batch: int) -> np.ndarray:
    head.eval()
    out = np.empty((len(idx), head.out_features), np.float32)
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            take = idx[start : start + batch]
            z = embed(model, params, torch.from_numpy(windows[take]).to("cuda"))
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


def train_frozen(model, horizons: list, windows, y_all, fit_idx, es_idx, plan: dict, out: Path, job: str, val_split: str) -> dict:
    batch = int(plan["batch"])
    lr0 = float(plan["head_lr"])
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
    scale = pool_scale(model, params, windows, fit_idx)
    head = nn.Linear(model.actor_net.encoder.features_dim, len(horizons)).to("cuda")
    torch.manual_seed(123)
    nn.init.normal_(head.weight, std=0.05)
    nn.init.zeros_(head.bias)
    chosen = encoder_parameters(model)
    for _, param in model.loss.named_parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr0, weight_decay=wd)
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
    best_opt = None
    best_train = None
    best_val_per = None
    lr = lr0
    rng = np.random.default_rng(20261123)
    y_es = y_all[es_idx]
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
            pred = head(embed(model, params, x) / scale)
            loss = torch.mean(torch.abs(pred - target))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(take)
            seen += len(take)
        train_price = float((total / seen) * float(np.mean(std)))
        pred_m = predict(model, params, head, scale, windows, es_idx, batch)
        val_price, val_per = price_mae(pred_m, y_es, mean, std)
        improved = val_price < best - min_delta
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
            if epoch >= es_start:
                wait += 1
            if epoch >= lr_grace:
                lr_wait += 1
            if epoch >= lr_grace and lr_wait >= lr_patience and lr * lr_factor >= lr_floor - 1e-15:
                lr = max(lr * lr_factor, lr_floor)
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
            "val_mae_definition": f"mean across horizons of MAE of the seasonal increment on {val_split}",
            "val_mae_per_horizon": {str(h): val_per[i] for i, h in enumerate(horizons)},
            "lr": lr,
            "early_stopping": {"wait": wait, "patience": es_patience, "started": epoch >= es_start, "best_epoch": best_epoch, "min_delta": min_delta},
            "reduce_lr": {"wait": lr_wait, "patience": lr_patience, "grace": lr_grace, "reductions": reductions, "factor": lr_factor, "floor": lr_floor},
            "job": job,
            "rows": {"train": int(len(fit_idx)), "validation": int(len(es_idx))},
            "holdout_2024": "not predicted",
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        write_status(out, record | {"state": "running"})
        print(f"{job} epoch {epoch} train {train_price:.6f} val {val_price:.6f} lr {lr:.3e} es {wait}/{es_patience}", flush=True)
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
        "champion_train_mae_definition": "pooled z-space L1 times mean train-target std at the champion epoch; not per horizon",
        "champion_val_mae": best,
        "champion_val_mae_per_horizon": {str(h): best_val_per[i] for i, h in enumerate(horizons)},
        "val_mae_split": val_split,
        "lr": lr,
        "early_stopping": {"wait": wait, "patience": es_patience, "best_epoch": best_epoch},
        "reduce_lr": {"reductions": reductions, "wait": lr_wait, "patience": lr_patience},
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "embedding_scale": scale,
        "train_rows": int(len(fit_idx)),
        "validation_rows": int(len(es_idx)),
        "holdout_2024": "not predicted",
        "encoder": "seed-7 fresh build, not updated; model.pt not loaded",
    }
    torch.save({"head": best_head, "encoder": best_enc, "optimizer": best_opt, "lr": lr, "record": done}, out / "champion.pt")
    write_status(out, done)
    print(f"JOB_DONE {job} champion_epoch {best_epoch} val {best:.6f}", flush=True)
    return done


def restore(model, bundle):
    chosen = encoder_parameters(model)
    saved = dict(bundle["encoder"])
    with torch.no_grad():
        for name, param in chosen:
            param.copy_(saved[name].to(param.device))
    head = nn.Linear(model.actor_net.encoder.features_dim, bundle["head"]["weight"].shape[0]).to("cuda")
    head.load_state_dict(bundle["head"])
    return head


def mask_family(model, windows, names, groups, y_all, score_idx, bundle, plan, task: str) -> dict:
    head = restore(model, bundle)
    params = model.loss.actor_network_params
    scale = float(bundle["record"]["embedding_scale"])
    mean = np.asarray(bundle["record"]["target_mean"], dtype=np.float64)
    std = np.asarray(bundle["record"]["target_std"], dtype=np.float64)
    y = y_all[score_idx]
    naive = [float(np.mean(np.abs(y[:, i]))) for i in range(y.shape[1])]
    index = {name: i for i, name in enumerate(names)}

    def evaluate(columns):
        keep = set(columns)
        dropped = [index[name] for name in names if name not in keep]
        view = np.array(windows[score_idx], copy=True)
        if dropped:
            view[:, :, dropped] = 0.0
        pred_z = predict_local(model, params, head, scale, view, int(plan["batch"]))
        if not np.isfinite(pred_z).all():
            return {"eligible": False, "score": None, "gap": None}
        mae = [float(np.mean(np.abs(pred_z[:, i] * std[i] + mean[i] - y[:, i]))) for i in range(y.shape[1])]
        score = float(np.mean([base - err for base, err in zip(naive, mae)]))
        return {"eligible": True, "score": score, "gap": score, "mae": mae, "naive": naive}

    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    blocks = [list(group["columns"]) for group in groups]
    dataset = {
        "schema": "feature_selector.dataset.v1",
        "scope": "development",
        "features": list(names),
        "groups": [{"name": group["name"], "columns": list(group["columns"]), "extractor": group["kind"]} for group in groups],
        "blocks": blocks,
        "support_hash": digest,
        "scaler_hash": hashlib.sha256(b"phase2-expanding-z-identity").hexdigest(),
        "representation": {"encoder": "DoubleTCNExtractor", "frozen": True},
        "splits": {
            "train": ["2016-01-01", "2023-01-01"],
            "selection": ["2023-01-01", "2023-07-01"],
            "test_start": "2023-07-01",
        },
        "constraints": {
            "agent_state": "none",
            "scaling": "fixed_train",
            "clipping": False,
            "fit_scope": "train_only",
            "test_access": False,
        },
    }
    result = TwoPassGreedy().select(
        dataset,
        evaluate,
        {"min_features": int(plan["min_features"]), "tolerance": float(plan["tolerance"]), "remove_on_tie": True},
    )
    if result.get("termination") != "TWO_PASSES_COMPLETE" or not result.get("selected"):
        raise SystemExit(f"MASK_{task}_{result.get('termination')}")
    print(
        f"MASK {task} {result['termination']} selected {len(result['selected'])} of {len(names)} evals {result['evaluations']}",
        flush=True,
    )
    return result


def predict_local(model, params, head, scale, rows: np.ndarray, batch: int) -> np.ndarray:
    head.eval()
    out = np.empty((len(rows), head.out_features), np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            x = torch.from_numpy(np.ascontiguousarray(rows[start : start + batch])).to("cuda")
            pred = head(embed(model, params, x) / scale)
            out[start : start + len(x)] = pred.float().cpu().numpy()
    return out


def seal_family(parent: dict, selected: list[str], task: str, plan: dict) -> dict:
    ordered = [name for name in parent["list"]["columns"] if name in set(selected)]
    if ordered != list(selected):
        raise SystemExit("MASK_ORDER")
    rec = subset_recipe(parent, ordered)
    rec["split"]["status"] = "phase2_columns_sealed_inner_train_mask"
    rec["split"]["support_status"] = "two_pass_greedy_mae_versus_naive_on_inner_train"
    rec["split"]["arm"] = f"phase2_mask_{task}"
    rec["split"]["selection_period"] = None
    rec["provenance"] = {
        "parent_recipe_identity": parent["identity"],
        "derivation": "phase2_two_pass_greedy_inner_train_seasonal_increment_mae",
        "score_window": [plan["mask_score_from"], plan["mask_score_until"]],
        "not_2024": True,
        "family": task,
    }
    rec["scope"] = "Columns kept by two_pass_greedy. Score cut is inside train, before 2023-07-01. 2024 was not a selection period."
    sealed = seal_recipe(rec)
    if sealed["identity"] in set(plan["forbidden_recipe"]) or sealed["identity"] == plan["parent_recipe"]:
        raise SystemExit("SEALED_A_CLOSED_RECIPE")
    return sealed


def slice_windows(windows: np.ndarray, names: list[str], selected: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    take = [index[name] for name in selected]
    return np.ascontiguousarray(windows[:, :, take])


def cuts(ts: np.ndarray, data: dict, ok: np.ndarray, plan: dict) -> dict[str, np.ndarray]:
    fit_before = ns(plan["frozen_fit_before"])
    es_from = ns(plan["frozen_es_from"])
    es_until = ns(plan["frozen_es_until"])
    score_from = ns(plan["mask_score_from"])
    score_until = ns(plan["mask_score_until"])
    monitor_from = ns(plan["monitor_from"])
    monitor_until = ns(plan["monitor_until"])
    test_from = ns(plan["test_from"])
    pool = data["pool"] & ok & ~data["monitor"] & ~data["holdout"]
    monitor = data["monitor"] & ok & ~data["holdout"] & (ts >= monitor_from) & (ts < monitor_until)
    hold = data["holdout"] & ok & (ts >= test_from)
    parts = {
        "fit": np.flatnonzero(pool & (ts < fit_before)),
        "es": np.flatnonzero(pool & (ts >= es_from) & (ts < es_until)),
        "score": np.flatnonzero(pool & (ts >= score_from) & (ts < score_until)),
        "refit": np.flatnonzero(pool & (ts < monitor_from)),
        "monitor": np.flatnonzero(monitor),
        "hold": np.flatnonzero(hold),
    }
    ids = {key: set(value.tolist()) for key, value in parts.items()}
    # The refit pool contains fit, the inner early-stopping cut and the score cut.
    # Those three are disjoint from each other and from H2 2023 and 2024.
    for left, right in (
        ("fit", "es"),
        ("fit", "score"),
        ("fit", "monitor"),
        ("fit", "hold"),
        ("es", "score"),
        ("es", "monitor"),
        ("es", "hold"),
        ("score", "monitor"),
        ("score", "hold"),
        ("monitor", "hold"),
        ("refit", "monitor"),
        ("refit", "hold"),
    ):
        if ids[left] & ids[right]:
            raise SystemExit(f"SPLIT_OVERLAP {left} {right}")
    if (ids["fit"] | ids["es"] | ids["score"]) - ids["refit"]:
        raise SystemExit("REFIT_MISSING_INNER")
    for key in ("fit", "es", "score", "refit", "monitor"):
        if len(parts[key]) < 200:
            raise SystemExit(f"ROWS_{key} {len(parts[key])}")
    return parts


def job_done(path: Path) -> bool:
    status = path / "STATUS.json"
    return status.exists() and (path / "champion.pt").exists() and json.loads(status.read_text()).get("state") == "job_done"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUIRED")
    plan = load_plan(Path(sys.argv[sys.argv.index("--plan") + 1]))
    phase = Path(plan["out_dir"]).resolve()
    finished = Path(plan["finished_dir"]).resolve()
    if phase == finished or str(phase).startswith(str(finished) + os.sep):
        raise SystemExit("REFUSING_FINISHED_DIR")
    protocol = json.loads((phase / "protocol.json").read_text())
    if protocol.get("model_mae_2024") != "not computed yet; one predict after the final champion restore":
        raise SystemExit("PROTOCOL_ALREADY_HAS_TEST_MAE")
    if protocol.get("identity") in set(plan["forbidden_recipe"]):
        raise SystemExit("FORBIDDEN_RECIPE")
    parent = json.loads((phase / "RECIPE.json").read_text())
    if parent["identity"] != protocol["identity"] or parent["list"]["n"] <= 61:
        raise SystemExit("RECIPE_IDENTITY")
    manifest = json.loads((phase / "MANIFEST.json").read_text())
    names = list(parent["list"]["columns"])
    if names != manifest["columns"]:
        raise SystemExit("COLUMN_ORDER")
    data = {key: array for key, array in np.load(phase / "dataset.npz").items()}
    loaded = np.load(phase / "windows.npz")
    if not np.array_equal(loaded["ts"], data["ts"]):
        raise SystemExit("WINDOWS_TS")
    windows = loaded["windows"]
    if windows.shape[2] != len(names) or windows.shape[1] != 13:
        raise SystemExit(f"WINDOWS_SHAPE {windows.shape}")
    if np.any(data["seasonal_origin"] != 0):
        raise SystemExit("ORIGIN_NOT_ZERO")
    effective = json.loads(MODEL_CONFIG.read_text())
    out = phase / "step3"
    out.mkdir(parents=True, exist_ok=True)
    declared = {
        "state": "declared",
        "step": 3,
        "recipe_identity": parent["identity"],
        "n_columns": len(names),
        "holdout_2024": "not predicted in this step",
        "score": "mean over horizons of (naive MAE - model MAE) on the inner train cut",
    }
    (out / "DECLARED.json").write_text(json.dumps(declared, indent=2) + "\n")
    print("STEP3_DECLARED", parent["identity"], flush=True)
    families = {
        "short": (plan["short_horizons"], "ok_short", data["y_short"], "inner train [2022-07-01, 2023-01-01)"),
        "long": (plan["long_horizons"], "ok_long", data["y_long"], "inner train [2022-07-01, 2023-01-01)"),
    }
    for task, (horizons, ok_name, y_all, val_split) in families.items():
        parts = cuts(data["ts"].astype(np.int64), data, data[ok_name], plan)
        print(
            f"ROWS {task} fit {len(parts['fit'])} es {len(parts['es'])} score {len(parts['score'])} refit {len(parts['refit'])} monitor {len(parts['monitor'])}",
            flush=True,
        )
        prep = out / "prep" / task
        if not job_done(prep):
            model = fresh_model(parent, effective)
            train_frozen(model, horizons, windows, y_all, parts["fit"], parts["es"], plan, prep, f"prep_{task}", val_split)
            del model
            torch.cuda.empty_cache()
        bundle = torch.load(prep / "champion.pt", map_location="cpu", weights_only=False)
        mask_path = out / "mask" / f"{task}.json"
        if not mask_path.exists():
            model = fresh_model(parent, effective)
            result = mask_family(model, windows, names, manifest["groups"], y_all, parts["score"], bundle, plan, task)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            kept = {key: result[key] for key in ("selected", "termination", "evaluations", "trace") if key in result}
            kept["naive_definition"] = "zero increment, seasonal-level persistence, on the inner score cut"
            kept["score_definition"] = "mean over horizons of (naive MAE - MAE of the seasonal increment)"
            mask_path.write_text(json.dumps(kept) + "\n")
            del model
            torch.cuda.empty_cache()
        chosen = json.loads(mask_path.read_text())["selected"]
        sealed = seal_family(parent, chosen, task, plan)
        (out / "mask" / f"RECIPE_{task}.json").write_text(json.dumps(sealed, indent=2) + "\n")
        refit = out / "refit" / task
        if not job_done(refit):
            model = fresh_model(sealed, effective)
            taken = slice_windows(windows, names, sealed["list"]["columns"])
            train_frozen(
                model,
                horizons,
                taken,
                y_all,
                parts["refit"],
                parts["monitor"],
                plan,
                refit,
                f"refit_{task}",
                "H2 2023 monitor",
            )
            del model
            torch.cuda.empty_cache()
    (out / "STEP3_REFIT_OK").write_text("refit champions restored in their job dirs; 2024 not predicted\n")
    print("STEP3_REFIT_OK", flush=True)


if __name__ == "__main__":
    main()
