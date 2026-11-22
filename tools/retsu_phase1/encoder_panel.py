#!/usr/bin/env python
"""Frozen-encoder panel plus a seed-7 encoder that never loads model.pt.

CPU only. Does not write STATUS.json. The pretrained embeddings stay in
dataset.npz. The fresh encoder is written to dataset_random_encoder.npz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import export_dataset as ex
from app.structural_refit import build_model

PHASE = Path("/var/tmp/retsu-phase1")
FOLLOW = PHASE / "followup"
BUNDLE = ex.ROOT / "model/61/bundle"


def summarize(tensor: torch.Tensor) -> dict:
    values = tensor.detach().float().cpu().numpy()
    return {
        "shape": list(values.shape),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "absmax": float(np.max(np.abs(values))),
        "l2": float(np.linalg.norm(values)),
        "finite_fraction": float(np.isfinite(values).mean()),
    }


def pick(avail: np.ndarray, n_train: int, n_monitor: int, n_hold: int) -> np.ndarray:
    train = np.flatnonzero(avail < ex.MONITOR_START)
    monitor = np.flatnonzero((avail >= ex.MONITOR_START) & (avail < ex.HOLDOUT_START))
    hold = np.flatnonzero(avail >= ex.HOLDOUT_START)

    def take(idx: np.ndarray, n: int) -> np.ndarray:
        if idx.size <= n:
            return idx
        choice = np.linspace(0, idx.size - 1, n).astype(np.int64)
        return idx[choice]

    return np.concatenate([take(train, n_train), take(monitor, n_monitor), take(hold, n_hold)])


def module_norms(encoder) -> list[dict]:
    rows = []
    for name, parameter in encoder.named_parameters():
        values = parameter.detach().float().cpu().numpy()
        rows.append({
            "name": name,
            "kind": "parameter",
            "numel": int(values.size),
            "l2": float(np.linalg.norm(values)),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "absmax": float(np.max(np.abs(values))),
            "finite": bool(np.isfinite(values).all()),
        })
    for name, buffer in encoder.named_buffers():
        values = buffer.detach().float().cpu().numpy()
        rows.append({
            "name": name,
            "kind": "buffer",
            "numel": int(values.size),
            "l2": float(np.linalg.norm(values)),
            "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
            "absmax": float(np.max(np.abs(values))) if values.size else 0.0,
            "finite": bool(np.isfinite(values).all()),
        })
    return rows


def main() -> None:
    FOLLOW.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    risk = json.loads((ex.PARENT / "RISK.json").read_text())
    joined, real = ex.assemble(ex.PARENT, BUNDLE, ex.ROOT / "history397-61-r2")
    rows_j, avail_j, _ = ex.strict_table(joined, risk, "2016-01-01", "2024-01-01")
    rows_r, avail_r, _ = ex.strict_table(real, risk, "2024-01-01", "2025-01-01")
    avail = np.concatenate([avail_j, avail_r])
    rows = np.concatenate([rows_j, rows_r])
    which = np.concatenate([
        np.zeros(len(rows_j), dtype=np.int8),
        np.ones(len(rows_r), dtype=np.int8),
    ])
    order = np.argsort(avail, kind="mergesort")
    avail, rows, which = avail[order], rows[order], which[order]
    base = np.load(PHASE / "dataset.npz")
    if not np.array_equal(avail, base["ts"]):
        raise RuntimeError("PANEL_TS_MISMATCH")
    chosen = pick(avail, 256, 60, 61)
    pretrained, _ = ex.load_bundle(BUNDLE, "cpu")
    # Rebuild the chosen windows in panel order from the owning market.
    windows = joined.inputs.windows
    real_windows = real.inputs.windows
    panel = np.empty((len(chosen),) + windows.shape[1:], dtype=windows.dtype)
    for position, (flag, row) in enumerate(zip(which[chosen], rows[chosen])):
        source = windows if flag == 0 else real_windows
        panel[position] = source[row]
    encoder_net = pretrained
    encoder = encoder_net.actor_net.encoder.eval()
    captured: dict[str, dict] = {}

    def keep(name):
        def hook(_module, _inputs, output):
            captured[name] = summarize(output)
            if name == "fusion" and hasattr(encoder, "fixed_fusion_offset"):
                captured["fusion_plus_offset"] = summarize(output + encoder.fixed_fusion_offset)
        return hook

    handles = [branch.register_forward_hook(keep(f"branch_{i}")) for i, branch in enumerate(encoder.branches)]
    handles.append(encoder.fusion.register_forward_hook(keep("fusion")))
    handles.append(encoder.post_fusion.register_forward_hook(keep("post_fusion")))
    handles.append(encoder.core.register_forward_hook(keep("core")))
    with encoder_net.loss.actor_network_params.to_module(encoder_net.actor), torch.inference_mode():
        observed = encoder({"features": torch.from_numpy(np.ascontiguousarray(panel))})
        captured["encoder_output"] = summarize(observed)
        if hasattr(encoder, "fixed_fusion_offset"):
            captured["fixed_fusion_offset_buffer"] = summarize(encoder.fixed_fusion_offset)
        weights = module_norms(encoder)
    for handle in handles:
        handle.remove()
    saved = base["z"][chosen]
    gap = float(np.max(np.abs(observed.float().cpu().numpy() - saved)))
    if gap > 1e-2:
        raise RuntimeError(f"PANEL_EMBEDDING_MISMATCH {gap}")
    (FOLLOW / "encoder_panel.json").write_text(json.dumps({
        "device": "cpu",
        "panel_rows": int(len(chosen)),
        "panel_train": int((avail[chosen] < ex.MONITOR_START).sum()),
        "panel_monitor": int(((avail[chosen] >= ex.MONITOR_START) & (avail[chosen] < ex.HOLDOUT_START)).sum()),
        "panel_holdout": int((avail[chosen] >= ex.HOLDOUT_START).sum()),
        "encoder_frozen_relative_l2": 0.0,
        "reason_relative_l2": "The encoder weights are the recipe-61 bundle. This phase does not train them, so the displacement from the loaded checkpoint is zero.",
        "panel_vs_saved_embedding_max_abs": gap,
        "activations": captured,
        "weights": weights,
    }, indent=2) + "\n")
    print("PANEL_OK", "gap", gap, "modules", len(captured), flush=True)
    del pretrained
    recipe = json.loads((BUNDLE / "RECIPE.json").read_text())
    effective = json.loads((BUNDLE / "MODEL_CONFIG.json").read_text())
    fresh = build_model(recipe, effective, seed=7, device="cpu")
    z_j = ex.encode(joined, rows_j, fresh, "cpu")
    z_r = ex.encode(real, rows_r, fresh, "cpu")
    z = np.concatenate([z_j, z_r])[order]
    delta = float(np.max(np.abs(z - base["z"])))
    if delta < 1e-3:
        raise RuntimeError(f"RANDOM_ENCODER_MATCHES_PRETRAINED {delta}")
    payload = {key: base[key] for key in base.files}
    payload["z"] = z.astype(np.float32)
    target = PHASE / "dataset_random_encoder.npz"
    np.savez_compressed(target, **payload)
    (FOLLOW / "random_encoder.json").write_text(json.dumps({
        "seed": 7,
        "loaded_model_pt": False,
        "max_abs_vs_pretrained_z": delta,
        "rows": int(z.shape[0]),
        "dim": int(z.shape[1]),
        "path": str(target),
    }, indent=2) + "\n")
    print("RANDOM_ENCODER_OK", delta, flush=True)


if __name__ == "__main__":
    main()
