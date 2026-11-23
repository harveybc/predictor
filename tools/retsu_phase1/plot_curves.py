#!/usr/bin/env python
"""TRAIN/VAL curves from the epoch logs already written. No fit and no predict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOTS = {
    "seasonal_pretrained": Path("/var/tmp/retsu-phase1/jobs"),
    "seasonal_seed7": Path("/var/tmp/retsu-phase1-control/jobs"),
    "close_seed7": Path("/var/tmp/retsu-phase1-close-seed7/jobs"),
    "close_pretrained": Path("/var/tmp/retsu-phase1-close-pretrained/jobs"),
    "change_seed7": Path("/var/tmp/retsu-phase1-dclose-seed7/jobs"),
    "change_pretrained": Path("/var/tmp/retsu-phase1-dclose-pretrained/jobs"),
    "unfreeze_seasonal_frozen": Path("/var/tmp/retsu-phase1-unfreeze-frozen/jobs"),
    "unfreeze_seasonal_tuned": Path("/var/tmp/retsu-phase1-unfreeze-tuned/jobs"),
    "unfreeze_change_frozen": Path("/var/tmp/retsu-phase1-unfreeze-change-frozen/jobs"),
    "unfreeze_change_tuned": Path("/var/tmp/retsu-phase1-unfreeze-change-tuned/jobs"),
}
OUT = Path("/var/tmp/retsu-phase1/followup/curves")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_one(family: str, name: str, rows: list[dict], dest: Path) -> dict:
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_mae"]) for row in rows]
    val = [float(row["val_mae"]) for row in rows]
    lr = [float(row["lr"]) for row in rows]
    reductions = [int((row.get("reduce_lr") or {}).get("reductions") or 0) for row in rows]
    best = int((rows[-1].get("early_stopping") or {}).get("best_epoch") or epochs[0])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(epochs, train, label="train MAE")
    axes[0].plot(epochs, val, label="val MAE")
    axes[0].axvline(best, color="black", linewidth=0.8, label=f"champion {best}")
    axes[0].set_ylabel("MAE")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(epochs, lr, color="tab:purple")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("LR")
    axes[1].set_xlabel("epoch")
    marked = False
    for epoch, previous, current in zip(epochs, [0, *reductions[:-1]], reductions):
        if current > previous:
            axes[1].axvline(epoch, color="tab:orange", linewidth=0.6, label="LR reduction" if not marked else None)
            marked = True
    if marked:
        axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{family} {name}")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=100)
    plt.close(fig)
    last = rows[-1]
    early = last.get("early_stopping") or {}
    return {
        "family": family,
        "job": name,
        "epochs": int(last["epoch"]),
        "best_epoch": best,
        "es_wait": early.get("wait"),
        "es_patience": early.get("patience"),
        "lr": last.get("lr"),
        "reductions": reductions[-1],
        "train_mae": last.get("train_mae"),
        "val_mae": last.get("val_mae"),
        "best_val_mae": last.get("best_val_mae"),
        "stop": "early_stopping" if early.get("wait") == early.get("patience") else "other",
        "png": str(dest),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family, root in ROOTS.items():
        if not root.exists():
            continue
        for job in sorted(path.parent for path in root.glob("*/epochs.jsonl")):
            rows = load_rows(job / "epochs.jsonl")
            if not rows:
                continue
            record = plot_one(family, job.name, rows, OUT / family / f"{job.name}.png")
            weights = job / "champion.weights.h5"
            if not weights.exists():
                weights = job / "champion.pt"
            record["champion_sha256"] = sha256(weights)
            record["champion_bytes"] = weights.stat().st_size if weights.exists() else None
            record["champion_file"] = weights.name if weights.exists() else None
            manifest.append(record)
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("curves", len(manifest), "manifest", path)
    print("CURVES_OK")


if __name__ == "__main__":
    main()
