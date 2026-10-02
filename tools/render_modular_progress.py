#!/usr/bin/env python3
"""Render the modular architecture progress figure from a retained summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _metric_panel(axis, title, block, naive_key, xlim=None):
    rows = block["rows"]
    labels = [row["label"] for row in rows]
    values = [row["mae_z"] for row in rows]
    colors = ["#177E89" if row["status"] == "verified" else "#5C677D" for row in rows]
    positions = np.arange(len(rows))
    axis.barh(positions, values, color=colors, height=0.62)
    naive = block[naive_key]
    axis.axvline(naive, color="#C33C54", linewidth=2, linestyle="--", label="Naive pareado")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("MAE_z (menor es mejor)")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    if xlim:
        axis.set_xlim(*xlim)
    for position, value in zip(positions, values):
        axis.text(value, position, f"  {value:.6f}", va="center", fontsize=8)


def render(summary_path, output_path):
    summary = json.loads(Path(summary_path).read_text())
    figure = plt.figure(figsize=(15, 10), facecolor="white", constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(3, 1.25))
    ecl = figure.add_subplot(grid[0, 0])
    eth = figure.add_subplot(grid[0, 1])
    progress = figure.add_subplot(grid[1, :])
    _metric_panel(ecl, "ECL: arquitectura y preentrenamiento", summary["ecl"],
                  "naive_seasonal_mae_z", xlim=(0.205, 0.252))
    _metric_panel(eth, "ETH 4h: primer contraste financiero", summary["eth"],
                  "naive_persistence_mae_z", xlim=(2.34, 2.45))
    milestones = summary["milestones"]
    palette = {"complete": "#2A9D8F", "active": "#E9C46A", "pending": "#D9D9D9"}
    x = np.arange(len(milestones))
    progress.scatter(x, np.zeros_like(x), s=950,
                     c=[palette[item["state"]] for item in milestones], edgecolors="white", linewidths=2)
    progress.plot(x, np.zeros_like(x), color="#667085", linewidth=2, zorder=0)
    for index, item in enumerate(milestones):
        progress.text(index, -0.18 if index % 2 else 0.18, item["label"], ha="center",
                      va="top" if index % 2 else "bottom", fontsize=8, wrap=True)
    completed = sum(item["state"] == "complete" for item in milestones)
    progress.set_title(f"Plan modular: {completed}/{len(milestones)} hitos completos; "
                       "seleccion progresiva activa", loc="left", fontweight="bold")
    progress.set_xlim(-0.5, len(milestones) - 0.5)
    progress.set_ylim(-0.48, 0.48)
    progress.axis("off")
    figure.suptitle("Arquitectura temporal diferenciada: estado experimental al 2026-10-01",
                    fontsize=16, fontweight="bold")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.summary, args.output)


if __name__ == "__main__":
    main()
