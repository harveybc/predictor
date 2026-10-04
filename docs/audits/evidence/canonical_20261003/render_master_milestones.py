#!/usr/bin/env python3
"""Render the owner's stable critical-path milestone view from authenticated JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "MASTER_MILESTONE_STATUS.json"
OUTPUT = HERE / "MASTER_MILESTONE_PROGRESS.png"

COLORS = {
    "CORE_IMPLEMENTED": "#16836f",
    "IN_PROGRESS": "#d98e04",
    "MULTIPLE_REPLICATIONS_COMPLETE": "#2878b5",
    "PARTIAL_MEASUREMENT": "#7656a8",
    "DESIGNED_NOT_DEFINITIVE": "#64748b",
    "SYNTHETIC_ONLY": "#b45309",
    "INTEGRATED_NOT_TRAINED_ON_FINAL_DATA": "#a23b72",
    "INTERFACES_ONLY": "#64748b",
}


def wrapped(text: str, width: int) -> str:
    words = text.split()
    lines, current = [], []
    for word in words:
        if current and len(" ".join(current + [word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def main() -> None:
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = doc["milestones"]
    fig, ax = plt.subplots(figsize=(18, 10), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(0.55, 9.52, "Predictor: critical path to business trading",
            fontsize=24, fontweight="bold", color="#132238", va="top")
    ax.text(0.55, 9.08,
            f"Measured status {doc['as_of']}  |  Critical-path ETA: {doc['critical_path_eta']}",
            fontsize=10.5, color="#526170", va="top")

    box_w, box_h = 4.05, 1.82
    xs = (0.55, 4.92, 9.29, 13.66)
    ys = (6.75, 4.45)
    for index, item in enumerate(items):
        row, col = divmod(index, 4)
        x, y = xs[col], ys[row]
        color = COLORS[item["state"]]
        if index:
            previous_row, previous_col = divmod(index - 1, 4)
            px, py = xs[previous_col], ys[previous_row]
            if row == previous_row:
                ax.annotate("", xy=(x - 0.08, y + box_h / 2),
                            xytext=(px + box_w + 0.08, py + box_h / 2),
                            arrowprops=dict(arrowstyle="-|>", color="#8b98a7", lw=1.8))

        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor="#f8fafc", edgecolor=color, linewidth=2.1))
        ax.text(x + 0.18, y + 1.63, item["id"], fontsize=8.2,
                fontweight="bold", color=color, va="top")
        ax.text(x + 0.18, y + 1.39, wrapped(item["title"], 36),
                fontsize=10.0, fontweight="bold", color="#132238", va="top")
        ax.text(x + box_w - 0.16, y + 1.62, f"{item['progress']}%",
                fontsize=11.2, fontweight="bold", color=color, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor=color, linewidth=0.8))
        ax.add_patch(FancyBboxPatch((x + 0.18, y + 0.92), box_w - 0.36, 0.13,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="#dfe5eb", edgecolor="none"))
        ax.add_patch(FancyBboxPatch((x + 0.18, y + 0.92),
                                    (box_w - 0.36) * item["progress"] / 100, 0.13,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 0.18, y + 0.73, wrapped(item["evidence"], 53),
                fontsize=7.35, color="#334155", va="top", linespacing=1.18)
        ax.text(x + 0.18, y + 0.12, wrapped(f"ETA: {item['eta']}", 48),
                fontsize=7.35, fontweight="bold", color=color, va="bottom",
                linespacing=1.05)

    ax.text(9.0, 6.42, "CRITICAL PATH CONTINUES BELOW", fontsize=7.8,
            fontweight="bold", color="#7b8794", ha="center")
    ax.text(0.55, 3.99, "NEXT GATE FOR EACH MILESTONE", fontsize=10.5,
            fontweight="bold", color="#132238")
    for index, item in enumerate(items):
        col = index % 2
        row = index // 2
        x = 0.65 + col * 8.7
        y = 3.62 - row * 0.56
        ax.text(x, y, f"{item['id']}", fontsize=8.2, fontweight="bold",
                color=COLORS[item["state"]], va="top")
        ax.text(x + 0.37, y, wrapped(item["next"], 95), fontsize=7.8,
                color="#334155", va="top")

    optional = "  |  ".join(doc["optional_after_critical_path"])
    ax.text(0.55, 0.78, "Optional after the spearhead:", fontsize=8.5,
            fontweight="bold", color="#526170")
    ax.text(3.22, 0.78, optional, fontsize=8.2, color="#64748b")
    ax.text(17.45, 0.28, "Percentages are engineering completion estimates, not scientific confidence.",
            fontsize=7.5, color="#7b8794", ha="right")

    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
