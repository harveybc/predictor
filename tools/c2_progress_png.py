"""Lane C2: PROGRESS.png from STATUS.json (orders section 6: milestones, completed/planned with denominator, current
work, next dependencies, ETA). One axis, one sequential hue for magnitude, text in ink tokens; no figure is a measurement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, HUE, HUE_LIGHT = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#cde2fb"


def render(status: dict, out: Path):
    items = status["deliverables"]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(items) + 2.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    names = [f"{d['id']} {d['name']}" for d in items][::-1]
    done = [d["completed"] for d in items][::-1]
    planned = [d["planned"] for d in items][::-1]
    frac = [c / p if p else 0 for c, p in zip(done, planned)]
    y = range(len(items))
    ax.barh(y, [1] * len(items), color=HUE_LIGHT, height=0.5, linewidth=0)
    ax.barh(y, frac, color=HUE, height=0.5, linewidth=0)
    for i, (c, p, d) in enumerate(zip(done, planned, items[::-1])):
        ax.text(1.02, i, f"{c}/{p}  {d['state']}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.set_xlim(0, 2.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Lane C2 (PS3-C/PS3-R over ETH 4h TRAIN) -- {status['label']} -- {status['updated_at']}",
                 fontsize=10, color=INK, loc="left")
    foot = (f"current: {status['current_work']}\nnext dependencies: {status['next_dependencies']}\n"
            f"ETA: {status['eta']}\nmilestones: {status['milestones']}")
    fig.text(0.01, 0.01, foot, fontsize=7.8, color=INK2, va="bottom", ha="left", wrap=True)
    fig.subplots_adjust(left=0.34, right=0.99, top=0.9, bottom=0.34)
    fig.savefig(out, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(render(status, Path(sys.argv[2])))
