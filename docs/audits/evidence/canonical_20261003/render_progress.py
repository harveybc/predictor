#!/usr/bin/env python3
"""Render the manually reconciled execution snapshot as a PNG."""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OBSERVED_AT = datetime(2026, 10, 3, 16, 34, tzinfo=timezone.utc)
STAMP = OBSERVED_AT.strftime("%Y%m%dT%H%MZ")
OUTPUT = ROOT / f"PROGRESS_{STAMP}.png"

fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(0.45, 8.58, "EURUSD feature selection → temporal representation",
        fontsize=24, weight="bold", color="#17212b")
ax.text(0.48, 8.20,
        f"Verified operating snapshot · {OBSERVED_AT:%Y-%m-%d %H:%M UTC} · stage states, not a made-up completion percentage",
        fontsize=11.5, color="#4a5a67")


def box(x, y, w, h, title, status, detail, edge, fill, status_offset=0.70, detail_offset=0.22):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                           linewidth=1.5, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + 0.20, y + h - 0.34, title, fontsize=12, weight="bold", color="#17212b")
    ax.text(x + 0.20, y + h - status_offset, status, fontsize=8.5, weight="bold", color=edge)
    ax.text(x + 0.20, y + detail_offset, detail, fontsize=8.2, color="#263541", va="bottom", linespacing=1.25)


def arrow(start, end, color="#778791", curve=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.35, color=color,
                                 connectionstyle=f"arc3,rad={curve}"))


blue = ("#08738d", "#e3f3f7")
green = ("#557421", "#eff5df")
orange = ("#a35b08", "#fff0d9")
purple = ("#755c91", "#f1ecf7")
gray = ("#61717b", "#f3f5f6")

box(0.5, 6.1, 3.25, 1.55, "PS0 / PS1", "PARTIAL", "366 admissible features;\nsource clocks and paid feeds open.", *blue)
box(4.45, 6.1, 3.25, 1.55, "PS2", "PARTIAL", "279 joined; 87 low-priority remain\nexplicitly pending, not rejected.", *green)
box(8.8, 6.55, 3.15, 1.48, "PS3-C · causal ladder", "REVIEWED · NO SELECTION", "Fix 48ae17c; 1,076 estimates.\nNone passes identification gate.", *orange)
box(8.8, 4.85, 3.15, 1.48, "PS3-R · extractibility", "RUNNING", "E: 33/33 tier-1; 4 in batch 003.\nF: 29 retained; RSI-14 done.", *blue)
box(13.0, 5.7, 2.5, 1.55, "PS4 / PS5", "WAITING", "Reconcile both arms;\nselected / rejected / pending.", *purple)

arrow((3.8, 6.88), (4.35, 6.88))
arrow((7.75, 6.98), (8.68, 7.20), curve=0.04)
arrow((7.75, 6.62), (8.68, 5.62), curve=-0.06)
arrow((12.02, 7.20), (12.90, 6.70), curve=-0.02)
arrow((12.02, 5.58), (12.90, 6.25), curve=0.02)

later = [
    (0.55, "ARCH / E1", "Controls; branch pretrain; R0/R1/R2"),
    (4.40, "H-CORE", "Core after branch selection"),
    (8.25, "DENSE vs NEAT", "Late head on frozen features"),
    (12.10, "SAC / DQN", "Raw vs modular after core"),
]
for x, title, detail in later:
    box(x, 3.00, 3.35, 1.25, title, "NOT ELIGIBLE YET", detail, *gray,
        status_offset=0.68, detail_offset=0.16)

ax.text(0.55, 2.65, "MACHINE SLOTS · same observation window", fontsize=12, weight="bold", color="#17212b")
machine_rows = [
    ("Gamma · RTX 5090", "E / rg.ema_alignment active · 46 C · 30.4 GiB VRAM", blue[0]),
    ("Gamma · RTX 5070 Ti", "F / ta.rsi_14 complete · 40 C · 14 MiB VRAM", green[0]),
    ("Dragon · RTX 4090", "idle · MT5 VM running · governed inputs and pins missing", orange[0]),
    ("Omega · RTX 4070", "desktop / OLAP activity · protected from long training", green[0]),
]
for i, (name, detail, color) in enumerate(machine_rows):
    y = 2.30 - i * 0.34
    ax.text(0.62, y, name, fontsize=9.8, weight="bold", color=color, va="center")
    ax.text(4.15, y, detail, fontsize=9.8, color="#263541", va="center")

ax.text(0.58, 0.60,
        "No selected feature and no new financial-accuracy result. Feature coverage reconciliation is not a selection result.",
        fontsize=11.2, weight="bold", color="#17212b")
ax.text(0.58, 0.28,
        "NEAT is not DEAP or DOIN; it remains a late predictive-head comparison after the frozen pretrained representation.",
        fontsize=9.7, color="#4a5a67")

plt.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
print(OUTPUT)
