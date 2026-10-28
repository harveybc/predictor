#!/usr/bin/env python3
"""Render the manually reconciled execution snapshot as a PNG."""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OBSERVED_AT = datetime(2026, 10, 7, 0, 19, tzinfo=timezone.utc)
STAMP = OBSERVED_AT.strftime("%Y%m%dT%H%MZ")
OUTPUT = ROOT / f"PROGRESS_{STAMP}.png"

fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(0.45, 8.58, "Feature selection: phases 1-3 closed → phase 4 design",
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

box(0.5, 6.1, 3.25, 1.55, "Phase 1 · profiles + causal", "CLOSED", "EURUSD 366/366, ETH 83/83;\n34 + 3 causal-supported pairs (inputs).", *green)
box(4.45, 6.1, 3.25, 1.55, "Phase 2 · pairwise matrix", "CLOSED 2026-10-06", "66,795 + 3,403 pairs, 0 failed;\n7.46M rows live, digests sealed.", *green)
box(8.8, 6.55, 3.15, 1.48, "Phase 3 · filters", "CLOSED 2026-10-06", "9 methods x 14 + 6 targets, K={4..32};\n686 + 294 candidates, no winner.", *green)
box(8.8, 4.85, 3.15, 1.48, "Warehouse + snapshot", "RECONCILED", "Live cube complete: true (both runs);\nrelease phase2-3-feature-selection-20261007.", *blue)
box(13.0, 5.7, 2.5, 1.55, "Phase 4 · extractibility", "DESIGN · NOT STARTED", "raw/random/trained on phase-3\ncandidates with matched controls.", *purple)

arrow((3.8, 6.88), (4.35, 6.88))
arrow((7.75, 6.98), (8.68, 7.20), curve=0.04)
arrow((7.75, 6.62), (8.68, 5.62), curve=-0.06)
arrow((12.02, 7.20), (12.90, 6.70), curve=-0.02)
arrow((12.02, 5.58), (12.90, 6.25), curve=0.02)

later = [
    (0.55, "Wrapper validation", "BUSINESS_WEEKLY_WALK_FORWARD on candidates"),
    (4.40, "ARCH / E1 · H-CORE", "Controls; branch pretrain; R0/R1/R2"),
    (8.25, "DENSE vs NEAT", "Late head on frozen features"),
    (12.10, "SAC / DQN", "Raw vs modular after core"),
]
for x, title, detail in later:
    box(x, 3.00, 3.35, 1.25, title, "NOT ELIGIBLE YET", detail, *gray,
        status_offset=0.68, detail_offset=0.16)

ax.text(0.55, 2.65, "HOST ROLES · phase 2/3 campaign (CPU only, admission-capped)", fontsize=12, weight="bold", color="#17212b")
machine_rows = [
    ("coordinator", "1 slot x 1 GiB · EURUSD 85 + ETH 11 shards (cheapest third) · followers + relay", blue[0]),
    ("worker_a", "1 slot x 2 GiB · EURUSD 86 + ETH 8 shards", green[0]),
    ("worker_b", "3 slots x 2 GiB · EURUSD 85 + ETH 13 shards (3 stolen)", green[0]),
    ("warehouse (coordinator)", "live cube reconciled complete for both runs · snapshot released", blue[0]),
]
for i, (name, detail, color) in enumerate(machine_rows):
    y = 2.30 - i * 0.34
    ax.text(0.62, y, name, fontsize=9.8, weight="bold", color=color, va="center")
    ax.text(4.15, y, detail, fontsize=9.8, color="#263541", va="center")

ax.text(0.58, 0.60,
        "No predictive winner was chosen and the test split was never read: phase-3 outputs are candidates for wrapper validation.",
        fontsize=11.2, weight="bold", color="#17212b")
ax.text(0.58, 0.28,
        "Next object: phase 4 extractibility DESIGN (raw / random / trained on phase-3 candidate subsets with matched controls), not started.",
        fontsize=9.7, color="#4a5a67")

plt.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
print(OUTPUT)
