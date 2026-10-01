"""Lane H: PROGRESS.png generated from STATUS.json (milestones with denominators, current work, dependencies)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(status_path, out_path):
    s = json.loads(Path(status_path).read_text())
    ms = s["milestones"]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(ms) + 2.2))
    for i, m in enumerate(reversed(ms)):
        frac = m["done"] / m["of"]
        ax.barh(i, 1.0, color="#e3e3e3", height=0.6)
        ax.barh(i, frac, color="#2a6f97" if frac >= 1 else "#c97a1d", height=0.6)
        ax.text(1.01, i, f"{m['done']}/{m['of']}", va="center", fontsize=9)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([m["name"] for m in reversed(ms)], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    ax.set_title(f"Lane H causal Kalman ({s['label']}): completed / planned, versioned scope {s['branch']}", fontsize=10)
    fig.text(0.01, 0.01, "Current: " + s["current_work"] + "   Next: " + "; ".join(s["next_dependencies"]) + "   ETA: none pending (no worker job running)",
             fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
