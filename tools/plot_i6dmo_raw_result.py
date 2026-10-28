"""Render retained master milestones and measured RAW-control results from JSON.

This read-only reporting utility never recomputes scientific metrics, updates
milestone percentages or accesses a campaign queue, model, credential or TEST.
"""
import argparse
import json
from pathlib import Path


def render(milestones_path, result_path, output):
    """Write a PNG using only the two explicit, public summary inputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    milestones = json.loads(Path(milestones_path).read_text())["milestones"]
    result = json.loads(Path(result_path).read_text())
    fig, (master, errors) = plt.subplots(2, 1, figsize=(13, 10),
                                        gridspec_kw={"height_ratios": [1.3, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Plan maestro y resultado del control RAW", fontsize=20, y=.97)
    publication = result["publication"]
    fig.text(.5, .925, f"{result['state']} / {result['disposition']}   |   "
             f"6/6 unidades   |   OLAP {publication['delivered']}/{publication['expected']}   |   "
             f"{result['mirrors']['files_each']} archivos por espejo", ha="center", fontsize=11)
    labels = [f"{m['id']}  {m['title']}" for m in milestones]
    positions = np.arange(len(milestones))
    master.barh(positions, [100] * len(milestones), color="#edf0f2", height=.6)
    master.barh(positions, [m["progress"] for m in milestones],
                color=["#bc5c2e" if m["id"] == "M6" else "#237e83" for m in milestones], height=.6)
    for i, milestone in enumerate(milestones):
        master.text(101, i, f"{milestone['progress']}%", va="center", fontsize=11)
    master.set(yticks=positions, yticklabels=labels, xlim=(0, 108), xticks=[0, 25, 50, 75, 100])
    master.invert_yaxis()
    master.set_title("Hitos M1-M8: porcentajes retenidos, sin incrementos inventados", fontsize=12, loc="left")
    master.grid(axis="x", alpha=.15)
    horizons = sorted(result["arms"]["mg12m"]["per_horizon"], key=int)
    x = np.arange(len(horizons))
    for arm, offset, color, label in (("mg12m", -.19, "#237e83", "MG12m: 13 observaciones"),
                                      ("mg1l", .19, "#bc5c2e", "MG1l: 145 observaciones")):
        values = [100 * result["arms"][arm]["per_horizon"][h]["relative_error"] for h in horizons]
        bars = errors.bar(x + offset, values, width=.36, color=color, label=label)
        errors.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    errors.axhline(0, color="#343434", linewidth=1)
    errors.set(xticks=x, xticklabels=[f"{h} h" for h in horizons], ylim=(-3, 70),
               ylabel="Error relativo al naive del horizonte (%)")
    errors.set_title("M6: ambas recetas largas pierden frente a su naive; trading no ejecutado", fontsize=12, loc="left")
    errors.legend(frameon=False, fontsize=10, loc="upper left")
    errors.grid(axis="y", alpha=.15)
    for axis in (master, errors):
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .018, "DEVELOPMENT 2024, no TEST | positivo = peor | modelos y dispositivos distintos | "
             "una semilla, no prueba de superioridad estadistica", ha="center", fontsize=9)
    fig.tight_layout(rect=(.02, .045, .98, .90), h_pad=2.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestones", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.milestones, args.result, args.output)


if __name__ == "__main__":
    main()
