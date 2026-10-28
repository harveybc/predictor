"""Render bounded R0/R2 execution counts and retained master milestones.

Read-only: no fitting, publication, credentials or source metric recomputation.
The public snapshot deliberately excludes deployment paths and private logs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.i6dmo import r2_downsampled_campaign as S, status as STATUS
from tools.i6n_swarm.common import atomic_json

STEPS = (("Donantes", "pretrain", 19), ("Ajustes R0/R2", "fit", 2),
         ("Informes", "report", 2), ("Sonda diagnostica", "evaluation", 1),
         ("Comparacion", "comparison", 1))


def snapshot(root, milestones):
    root = Path(root)
    sequence = S.status(root)
    families = {}
    for family, campaign in sequence["families"].items():
        units = [] if campaign is None else STATUS.compute(root / family / "campaign", level="all")["units"]
        families[family] = dict(
            state="PENDING_SHORT_CLOSE" if campaign is None else campaign["state"],
            campaign_id=None if campaign is None else campaign["campaign_id"],
            steps=[dict(label=label, complete=sum(u["state"] == "COMPLETE" for u in units if u["kind"] == kind),
                        total=total) for label, kind, total in STEPS],
            failed=sum(u["state"] in ("FAILED", "BLOCKED") for u in units),
            running=[u["unit_id"] for u in units if u["state"] == "RUNNING"],
            adopted=sum(u.get("reuse") is not None and u["state"] == "COMPLETE" for u in units))
    return dict(schema="i6dmo.r2_public_progress.v1", as_of=datetime.now(timezone.utc).isoformat(),
        action=sequence["action"], test_open=False, families=families,
        milestones=json.loads(Path(milestones).read_text())["milestones"])


def render(doc, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    master = fig.add_subplot(grid[0, :])
    rows = doc["milestones"]
    y = list(range(len(rows)))
    master.barh(y, [100] * len(rows), color="#e8edef", height=.6)
    master.barh(y, [m["progress"] for m in rows], height=.6,
                color=["#b65132" if m["id"] == "M4" else "#247b80" for m in rows])
    master.set(yticks=y, yticklabels=[f"{m['id']}  {m['title']}" for m in rows], xlim=(0, 110))
    master.invert_yaxis()
    for i, m in enumerate(rows):
        master.text(101, i, f"{m['progress']}%", va="center", fontsize=11)
    master.set_title("Hitos maestros: porcentajes retenidos, no recalculados por este arranque", loc="left")
    for col, (family, title, color) in enumerate((("short", "Corto MG2m: 2/3/4/5/6/12 h", "#247b80"),
                                               ("long", "Largo MG12m: 18/24/36/48/60/72 h", "#b65132"))):
        axis = fig.add_subplot(grid[1, col])
        item = doc["families"][family]
        for i, step in enumerate(item["steps"]):
            axis.barh(i, 1, color="#e8edef", height=.55)
            axis.barh(i, step["complete"] / step["total"], color=color, height=.55)
            axis.text(1.02, i, f"{step['complete']}/{step['total']}", va="center", fontsize=11)
        axis.set(yticks=range(len(STEPS)), yticklabels=[s["label"] for s in item["steps"]],
                 xlim=(0, 1.18), xticks=[])
        axis.invert_yaxis()
        axis.set_title(f"{title}\n{item['state']} | adoptados {item['adopted']} | fallos {item['failed']}",
                       fontsize=11, loc="left", pad=14)
    for axis in fig.axes:
        axis.spines[["top", "right", "bottom"]].set_visible(False)
    fig.suptitle("R0 frente a R2: corto -> cierre verificado -> largo\n"
                 + doc["as_of"][:19].replace("T", " ") + " UTC | TEST cerrado", fontsize=17)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--milestones", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    doc = snapshot(args.root, args.milestones)
    atomic_json(args.output.with_suffix(".json"), doc)
    render(doc, args.output)


if __name__ == "__main__":
    main()
