"""Generate pooled horizon tables, compact milestones and PNG from receipts."""

from collections import defaultdict
import json

from tools.i6m_queue import atomic_json, seal


def close(root, queue):
    """No inference or fitting: row-weighted reductions from verified terminals."""
    with queue.connect() as conn:
        rows = conn.execute("SELECT terminal FROM jobs WHERE terminal IS NOT NULL").fetchall()
        remaining = conn.execute("SELECT count(*) FROM jobs WHERE published=0").fetchone()[0]
    groups = defaultdict(list)
    for row in rows:
        cell = json.loads(row[0])
        groups[(cell["model"], cell["arm"]["arm_id"])].append(cell)
    summary = []
    for (model, arm_id), cells in sorted(groups.items()):
        arm = cells[0]["arm"]
        complete = len(cells) == len(queue.design["weeks"]) and not any(c["status"] == "FAILED" for c in cells)
        scored = [c["metrics"] for c in cells if c["status"] == "COMPLETED"]
        n = sum(m["n"] for m in scored)
        mae = sum(m["mae"]*m["n"] for m in scored)/n if n else None
        naive = sum(m["naive_mae"]*m["n"] for m in scored)/n if n else None
        summary.append(dict(model=model, arm_id=arm_id, horizon_hours=arm["horizon_hours"], samples=arm["samples"],
                            spacing_hours=arm["spacing_hours"], dispositions=len(cells), n=n,
                            status="COMPLETE" if complete else "INCOMPLETE_NOT_ELIGIBLE",
                            mae=mae if complete else None, naive_mae=naive if complete else None,
                            skill=None if not complete or not naive else (mae-naive)/naive,
                            skill_formula="(MAE_h - naive_h) / naive_h", unit="USD_per_EUR"))
    complete = len(rows) == 2*38*53 and not remaining
    atomic_json(root/"RESULTS.json", seal(dict(status="COMPLETE" if complete else "COMPUTE_CLOSED_PUBLICATION_PENDING",
                design_sha256=queue.identity, summary=summary, test_read=False, missing_publications=remaining)))
    body = "# I6-E-M: resultados por horizonte\n\nSkill = (MAE_h - naive_h) / naive_h; negativo mejora. Unidad MAE: USD/EUR. TEST no leido.\n\n"
    body += "| Modelo | h (horas) | delta (horas) | Muestras | Filas | MAE | Naive para h | Skill | Estado |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
    for r in summary:
        fmt = lambda v: "--" if v is None else f"{v:.9g}"
        body += f"| {r['model']} | {r['horizon_hours']} | {r['spacing_hours']} | {r['samples']} | {r['n']} | {fmt(r['mae'])} | {fmt(r['naive_mae'])} | {fmt(r['skill'])} | {r['status']} |\n"
    (root/"RETURN.md").write_text(body)
    plot(root, queue)


def plot(root, queue):
    """Progress is explicit completed/expected dispositions, not invented percent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.i6e_horizon_plan import HORIZONS
    state = queue.status()
    with queue.connect() as conn:
        rows = conn.execute("SELECT spec,state FROM jobs WHERE model='NEURAL'").fetchall()
    counts = {h:[0,0,0] for h in HORIZONS}
    for text, status in rows:
        h = json.loads(text)["arm"]["horizon_hours"]
        counts[h][0] += status in ("COMPLETED", "NOT_APPLICABLE")
        counts[h][1] += status == "FAILED"
        counts[h][2] += 1
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    y = list(range(len(HORIZONS)))
    ax.barh(y, [100*counts[h][0]/counts[h][2] for h in HORIZONS], color="#14866d")
    ax.barh(y, [100*counts[h][1]/counts[h][2] for h in HORIZONS], left=[100*counts[h][0]/counts[h][2] for h in HORIZONS], color="#bf4545")
    ax.set_yticks(y, [f"{h} h   ({counts[h][0]}/{counts[h][2]}; fallos {counts[h][1]})" for h in HORIZONS])
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Disposiciones verificadas (%)")
    ax.set_title("I6-E-M: downsampling → controles horarios\nSelección cerrada · RAW · 4 años TRAIN · reajuste semanal · VALIDATION 2024")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root/"PROGRESS.png", dpi=160)
    plt.close(fig)
