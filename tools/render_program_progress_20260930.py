"""Render a dated program snapshot with an explicit, count-based denominator."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROWS = [
    ("REFERENCIAS", "TimeFilter · ECL", "ACUERDO OPERATIVO", "12 celdas; no identidad con paper", "#087f5b"),
    ("", "TimeFilter · Weather", "PARIDAD TÉCNICA", "12 celdas; falta custodia científica", "#1678a8"),
    ("", "TimeFilter · Traffic", "EN CURSO", "h192/h336 coste; 2 scores h96 activos", "#b56800"),
    ("ARQUITECTURA", "Modular R0 / R1 / R2", "DEV MEDIDO", "ECL: 3 semillas; falta confirmación externa", "#1678a8"),
    ("", "Branching vs referencia exacta", "PENDIENTE", "falta contraste emparejado ECL/Traffic", "#b43b3b"),
    ("", "Autoencoder por extractor", "PENDIENTE", "sin AE por branch ni ablación", "#b43b3b"),
    ("", "Autoencoder del núcleo", "PENDIENTE", "sin entrenamiento ni comparación", "#b43b3b"),
    ("PLATAFORMA", "DOIN · archivo externo", "PROTOTIPO", "24 pruebas; store desechable, sin cadena real", "#1678a8"),
    ("", "DOIN · optimización modular", "PENDIENTE", "sin campaña integrada", "#b43b3b"),
    ("", "LTS · selección de modelos", "EN INTEGRACIÓN", "guardas paper; sin promoción validada", "#b56800"),
    ("TRADING", "Alpaca Paper", "ACTIVO", "1 orden y 1 posición; modelo no promovido", "#1678a8"),
    ("", "MT5 Paper", "INACTIVO", "bridge y runners detenidos", "#b43b3b"),
    ("", "Trading con modelos nuevos", "PENDIENTE", "sin cartera live basada en ganadores", "#b43b3b"),
]

# Evidence deliverables, not effort or expected financial utility. Seven have
# retained evidence; the nine remaining are distinct end-to-end claims.
DONE = ("ECL L96", "ECL L512", "Weather replay", "Traffic cost", "ECL DEV",
        "DOIN archive prototype", "Alpaca paper baseline")
OPEN = ("Traffic 12 scores", "matched modular comparison", "branch AE",
        "core AE", "DOIN optimization", "LTS model handoff", "MT5 paper",
        "financial model evaluation", "validated live model")


def main():
    out = Path(__file__).resolve().parents[1] / "docs" / "audits" / "work_plan" / "PROGRAM_PROGRESS_2026_09_30.png"
    fig, ax = plt.subplots(figsize=(18, 11.3), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 17.5)
    ax.axis("off")
    ax.text(0.7, 16.92, "De la réplica al trading", fontsize=29, weight="bold", color="#17232b")
    total = len(DONE) + len(OPEN)
    ax.text(17.25, 16.92, f"{len(DONE)}/{total}  ·  {len(DONE)/total:.0%}", fontsize=20,
            weight="bold", color="#087f5b", ha="right")
    ax.text(0.72, 16.38, "Entregables con evidencia / entregables definidos · conteo, no horas ni rentabilidad", fontsize=11.8, color="#52636b")
    ax.add_patch(Rectangle((0.72, 16.05), 16.55, 0.13, color="#e7ecee", lw=0))
    ax.add_patch(Rectangle((0.72, 16.05), 16.55 * len(DONE) / total, 0.13, color="#087f5b", lw=0))
    history = [(1.1, "22 SEP", "ECL L96"), (5.0, "24 SEP", "Modular DEV"),
               (8.9, "29 SEP", "Weather"), (12.8, "30 SEP", "Traffic coste"),
               (16.7, "~22:10Z", "primer score")]
    ax.plot([1.1, 16.7], [15.35, 15.35], color="#aab8be", lw=2, zorder=1)
    for x, day, label in history:
        color = "#b56800" if x == 16.7 else "#087f5b"
        ax.scatter([x], [15.35], s=75, color=color, zorder=2)
        ax.text(x, 15.68, day, fontsize=10, weight="bold", color=color, ha="center")
        ax.text(x, 14.94, label, fontsize=9.5, color="#33454e", ha="center")
    ax.plot([0.72, 17.3], [14.55, 14.55], color="#17232b", lw=1.2)
    last_group = None
    for index, (group, name, status, detail, color) in enumerate(ROWS):
        y = 13.13 - index * 0.91
        if group:
            if last_group is not None:
                ax.plot([0.72, 17.3], [y + 0.52, y + 0.52], color="#d9e0e3", lw=1)
            ax.text(0.72, y + 0.12, group, fontsize=9.5, weight="bold", color="#72818a")
            last_group = group
        ax.add_patch(Rectangle((3.17, y - 0.17), 0.065, 0.58, color=color, lw=0))
        ax.text(3.45, y + 0.10, name, fontsize=14.1, weight="bold", color="#17232b", va="center")
        ax.text(10.4, y + 0.11, status, fontsize=10.3, weight="bold", color=color, va="center")
        ax.text(13.55, y + 0.11, detail, fontsize=9.0, color="#3c4d55", va="center")
    ax.plot([0.72, 17.3], [0.87, 0.87], color="#17232b", lw=1.2)
    ax.text(0.72, 0.47, "Siguiente evidencia: score Traffic → referencia emparejada → branching → preentrenamiento → promoción paper", fontsize=12, color="#17232b")
    ax.text(0.72, 0.13, "Fuentes: cierres RP140–143, ECL R0/R1/R2, Weather 2026-09-29, Traffic TRAIN_PILOT h192, DOIN 4c23a6f, LTS 0a36609.", fontsize=8.8, color="#66777e")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
