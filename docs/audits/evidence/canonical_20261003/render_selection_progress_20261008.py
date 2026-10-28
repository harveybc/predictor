#!/usr/bin/env python3
"""Render the feature-selection milestone line from retained closure artifacts."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def draw(closure: dict, frozen: dict, wave: dict, output: Path) -> None:
    if closure["closure_sha256"] != frozen["closure_sha256"]:
        raise ValueError("freeze does not bind the supplied weekly closure")
    if not closure["warehouse"]["readback"]["matches_store"]:
        raise ValueError("warehouse readback did not reconcile")
    if closure["denominator"]["tasks"] != closure["denominator"]["terminal"]:
        raise ValueError("weekly denominator is incomplete")
    if frozen["firewall_phase"] != "PROCEDURE_SEALED":
        raise ValueError("TEST procedure is not sealed")

    winners = closure["winners"]
    union = {
        population: len({feature for target in targets.values()
                         for feature in target["RAW"]["members"]})
        for population, targets in winners.items()
    }
    n_targets = sum(len(targets) for targets in winners.values())
    positive = sum(target["RAW"]["score"] > 0
                   for targets in winners.values() for target in targets.values())
    random_eur = sum("RANDOM_K" in target["RAW"]["methods"]
                     for target in winners["EURUSD"].values())
    counts = wave["wave_status"]["counts"]
    extractibility = sum(sum(states.values()) for states in counts.values())
    extracted = sum(states.get("COMPLETE", 0) for states in counts.values())
    refused = extractibility - extracted
    staged = closure["denominator"]["tasks"]
    shortlist = wave["wave_status"]["features"]

    dark = "#17232b"
    muted = "#52616b"
    green = "#087a5a"
    amber = "#a45b00"
    blue = "#166b9a"
    line = "#d8e0e3"

    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(0.65, 9.45, "Selección de características", fontsize=27, weight="bold", color=dark)
    ax.text(0.67, 9.06, "Hitos observados  |  " + datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC"),
            fontsize=11, color=muted)
    ax.plot([0.65, 15.35], [8.80, 8.80], color=dark, linewidth=2)
    ax.text(0.65, 8.56, "DE LOS DATOS A LA ELECCIÓN", fontsize=10, weight="bold", color=muted)

    rows = [
        (8.10, "Inventario", "449 series admisibles · EURUSD 366 / ETH 83", "COMPLETO"),
        (7.43, "Perfiles y causalidad", "449 perfiles · resultados en OLAP", "COMPLETO"),
        (6.76, "Dependencias cruzadas", "70.198 pares evaluados · redundancia y alias", "COMPLETO"),
        (6.09, "Filtros preliminares", f"EURUSD {shortlist['EURUSD']} / ETH {shortlist['ETH']} candidatas", "COMPLETO"),
        (5.42, "Extractibilidad", f"{extractibility:,} terminales · {extracted:,} completos / {refused} rechazos", "COMPLETO"),
        (4.75, "Validación semanal", f"{staged:,}/{staged:,} tareas · {n_targets} objetivos · warehouse reconciliado", "CONGELADA"),
    ]
    ax.plot([0.91, 0.91], [4.75, 8.10], color=line, linewidth=3, zorder=1)
    for index, (y, title, detail, status) in enumerate(rows, 1):
        ax.add_patch(Circle((0.91, y), 0.16, color=green, zorder=2))
        ax.text(0.91, y, str(index), color="white", fontsize=8.5, ha="center", va="center", weight="bold", zorder=3)
        ax.text(1.38, y + 0.08, title, fontsize=13.2, weight="bold", color=dark, va="center")
        ax.text(5.02, y + 0.08, detail, fontsize=11.4, color=muted, va="center")
        ax.text(15.22, y + 0.08, status, fontsize=9, weight="bold", color=green, ha="right", va="center")
        if index < len(rows):
            ax.plot([1.36, 15.35], [y - 0.28, y - 0.28], color=line, linewidth=0.8)

    ax.add_patch(Rectangle((0.64, 3.55), 14.72, 0.63, facecolor="#fff4df", edgecolor="none"))
    ax.text(0.82, 3.87, "PUNTO ACTUAL", fontsize=9.5, weight="bold", color=amber, va="center")
    ax.text(2.34, 3.87, f"{n_targets} conjuntos elegidos · unión: EURUSD {union['EURUSD']} / ETH {union['ETH']}",
            fontsize=13, weight="bold", color=dark, va="center")
    ax.text(0.82, 3.32,
            f"Utilidad pendiente: {positive}/{n_targets} ganadores con skill semanal medio > 0; "
            f"{random_eur}/{len(winners['EURUSD'])} ganadores EURUSD son control aleatorio.",
            fontsize=11, color=amber)

    ax.text(0.65, 2.86, "PRÓXIMA LÍNEA DE TRABAJO", fontsize=10, weight="bold", color=muted)
    next_rows = [
        (2.43, "01", "Auditar el contraste", "rezago de 3 h, naive pareado y dominio del control aleatorio"),
        (1.90, "02", "Ablación sobre entradas", "RAW vs reconstrucción solo con prefijo; target y naive intactos"),
        (1.37, "03", "Representación modular", "branches + fusión + núcleo frente a controles emparejados"),
    ]
    for y, number, title, detail in next_rows:
        ax.text(0.72, y, number, fontsize=12, weight="bold", color=blue, va="center")
        ax.text(1.35, y, title, fontsize=12.2, weight="bold", color=dark, va="center")
        ax.text(5.02, y, detail, fontsize=10.8, color=muted, va="center")
        ax.plot([0.68, 15.35], [y - 0.25, y - 0.25], color=line, linewidth=0.8)

    ax.text(0.68, 0.73, "DESPUÉS  ·  R0/R1/R2 → preentrenamiento del núcleo → DOIN → SAC/DQN → paper trading",
            fontsize=11, weight="bold", color=dark)
    ax.text(0.68, 0.35, "TEST cerrado. Ningún pronóstico de este banco pasa a la estrategia heurística sin superar el naive en las mismas filas.",
            fontsize=9.8, color=muted)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--wave-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(read_json(args.closure), read_json(args.freeze), read_json(args.wave_status), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
