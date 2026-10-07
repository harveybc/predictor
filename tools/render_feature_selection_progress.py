#!/usr/bin/env python3
"""Render the current feature-selection milestones from retained FS4 status."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


def render(status_path: Path, output_path: Path, wave_status_path: Path | None = None) -> None:
    status = json.loads(status_path.read_text())
    not_available = status.get("not_available_for_train", 0)
    if isinstance(not_available, dict):
        not_available = not_available["total"]
    done = status["complete"] + not_available
    total = status.get("total", status.get("expected", {}).get("total"))
    if not 0 <= done <= total or total <= 0:
        raise ValueError("Invalid FS4 denominator")
    if wave_status_path is not None:
        wave = json.loads(wave_status_path.read_text())
        wave_total = wave["tasks_per_arm"] * 3
        wave_done = sum(wave["done_per_arm"].values())
        trained_measured = wave["counts"]["TRAINED_ENCODER"]["COMPLETE"]
        trained_unavailable = wave["typed_refused_per_arm"]["TRAINED_ENCODER"]
        wave_detail = (f"{wave_done:,}/{wave_total:,} tareas resueltas; encoder: "
                       f"{trained_measured} medidos, {trained_unavailable} sin TRAIN")
        wave_progress = wave_done / wave_total
        if wave_done == wave_total:
            footer = "Primera ola cerrada; faltan cierre parcial y comparacion semanal para seleccionar."
        elif wave.get("gpu_eta_seconds") is not None:
            eta_hours = wave["gpu_eta_seconds"] / 3600
            footer = (f"ETA ola GPU ~{eta_hours:.0f} h si ambos workers siguen sanos; "
                      "la seleccion final requiere walk-forward semanal.")
        else:
            footer = "ETA de ola pendiente de cinco tiempos GPU recientes; la seleccion final requiere walk-forward."
    else:
        wave_detail = f"{done:,}/{total:,} tareas del plan amplio; no es seleccion final"
        wave_progress = done / total
        footer = "ETA de seleccion final: sin base fiable hasta medir la comparacion semanal."
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ink = "#182b36"
    teal = "#087f79"
    blue = "#3b6582"
    orange = "#dc8046"
    light = "#e5ebed"
    muted = "#61727b"
    rows = [
        ("01", "Conocimiento del negocio", "Contrato semanal definido; ejecucion walk-forward pendiente", None, blue),
        ("02", "Perfil individual y causal", "Fase 1 cerrada para EURUSD y ETH", 1.0, teal),
        ("03", "Matriz entre caracteristicas", "70 198 pares; 7 458 876 metricas cruzadas", 1.0, teal),
        ("04", "Rankings por objetivo", "20/20 objetivos; 980 subconjuntos candidatos", 1.0, teal),
        ("05", "Preseleccion preliminar", "126 rasgos en primera ola; despacho GPU ligado al manifiesto", 1.0, teal),
        ("06", "Extractibilidad / reconstruccion", wave_detail, wave_progress, orange),
        ("07", "Validacion semanal y seleccion", "Sin cierre comparativo ni manifiesto de seleccion", 0.0, muted),
        ("08", "Representacion modular y trading", "Depende del manifiesto seleccionado y del walk-forward", None, muted),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 10.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    ax.text(0.75, 10.05, "SELECCION DE CARACTERISTICAS", fontsize=25, weight="bold", color=ink)
    ax.text(0.76, 9.66, "Hitos reales, no porcentaje de investigacion inventado", fontsize=12, color=muted)
    ax.text(14.8, 9.72, generated, fontsize=9, color=muted, ha="right")
    ax.plot([1.09, 1.09], [1.18, 8.96], color=light, lw=4, zorder=1)

    for i, (number, title, detail, progress, color) in enumerate(rows):
        y = 8.87 - i * 1.02
        ax.add_patch(Circle((1.09, y), 0.21, facecolor=color, edgecolor="white", lw=2, zorder=3))
        ax.text(1.09, y, number, fontsize=9, weight="bold", color="white", ha="center", va="center", zorder=4)
        ax.text(1.52, y + 0.16, title, fontsize=15, weight="bold", color=ink, va="center")
        ax.text(1.53, y - 0.21, detail, fontsize=10.5, color=muted, va="center")
        if progress is not None:
            x, width = 11.43, 2.42
            ax.add_patch(FancyBboxPatch((x, y - 0.085), width, 0.18,
                boxstyle="round,pad=0.01,rounding_size=0.07", facecolor=light, edgecolor="none"))
            if progress:
                ax.add_patch(FancyBboxPatch((x, y - 0.085), max(0.06, width * progress), 0.18,
                    boxstyle="round,pad=0.01,rounding_size=0.07", facecolor=color, edgecolor="none"))
            ax.text(14.75, y, f"{progress:.1%}", fontsize=12, weight="bold", color=color,
                    ha="right", va="center")
        else:
            label = "CONTRATO" if number == "01" else "POST-SELECCION"
            ax.text(14.75, y, label, fontsize=10, weight="bold", color=color,
                    ha="right", va="center")

    ax.plot([0.76, 14.75], [0.69, 0.69], color=light, lw=1)
    ax.text(0.76, 0.38, footer,
            fontsize=10, color=muted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--wave-status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.status, args.output, args.wave_status)
