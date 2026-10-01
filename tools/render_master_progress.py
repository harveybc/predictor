"""Render a bounded, evidence-derived program snapshot without importing ML engines.

Reads campaign metadata and verification receipts, not prediction arrays or test
data. Percentages have explicit denominators; there is no invented overall score.
The JSON sidecar contains only the fields used in the image and input digests.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_json(path, inputs):
    """Bind each parsed metadata file to its exact consumed bytes."""
    raw = path.read_bytes()
    inputs[path.name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def collect(root):
    """Reconcile current candidate objectives against their verification receipts."""
    inputs = {}
    status = read_json(root / "STATUS.json", inputs)
    results = root / "RESULTS"
    queue = read_json(results / "corrected_QUEUE_r0_v1.export.json", inputs)
    coverage = read_json(results / "coverage_laneB.json", inputs)
    traffic = read_json(results / "traffic_h96_closure_table.json", inputs)
    current = [c for c in queue["candidates"] if c["status"] != "SUPERSEDED_OLD_ARCH"]
    verified = [c for c in current if c["status"] == "verified"]
    groups = defaultdict(list)
    negatives = defaultdict(int)
    baselines = set()
    population_ids = set()
    for candidate in verified:
        receipt = read_json(results / "corrected_verify_receipts" /
                            (candidate["cid"][:16] + ".json"), inputs)
        if receipt.get("exact_match") is not True:
            raise ValueError("Candidate has no exact replay receipt")
        if receipt["objective"]["rescored_value"] != candidate["objective"]:
            raise ValueError("Queue objective differs from receipt")
        if receipt["objective"]["split"] != "validation":
            raise ValueError("Unexpected candidate split")
        baselines.add(receipt["metrics"]["baseline_MAE"])
        population_ids.add(receipt["digests"]["validation_sha256"])
        groups[candidate["config_id"]].append(candidate)
        for horizon, metrics in receipt["per_horizon"].items():
            negatives[horizon] += int(metrics["skill_MAE"] < 0)
    if len(baselines) != 1 or len(population_ids) != 1:
        raise ValueError("Candidates do not share a baseline and validation identity")
    paired = [rows for rows in groups.values()
              if len(rows) == 2 and {r["seed"] for r in rows} == {2021, 2022}]
    if not paired:
        raise ValueError("No complete paired-seed configuration")
    best = min(paired, key=lambda rows: sum(c["objective"] for c in rows) / 2)
    donors = next(j for j in status["jobs"] if j["id"] == "m02-ecl-v2-donors-r7g2")
    return {
        "observed_at": status["observed_at"],
        "scope": "Metadata/receipt reconciliation; no new inference or training",
        "inputs_sha256": inputs,
        "rows": [
            {"name": "Motor temporal integrado", "done": 8, "total": 8,
             "detail": "MS01-MS08 declarados verificados | codigo 3ecdb256", "color": "#18785e"},
            {"name": "Traffic / replica de referencia", "done": len(traffic["rows"]), "total": 12,
             "detail": "H96 listo; H192 / H336 / H720 pendientes", "color": "#347ac1"},
            {"name": "DOIN / lote modular corregido", "done": len(verified), "total": len(current),
             "detail": "16 R0 listos; 8 por rasgo + 12 R1/R2 pendientes", "color": "#347ac1"},
            {"name": "Preentrenamiento de ramas", "done": donors["progress"]["completed"],
             "total": donors["progress"]["total"],
             "detail": "En CPU; el nucleo es una etapa posterior", "color": "#b78418"},
            {"name": "Perfiles / indice de columnas", "done": coverage["old_denominator"]["covered"],
             "total": coverage["old_denominator"]["distinct_rows"],
             "detail": "Cobertura del indice; no de todas las fuentes ni metricas", "color": "#b78418"},
        ],
        "best": {"label": best[0]["label"],
                 "mae": sum(c["objective"] for c in best) / 2,
                 "naive": next(iter(baselines)), "n": 2,
                 "scope": "ECL L24/H1..24, validation; no published comparable row"},
        "negative_skill_counts": dict(negatives),
        "traffic": {k: {a: traffic["mean"][k][a] for a in ["mean", "naive", "published"]}
                    for k in ["mse", "mae"]},
        "coverage": {"files": coverage["new_denominator"]["files"],
                     "providers": coverage["sources"]["rows"],
                     "transform_rows": coverage["transform_ledger"]["rows"],
                     "evaluated": coverage["transform_ledger"]["evaluated_true"]},
        "donor_eta": donors["eta"],
        "devices": [{k: d.get(k) for k in ["host_alias", "name", "state", "temperature_c", "observed_at"]}
                    for d in status["devices"]],
        "next_actions": ["Aplicar fix de admision 775c5545 en el obrero preferido",
                         "Continuar piloto por rasgo y cola DOIN sin duplicados",
                         "Terminar donantes, preentrenar nucleo, comparar R1/R2",
                         "Conciliar fuentes y avanzar seleccion / causalidad",
                         "Conectar finalista financiero al runner shadow/paper"],
    }


def local_time(value):
    """Display operational times in the owner's fixed UTC-05 timezone."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone(timedelta(hours=-5))).strftime("%H:%M")


def render(data, output):
    """Draw status bars, current bottlenecks, measured scores and next milestones."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, muted = "#18252b", "#59676e"

    def text(x, y, value, size=11, color=ink, weight="normal", **kwargs):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                va="top", transform=ax.transAxes, **kwargs)

    def line(y):
        ax.plot([.045, .955], [y, y], color="#d9e0e2", lw=1)

    text(.045, .962, "DEL DATO AL MODELO DE TRADING", 24, weight="bold")
    text(.045, .923, "PROGRESO DEL PROGRAMA  |  " + data["observed_at"] +
         "  |  " + local_time(data["observed_at"]) + " Colombia", 10, muted)
    text(.045, .898, "Porcentajes por entregable y denominador. Software, datos y resultados se muestran por separado.",
         10, muted)
    line(.872)
    text(.045, .850, "ENTREGABLES Y MEDICIONES", 12, weight="bold")
    text(.685, .850, "AHORA / SIGUIENTE", 12, weight="bold")
    for index, row in enumerate(data["rows"]):
        y = .809 - index * .101
        ratio = row["done"] / row["total"]
        if not 0 <= ratio <= 1:
            raise ValueError("Invalid progress denominator")
        text(.045, y, row["name"], 12, weight="bold")
        text(.633, y, f"{100*ratio:.1f}%", 12, row["color"], "bold", ha="right")
        ax.add_patch(Rectangle((.045, y-.037), .588, .011, color="#e9edef", lw=0))
        ax.add_patch(Rectangle((.045, y-.037), .588*ratio, .011, color=row["color"], lw=0))
        text(.045, y-.049, f"{row['done']:,} / {row['total']:,}  |  {row['detail']}", 9, muted)
    eta = data["donor_eta"]
    interval = (local_time(eta["earliest"]) + " - " + local_time(eta["latest"])) if eta["earliest"] else "Sin estimacion"
    text(.685, .804, "RAM / ADMISION", 10, "#b65b37", "bold")
    text(.685, .779, "5090 detectada: 37 C, sin ajuste activo.\nCache residual limita la admision.\nNo se requiere reinicio para aplicar el fix.", 10)
    text(.685, .704, "ETA DE RAMAS  |  " + interval, 11, "#18785e", "bold")
    text(.685, .678, "Hora Colombia; estimacion por ritmo observado.\nNo incluye nucleo ni R1/R2.\nPiloto por rasgo: espera admision.\nETA total: requiere medir las etapas restantes.", 10)
    text(.685, .584, "FUENTES Y TRANSFORMACIONES", 10, weight="bold")
    c = data["coverage"]
    text(.685, .558, f"{c['providers']} proveedores / {c['files']:,} archivos descubiertos.\n"
         f"{c['transform_rows']} filas del registro de transformaciones.\n{c['evaluated']} evaluadas en esa ampliacion.\nDescubrir no equivale a validar ni seleccionar.", 10)
    text(.685, .465, "FINANZAS / LTS / M5PHET", 10, weight="bold")
    text(.685, .440, "SAC / DQN x con / sin ramas: plan paralelo.\nArranque: seleccion + datasets del lago.\nHeuristica: pronosticos deben superar naive.\nSin ganador financiero modular demostrado.", 10)
    line(.361)
    text(.045, .341, "RESULTADOS RETENIDOS", 12, weight="bold")
    b = data["best"]
    text(.045, .307, "MODULAR CORREGIDO / ECL", 11, "#347ac1", "bold")
    text(.045, .279, f"MAE_z  {b['mae']:.6f}     |     Naive  {b['naive']:.6f}", 17, weight="bold")
    text(.045, .243, "Mejor configuracion, dos semillas; validacion L24 / H1..24.\n"
         "Sin fila publicada comparable. Skill negativo en h1, h23 y h24\nen las 16 celdas verificadas; h2 negativo en 2/16.", 10, muted)
    t = data["traffic"]
    text(.55, .307, "TIMEFILTER / TRAFFIC H96", 11, "#18785e", "bold")
    text(.55, .279, f"MSE  {t['mse']['mean']:.6f}     MAE  {t['mae']['mean']:.6f}", 16, weight="bold")
    text(.55, .243, f"Publicado: {t['mse']['published']:.3f} / {t['mae']['published']:.3f}   |   Tres semillas\n"
         f"Naive: {t['mse']['naive']:.6f} / {t['mae']['naive']:.6f}\n"
         "Espacio normalizado y receta del autor; no equivalencia estadistica.", 10, muted)
    line(.168)
    text(.045, .148, "RUTA DE CONTINUACION", 11, weight="bold")
    stages = [("01", "Admitir 5090", "Fix + prueba CUDA"),
              ("02", "Completar el lote", "R0 + ramas + nucleo"),
              ("03", "Comparar R1/R2", "Mismas filas y coste"),
              ("04", "Validar finanzas", "Seleccion y riesgo"),
              ("05", "Consumir en LTS", "Shadow / paper")]
    for i, (number, title, detail) in enumerate(stages):
        x = .045 + i*.183
        text(x, .114, number, 14, "#347ac1", "bold")
        text(x+.031, .112, title, 10, weight="bold")
        text(x, .082, detail, 9, muted)
    text(.045, .031, "Snapshot de evidencia, no monitor en vivo. Fuentes y hashes en el JSON adjunto. No se reentreno para producir este grafico.", 9, muted)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    render(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
