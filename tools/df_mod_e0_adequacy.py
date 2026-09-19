#!/usr/bin/env python3
"""RP21: adequacy and costs of the ARCH stage, from the closure, the design and the retained records.
Allowance is not capacity: per cell, the references are published apart (persistence naive, the MASE
denominator = train seasonal-naive MAE which is NOT a predictor score, the linear reference with its
point gap AND the historic +0.03 criterion, the oracle); the curves give best update, last update, stop
reason, margin to the allowance and whether the best checkpoint touches the ceiling (then convergence
is NOT declared); reach, W/P per group and the task; seed variation per task. Costs are compared on the
exact intersection of tasks x seeds x arms across architectures and per host stratum, with the donor's
initial cost and its amortisation over its real uses; pilots and the A/B-only donor sensitivity are
excluded from the architecture comparison. A learning-curve design is PREPARED, not launched.

    python tools/df_mod_e0_adequacy.py --close CLOSE.json --design DESIGN.json --root RUN_ROOT --out OUT.json [--tables OUT.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

LINEAR_TOLERANCE = 0.03
PERIOD_A = 24.0


def _parse(cid):
    p = cid.split("__")
    return {"hyp": p[0], "cond": p[1], "seed": int(p[2][1:]), "arch": p[3], "arm": p[4], "dsum": len(p) > 5}


def cell_rows(close_local: dict, design: dict, root: Path, split: str = "validation") -> list:
    by_id = {c["cell_id"]: c for c in design["cells"]}
    rows = []
    for cid, u in sorted(close_local["units"].items()):
        if u.get("role") != "CELL":
            continue
        c = by_id.get(cid) or {}
        m = _parse(cid)
        row = {"cell": cid, "task": f"{m['hyp']}/{m['cond']}/{m['arm']}" + ("/dsum" if m["dsum"] else ""), "arch": m["arch"], "seed": m["seed"],
               "host_role": c.get("host_role"), "status": u["status"]}
        r = u.get("record")
        if not r:
            row["state"] = u["status"]
            rows.append(row)
            continue
        rec = json.loads((root / "attempts" / cid / "cell.json").read_bytes()) if (root / "attempts" / cid / "cell.json").is_file() else {}
        tr = rec.get("training") or {}
        steps = int(tr.get("steps_per_epoch") or 0)
        best_epoch = int(tr.get("restored_checkpoint_epoch") or 0)
        epochs = int(tr.get("epochs") or 0)
        allowance = int((tr.get("rule") or {}).get("max_updates") or 0)
        updates = int(r["updates"])
        best_update = best_epoch * steps
        row.update({"model": r["mase"][split], "naive_persistence": r["naive_mase"][split], "linear": r["linear_mase"][split], "oracle": r["oracle_mase"][split],
                    "mase_denominator_train_seasonal_naive_mae": float(np.mean(r["denominator"])), "seasonal_naive_as_predictor": "NOT_MEASURED",
                    "gap_to_linear": r["mase"][split] - r["linear_mase"][split], "within_linear_plus_0_03": r["mase"][split] <= r["linear_mase"][split] + LINEAR_TOLERANCE,
                    "reaches_linear_point": r["mase"][split] <= r["linear_mase"][split], "beats_persistence": r["mase"][split] < r["naive_mase"][split],
                    "curve_train": (tr.get("curve") or {}).get("train"), "curve_validation": (tr.get("curve") or {}).get("validation"),
                    "best_epoch": best_epoch, "epochs": epochs, "steps_per_epoch": steps, "best_update": best_update, "last_update": updates,
                    "allowance": allowance, "margin_to_allowance": allowance - updates, "stop_reason": r["stop_reason"],
                    "best_at_ceiling": bool(r["stop_reason"] == "UPDATE_BUDGET" and best_epoch >= epochs),
                    "convergence_declared": False if (r["stop_reason"] == "UPDATE_BUDGET" and best_epoch >= epochs) else None,
                    "reach": r.get("support_reach"), "branch_reach": r.get("branch_reach"), "window": r.get("window"),
                    "periods": (rec.get("generator") or {}).get("groups"), "cost_cpu_seconds": r["cost"]["cpu_seconds"], "fit_seconds": r["cost"].get("fit_seconds"),
                    "extractor_weight_change": r.get("extractor_weight_change"), "assignment": r.get("assignment")})
        per = (rec.get("generator") or {}).get("groups") or {}
        if per and row["reach"]:
            row["reach_over_period"] = {g: row["reach"] / float(v["period"]) for g, v in per.items()}
            row["W_over_period"] = {g: float(row["window"]) / float(v["period"]) for g, v in per.items()}
        rows.append(row)
    return rows


def classify(rows: list) -> dict:
    """Separate causes per task: limited optimisation (best at ceiling), insufficient context (reach below the
    slowest period), seed variation (SD across seeds of the same task/arch), lack of support (few updates)."""
    out = {}
    by_task = {}
    for r in rows:
        if "model" not in r:
            continue
        by_task.setdefault((r["task"], r["arch"]), []).append(r)
    for (task, arch), rs in sorted(by_task.items()):
        vals = [r["model"] for r in rs]
        slow = max((v for r in rs for v in (r.get("W_over_period") or {}).values()), default=None)
        out[f"{task}|{arch}"] = {"n_seeds": len(rs), "mase_mean": float(np.mean(vals)), "mase_sd_seeds": float(statistics.stdev(vals)) if len(vals) > 1 else None,
                                 "best_at_ceiling_count": sum(r["best_at_ceiling"] for r in rs), "stop_reasons": sorted({r["stop_reason"] for r in rs}),
                                 "reach_below_slowest_period": any((r.get("reach") or 0) < max(float(v["period"]) for v in (r.get("periods") or {}).values()) for r in rs if r.get("periods")),
                                 "W_over_slowest_period": slow, "flags": []}
        e = out[f"{task}|{arch}"]
        if e["best_at_ceiling_count"]:
            e["flags"].append(f"LIMITED_OPTIMISATION ({e['best_at_ceiling_count']}/{len(rs)} best checkpoints at the allowance)")
        if e["reach_below_slowest_period"]:
            e["flags"].append(f"CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach {max((r.get('reach') or 0) for r in rs)} < slowest period {max(float(v['period']) for r in rs for v in (r.get('periods') or {}).values()):.0f}; by construction for ARCH-0/A, and for B/C only against P_B at h3)")
        if e["mase_sd_seeds"] is not None and e["mase_sd_seeds"] > 0.01:
            e["flags"].append(f"SEED_VARIATION (sd {e['mase_sd_seeds']:.4f} across {len(rs)} seeds)")
    return out


def costs(rows: list, design: dict) -> dict:
    """Costs on the exact intersection: tasks x seeds present for EVERY architecture; per host stratum;
    donor initial cost amortised over its real uses."""
    archs = design["archs"]
    rows_ok = [r for r in rows if "model" in r and not r["task"].endswith("/dsum") and not r["task"].endswith("/extractor_summary")]
    keys = {}
    for r in rows_ok:
        keys.setdefault((r["task"], r["seed"]), {})[r["arch"]] = r
    common = {k: v for k, v in keys.items() if set(v) == set(archs)}
    per_arch = {a: {"cells": 0, "cpu": 0.0, "fit": 0.0, "by_host": {}} for a in archs}
    for (task, seed), by_arch in common.items():
        for a, r in by_arch.items():
            p = per_arch[a]
            p["cells"] += 1
            p["cpu"] += r["cost_cpu_seconds"]
            p["fit"] += r.get("fit_seconds") or 0.0
            h = p["by_host"].setdefault(r["host_role"], {"cells": 0, "cpu": 0.0})
            h["cells"] += 1
            h["cpu"] += r["cost_cpu_seconds"]
    # donor amortisation: extractor cells and the arms that used them (same arch, r, seed)
    donors = {}
    for r in rows:
        if "model" in r and r["task"].endswith("/extractor"):
            donors[(r["arch"], r["task"].split("/")[1], r["seed"])] = {"cost": r["cost_cpu_seconds"], "uses": 0}
    for r in rows:
        if "model" in r and r["task"].split("/")[0] == "H3" and r["task"].split("/")[2] in ("sequence", "sequence_gap", "summary", "summary_last") and not r["task"].endswith("/dsum"):
            k = (r["arch"], r["task"].split("/")[1], r["seed"])
            if k in donors:
                donors[k]["uses"] += 1
    amort = {a: {"donor_cells": 0, "donor_cpu": 0.0, "uses": 0} for a in archs}
    for (a, cond, seed), d in donors.items():
        amort[a]["donor_cells"] += 1
        amort[a]["donor_cpu"] += d["cost"]
        amort[a]["uses"] += d["uses"]
    for a in archs:
        amort[a]["donor_cpu_per_use"] = amort[a]["donor_cpu"] / amort[a]["uses"] if amort[a]["uses"] else None
    return {"intersection": {"tasks_x_seeds": len(common), "rule": "only (task, seed) pairs executed by EVERY architecture; pilots, donor-sensitivity cells and the "
                                                                    "summary-trained donors excluded; hosts differ, so per-host strata are shown apart"},
            "per_arch": {a: {**p, "cpu_per_cell": (p["cpu"] / p["cells"]) if p["cells"] else None} for a, p in per_arch.items()},
            "hosts_note": "host strata: the same cell type on different hosts is not the same hardware; compare within a stratum",
            "donor_amortisation": amort}


def learning_curve_design(rows: list, design: dict, measured: dict | None) -> dict:
    """PREPARED, not launched: allowances derived from the observed curves (best updates) and the pilot costs."""
    best = [r["best_update"] for r in rows if "model" in r and r["best_update"]]
    q = np.quantile(best, [0.5, 0.9, 1.0]).tolist() if best else []
    allowances = sorted({300, 600, 1100, int(2 * 1100)})
    per_update = {}
    for k, v in (measured or {}).items():
        per_update[k] = v.get("seconds_per_update")
    projection = {}
    for a in design["archs"]:
        spu = per_update.get(f"pilot__{a}__H2_profiles")
        projection[a] = {u: (spu * u + 12.0) if spu else None for u in allowances}
    return {"purpose": "distinguish limited optimisation from lack of capacity: one task (H2 h3 profiles), every architecture, the same seeds, "
                       "allowances on a ladder; the curve of best validation MASE vs allowance decides, not a universal update number",
            "observed_best_updates_quantiles_50_90_100": q, "allowances": allowances, "task": "H2__h3__s{seed}__{arch}__profiles",
            "seeds": design["replicates"], "cells": len(allowances) * len(design["archs"]) * len(design["replicates"]),
            "projected_cpu_seconds_per_cell": projection,
            "projected_total_cpu_seconds": float(sum(v for a in projection.values() for v in a.values() if v) * len(design["replicates"])),
            "status": "PREPARED_NOT_LAUNCHED"}


def tables(doc: dict) -> str:
    L = ["# Adequacy and costs (RP21)", "", "## Per cell", "",
         "| cell | task | arch | host | model | persistence | linear | gap | ≤ lin+0.03 | oracle | best upd | last upd | allowance | stop | best at ceiling | reach | W/P | cpu s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["cells"]:
        if "model" not in r:
            L.append(f"| {r['cell']} | {r['task']} | {r['arch']} | {r['host_role']} | {r['status']} | | | | | | | | | | | | | |")
            continue
        wp = ", ".join(f"{g}: {v:.2f}" for g, v in (r.get("W_over_period") or {}).items())
        L.append(f"| {r['cell']} | {r['task']} | {r['arch']} | {r['host_role']} | {r['model']:.4f} | {r['naive_persistence']:.4f} | {r['linear']:.4f} | {r['gap_to_linear']:+.4f} | "
                 f"{r['within_linear_plus_0_03']} | {r['oracle']:.4f} | {r['best_update']} | {r['last_update']} | {r['allowance']} | {r['stop_reason']} | {r['best_at_ceiling']} | "
                 f"{r['reach']} | {wp} | {r['cost_cpu_seconds']:.1f} |")
    L += ["", "## Causes per task and architecture", "", "| task | arch | n | MASE mean | sd seeds | best at ceiling | flags |", "|---|---|---|---|---|---|---|"]
    for k, e in doc["classification"].items():
        task, arch = k.split("|")
        L.append(f"| {task} | {arch} | {e['n_seeds']} | {e['mase_mean']:.4f} | {e['mase_sd_seeds'] if e['mase_sd_seeds'] is None else f'{e['mase_sd_seeds']:.4f}'} | {e['best_at_ceiling_count']} | {'; '.join(e['flags'])} |")
    c = doc["costs"]
    L += ["", f"## Costs on the exact intersection ({c['intersection']['tasks_x_seeds']} task × seed pairs; {c['intersection']['rule']})", "",
          "| arch | cells | cpu total | cpu per cell | fit total | by host | donor cells | donor cpu | uses | donor cpu per use |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a, p in c["per_arch"].items():
        d = c["donor_amortisation"][a]
        L.append(f"| {a} | {p['cells']} | {p['cpu']:.0f} | {p['cpu_per_cell'] or 0:.1f} | {p['fit']:.0f} | {json.dumps({h: round(v['cpu']) for h, v in p['by_host'].items()})} | "
                 f"{d['donor_cells']} | {d['donor_cpu']:.0f} | {d['uses']} | {d['donor_cpu_per_use'] if d['donor_cpu_per_use'] is None else f'{d['donor_cpu_per_use']:.1f}'} |")
    lc = doc["learning_curve_design"]
    L += ["", f"## Learning-curve design ({lc['status']})", "", f"- allowances {lc['allowances']}; observed best updates quantiles 50/90/100: {lc['observed_best_updates_quantiles_50_90_100']}",
          f"- cells {lc['cells']}; projected CPU {lc['projected_total_cpu_seconds']:.0f} s from the pilots' per-update cost"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None, help="REPORT.json with the measured pilot costs (for the learning-curve projection)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=None)
    args = parser.parse_args(argv)
    close = json.loads(args.close.read_text())
    local = close.get("local") or close
    design = json.loads(args.design.read_text())
    if local.get("design_sha256") != design["design_sha256"]:
        raise SystemExit("REFUSED: the closure is not of this design")
    rows = cell_rows(local, design, args.root)
    measured = (json.loads(args.report.read_text()).get("cost_pilot") if args.report else None) or None
    doc = {"schema": "df_mod_e0_adequacy.v1", "design_sha256": design["design_sha256"], "cells": rows, "classification": classify(rows), "costs": costs(rows, design),
           "learning_curve_design": learning_curve_design(rows, design, measured),
           "notes": ["persistence naive = the naive of prepare() (last observation); the seasonal naive is only the MASE denominator on train",
                     "convergence is never declared where the best checkpoint touches the allowance",
                     "DX is a distinct condition and decides no regime"]}
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    args.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if args.tables:
        args.tables.write_text(tables(doc))
    print(json.dumps({"cells": len(rows), "classification": {k: v["flags"] for k, v in doc["classification"].items() if v["flags"]}, "costs": doc["costs"]["per_arch"],
                      "learning_curve": {k: doc["learning_curve_design"][k] for k in ("cells", "projected_total_cpu_seconds")}}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
