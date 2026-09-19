#!/usr/bin/env python3
"""RP14/RP16: effects and tables of the ARCH comparison from a closure (df_mod_e0_close CLOSE.json).

Only VERIFIED cells enter; the denominator (design population) is stated with every table. Effects
per architecture: H2 e_a(h) and slope; H3 d_a,r over the fusion arms (averaged over readouts where
the controls exist) and gamma_a; READOUT rho_a,r (last - pooled, averaged over fusions); DONOR delta_a;
DX adequacy rows; receiver adequacy (model < naive on validation at h = 3, r = 1) gates the
interpretation. Replicate SDs and paired bootstrap over replicates are DESCRIPTIVE (n replicates is
stated). Costs per phase and stop reasons are tabulated per architecture, never only mean MASE.

    python tools/df_mod_e0_arch_verify.py --close CLOSE.json --design DESIGN.json --out ARCH_EFFECTS.json [--tables TABLES.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEQ_FUSIONS = ("sequence", "sequence_gap")
SUM_FUSIONS = ("summary", "summary_last")
LAST_READOUTS = ("sequence", "summary_last")
POOLED_READOUTS = ("sequence_gap", "summary")


def _mean(values):
    return float(np.mean(values)) if values else None


def _sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _slope(levels, values):
    if len(levels) < 2:
        return None
    A = np.vstack([np.asarray(levels, dtype=float), np.ones(len(levels))]).T
    return float(np.linalg.lstsq(A, np.asarray(values), rcond=None)[0][0])


def _parse(cell_id: str) -> dict:
    parts = cell_id.split("__")
    if parts[0] in ("H2", "H3"):
        cond, seed, arch, arm = parts[1], int(parts[2][1:]), parts[3], parts[4]
        return {"hypothesis": parts[0], "cond": cond, "seed": seed, "arch": arch, "arm": arm, "dsum": len(parts) > 5 and parts[5] == "dsum"}
    if parts[0] == "DX":
        return {"hypothesis": "DX", "cond": parts[1], "seed": int(parts[2][1:]), "arch": parts[3], "arm": parts[4], "dsum": False}
    return {"hypothesis": parts[0], "cond": None, "seed": None, "arch": None, "arm": None, "dsum": False}


def effects(close_local: dict, design: dict, split: str = "validation") -> dict:
    units = close_local["units"]
    verified = {k: u for k, u in units.items() if u["status"] == "VERIFIED" and u.get("role") == "CELL"}
    archs = design["archs"]
    out = {"schema": "df_mod_e0_arch_effects.v1", "split": split, "population": {"cells": len(close_local["population"]["members"]), "verified_cells": len(verified),
                                                                                     "closure": close_local["closure"]},
           "replicates": design["replicates"], "per_arch": {}, "adequacy": {}, "readout": {}, "donor": {}, "dx": {}, "costs": {}}
    for a in archs:
        cells = {k: u for k, u in verified.items() if u["record"].get("arch") == a}
        rec = lambda k: cells[k]["record"]
        # --- adequacy: H2 profiles at h=3 (or the max level), r=1: model < naive ---
        lv = max(design["levels"])
        adequate = []
        for k, u in cells.items():
            m = _parse(k)
            if m["hypothesis"] == "H2" and m["arm"] == "profiles" and m["cond"] == f"h{lv}":
                r = u["record"]
                adequate.append({"cell": k, "model": r["mase"][split], "naive": r["naive_mase"][split], "linear": r["linear_mase"][split], "oracle": r["oracle_mase"][split],
                                 "beats_naive": r["mase"][split] < r["naive_mase"][split]})
        out["adequacy"][a] = {"cells": adequate, "receiver_adequate": bool(adequate) and all(x["beats_naive"] for x in adequate),
                              "rule": f"every H2 profiles cell at h = {lv}, r = 1 beats the seasonal-naive on {split}"}
        # --- H2 within arch ---
        by = {}
        for k, u in cells.items():
            m = _parse(k)
            if m["hypothesis"] == "H2":
                by.setdefault((int(m["cond"][1:]), m["seed"]), {})[m["arm"]] = u["record"]["mase"][split]
        e_by, sd_by = {}, {}
        for h in sorted({lv_ for lv_, _ in by}):
            diffs = [arms["profiles"] - _mean([v for a_, v in arms.items() if a_.startswith("random_")])
                     for (lv_, s), arms in by.items() if lv_ == h and "profiles" in arms and any(a_.startswith("random_") for a_ in arms)]
            if diffs:
                e_by[h], sd_by[h] = _mean(diffs), _sd(diffs)
        h2 = {"e": e_by, "sd_replicates": sd_by, "slope": _slope(sorted(e_by), [e_by[h] for h in sorted(e_by)]), "n_replicates": len({s for _, s in by})}
        # --- H3 within arch: fusion contrast averaged over readouts, per (r, seed); readout contrast; donor ---
        h3 = {}
        by3 = {}
        for k, u in cells.items():
            m = _parse(k)
            if m["hypothesis"] == "H3" and m["arm"] not in ("extractor", "extractor_summary"):
                by3.setdefault((int(m["cond"][1:]), m["seed"], "dsum" if m["dsum"] else "seq"), {})[m["arm"]] = u["record"]["mase"][split]
        d, sd_d, rho, sd_rho = {}, {}, {}, {}
        for r in design["r_values"]:
            diffs, rdiffs = [], []
            for (rr, s, donor), arms in by3.items():
                if rr != r or donor != "seq":
                    continue
                seq = [v for f, v in arms.items() if f in SEQ_FUSIONS]
                summ = [v for f, v in arms.items() if f in SUM_FUSIONS]
                if seq and summ:
                    diffs.append(_mean(seq) - _mean(summ))
                last = [v for f, v in arms.items() if f in LAST_READOUTS]
                pooled = [v for f, v in arms.items() if f in POOLED_READOUTS]
                if last and pooled and len(arms) >= 4:
                    rdiffs.append(_mean(last) - _mean(pooled))
            if diffs:
                d[r], sd_d[r] = _mean(diffs), _sd(diffs)
            if rdiffs:
                rho[r], sd_rho[r] = _mean(rdiffs), _sd(rdiffs)
        h3 = {"d": d, "sd_replicates": sd_d, "gamma": (d[1] - d[0]) if 0 in d and 1 in d else None, "n_replicates": len({s for _, s, _ in by3}),
              "definition": "d_r = mean(sequence-fusion arms) - mean(summary-fusion arms) per replicate (readouts averaged where present)"}
        out["readout"][a] = {"rho": rho, "sd_replicates": sd_rho, "definition": "rho_r = mean(last readouts) - mean(pooled readouts) per replicate (fusions averaged); "
                                                                                  "only where the 2 x 2 exists"}
        dd = []
        for (rr, s, donor), arms in by3.items():
            if donor != "dsum" or rr != 1:
                continue
            seq_arms = by3.get((rr, s, "seq"), {})
            if "sequence" in arms and "summary" in arms and "sequence" in seq_arms and "summary" in seq_arms:
                dd.append((arms["sequence"] - arms["summary"]) - (seq_arms["sequence"] - seq_arms["summary"]))
        out["donor"][a] = {"delta": _mean(dd), "sd_replicates": _sd(dd), "n": len(dd),
                           "definition": "delta = d_1(donor trained with the summary receiver) - d_1(donor trained with the sequence receiver), paired by replicate"}
        dx = []
        for k, u in cells.items():
            m = _parse(k)
            if m["hypothesis"] == "DX":
                r = u["record"]
                dx.append({"cell": k, "seed": m["seed"], "model": r["mase"][split], "naive": r["naive_mase"][split], "linear": r["linear_mase"][split],
                           "oracle": r["oracle_mase"][split], "beats_naive": r["mase"][split] < r["naive_mase"][split],
                           "within_linear": r["mase"][split] <= r["linear_mase"][split] + 0.03})
        out["dx"][a] = dx
        costs = [u["record"]["cost"]["cpu_seconds"] for u in cells.values()]
        fits = [u["record"]["cost"].get("fit_seconds") for u in cells.values()]
        stops = {}
        for u in cells.values():
            stops[u["record"]["stop_reason"]] = stops.get(u["record"]["stop_reason"], 0) + 1
        out["costs"][a] = {"cells": len(cells), "cpu_seconds_total": float(sum(costs)), "cpu_seconds_mean": _mean(costs), "fit_seconds_mean": _mean([f for f in fits if f]),
                           "updates_mean": _mean([u["record"]["updates"] for u in cells.values()]), "stop_reasons": stops,
                           "parameters": sorted({json.dumps(u["record"].get("parameters"), sort_keys=True) for u in cells.values()})[:4]}
        out["per_arch"][a] = {"H2": h2, "H3": h3, "interpretable": out["adequacy"][a]["receiver_adequate"],
                              "mase_mean_validation_by_arm": {arm: _mean([u["record"]["mase"][split] for k, u in cells.items() if _parse(k)["arm"] == arm])
                                                              for arm in sorted({_parse(k)["arm"] for k in cells})}}
    # bootstrap over replicates (paired within replicate), descriptive
    rng = np.random.default_rng(0)
    seeds = sorted(design["replicates"])
    boot = {a: {"H2_slope": [], "H3_gamma": [], "H3_d1": []} for a in archs}
    for _ in range(1000):
        pick = list(rng.choice(seeds, size=len(seeds), replace=True))
        sub = {}
        for i, s in enumerate(pick):
            for k, u in verified.items():
                m = _parse(k)
                if m["seed"] == s:
                    kk = k.replace(f"__s{s}__", f"__s{100 + i}__")
                    sub[kk] = {**u, "record": {**u["record"], "seed": 100 + i}}
        eff = effects({"units": sub, "population": close_local["population"], "closure": close_local["closure"]}, {**design, "replicates": [100 + i for i in range(len(pick))]}, split) \
            if False else None
        # (a full recursive bootstrap is avoided: the per-arch estimators are recomputed inline)
        for a in archs:
            e_by = {}
            by = {}
            d = {}
            by3 = {}
            for kk, u in sub.items():
                m = _parse(kk)
                if u["record"].get("arch") != a:
                    continue
                if m["hypothesis"] == "H2":
                    by.setdefault((int(m["cond"][1:]), m["seed"]), {})[m["arm"]] = u["record"]["mase"][split]
                elif m["hypothesis"] == "H3" and m["arm"] not in ("extractor", "extractor_summary") and not m["dsum"]:
                    by3.setdefault((int(m["cond"][1:]), m["seed"]), {})[m["arm"]] = u["record"]["mase"][split]
            for h in sorted({lv_ for lv_, _ in by}):
                diffs = [arms["profiles"] - _mean([v for a_, v in arms.items() if a_.startswith("random_")])
                         for (lv_, s), arms in by.items() if lv_ == h and "profiles" in arms and any(a_.startswith("random_") for a_ in arms)]
                if diffs:
                    e_by[h] = _mean(diffs)
            sl = _slope(sorted(e_by), [e_by[h] for h in sorted(e_by)])
            if sl is not None:
                boot[a]["H2_slope"].append(sl)
            for r in design["r_values"]:
                diffs = []
                for (rr, s), arms in by3.items():
                    if rr != r:
                        continue
                    seq = [v for f, v in arms.items() if f in SEQ_FUSIONS]
                    summ = [v for f, v in arms.items() if f in SUM_FUSIONS]
                    if seq and summ:
                        diffs.append(_mean(seq) - _mean(summ))
                if diffs:
                    d[r] = _mean(diffs)
            if 1 in d:
                boot[a]["H3_d1"].append(d[1])
            if 0 in d and 1 in d:
                boot[a]["H3_gamma"].append(d[1] - d[0])
    out["bootstrap"] = {a: {k: ({"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n": len(v)} if v else None) for k, v in b.items()}
                        for a, b in boot.items()}
    out["bootstrap_note"] = f"percentile intervals from resampling {len(seeds)} replicates with replacement: descriptive precision, not confirmation"
    return out


def tables(eff: dict, close_local: dict, design: dict) -> str:
    split = eff["split"]
    lines = [f"# ARCH comparison — tables ({split}; population {eff['population']['cells']} cells, verified {eff['population']['verified_cells']}, closure {eff['population']['closure']})", ""]
    lines.append("## Receiver adequacy (H2 profiles at the top level, r = 1)")
    lines.append("| arch | cell | model MASE | naive | linear | oracle | beats naive |")
    lines.append("|---|---|---|---|---|---|---|")
    for a, ad in eff["adequacy"].items():
        for x in ad["cells"]:
            lines.append(f"| {a} | {x['cell']} | {x['model']:.4f} | {x['naive']:.4f} | {x['linear']:.4f} | {x['oracle']:.4f} | {x['beats_naive']} |")
    lines.append("")
    lines.append("## Effects per architecture (replicate = unit; SD with n replicates; descriptive)")
    lines.append("| arch | interpretable | e(h) | slope | d_0 | d_1 | gamma | rho_1 (last − pooled) | donor delta | n rep |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a, p in eff["per_arch"].items():
        h2, h3 = p["H2"], p["H3"]
        e = ", ".join(f"h{h}: {v:+.4f}" for h, v in h2["e"].items())
        rho = eff["readout"][a]["rho"].get(1)
        dd = eff["donor"][a]["delta"]
        lines.append(f"| {a} | {p['interpretable']} | {e} | {h2['slope'] if h2['slope'] is None else f'{h2['slope']:+.4f}'} | "
                     f"{h3['d'].get(0) if h3['d'].get(0) is None else f'{h3['d'][0]:+.4f}'} | {h3['d'].get(1) if h3['d'].get(1) is None else f'{h3['d'][1]:+.4f}'} | "
                     f"{h3['gamma'] if h3['gamma'] is None else f'{h3['gamma']:+.4f}'} | {rho if rho is None else f'{rho:+.4f}'} | {dd if dd is None else f'{dd:+.4f}'} | {h3['n_replicates']} |")
    lines.append("")
    lines.append("## Cells (raw error and MASE, denominator, control delta, support, updates, stop, cost)")
    lines.append("| cell | arch | arm | status | MAE val | MASE val | naive | linear | oracle | MASE test | denom mean | reach | updates | stop | cpu s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, u in sorted(close_local["units"].items()):
        if u.get("role") != "CELL":
            continue
        r = u.get("record") or {}
        if not r:
            lines.append(f"| {k} | | | {u['status']} | | | | | | | | | | | |")
            continue
        lines.append(f"| {k} | {r.get('arch')} | {r['arm']} | {u['status']} | {r['mae'][split]:.4f} | {r['mase'][split]:.4f} | {r['naive_mase'][split]:.4f} | "
                     f"{r['linear_mase'][split]:.4f} | {r['oracle_mase'][split]:.4f} | {r['mase'].get('test', float('nan')):.4f} | {np.mean(r['denominator']):.4f} | "
                     f"{r.get('support_reach')} | {r['updates']} | {r['stop_reason']} | {r['cost']['cpu_seconds']:.1f} |")
    lines.append("")
    lines.append("## Costs per architecture")
    lines.append("| arch | cells | cpu total s | cpu mean s | fit mean s | updates mean | stop reasons |")
    lines.append("|---|---|---|---|---|---|---|")
    for a, c in eff["costs"].items():
        lines.append(f"| {a} | {c['cells']} | {c['cpu_seconds_total']:.0f} | {c['cpu_seconds_mean'] or 0:.1f} | {c['fit_seconds_mean'] or 0:.1f} | {c['updates_mean'] or 0:.0f} | {c['stop_reasons']} |")
    lines.append("")
    lines.append("## Diagnostic trend/event (adequacy only)")
    lines.append("| arch | cell | model | naive | linear | oracle | beats naive | within linear + 0.03 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a, rows in eff["dx"].items():
        for x in rows:
            lines.append(f"| {a} | {x['cell']} | {x['model']:.4f} | {x['naive']:.4f} | {x['linear']:.4f} | {x['oracle']:.4f} | {x['beats_naive']} | {x['within_linear']} |")
    lines.append("")
    lines.append(f"Bootstrap: {eff['bootstrap_note']}")
    for a, b in eff["bootstrap"].items():
        lines.append(f"- {a}: " + ", ".join(f"{k} {v['ci95'][0]:+.4f}..{v['ci95'][1]:+.4f}" if v else f"{k} —" for k, v in b.items()))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=None)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)
    close = json.loads(args.close.read_text())
    local = close.get("local") or close
    design = json.loads(args.design.read_text())
    eff = effects(local, design, args.split)
    eff["effects_test"] = effects(local, design, "test") if args.split != "test" else None
    if eff["effects_test"]:
        eff["effects_test"] = {k: eff["effects_test"][k] for k in ("per_arch", "readout", "donor")}
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    args.out.write_text(json.dumps(eff, indent=1, sort_keys=True, default=str) + "\n")
    if args.tables:
        args.tables.write_text(tables(eff, local, design))
    print(json.dumps({a: {"interpretable": p["interpretable"], "H2_slope": p["H2"]["slope"], "H3_d": p["H3"]["d"], "gamma": p["H3"]["gamma"],
                          "rho_1": eff["readout"][a]["rho"].get(1), "donor_delta": eff["donor"][a]["delta"]} for a, p in eff["per_arch"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
