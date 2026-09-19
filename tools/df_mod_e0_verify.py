#!/usr/bin/env python3
"""MOD-E0-DEV independent verification and effects (RP5): MASE/MAE recomputed from the persisted
arrays with the shared denominators, extractor freezing and H2 information equality re-checked
from the records, e(h) / slope / d_r / gamma with the declared sign and replicate as the unit,
parent and live-warehouse content compared, tables and figures written.

    python tools/df_mod_e0_verify.py --root RUN_ROOT [--no-warehouse] [--out VERIFY.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("df_mod_e0")


# --- effects with the declared sign; the unit is the replicate ---------------------------------------------

def effects(cells: dict) -> dict:
    """`cells`: {cell_id: {hypothesis, level|r, seed, arm, mase}} (validation or test MASE of one split).
    H2: e(h) = mean over replicates of [MASE(profiles) - mean over random assignments MASE(random)];
    slope by least squares over h. H3: d_r = mean over replicates of [MASE(sequence) - MASE(summary)];
    gamma = d_1 - d_0. Replicate SDs and the number of units are reported, never inflated by
    assignments, optimiser seeds or windows."""
    out = {}
    h2 = [c for c in cells.values() if c["hypothesis"] == "H2" and c.get("mase") is not None]
    if h2:
        by = {}
        for c in h2:
            by.setdefault((c["level"], c["seed"]), {})[c["arm"]] = c["mase"]
        e_by_level, sd_by_level = {}, {}
        n_random = 0
        for level in sorted({k[0] for k in by}):
            diffs = []
            for (lv, seed), arms in by.items():
                if lv != level or "profiles" not in arms:
                    continue
                rand = [v for a, v in arms.items() if a.startswith("random_")]
                if not rand:
                    continue
                n_random = max(n_random, len(rand))
                diffs.append(arms["profiles"] - float(np.mean(rand)))
            if diffs:
                e_by_level[level] = float(np.mean(diffs))
                sd_by_level[level] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
        levels = sorted(e_by_level)
        slope = None
        if len(levels) >= 2:
            A = np.vstack([np.asarray(levels, dtype=float), np.ones(len(levels))]).T
            slope = float(np.linalg.lstsq(A, np.asarray([e_by_level[h] for h in levels]), rcond=None)[0][0])
        out["H2"] = {"e": e_by_level, "sd_replicates": sd_by_level, "slope": slope, "levels": levels,
                     "replicates": len({k[1] for k in by}), "random_assignments": n_random,
                     "unit": "replicate (independent trajectory); random assignments averaged within the replicate",
                     "sign": "negative favours profile grouping; a negative slope means the advantage grows with heterogeneity"}
    h3 = [c for c in cells.values() if c["hypothesis"] == "H3" and c.get("mase") is not None and c["arm"] in ("sequence", "summary")]
    if h3:
        by = {}
        for c in h3:
            by.setdefault((c["r"], c["seed"]), {})[c["arm"]] = c["mase"]
        d, sd = {}, {}
        for r in (0, 1):
            diffs = [arms["sequence"] - arms["summary"] for (rr, seed), arms in by.items() if rr == r and {"sequence", "summary"} <= set(arms)]
            if diffs:
                d[r] = float(np.mean(diffs))
                sd[r] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
        gamma = (d[1] - d[0]) if 0 in d and 1 in d else None
        out["H3"] = {"d": d, "sd_replicates": sd, "gamma": gamma, "n_units": len({k[1] for k in by}),
                     "unit": "replicate; both arms of a replicate share the frozen extractor",
                     "sign": "negative d favours sequence fusion; negative gamma means the advantage is larger with lagged dependence"}
    return out


def bootstrap_effects(cells: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile intervals by resampling REPLICATES with their paired arms (descriptive precision)."""
    rng = np.random.default_rng(seed)
    seeds = sorted({c["seed"] for c in cells.values()})
    samples = {"H2_slope": [], "H3_d1": [], "H3_gamma": []}
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sub = {}
        for i, s in enumerate(pick):
            for k, c in cells.items():
                if c["seed"] == s:
                    sub[f"{k}__b{i}"] = {**c, "seed": f"{s}_{i}"}
        eff = effects(sub)
        if "H2" in eff and eff["H2"]["slope"] is not None:
            samples["H2_slope"].append(eff["H2"]["slope"])
        if "H3" in eff and eff["H3"]["gamma"] is not None:
            samples["H3_d1"].append(eff["H3"]["d"][1])
            samples["H3_gamma"].append(eff["H3"]["gamma"])
    return {k: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None, "n": len(v)} for k, v in samples.items()}


# --- verification of cells ------------------------------------------------------------------------------------

def verify_cell(attempt: Path) -> dict:
    entry = {"attempt": attempt.name, "problems": [], "record": None}
    if not (attempt / "outcome.json").is_file():
        entry["problems"].append("no outcome.json")
        return entry
    outcome = json.loads((attempt / "outcome.json").read_text())
    if outcome.get("status") != "COMPLETED":
        entry["carried"] = (outcome.get("summary") or {}).get("outcome")
        return entry
    result = json.loads((attempt / "result.json").read_text())
    body = (attempt / "cell.json").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (outcome.get("verified") or {}).get("output_sha256"):
        entry["problems"].append("cell.json bytes differ from the declared/verified digest")
        return entry
    rec = json.loads(body)
    arrays = attempt / "arrays.npz"
    if hashlib.sha256(arrays.read_bytes()).hexdigest() != rec["arrays_sha256"]:
        entry["problems"].append("arrays altered")
        return entry
    arr = np.load(arrays)
    denom = arr["denominator"].tolist()
    if denom != rec["mase_denominator"]:
        entry["problems"].append("denominator differs")
    recomputed = {}
    for part in [q for q in ("train", "validation", "test") if f"{q}_y" in arr.files]:
        y = arr[f"{part}_y"]
        if not np.isfinite(y).all() or not np.isfinite(arr[f"{part}_pred"]).all():
            entry["problems"].append(f"{part}: non-finite arrays")
            continue
        recomputed[part] = {k: E.mase(arr[f"{part}_{src}"], y, denom) for k, src in (("model", "pred"), ("naive", "naive"), ("oracle", "oracle"))}
        for k in ("model", "naive", "oracle"):
            a, b = recomputed[part][k]["mase_mean"], rec["scores"][part][k]["mase_mean"]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                entry["problems"].append(f"{part} {k}: record {b} vs recomputed {a}")
    # the generator identity: regenerate and compare the digest of x (the data really consumed)
    g = E.generate(rec["level"], rec["r"], rec["seed"])
    if hashlib.sha256(np.ascontiguousarray(g["x"]).tobytes()).hexdigest() != rec["x_sha256"]:
        entry["problems"].append("consumed data are not the generator's output for (level, r, seed)")
    if rec["hypothesis"] == "H3" and rec["arm"] in ("sequence", "summary"):
        if not rec.get("frozen") or rec.get("extractor_weight_change", 1.0) != 0.0:
            entry["problems"].append("H3 arm: the extractor was not frozen")
    if rec["hypothesis"] == "H2":
        if rec["arm"] == "profiles" and not rec["assignment_is_profile"]:
            entry["problems"].append("H2 profiles arm did not use the profile partition")
        if rec["arm"].startswith("random") and rec["assignment_is_profile"]:
            entry["problems"].append("H2 random arm equals the profile partition (a relabeling)")
    if not rec.get("prediction_parity_after_reload"):
        entry["problems"].append("reload parity failed")
    entry["recomputed"] = {p: {k: v["mase_mean"] for k, v in d.items()} for p, d in recomputed.items()}
    entry["record"] = {k: rec.get(k) for k in ("cell_id", "hypothesis", "level", "r", "seed", "arm", "role", "assignment", "assignment_is_profile",
                                              "fusion", "parameters", "exposure", "extractor_weight_change", "receptive_field", "window")}
    entry["record"]["profiles_ari"] = rec["profiles"]["ari_vs_latent"]
    entry["record"]["updates"] = rec["training"]["updates"]
    entry["record"]["stop_reason"] = rec["training"]["stop_reason"]
    entry["record"]["cost"] = rec["cost"]
    entry["record"]["mase"] = {p: rec["scores"][p]["model"]["mase_mean"] for p in rec["scores"]}
    entry["record"]["mae"] = {p: rec["scores"][p]["model"]["mae_mean"] for p in rec["scores"]}
    entry["record"]["naive_mase"] = {p: rec["scores"][p]["naive"]["mase_mean"] for p in rec["scores"]}
    entry["record"]["oracle_mase"] = {p: rec["scores"][p]["oracle"]["mase_mean"] for p in rec["scores"]}
    entry["record"]["curve"] = rec["training"]["curve"]
    return entry


def cube_rows(url: str, token: str, key: str) -> dict:
    sql = (f"SELECT t.unit_id, m.metric, m.value FROM \"main\".\"gov_terminal_metric\" m "
           f"JOIN \"main\".\"gov_terminal\" t ON m.terminal_sha256 = t.terminal_sha256 WHERE t.campaign_key = '{key}' LIMIT 5000")
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}), headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        rows = json.loads(answer.read())["rows"]
    held = {}
    for r in rows:
        held.setdefault(r["unit_id"], {})[r["metric"]] = r["value"]
    return held


def verify(root: Path, warehouse_url: str | None, token: str | None, split: str = "validation") -> dict:
    report = json.loads((root / "REPORT.json").read_text())
    out = {"schema": "df_mod_e0_verify.v1", "run_id": report["run_id"], "design_sha256": report["design_sha256"], "stopped": report.get("stopped"),
           "cells": {}, "all_verified": True, "parent_equal": True, "warehouse": None, "live_query": bool(warehouse_url and token),
           "effects_split": split}
    for attempt in sorted(p for p in (root / "attempts").iterdir() if p.is_dir()):
        e = verify_cell(attempt)
        out["cells"][attempt.name] = e
        if e["problems"]:
            out["all_verified"] = False
        parent = (report.get("cells") or {}).get(attempt.name)
        if e.get("record") and parent and parent.get("mase_validation") is not None \
                and abs(parent["mase_validation"] - e["record"]["mase"]["validation"]) > 1e-9:
            out["parent_equal"] = False
            e["problems"].append("parent's MASE differs from the file")
    usable = {k: {"hypothesis": e["record"]["hypothesis"], "level": e["record"]["level"], "r": e["record"]["r"], "seed": e["record"]["seed"],
                  "arm": e["record"]["arm"], "mase": e["record"]["mase"].get(split)}
              for k, e in out["cells"].items() if e.get("record") and not e["problems"] and e["record"]["role"] == "CELL"}
    out["effects"] = effects(usable)
    out["bootstrap"] = bootstrap_effects(usable) if usable else None
    out["effects_test"] = effects({k: {**v, "mase": out["cells"][k]["record"]["mase"].get("test")} for k, v in usable.items()}) if usable else None
    if warehouse_url and token:
        keys = [k for k in ((report.get("campaign") or {}).get("key"), f"{report['run_id']}-mod-e0-cost-pilot", f"{report['run_id']}-mod-e0-cost-pilot-arm") if k]
        held = {}
        for k in keys:
            held.update(cube_rows(warehouse_url, token, k))
        wh = {"units": {}, "all_equal": True, "covered": 0}
        for name, e in out["cells"].items():
            if not e.get("record"):
                continue
            got = held.get(name, {})
            expected = {"mod_e0.mase_validation": e["record"]["mase"]["validation"], "mod_e0.mae_validation": e["record"]["mae"]["validation"],
                        "mod_e0.updates": float(e["record"]["updates"])}
            if "test" in e["record"]["mase"]:
                expected["mod_e0.mase_test"] = e["record"]["mase"]["test"]
            expected = {k: v for k, v in expected.items() if v is not None}
            equal = bool(got) and all(k in got and abs(float(got[k]) - v) < 1e-9 for k, v in expected.items())
            wh["units"][name] = {"equal": equal, "cube": got, "expected": expected}
            wh["all_equal"] &= equal
            wh["covered"] += int(bool(got))
        out["warehouse"] = wh
        out["all_verified"] &= wh["all_equal"]
    return out


def tables(out: dict) -> str:
    lines = [f"# MOD-E0-DEV `{out['run_id']}` — verification and effects ({out['effects_split']} split; descriptive, no confirmation)", "",
             f"all_verified = {out['all_verified']}, parent_equal = {out['parent_equal']}, live warehouse query = {out['live_query']}"
             + (f", warehouse equal = {out['warehouse']['all_equal']} ({out['warehouse']['covered']} units)" if out.get("warehouse") else ""), "",
             "## Cells", "", "| cell | arm | ARI(profiles, latent) | params (trainable) | updates | stop | MASE val | naive val | oracle val | MASE test | cpu s |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for k, e in out["cells"].items():
        r = e.get("record")
        if not r:
            lines.append(f"| `{k}` | — | — | — | — | — | {'; '.join(e['problems']) or e.get('carried')} | | | | |")
            continue
        f = lambda v: "—" if v is None else f"{v:.4f}"
        lines.append(f"| `{k}` | {r['arm']} | {r['profiles_ari']:.2f} | {r['parameters']['trainable']} | {r['updates']} | {r['stop_reason']} | "
                     f"{f(r['mase'].get('validation'))} | {f(r['naive_mase'].get('validation'))} | {f(r['oracle_mase'].get('validation'))} | "
                     f"{f(r['mase'].get('test'))} | {r['cost']['cpu_seconds']:.1f} |")
    eff = out.get("effects") or {}
    lines += ["", "## Effects (unit = replicate)", ""]
    if "H2" in eff:
        h2 = eff["H2"]
        lines += ["| h | e(h) = MASE(profiles) − MASE(random) | SD over replicates |", "|---:|---:|---:|"]
        for h in h2["levels"]:
            sd = h2["sd_replicates"].get(h)
            lines.append(f"| {h} | {h2['e'][h]:+.4f} | {'—' if sd is None else f'{sd:.4f}'} |")
        lines += [f"", f"slope of e(h): {h2['slope']:+.4f} (negative = advantage grows with heterogeneity); replicates {h2['replicates']}, random assignments {h2['random_assignments']}", ""]
    if "H3" in eff:
        h3 = eff["H3"]
        lines += ["| r | d_r = MASE(sequence) − MASE(summary) | SD |", "|---:|---:|---:|"]
        for r in sorted(h3["d"]):
            sd = h3["sd_replicates"].get(r)
            lines.append(f"| {r} | {h3['d'][r]:+.4f} | {'—' if sd is None else f'{sd:.4f}'} |")
        lines += ["", f"gamma = d_1 − d_0 = {h3['gamma']:+.4f} (negative = advantage larger with lagged dependence); units {h3['n_units']}", ""]
    if out.get("bootstrap"):
        lines += ["Bootstrap over replicates (95 % percentile, descriptive precision): " + ", ".join(
            f"{k}: [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" if v["ci95"] else f"{k}: —" for k, v in out["bootstrap"].items()), ""]
    lines += ["Negative favours the method. This pilot reports effects, dispersion, precision and cost; it does not support H2/H3.", ""]
    return "\n".join(lines)


def figures(root: Path, out: dict) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    made = []
    eff = out.get("effects") or {}
    if "H2" in eff and eff["H2"]["levels"]:
        h2 = eff["H2"]
        plt.figure(figsize=(5, 3.5))
        xs = h2["levels"]
        ys = [h2["e"][h] for h in xs]
        err = [h2["sd_replicates"].get(h) or 0 for h in xs]
        plt.errorbar(xs, ys, yerr=err, marker="o")
        plt.axhline(0, color="grey", lw=0.8)
        plt.xlabel("heterogeneity level h")
        plt.ylabel("e(h) = MASE(profiles) − MASE(random)")
        plt.title(f"H2 (slope {h2['slope']:+.3f})")
        plt.tight_layout()
        p = root / "fig_h2_e_h.png"
        plt.savefig(p, dpi=120)
        plt.close()
        made.append(str(p))
    if "H3" in eff and eff["H3"]["d"]:
        h3 = eff["H3"]
        plt.figure(figsize=(4, 3.5))
        xs = sorted(h3["d"])
        plt.bar([str(r) for r in xs], [h3["d"][r] for r in xs], yerr=[h3["sd_replicates"].get(r) or 0 for r in xs])
        plt.axhline(0, color="grey", lw=0.8)
        plt.xlabel("r (lagged dependence)")
        plt.ylabel("d_r = MASE(sequence) − MASE(summary)")
        plt.title(f"H3 (gamma {h3['gamma']:+.3f})")
        plt.tight_layout()
        p = root / "fig_h3_d_r.png"
        plt.savefig(p, dpi=120)
        plt.close()
        made.append(str(p))
    # learning curves of every cell
    curves = [(k, e["record"]["curve"]) for k, e in out["cells"].items() if e.get("record")]
    if curves:
        plt.figure(figsize=(7, 4))
        for k, c in curves:
            plt.plot(c["validation"], lw=0.7, alpha=0.7)
        plt.xlabel("epoch")
        plt.ylabel("validation loss (mse on scaled targets)")
        plt.title("validation curves of every cell")
        plt.tight_layout()
        p = root / "fig_curves.png"
        plt.savefig(p, dpi=120)
        plt.close()
        made.append(str(p))
    return made


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--no-warehouse", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", default="VERIFY.json")
    args = parser.parse_args(argv)
    token = None if args.no_warehouse else os.environ.get(args.token_env, "")
    out = verify(args.root, None if args.no_warehouse else args.warehouse_url, token or None, args.split)
    target = args.root / args.out
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; a verification is never written over")
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    target.with_suffix(".md").write_text(tables(out))
    out["figures"] = figures(args.root, out)
    print(json.dumps({"all_verified": out["all_verified"], "parent_equal": out["parent_equal"], "live_query": out["live_query"],
                      "warehouse": {k: v for k, v in (out["warehouse"] or {}).items() if k != "units"},
                      "effects": out["effects"], "bootstrap": out["bootstrap"], "figures": out["figures"],
                      "problems": {k: v["problems"] for k, v in out["cells"].items() if v["problems"]}}, indent=1, default=str))
    return 0 if out["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
