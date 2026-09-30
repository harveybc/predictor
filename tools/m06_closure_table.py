#!/usr/bin/env python3
"""Closure table for a sealed multi-seed TSL cell group, GENERATED from the retained records.

Owner's closure-table rule: every closure carries model error with its scale, the naive on the
same rows (with the proof of pairing), skill, the literature value with its source, and the
comparability class, generated from artifacts, never typed.

This generator is independent of the sealing executor: it re-derives every check itself and
REFUSES (raises ClosureRefusal) on
  * a forged record: its record_sha256 does not recompute from its content;
  * a record produced under another design;
  * a missing or duplicated sealed seed for the horizon;
  * a population mismatch: records, or a record's naive, not on the same rows;
  * non-finite predictions or an element count that does not match windows x steps x channels;
  * disagreeing comparison classes across seeds.
The agreement class follows the design lock's own predeclared rule:
  |mean - published| <= k_agree * std_paper + rounding  -> OPERATIONAL_AGREEMENT
                     <= k_partial * std_paper + rounding -> OPERATIONAL_PARTIAL
                     otherwise                           -> OUTSIDE_OPERATIONAL_MARGIN

Usage: m06_closure_table.py --design DESIGN.json --horizon 96 --out-dir DIR RECORD.json ...
Writes DIR/<dataset>_h<H>_closure_table.md and .json (with the digest of every input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys


class ClosureRefusal(Exception):
    pass


def sha_obj(obj) -> str:
    """The sealing code's canonical digest: sorted keys, compact separators, str() fallback."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def sha_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _check_record(rec: dict, design: dict, horizon: int):
    body = {k: v for k, v in rec.items() if k != "record_sha256"}
    if rec.get("record_sha256") != sha_obj(body):
        raise ClosureRefusal(f"FORGED_OR_ALTERED_RECORD: {rec.get('cell_id')} record_sha256 does not recompute")
    if rec.get("design_sha256") != design.get("design_sha256"):
        raise ClosureRefusal(f"FOREIGN_DESIGN: {rec.get('cell_id')} was not produced under this design")
    if int(rec.get("horizon_steps", -1)) != horizon:
        raise ClosureRefusal(f"WRONG_HORIZON: {rec.get('cell_id')} is h{rec.get('horizon_steps')}, not h{horizon}")
    pop = rec["population"]
    if pop.get("all_finite") is not True:
        raise ClosureRefusal(f"NONFINITE: {rec.get('cell_id')}")
    if int(pop["windows"]) * horizon * int(pop["target_channels"]) != int(pop["elements"]):
        raise ClosureRefusal(f"ELEMENT_COUNT: {rec.get('cell_id')} windows x steps x channels != elements")
    proof = rec["naive"]["paired_on_the_same_rows_proved_by"]
    if not (proof.get("equal") is True and proof.get("target_population_sha256") == proof.get("naive_target_sha256")
            == pop["sha256"]):
        raise ClosureRefusal(f"NAIVE_NOT_PAIRED: {rec.get('cell_id')} naive is not proven on the model's rows")
    for m in ("mse", "mae"):
        for v in (rec["metric"]["author_float32"][m], rec["naive"][m]):
            if not math.isfinite(float(v)):
                raise ClosureRefusal(f"NONFINITE_METRIC: {rec.get('cell_id')} {m}")


def _classify(design: dict, mean: float, published: float, metric: str) -> dict:
    ag = design["lock"]["agreement"]
    std = float(ag["std_paper"][metric])
    tol_a = float(ag["k_agree"]) * std + float(ag["rounding"])
    tol_p = float(ag["k_partial"]) * std + float(ag["rounding"])
    diff = mean - published
    cls = ("OPERATIONAL_AGREEMENT" if abs(diff) <= tol_a else
           "OPERATIONAL_PARTIAL" if abs(diff) <= tol_p else "OUTSIDE_OPERATIONAL_MARGIN")
    return {"class": cls, "difference": diff, "tolerance_agreement": tol_a, "tolerance_partial": tol_p,
            "std_paper": std, "margin_source": ag.get("source"), "rule": ag.get("rule")}


def build(design: dict, records: list, horizon: int) -> dict:
    for r in records:
        _check_record(r, design, horizon)
    seeds = [int(r["seed"]) for r in records]
    sealed = sorted(int(s) for s in design["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ClosureRefusal(f"DUPLICATE_SEED: {sorted(seeds)}")
    if sorted(seeds) != sealed:
        raise ClosureRefusal(f"MISSING_OR_EXTRA_SEED: have {sorted(seeds)}, sealed {sealed}; no class without every sealed seed")
    pops = {r["population"]["sha256"] for r in records}
    if len(pops) != 1:
        raise ClosureRefusal(f"POPULATION_MISMATCH: {len(pops)} distinct population digests across seeds")
    for m in ("mse", "mae"):
        if len({float(r["naive"][m]) for r in records}) != 1:
            raise ClosureRefusal(f"NAIVE_DIFFERS_ACROSS_SEEDS: {m} (same rows must give one naive)")
    classes = {r["receipt"]["tags"]["comparison_class"] for r in records}
    if len(classes) != 1:
        raise ClosureRefusal(f"COMPARISON_CLASS_DISAGREES: {sorted(classes)}")
    pub = design["lock"]["published"]["per_horizon"][str(horizon)]
    r0 = sorted(records, key=lambda r: int(r["seed"]))
    rows = []
    for r in r0:
        m, n = r["metric"]["author_float32"], r["naive"]
        rows.append({"seed": int(r["seed"]), "cell_id": r["cell_id"],
                     "mse": float(m["mse"]), "mae": float(m["mae"]),
                     "independent_float64": r["metric"]["independent_float64"],
                     "naive_mse": float(n["mse"]), "naive_mae": float(n["mae"]),
                     "skill_mse": 1 - float(m["mse"]) / float(n["mse"]),
                     "skill_mae": 1 - float(m["mae"]) / float(n["mae"]),
                     "diff_mse": float(m["mse"]) - float(pub["mse"]), "diff_mae": float(m["mae"]) - float(pub["mae"]),
                     "wall_seconds": r["resources"]["wall_seconds"],
                     "device_uuid": r.get("device_uuid_measured_inside_child"),
                     "checkpoint_sha256": r.get("checkpoint_sha256"), "record_sha256": r["record_sha256"]})
    mean = {}
    for m in ("mse", "mae"):
        vals = [x[m] for x in rows]
        mu = statistics.fmean(vals)
        mean[m] = {"mean": mu, "seed_sd_ddof1": statistics.stdev(vals) if len(vals) > 1 else None,
                   "naive": rows[0][f"naive_{m}"], "skill": 1 - mu / rows[0][f"naive_{m}"],
                   "published": float(pub[m]), **_classify(design, mu, float(pub[m]), m)}
    first = r0[0]
    return {"schema": "m06.closure_table.v1", "dataset": first["dataset"], "horizon_steps": horizon,
            "input_window_steps": first["seq_len"], "design_sha256": design["design_sha256"],
            "scale": {"space": first["metric"]["space"], "reduction": first["metric"]["reduction"],
                      "dtype": "float32, the author's own reduction; an independent float64 reduction is carried beside it",
                      "elements": first["population"]["elements"], "windows": first["population"]["windows"],
                      "target_channels": first["population"]["target_channels"]},
            "naive": {"definition": first["naive"]["definition"],
                      "pairing_proof": first["naive"]["paired_on_the_same_rows_proved_by"],
                      "population_sha256": first["population"]["sha256"]},
            "literature": {"source": design["lock"]["paper"], "table": design["lock"]["published"]["table"],
                           "read_at": design["lock"].get("paper_read_at"), "values": pub,
                           "margin_source": design["lock"]["agreement"].get("source")},
            "comparability_class": classes.pop(), "rows": rows, "mean": mean,
            "evidence": "measured; per-seed rows carry a difference, the class is on the seed mean only"}


def render_md(t: dict) -> str:
    s, lit = t["scale"], t["literature"]
    L = [f"# Closure table: {t['dataset']} L{t['input_window_steps']} -> h{t['horizon_steps']} (generated, not typed)", "",
         f"- Scale: {s['space']}; {s['reduction']}; {s['dtype']}; {s['elements']:,} elements "
         f"({s['windows']:,} windows x {t['horizon_steps']} steps x {s['target_channels']:,} channels).",
         f"- Naive: {t['naive']['definition']}",
         f"- Pairing proof: population sha {t['naive']['population_sha256'][:16]}… equals the naive target sha "
         f"(equal={t['naive']['pairing_proof'].get('equal')}).",
         f"- Literature: {lit['source']}, {lit['table']} (read {lit['read_at']}); margin from {lit['margin_source']}.",
         f"- Comparability class: {t['comparability_class']}.", "",
         "| seed | MSE | MAE | naive MSE | naive MAE | skill MSE | skill MAE | published MSE / MAE | diff MSE / MAE | record sha |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    pm, pa = lit["values"]["mse"], lit["values"]["mae"]
    for r in t["rows"]:
        L.append(f"| {r['seed']} | {r['mse']:.10f} | {r['mae']:.10f} | {r['naive_mse']:.10f} | {r['naive_mae']:.10f} | "
                 f"{r['skill_mse']:.4f} | {r['skill_mae']:.4f} | {pm} / {pa} | {r['diff_mse']:+.5f} / {r['diff_mae']:+.5f} | "
                 f"{r['record_sha256'][:12]}… |")
    mm, ma = t["mean"]["mse"], t["mean"]["mae"]
    L.append(f"| **mean** | {mm['mean']:.10f} (sd {mm['seed_sd_ddof1']:.5f}) | {ma['mean']:.10f} (sd {ma['seed_sd_ddof1']:.5f}) | "
             f"{mm['naive']:.10f} | {ma['naive']:.10f} | {mm['skill']:.4f} | {ma['skill']:.4f} | {pm} / {pa} | "
             f"{mm['difference']:+.5f} / {ma['difference']:+.5f} | — |")
    L += ["", f"Class on the seed mean: MSE **{mm['class']}** (tolerance {mm['tolerance_agreement']:.4f}), "
              f"MAE **{ma['class']}** (tolerance {ma['tolerance_agreement']:.4f}). Rule: {mm['rule']}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("records", nargs="+")
    a = ap.parse_args(argv)
    design = json.load(open(a.design))
    recs = [json.load(open(p)) for p in a.records]
    try:
        t = build(design, recs, a.horizon)
    except ClosureRefusal as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return 3
    t["inputs"] = {"design_file_sha256": sha_file(a.design),
                   "record_files_sha256": {os.path.basename(p): sha_file(p) for p in a.records}}
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{t['dataset']}_h{a.horizon}_closure_table")
    for ext, text in ((".json", json.dumps(t, indent=1)), (".md", render_md(t))):
        tmp = stem + ext + ".tmp"
        open(tmp, "w").write(text)
        os.replace(tmp, stem + ext)
    print(stem + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
