#!/usr/bin/env python3
"""Seed-interval table across campaign queues (from receipts only). Label TABLE_FROM_RECEIPTS.

Joins every VERIFIED row of one or more campaign queues by its flat parameters without
``train.seed`` (``model.target_residual: none`` is the same configuration as an absent key,
and the search-space identity is ignored), so a configuration re-run in a later campaign
pairs with its earlier seeds. Per configuration: per-seed MAE with the producing pin
(``bridge.predictor_revision`` from the receipt) and queue, n, mean, sample sd (ddof=1),
min, max, the same-row persistence and seasonal-naive means, skill against both, and the
horizons where the mean does not beat the seasonal naive. Contrasts are paired BY SEED
against a named reference; the label is STRICT_MINIMUM and says whether the mean paired
difference exceeds both seed ranges (stated with numbers, never as proof).

usage: modular_doin_four_seed_table.py OUT --queue Q --per-horizon P [--queue Q2 --per-horizon P2 ...]
       [--reference LABEL_OR_CONFIG_PREFIX ...]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

HORIZONS = range(1, 25)


def config_key(flat):
    flat = {k: v for k, v in flat.items() if k != "train.seed"}
    if flat.get("model.target_residual") == "none":
        flat.pop("model.target_residual")
    return json.dumps(flat, sort_keys=True, separators=(",", ":"))


def _pin(receipt_path):
    try:
        return json.loads(Path(receipt_path).read_text())["bridge"]["predictor_revision"][:8]
    except (OSError, KeyError, ValueError, TypeError):
        return "UNKNOWN_RECEIPT_NOT_ON_HOST"


def collect(pairs):
    configs = {}
    for queue, per_horizon in pairs:
        records = {r["cid"]: r for r in json.loads(Path(per_horizon).read_text())["rows"]}
        db = sqlite3.connect(f"file:{queue}?mode=ro", uri=True)
        for cid, label, seed, flat, receipt in db.execute(
                "select c.cid, c.label, c.seed, c.flat, (select a.receipt_path from attempts a where a.cid=c.cid and"
                " a.kind='train' and a.status='completed' order by a.attempt desc limit 1)"
                " from candidates c where c.status='verified' order by c.position"):
            rec = records.get(cid)
            if rec is None:
                continue
            key = config_key(json.loads(flat))
            entry = configs.setdefault(key, {"labels": [], "seeds": {}})
            if label not in entry["labels"]:
                entry["labels"].append(label)
            if seed in entry["seeds"]:
                raise ValueError(f"seed {seed} of {label} verified twice across queues; refusing to choose")
            entry["seeds"][seed] = {"MAE": rec["MAE"], "pin": _pin(receipt), "queue": str(queue), "cid": cid,
                                    "persistence": rec["naive_MAE"],
                                    "seasonal": {h: rec.get(f"h{h}_seasonal_MAE") for h in HORIZONS},
                                    "per_h": {h: rec.get(f"h{h}_MAE") for h in HORIZONS}}
    return configs


def summarize(entry):
    seeds = entry["seeds"]
    vals = [seeds[s]["MAE"] for s in sorted(seeds)]
    mean = statistics.fmean(vals)
    seasonal_h = {h: statistics.fmean(seeds[s]["seasonal"][h] for s in seeds) for h in HORIZONS
                  if all(seeds[s]["seasonal"][h] is not None for s in seeds)}
    model_h = {h: statistics.fmean(seeds[s]["per_h"][h] for s in seeds) for h in HORIZONS
               if all(seeds[s]["per_h"][h] is not None for s in seeds)}
    seasonal = statistics.fmean(seasonal_h.values()) if seasonal_h else None
    persistence = statistics.fmean(seeds[s]["persistence"] for s in seeds)
    return {"labels": entry["labels"], "n": len(vals),
            "per_seed": {str(s): {"MAE": seeds[s]["MAE"], "pin": seeds[s]["pin"]} for s in sorted(seeds)},
            "mean": mean, "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "min": min(vals), "max": max(vals), "range": max(vals) - min(vals),
            "pins": sorted({seeds[s]["pin"] for s in seeds}),
            "persistence_same_rows": persistence, "skill_vs_persistence": 1 - mean / persistence,
            "seasonal_same_rows": seasonal, "skill_vs_seasonal": None if seasonal is None else 1 - mean / seasonal,
            "horizons_not_beating_seasonal": sorted(h for h in model_h if h in seasonal_h
                                                    and model_h[h] >= seasonal_h[h])}


def contrast(entry, ref):
    shared = sorted(set(entry["seeds"]) & set(ref["seeds"]))
    diffs = {str(s): entry["seeds"][s]["MAE"] - ref["seeds"][s]["MAE"] for s in shared}
    if not diffs:
        return {"paired_seeds": [], "label": "NO_SHARED_SEEDS"}
    mean_diff = statistics.fmean(diffs.values())
    ranges = [max(e["seeds"][s]["MAE"] for s in e["seeds"]) - min(e["seeds"][s]["MAE"] for s in e["seeds"])
              for e in (entry, ref)]
    exceeds = abs(mean_diff) > max(ranges)
    return {"paired_seeds": shared, "paired_difference": diffs, "mean_difference": mean_diff,
            "seed_ranges": ranges,
            "label": "STRICT_MINIMUM; mean paired difference exceeds both seed ranges (numbers, not proof)"
            if exceeds else "STRICT_MINIMUM; mean paired difference within the seed ranges"}


def build(pairs, references=()):
    configs = collect(pairs)
    table = []
    for key, entry in configs.items():
        row = {"config": json.loads(key), **summarize(entry)}
        row["contrasts"] = {}
        for ref_name in references:
            ref = next((e for e in configs.values() if ref_name in e["labels"]), None)
            if ref is not None and ref is not entry:
                row["contrasts"][ref_name] = contrast(entry, ref)
        table.append(row)
    table.sort(key=lambda r: (r["mean"], -r["n"]))
    for rank, row in enumerate(table, 1):
        row["strict_minimum_rank"] = rank
    return {"schema": "m04.seed_interval_table.v1", "label": "TABLE_FROM_RECEIPTS", "queues": [str(q) for q, _ in pairs],
            "references": list(references), "rows": table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--per-horizon", action="append", required=True)
    parser.add_argument("--reference", action="append", default=[])
    args = parser.parse_args()
    if len(args.queue) != len(args.per_horizon):
        parser.error("one --per-horizon per --queue")
    report = build(list(zip(args.queue, args.per_horizon)), args.reference)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for r in report["rows"]:
        print(r["strict_minimum_rank"], "|".join(r["labels"])[:60], r["n"], round(r["mean"], 6),
              None if r["sd"] is None else round(r["sd"], 6), round(r["min"], 6), round(r["max"], 6), r["pins"])


if __name__ == "__main__":
    main()
