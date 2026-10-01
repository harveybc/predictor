#!/usr/bin/env python3
"""Closure table for the lane F2 financial campaign (owner closure-table rule), from receipts only.

Reads one or more campaign queues (``queue.sqlite``; one per host) and, for every
configuration (cell name = architecture_loss_optimizer), its VERIFIED seeds' accepted
receipts. Emits per configuration and per horizon:

* model error (MAE, MSE) with its scale: ``z_train`` where 1 unit = sigma of the 4h
  close log-return (train), and the same numbers converted to log-return units;
* every same-row naive (persistence, zero-return, train-mean, seasonal) with the
  strict-minimum naive named, skill = 1 - model/naive and delta = model - naive;
* ``beats_zero_return`` (the mandatory first bar) and ``beats_strict_minimum`` per seed;
* the requested-seed spread, and for every pair of configurations sharing loss and optimizer
  (control vs grouped32 vs per_feature) the paired-by-seed difference with the STRICT_MINIMUM
  label: the word "advantage" appears only when the gap exceeds BOTH seed spreads, and even
  then only beside the numbers;
* literature value: NOT_AVAILABLE with the reason (no published row matches this asset,
  view, split, target, scaler and horizon set), comparability NOT_COMPARABLE for the same
  reason derived from the accepted receipts; the only comparable rows are the
  same-row naives and the paired cells.

Unverified cells are listed by status; nothing is computed from an unverified receipt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import time
from pathlib import Path

NAIVE_NAMES = ("persistence_last_value", "zero_return", "train_mean")  # + the seasonal_<P> naive found in the receipt
COMPARABILITY_REASON = ("only the same-row naives and paired cells carrying this exact campaign identity are "
                        "comparable; see literature.reason")


def _receipt_identity(receipt):
    """Return the scientific population identity asserted by an accepted receipt."""
    artifact = receipt.get("artifact", {})
    population = receipt.get("population", {})
    scale = receipt.get("scale", {})
    per_horizon = receipt.get("per_horizon", {})
    if isinstance(per_horizon, dict):
        horizons = sorted(int(h) for h in per_horizon)
    elif isinstance(per_horizon, list):
        horizons = sorted(int(row["horizon"]) for row in per_horizon)
    else:
        raise ValueError("accepted receipt has an invalid per_horizon collection")
    identity = {
        "campaign_id": artifact.get("campaign_id"),
        "asset": population.get("asset"),
        "dataset_id": population.get("dataset_id"),
        "sample_hours": population.get("sample_hours"),
        "targets": population.get("targets"),
        "horizons": horizons,
        "metric_space": scale.get("metric_space"),
        "scaler_identity": scale.get("scaler_identity"),
    }
    missing = [name for name, value in identity.items() if value in (None, [], "")]
    if missing:
        raise ValueError(f"accepted receipt lacks campaign identity fields: {missing}")
    return identity


def _literature(identity):
    reason = (
        "no published row has been registered with the exact accepted identity: "
        f"campaign={identity['campaign_id']}, asset={identity['asset']}, dataset={identity['dataset_id']}, "
        f"sample_hours={identity['sample_hours']}, targets={identity['targets']}, "
        f"horizons={identity['horizons']}, metric_space={identity['metric_space']}, "
        f"scaler={identity['scaler_identity']}"
    )
    return {"value": None, "status": "NOT_AVAILABLE", "reason": reason}


def load_cells(queues):
    cells = {}
    unverified = []
    for queue in queues:
        db = sqlite3.connect(f"file:{queue}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT * FROM candidates ORDER BY position"):
            if row["status"] != "verified":
                unverified.append({"label": row["label"], "seed": row["seed"], "status": row["status"],
                                   "cid": row["cid"][:16], "queue": str(queue)})
                continue
            train = db.execute("SELECT receipt_path FROM attempts WHERE cid=? AND kind='train' AND status='completed' "
                               "ORDER BY attempt DESC LIMIT 1", (row["cid"],)).fetchone()
            verify = db.execute("SELECT receipt_path, verdict, exit_code FROM attempts WHERE cid=? AND kind='verify' "
                                "ORDER BY attempt DESC LIMIT 1", (row["cid"],)).fetchone()
            receipt = json.loads(Path(train["receipt_path"]).read_text())
            verification = json.loads(Path(verify["receipt_path"]).read_text())
            if verification.get("verdict") != "VERIFIED" or verification.get("exact_match") is not True:
                raise ValueError(f"{row['cid'][:16]} is marked verified without an exact-match verification")
            if receipt.get("candidate", {}).get("cid") != row["cid"]:
                raise ValueError(f"receipt {train['receipt_path']} belongs to another candidate")
            cells.setdefault(row["label"], {})[int(row["seed"])] = {
                "cid": row["cid"], "receipt": receipt, "verification": verification, "queue": str(queue)}
    return cells, unverified


def _pair(model, naive):
    if model is None or naive is None or not (math.isfinite(model) and math.isfinite(naive)):
        return {"skill": None, "delta": None, "status": "NOT_AVAILABLE"}
    if naive == 0:
        return {"skill": None, "delta": model - naive, "status": "ZERO_NAIVE"}
    return {"skill": 1 - model / naive, "delta": model - naive, "status": "OK"}


def seed_rows(entry, sigma):
    receipt = entry["receipt"]
    naives = receipt["naives"]
    out = {"cid": entry["cid"], "objective_MAE_z": receipt["objective"]["value"],
           "selected_epoch": receipt["training"]["selected_epoch"], "observed_updates": receipt["training"]["observed_updates"],
           "stop_reason": receipt["training"]["stop_reason"], "weights_sha256": receipt["digests"]["weights_sha256"],
           "model_sha256": receipt["digests"]["model_sha256"], "per_horizon": {}}
    for h, m in receipt["per_horizon"].items():
        strict = naives["strict_minimum"][h]
        row = {"model_MAE_z": m["MAE"], "model_MSE_z": m["MSE"], "model_MAE_logret": m["MAE"] * sigma,
               "naives": {}, "strict_naive": strict["naive"], "strict_naive_MAE_z": strict["MAE"],
               "vs_strict": _pair(m["MAE"], strict["MAE"]),
               "beats_zero_return": m["MAE"] < naives["per_naive"]["zero_return"][h]["MAE"],
               "beats_strict_minimum": m["MAE"] < strict["MAE"]}
        seasonal = next((k for k in naives["per_naive"] if k.startswith("seasonal_")), None)
        for name in (*NAIVE_NAMES, *([seasonal] if seasonal else [])):
            n = naives["per_naive"].get(name, {}).get(h, {})
            row["naives"][name] = {"MAE_z": n.get("MAE"), "MSE_z": n.get("MSE"), **_pair(m["MAE"], n.get("MAE"))}
        out["per_horizon"][h] = row
    return out


def closure(queues, *, sigma, seeds=(2021,)):
    cells, unverified = load_cells(queues)
    identities = {_canonical_identity(_receipt_identity(entry["receipt"]))
                  for by_seed in cells.values() for entry in by_seed.values()}
    if not identities:
        raise ValueError("closure has no verified receipt from which to derive campaign identity")
    if len(identities) != 1:
        raise ValueError("mixed campaign identity across verified receipts")
    identity = json.loads(next(iter(identities)))
    literature = _literature(identity)
    comparability = {"status": "NOT_COMPARABLE", "reason": COMPARABILITY_REASON}
    table = {"schema": "f2.closure_table.v1", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "scale": {"metric_space": "z_train", "sigma_logret_4h_train": sigma,
                       "note": "1 z unit = sigma of the 4h close log-return on train rows; MAE_logret = MAE_z * sigma"},
             "label": "DEVELOPMENT", "campaign_identity": identity,
             "literature": literature, "comparability": comparability,
             "configurations": {}, "contrasts": [], "unverified": unverified}
    for label, by_seed in cells.items():
        rows = {s: seed_rows(by_seed[s], sigma) for s in seeds if s in by_seed}
        complete = set(rows) == set(seeds)
        values = [rows[s]["objective_MAE_z"] for s in rows]
        conf = {"eligible": complete, "seeds": sorted(rows), "per_seed": rows,
                "mean_MAE_z": sum(values) / len(values) if values else None,
                "spread": max(values) - min(values) if len(values) > 1 else None,
                "beats_zero_return_all_horizons_all_seeds": complete and all(
                    r["beats_zero_return"] for s in rows for r in rows[s]["per_horizon"].values()),
                "beats_strict_minimum_by_horizon": {h: [rows[s]["per_horizon"][h]["beats_strict_minimum"] for s in rows]
                                                    for h in next(iter(rows.values()))["per_horizon"]} if rows else {},
                "literature": literature, "comparability": comparability}
        table["configurations"][label] = conf
    labels = [l for l, c in table["configurations"].items() if c["eligible"]]
    for a in labels:
        for b in labels:
            if a >= b or a.split("_", 1)[1] != b.split("_", 1)[1] and not _same_axis(a, b):
                continue
            ca, cb = table["configurations"][a], table["configurations"][b]
            diffs = {s: ca["per_seed"][s]["objective_MAE_z"] - cb["per_seed"][s]["objective_MAE_z"] for s in seeds}
            mean = sum(diffs.values()) / len(diffs)
            exceeds = abs(mean) > max(ca["spread"], cb["spread"])
            table["contrasts"].append({
                "a": a, "b": b, "paired_difference_a_minus_b": diffs, "mean_difference": mean,
                "spread_a": ca["spread"], "spread_b": cb["spread"],
                "label_rule": ("STRICT_MINIMUM; gap exceeds both seed spreads: lower MAE for "
                               f"{a if mean < 0 else b} by {abs(mean):.6f} z") if exceeds
                else "STRICT_MINIMUM; gap within the requested-seed spread"})
    return table


def _canonical_identity(identity):
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _same_axis(a, b):
    """Contrast cells that differ in exactly one axis (architecture, loss or optimizer)."""
    pa, pb = a.rsplit("_", 2), b.rsplit("_", 2)
    return len(pa) == 3 and len(pb) == 3 and sum(x != y for x, y in zip(pa, pb)) == 1


def write_csv(table, path):
    with open(path, "w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["configuration", "seed", "horizon", "model_MAE_z", "model_MAE_logret", "strict_naive",
                    "strict_naive_MAE_z", "skill_vs_strict", "zero_return_MAE_z", "skill_vs_zero_return",
                    "persistence_MAE_z", "seasonal_MAE_z", "beats_zero_return", "beats_strict_minimum"])
        for label, conf in table["configurations"].items():
            for seed, rows in conf["per_seed"].items():
                for h, r in rows["per_horizon"].items():
                    w.writerow([label, seed, h, f"{r['model_MAE_z']:.6f}", f"{r['model_MAE_logret']:.8f}",
                                r["strict_naive"], f"{r['strict_naive_MAE_z']:.6f}",
                                _fmt(r["vs_strict"]["skill"]), _fmt(r["naives"]["zero_return"]["MAE_z"]),
                                _fmt(r["naives"]["zero_return"]["skill"]),
                                _fmt(r["naives"]["persistence_last_value"]["MAE_z"]),
                                _fmt(next((v["MAE_z"] for k, v in r["naives"].items() if k.startswith("seasonal_")), None)), r["beats_zero_return"], r["beats_strict_minimum"]])


def _fmt(v):
    return "NOT_AVAILABLE" if v is None else f"{v:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--manifest", required=True, help="data MANIFEST.json (sigma)")
    parser.add_argument("--out", required=True, help="output prefix (.json and .csv)")
    parser.add_argument("--seeds", default="2021", help="trained seeds to aggregate; default is the screening seed")
    args = parser.parse_args()
    sigma = json.loads(Path(args.manifest).read_text())["target"]["sigma"]
    table = closure(args.queue, sigma=sigma, seeds=tuple(int(s) for s in args.seeds.split(",")))
    Path(args.out + ".json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    write_csv(table, args.out + ".csv")
    print(json.dumps({label: {"mean_MAE_z": c["mean_MAE_z"], "spread": c["spread"],
                              "beats_zero_return": c["beats_zero_return_all_horizons_all_seeds"]}
                      for label, c in table["configurations"].items()}, indent=1))


if __name__ == "__main__":
    main()
