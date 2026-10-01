"""Per-horizon model / same-row persistence / seasonal-naive table for every VERIFIED candidate.

Read from the accepted receipts named in the campaign queue; with ``--validation`` and
``--seasonal-period`` the declared seasonal naive (value one period before the target
time, taken from the input window) is computed on the same rows. Output: .json and .csv,
plus a paired-configuration summary (mean, spread) and the count of negative-skill
horizons. Nothing is retrained or rescored here.

usage: modular_doin_per_horizon_table.py <queue.sqlite> <output prefix> [--validation NPZ --seasonal-period P]
"""
import argparse
import csv
import json
import sqlite3

import numpy as np


def seasonal_naive(validation, period):
    with np.load(validation, allow_pickle=False) as z:
        x, y = z["windows"].astype(np.float64), z["targets"].astype(np.float64)
        horizons = z["horizons"].astype(int).tolist()
        names = z["feature_names"].astype(str).tolist()
        idx = [names.index(t) for t in z["target_names"].astype(str).tolist()]
    out = {}
    for k, h in enumerate(horizons):
        pos = x.shape[1] - 1 - (period - h)
        if h > period or pos < 0:
            out[h] = None
        else:
            err = x[:, pos, :][:, idx] - y[:, k, :]
            out[h] = {"MAE": float(np.abs(err).mean()), "MSE": float((err ** 2).mean())}
    return out


def build(queue, validation=None, period=None):
    db = sqlite3.connect(f"file:{queue}?mode=ro", uri=True)
    rows = db.execute("select c.cid, c.label, c.seed, c.config_id, a.receipt_path from candidates c join attempts a "
                      "using(cid) where c.status='verified' and a.kind='train' and a.status='completed' "
                      "order by c.position").fetchall()
    seasonal = seasonal_naive(validation, period) if validation and period else {}
    out = []
    for cid, label, seed, config_id, receipt in rows:
        r = json.load(open(receipt))
        if r.get("candidate", {}).get("cid") not in (None, cid):
            raise ValueError(f"receipt {receipt} belongs to another candidate")
        rec = {"cid": cid, "config_id": config_id, "label": label, "seed": seed, "MAE": r["metrics"]["MAE"],
               "naive_MAE": r["metrics"]["baseline_MAE"], "skill_MAE": r["metrics"]["skill_MAE"],
               "model_sha256": r["digests"]["model_sha256"]}
        for h, m in r["per_horizon"].items():
            rec[f"h{h}_MAE"] = m["MAE"]
            rec[f"h{h}_naive_MAE"] = m["baseline_MAE"]
            rec[f"h{h}_skill_MAE"] = m["skill_MAE"]
            s = seasonal.get(int(h))
            if seasonal:
                rec[f"h{h}_seasonal_MAE"] = None if s is None else s["MAE"]
                rec[f"h{h}_skill_vs_seasonal"] = None if not s or not s["MAE"] else 1 - m["MAE"] / s["MAE"]
        out.append(rec)
    pairs = {}
    for r in out:
        pairs.setdefault(r["config_id"], {"label": r["label"].rsplit("_", 0)[0], "seeds": {}})["seeds"][r["seed"]] = r["MAE"]
    summary = [{"config_id": k, "label": v["label"], "seeds": v["seeds"],
                "mean": sum(v["seeds"].values()) / len(v["seeds"]),
                "spread": max(v["seeds"].values()) - min(v["seeds"].values())} for k, v in pairs.items()]
    negative = {}
    for r in out:
        for key, value in r.items():
            if key.endswith("_skill_MAE") and key.startswith("h") and value is not None and value < 0:
                h = int(key[1:].split("_")[0])
                negative[h] = negative.get(h, 0) + 1
    return {"rows": out, "pairs": summary, "negative_skill_vs_persistence_by_horizon": negative,
            "seasonal_period": period, "candidates": len(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queue")
    parser.add_argument("prefix")
    parser.add_argument("--validation")
    parser.add_argument("--seasonal-period", type=int)
    args = parser.parse_args()
    report = build(args.queue, args.validation, args.seasonal_period)
    json.dump(report, open(args.prefix + ".json", "w"), indent=1)
    keys = list(report["rows"][0])
    with open(args.prefix + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps({"candidates": report["candidates"], "negative": report["negative_skill_vs_persistence_by_horizon"]}))


if __name__ == "__main__":
    main()
