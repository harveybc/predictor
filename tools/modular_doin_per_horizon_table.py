"""Per-horizon MAE / same-row naive / skill table for every VERIFIED candidate, read from the accepted receipts.

usage: modular_doin_per_horizon_table.py <queue.sqlite> <output prefix>  (writes .json and .csv)
"""
import csv, glob, json, os, sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
rows = db.execute("select c.cid, c.label, c.seed, c.config_id, a.receipt_path from candidates c join attempts a using(cid) where c.status='verified' and a.kind='train' and a.status='completed' order by c.position").fetchall()
out = []
for cid, label, seed, config_id, receipt in rows:
    r = json.load(open(receipt))
    assert r["candidate"]["cid"] == cid
    rec = {"cid": cid, "config_id": config_id, "label": label, "seed": seed, "MAE": r["metrics"]["MAE"],
           "baseline_MAE": r["metrics"]["baseline_MAE"], "skill_MAE": r["metrics"]["skill_MAE"],
           "receipt_sha256_model": r["digests"]["model_sha256"]}
    for h, m in r["per_horizon"].items():
        rec[f"h{h}_MAE"] = m["MAE"]; rec[f"h{h}_naive_MAE"] = m["baseline_MAE"]; rec[f"h{h}_skill_MAE"] = m["skill_MAE"]
    out.append(rec)
json.dump(out, open(sys.argv[2] + ".json", "w"), indent=1)
keys = list(out[0])
with open(sys.argv[2] + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
neg = {h: sum(1 for r in out if r[f"h{h}_skill_MAE"] < 0) for h in range(1, 25)}
print(len(out), "negative-skill counts by horizon:", {h: n for h, n in neg.items() if n})
