#!/usr/bin/env python3
"""Export a campaign queue.sqlite (read-only) to the queue-export JSON shape the table generator
reads: campaign, meta, counts, objective, candidates, attempts, standings, incumbent,
incumbent_history.  Nothing in the database is written."""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict


def export(path, campaign, objective):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    meta = {r["key"]: r["value"] for r in c.execute("select key, value from meta")}
    cands = [dict(r) for r in c.execute("select cid, position, config_id, seed, label, flat, status, blocked_reason, objective "
                                        "from candidates order by position")]
    atts = [dict(r) for r in c.execute("select * from attempts order by started")]
    inc = [dict(r) for r in c.execute("select * from incumbent_changes order by seq")]
    c.close()
    counts = defaultdict(int)
    for x in cands:
        counts[x["status"]] += 1
    by_cfg = defaultdict(list)
    for x in cands:
        by_cfg[x["config_id"]].append(x)
    standings = []
    for cfg, xs in by_cfg.items():
        ok = all(x["status"] == "verified" for x in xs)
        standings.append({"config_id": cfg, "label": xs[0]["label"], "eligible": ok,
                          "mean_objective": (sum(x["objective"] for x in xs) / len(xs)) if ok else None,
                          "per_seed": {str(x["seed"]): x["objective"] for x in xs},
                          "statuses": {str(x["seed"]): x["status"] for x in xs}})
    current = (meta.get("amendment_2_campaign_sha256") or meta.get("amendment_1_campaign_sha256")
               or meta.get("campaign_sha256"))
    return {"campaign": campaign, "meta": {"campaign_sha256": current, "base_campaign_sha256": meta.get("campaign_sha256"),
                                           "amendments": {k: v for k, v in meta.items() if k.startswith("amendment")}},
            "counts": dict(counts), "total": len(cands), "objective": objective, "candidates": cands, "attempts": atts,
            "standings": standings, "incumbent": inc[-1] if inc else None, "incumbent_history": inc}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    q = export(os.path.expanduser(a.db), a.campaign,
               {"metric": "MAE", "split": "validation", "unit": "z_train", "higher_is_better": False})
    json.dump(q, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(q["counts"], len(q["attempts"]), "attempts")


if __name__ == "__main__":
    sys.exit(main())
