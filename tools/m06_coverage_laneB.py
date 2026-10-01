#!/usr/bin/env python3
"""Coverage denominators GENERATED from lane B's published files (never typed).

Inputs: DENOMINATORS.v1.json, TRANSFORM_LEDGER.v1.csv, SOURCE_TABLE.v1.csv (feature-eng) and
FEATURE_DAG.v3.json (financial-data).  Output: one JSON with the old and new denominators, the
transform ledger counted by state, sources and their point-in-time admissibility, the DAG's
denominators, each input's sha256 and the identity (repo@commit:path) it was read from.
The old (column-row) and new (file) grains are kept side by side; neither replaces the other.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import sys


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build(den_b, ledger_b, sources_b, dag_b, ids):
    den = json.loads(den_b)
    led = list(csv.DictReader(io.StringIO(ledger_b.decode())))
    src = list(csv.DictReader(io.StringIO(sources_b.decode())))
    dag = json.loads(dag_b)
    pit = [s["provider"] for s in src if s.get("point_in_time", "").startswith("POINT_IN_TIME_ADMISSIBLE")]
    return {"schema": "m06.coverage_laneB.v1",
            "old_denominator": den["old"], "new_denominator": den["new"], "grain_note": den.get("note"),
            "transform_ledger": {"rows": len(led), "by_state": dict(collections.Counter(r["state"] for r in led).most_common()),
                                 "evaluated_true": sum(r.get("evaluated") == "True" for r in led),
                                 "selected_true": sum(r.get("selected") == "True" for r in led)},
            "sources": {"rows": len(src), "by_status": dict(collections.Counter(s["status"] for s in src).most_common()),
                        "point_in_time_admissible": pit, "point_in_time_admissible_count": len(pit)},
            "dag": {"dag_sha256": dag.get("dag_sha256"), "denominators": dag.get("denominators"),
                    "not_claimed": dag.get("not_claimed")},
            "inputs": {k: {"from": ids[k], "sha256": sha(b)} for k, b in
                       (("denominators", den_b), ("transform_ledger", ledger_b), ("source_table", sources_b), ("feature_dag", dag_b))}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("denominators", "ledger", "sources", "dag"):
        ap.add_argument(f"--{k}", required=True, help="local file")
        ap.add_argument(f"--{k}-id", required=True, help="repo@commit:path it was read from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rd = lambda p: open(p, "rb").read()  # noqa: E731
    out = build(rd(a.denominators), rd(a.ledger), rd(a.sources), rd(a.dag),
                {"denominators": a.denominators_id, "transform_ledger": a.ledger_id,
                 "source_table": a.sources_id, "feature_dag": a.dag_id})
    tmp = a.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"new": out["new_denominator"], "ledger": out["transform_ledger"]["by_state"],
                      "pit": out["sources"]["point_in_time_admissible_count"], "dag": out["dag"]["denominators"]})[:800])


if __name__ == "__main__":
    sys.exit(main())
