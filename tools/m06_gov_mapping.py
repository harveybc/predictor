#!/usr/bin/env python3
"""PROPOSAL: map M04 batch-1 v3 verification receipts onto the governed warehouse grain.

Target grain (data-gov lake_plugins/sql_lake.py):
  gov_terminal          one row per campaign_sha256 x unit_id x generation
  gov_terminal_metric   terminal_sha256 x metric x split x horizon x unit -> value
  gov_terminal_dataset  terminal_sha256 x delivery_id (governed deliveries only)
  gov_terminal_artifact terminal_sha256 x role -> sha256, bytes

This tool WRITES NO ROWS and sends nothing.  It emits, per verified candidate, the terminal it
would propose, its metric rows, and the GAPS that block a governed write: in particular the inputs
were bound to an M03 declaration and a local NPZ, not to a governed delivery id, so
gov_terminal_dataset cannot be filled and the terminal is classified NON_GOVERNING.

Unit identity: unit_id = candidate id (cid); config identity: config_sha256 = config_id;
code identity: the receipt's predictor_revision.  Metric keys follow the terminal-key rule
[A-Za-z0-9._:-]: MAE, MSE, Naive_MAE, Naive_MSE, Skill_MAE, Skill_MSE; horizon NULL = aggregate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

METRIC_KEYS = {"MAE": "MAE", "MSE": "MSE", "baseline_MAE": "Naive_MAE", "baseline_MSE": "Naive_MSE",
               "skill_MAE": "Skill_MAE", "skill_MSE": "Skill_MSE"}
KEY_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def iso(t):
    import datetime as dt
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def propose(queue: dict, receipts: dict, binding: dict | None) -> dict:
    att = {}
    for a in queue["attempts"]:
        att.setdefault((a["cid"], a["kind"]), []).append(a)
    obj = queue["objective"]
    units, gaps = [], set()
    if not binding or not binding.get("governed_delivery_ids"):
        gaps.add("NO_GOVERNED_DELIVERY: inputs are bound to an M03 admissible-input declaration and a local NPZ, "
                 "not to a data-gov delivery id; gov_terminal_dataset cannot be filled, so the terminal is NON_GOVERNING")
    gaps.add("NO_SERVICE_ROUTE_USED: rows are written only through the governed route (tools/governed_run.py / "
             "flush_governed_terminals.py); this proposal sends nothing")
    for c in queue["candidates"]:
        if c["status"] != "verified":
            continue
        rc = receipts.get(c["cid"][:16])
        tr = [a for a in att.get((c["cid"], "train"), []) if a["status"] == "completed"][-1]
        ve = [a for a in att.get((c["cid"], "verify"), []) if a["status"] == "completed"][-1]
        if rc is None:
            raise SystemExit(f"missing receipt for {c['cid'][:16]}")
        rows = []
        for src, key in METRIC_KEYS.items():
            assert KEY_RE.match(key)
            rows.append({"metric": key, "split": obj["split"], "horizon": None, "unit": obj["unit"],
                         "value": rc["metrics"][src]})
            for h, m in sorted(rc["per_horizon"].items(), key=lambda kv: int(kv[0])):
                rows.append({"metric": key, "split": obj["split"], "horizon": int(h), "unit": obj["unit"], "value": m[src]})
        units.append({
            "gov_terminal": {"campaign_key": queue["campaign"], "campaign_sha256": queue["meta"]["campaign_sha256"],
                             "unit_id": c["cid"], "generation": 1, "actor": "satoshi", "project": "predictor",
                             "classification": "NON_GOVERNING", "status": "COMPLETED", "reason": None,
                             "started_at": iso(tr.get("started")), "finished_at": iso(ve.get("finished")),
                             "terminal_lake": "olap_cube", "config_sha256": c["config_id"],
                             "code_identity_json": {"kind": "git_commit", "value": rc["bridge"]["predictor_revision"]},
                             "costs_json": {"train_elapsed_seconds": tr.get("elapsed_seconds"),
                                            "train_cgroup_peak_bytes": tr.get("cgroup_peak_bytes"),
                                            "observed_updates": tr.get("observed_updates"),
                                            "per_update_seconds": tr.get("per_update_seconds"),
                                            "verify_elapsed_seconds": ve.get("elapsed_seconds"),
                                            "verify_cgroup_peak_bytes": ve.get("cgroup_peak_bytes")},
                             "tags_json": {"label": c["label"], "seed": c["seed"], "host_role_train": tr.get("host"),
                                           "host_role_verify": ve.get("host"), "verdict": ve.get("verdict"),
                                           "exact_match": rc.get("exact_match"), "selected_epoch": tr.get("selected_epoch"),
                                           "stop_reason": tr.get("stop_reason"),
                                           "comparability": "NOT_COMPARABLE (ECL L24->H1..24 validation; published rows are L96->H96 test)",
                                           "architecture": "superseded old design (branch_steps=12, core time_factors [2,1,1])"},
                             "synthetic_spec_sha256": None},
            "gov_terminal_metric": rows,
            "gov_terminal_dataset": [],
            "gov_terminal_artifact": [
                {"role": "model", "sha256": rc["digests"]["model_sha256"], "bytes": None},
                {"role": "weights", "sha256": tr.get("weights_sha256"), "bytes": None},
                {"role": "validation_population", "sha256": rc["digests"]["validation_sha256"], "bytes": None},
                {"role": "predictions", "sha256": rc["digests"]["predictions_sha256"], "bytes": None},
                {"role": "verification_receipt", "sha256": rc.get("receipt_sha256"), "bytes": None}]})
    gaps.add("ARTIFACT_BYTES_UNKNOWN: receipts carry digests, not byte sizes; gov_terminal_artifact.bytes is NOT NULL")
    return {"schema": "m06.gov_mapping_proposal.v1", "status": "PROPOSAL_NO_ROWS_WRITTEN",
            "grain": {"gov_terminal": "campaign_sha256 x unit_id(cid) x generation",
                      "gov_terminal_metric": "terminal x metric x split(validation) x horizon(NULL|1..24) x unit(z_train)"},
            "units": units, "gaps": sorted(gaps),
            "counts": {"terminals": len(units), "metric_rows": sum(len(u["gov_terminal_metric"]) for u in units)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--binding", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    q = json.load(open(a.queue))
    rec = {f[:-5]: json.load(open(os.path.join(a.receipts, f))) for f in os.listdir(a.receipts) if f.endswith(".json")}
    b = json.load(open(a.binding)) if a.binding else None
    p = propose(q, rec, b)
    tmp = a.out + ".tmp"
    json.dump(p, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(p["counts"], len(p["gaps"]), "gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
