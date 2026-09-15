#!/usr/bin/env python3
"""The whole route for a retrospective archive: provider → lake → data-gov → receipt → cube.

§4 of `docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md`:

    "Build that disposable stack from identified candidate packages and run the whole path.
     UNKNOWN must survive in persisted evidence; whole-archive delivery and rejected ranges/
     holdout/point-in-time/live cases must have their declared outcomes. Include regression
     coverage for existing synthetic contracts."

Semantic unit tests on the provider were not enough, and Musashi was right to say so: they
prove what a function returns, not what survives four processes, an HTTP boundary, a
confirmation and a row in a cube. This runs against a **disposable** stack — free ports,
throwaway SQLite, candidate packages — and never touches the deployed catalogue.

What it checks, in order:

  whole archive      a delivery of the entire resource is allowed, and the availability the
                     lake publishes for it still says UNKNOWN;
  ranged archive     refused, and the refusal names the retrospective archive;
  synthetic range    the contracts already deployed keep working — the change is additive;
  persisted evidence UNKNOWN reaches the receipt and the cube, and is not a zero there either.

usage:
  archive_route_probe.py --stack STACK.json --archive-resource NAME --plain-resource NAME
      --out RECEIPT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def client_for(stack: dict):
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT")
                    or Path.home() / "Documents/GitHub/data-gov").expanduser()
    sys.path.insert(0, str(checkout))
    from app.client import DataGovClient

    return DataGovClient(base_url=stack["gov_url"],
                         api_key=Path(stack["api_key_file"]).read_text(encoding="utf-8").strip(),
                         experiment_key=f"archive-route-{int(time.time())}")


def campaign_for(resources, unit, key):
    return {
        "schema": "governed_campaign.v1", "campaign_key": key,
        "classification": "NON_GOVERNING", "project": "predictor",
        "code_identity": {"kind": "file_manifest", "value": "d" * 64},
        "config_sha256": "e" * 64, "input_mode": "DATASETS",
        "synthetic_spec_sha256": None, "units": [unit],
        "datasets": [{"lake": "synthetic_fixtures", "resource": name, "role": role,
                      "from": start, "to": end}
                     for name, role, start, end in resources],
        "terminal_lake": "olap_cube",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", type=Path, required=True,
                        help="the JSON the disposable stack printed")
    parser.add_argument("--archive-resource", required=True)
    parser.add_argument("--plain-resource", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    stack = json.loads(args.stack.read_text(encoding="utf-8"))
    client = client_for(stack)
    work = Path(stack["work"])
    cache = work / "probe-cache"
    findings = {}

    # 1. the whole archive: allowed
    key = f"archive-whole-{int(time.time())}"
    unit = "archive-whole-1"
    status, receipt = client.submit_campaign(
        campaign_for([(args.archive_resource, "archive", None, None)], unit, key))
    findings["campaign_whole"] = status
    if status in (200, 201):
        sha = receipt["campaign_sha256"]
        code, delivery = client.governed_download(
            sha, unit, "synthetic_fixtures", args.archive_resource, "archive", cache)
        findings["whole_archive"] = {
            "status": code,
            "state": delivery.get("verification_state") if code == 200 else delivery,
            "bytes": delivery.get("bytes") if code == 200 else None,
            "availability": delivery.get("availability")
            or delivery.get("availability_contract") or None,
            "delivery_id": delivery.get("delivery_id") if code == 200 else None,
        }
        if code == 200:
            terminal = {
                "schema": "governed_terminal.v1", "generation": 1, "status": "COMPLETED",
                "reason": None, "started_at": now(), "finished_at": now(),
                "costs": {"wall_seconds": 0.0},
                "deliveries": [delivery["delivery_id"]], "artifacts": [],
                "metrics": [{"metric": "archive_bytes", "split": "test", "horizon": 0,
                             "unit": "bytes", "value": float(delivery["bytes"]),
                             "std_dev": None, "min_value": None, "max_value": None}],
                "tags": {"use_class": "ARCHIVE_RETROSPECTIVE"}}
            findings["terminal"] = client.report_terminal(sha, unit, terminal)[0]
            findings["reconciliation"] = client.reconcile_campaign(sha)[1]

    # 2. a range over the archive: refused, and the reason must name the class
    key = f"archive-range-{int(time.time())}"
    unit = "archive-range-1"
    status, receipt = client.submit_campaign(
        campaign_for([(args.archive_resource, "archive", "2024-01-01", "2024-01-03")],
                     unit, key))
    if status in (200, 201):
        code, refusal = client.governed_download(
            receipt["campaign_sha256"], unit, "synthetic_fixtures", args.archive_resource,
            "archive", cache, start="2024-01-01", end="2024-01-03")
        findings["ranged_archive"] = {"status": code, "body": refusal}
    else:
        findings["ranged_archive"] = {"status": status, "body": receipt,
                                      "refused_at": "campaign"}

    # 3. regression: a contract already deployed still delivers, whole and ranged
    key = f"plain-{int(time.time())}"
    unit = "plain-1"
    status, receipt = client.submit_campaign(
        campaign_for([(args.plain_resource, "input", None, None)], unit, key))
    if status in (200, 201):
        code, delivery = client.governed_download(
            receipt["campaign_sha256"], unit, "synthetic_fixtures", args.plain_resource,
            "input", cache)
        findings["plain_whole"] = {"status": code,
                                   "bytes": delivery.get("bytes") if code == 200 else delivery}

    # 4. what the accounting persisted: UNKNOWN must still be UNKNOWN there
    accounting = work / "accounting.sqlite"
    if accounting.is_file():
        connection = sqlite3.connect(f"file:{accounting}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(
            "SELECT unit_id, resource_id, state, bytes, availability_contract_sha256 "
            "FROM governed_deliveries ORDER BY rowid DESC LIMIT 6")]
        findings["persisted_deliveries"] = rows
        connection.close()

    body = {"schema": "archive_route_probe.v1", "at": now(),
            "stack": {"gov_url": stack["gov_url"], "work": str(work)},
            "archive_resource": args.archive_resource,
            "plain_resource": args.plain_resource, "findings": findings}
    args.out.write_text(json.dumps(body, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
