#!/usr/bin/env python3
"""Can the warehouse still say "retrospective archive" after the producer is gone?

S2 of `docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md`:

    "run archive delivery -> terminal -> cube; stop the producer and remove the test's
     temporary source configuration; a fresh reader still resolves the retained contract and
     verifies UNKNOWN. Missing or mismatched references yield an explicit unresolved outcome,
     never zero lag or a guessed use class."

The gap being closed is one I reported in R4: the cube stored the contract DIGEST and nothing
else, so it could show that a delivery referenced *some* contract and could not say what that
contract said. Resolving it meant asking the producer — which is precisely what may no longer
exist when someone reads the cube a year later.

The proof is deliberately destructive in the right order: deliver, close the unit, persist,
then **kill the lake host and delete its configuration file**, and only then read. A reader
that still answers has read the warehouse, because there is nothing else left to read.

usage:
  archive_contract_persistence_probe.py --stack STACK.json --archive-resource NAME
      --plain-resource NAME --out RECEIPT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def client_for(stack: dict, key: str):
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT")
                    or Path.home() / "Documents/GitHub/data-gov").expanduser()
    sys.path.insert(0, str(checkout))
    from app.client import DataGovClient

    return DataGovClient(base_url=stack["gov_url"],
                         api_key=Path(stack["api_key_file"]).read_text(encoding="utf-8").strip(),
                         experiment_key=key)


def campaign_for(datasets, unit, key):
    return {
        "schema": "governed_campaign.v1", "campaign_key": key,
        "classification": "NON_GOVERNING", "project": "predictor",
        "code_identity": {"kind": "file_manifest", "value": "d" * 64},
        "config_sha256": "e" * 64, "input_mode": "DATASETS",
        "synthetic_spec_sha256": None, "units": [unit],
        "datasets": datasets, "terminal_lake": "olap_cube",
    }


def dataset(resource, role="input", start=None, end=None):
    return {"lake": "synthetic_fixtures", "resource": resource, "role": role,
            "from": start, "to": end}


def terminal_for(delivery, unit_tag):
    return {"schema": "governed_terminal.v1", "generation": 1, "status": "COMPLETED",
            "reason": None, "started_at": now(), "finished_at": now(),
            "costs": {"wall_seconds": 0.0}, "deliveries": [delivery["delivery_id"]],
            "artifacts": [],
            "metrics": [{"metric": "delivered_bytes", "split": "test", "horizon": 0,
                         "unit": "bytes", "value": float(delivery["bytes"]),
                         "std_dev": None, "min_value": None, "max_value": None}],
            "tags": {"case": unit_tag}}


def run_case(client, findings, name, datasets, cache, *, expect_delivery=True,
             download=None, close=False):
    """One campaign, one unit, and whatever the governance actually answered."""
    key = f"s2-{name}-{int(time.time() * 1000)}"
    unit = f"{name}-1"
    status, receipt = client.submit_campaign(campaign_for(datasets, unit, key))
    case = {"campaign_key": key, "unit_id": unit, "campaign_status": status}
    if status not in (200, 201):
        case.update(refused_at="campaign", body=receipt, terminal=None)
        findings[name] = case
        return None, None
    sha = receipt["campaign_sha256"]
    case["campaign_sha256"] = sha
    if not expect_delivery:
        findings[name] = case
        return sha, None
    item = download or datasets[0]
    code, body = client.governed_download(
        sha, unit, item["lake"], item["resource"], item["role"], cache,
        start=item.get("from"), end=item.get("to"))
    case["download_status"] = code
    if code != 200:
        # a refusal is an outcome, and the unit is closed as REFUSED rather than abandoned
        case["refusal"] = body
        refused = {"schema": "governed_terminal.v1", "generation": 1, "status": "REFUSED",
                   "reason": str(body.get("error") or body)[:300], "started_at": now(),
                   "finished_at": now(), "costs": {"wall_seconds": 0.0}, "deliveries": [],
                   "artifacts": [], "metrics": [], "tags": {"case": name}}
        case["terminal"] = client.report_terminal(sha, unit, refused)[0]
        case["reconciliation"] = client.reconcile_campaign(sha)[1]
        findings[name] = case
        return sha, None
    case.update(delivery_id=body["delivery_id"], bytes=body["bytes"],
                verification_state=body.get("verification_state"),
                availability_contract_sha256=body.get("availability_contract_sha256"),
                availability=body.get("availability"))
    if close:
        case["terminal"] = client.report_terminal(sha, unit, terminal_for(body, name))[0]
        case["reconciliation"] = client.reconcile_campaign(sha)[1]
    findings[name] = case
    return sha, body


def fresh_reader(stack, deliveries):
    """A NEW process-local provider over the same database, and nothing else.

    Not the stack's warehouse host, not data-gov, not the lake: a reader that only has the
    database. This is what an operator has a year later.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "olap" / "store" / "src"))
    from predictor_olap_store.query import Plugin

    plugin = Plugin()
    if stack.get("cube_postgres"):
        os.environ["PGDATABASE"] = stack["cube_postgres"]
        plugin.set_params(sqlite_path=None, schema="public")
    else:
        plugin.set_params(sqlite_path=stack["cube"])
    out = {}
    for name, delivery_id in deliveries.items():
        row = dict(plugin.resolve_delivery_availability(delivery_id))
        canonical = row.pop("canonical_bytes", None)
        if canonical:
            # An INDEPENDENT recomputation beside the reader's own. The reader now verifies
            # before it answers (U2); this second hash is kept so the receipt shows the two
            # agreeing rather than asking anyone to take the reader's word for it.
            row["probe_recomputed_sha256"] = hashlib.sha256(
                canonical.encode("ascii")).hexdigest()
            row["canonical_says_lag"] = json.loads(canonical)["availability"][
                "completion_lag_max"]
            row["canonical_says_use_class"] = json.loads(canonical)["availability"]["use_class"]
        out[name] = row
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--archive-resource", required=True)
    parser.add_argument("--plain-resource", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", metavar="YYYY-MM-DD",
                        help="the holdout the stack declares, if any")
    args = parser.parse_args(argv)

    stack = json.loads(args.stack.read_text(encoding="utf-8"))
    stack.setdefault("api_key_file", stack["key_file"])
    client = client_for(stack, f"s2-archive-{int(time.time())}")
    work = Path(stack["work"])
    cache = work / "s2-cache"
    findings, deliveries = {}, {}

    # 1. the whole archive: delivered, closed, persisted
    _sha, delivered = run_case(client, findings, "archive-whole",
                               [dataset(args.archive_resource, "archive")], cache, close=True)
    if delivered:
        deliveries["archive-whole"] = delivered["delivery_id"]

    # 2. the refusal matrix, each closed as REFUSED so no campaign is left open
    run_case(client, findings, "archive-range",
             [dataset(args.archive_resource, "archive", "2024-01-01", "2024-01-03")], cache)
    run_case(client, findings, "archive-point-in-time",
             [dataset(args.archive_resource, "archive", "2024-01-02", "2024-01-02")], cache)
    # a range whose end is in the future: "give me everything up to tomorrow". An open end was
    # refused by the campaign schema as malformed, which tested the schema and not the class.
    run_case(client, findings, "archive-live-tail",
             [dataset(args.archive_resource, "archive", "2026-09-01", "2027-01-01")], cache)
    # a range that crosses the declared holdout: refused for the holdout's own reason, and
    # only meaningful when the stack declares one. When it does not, that is said, not faked.
    if args.holdout:
        # a range that reaches past the declared holdout. The holdout is placed AFTER the
        # fixture data on purpose: a holdout inside the data refuses the whole-archive
        # delivery too, which was measured, and would have hidden what this case tests.
        run_case(client, findings, "plain-crosses-holdout",
                 [dataset(args.plain_resource, "input", "2024-01-01", args.holdout)], cache)
    else:
        findings["plain-crosses-holdout"] = {
            "not_exercised": "the disposable stack declares no holdout_start, so no range can "
                             "cross one; run with --holdout to exercise this refusal"}

    # 3. regression: a contract already in the catalogue is unaffected, whole and ranged
    _sha, plain = run_case(client, findings, "plain-whole",
                           [dataset(args.plain_resource)], cache, close=True)
    if plain:
        deliveries["plain-whole"] = plain["delivery_id"]

    # 4. the producer is stopped and its configuration deleted BEFORE anything is read
    lake_pid = stack.get("lake_pid")
    teardown = {"lake_pid": lake_pid}
    if lake_pid:
        try:
            os.kill(int(lake_pid), signal.SIGTERM)
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(int(lake_pid), 0)
                except OSError:
                    break
            teardown["stopped"] = True
        except OSError as exc:
            teardown["stopped"] = False
            teardown["error"] = str(exc)
    lake_config = work / "lake.json"
    teardown["config_existed"] = lake_config.is_file()
    if lake_config.is_file():
        teardown["config_sha256"] = hashlib.sha256(lake_config.read_bytes()).hexdigest()
        lake_config.unlink()
    teardown["config_present_after"] = lake_config.is_file()
    try:
        import urllib.request

        urllib.request.urlopen(stack["lake_url"] + "/healthz", timeout=3)
        teardown["lake_still_answers"] = True
    except Exception as exc:
        teardown["lake_still_answers"] = False
        teardown["lake_error"] = type(exc).__name__
    findings["producer_teardown"] = teardown

    # 5. the fresh reader, with only the database left
    findings["fresh_reader"] = fresh_reader(stack, deliveries)
    # 6. and a reference nobody retained: it must be UNRESOLVED, not a zero
    findings["fresh_reader"]["never-retained"] = fresh_reader(
        stack, {"x": "delivery-that-never-existed"})["x"]

    body = {"schema": "archive_contract_persistence_probe.v1", "at": now(),
            "stack": {"gov_url": stack["gov_url"], "cube": stack.get("cube"),
                      "cube_postgres": stack.get("cube_postgres"), "work": str(work)},
            "archive_resource": args.archive_resource,
            "plain_resource": args.plain_resource, "findings": findings}
    args.out.write_text(json.dumps(body, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
