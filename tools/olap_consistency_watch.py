#!/usr/bin/env python3
"""Watch the live cube for surplus multiplicity and for a store that answers two ways.

Block 2 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`:

    "Keep a bounded read-only reconciliation/monitoring check for warehouse multiplicity and
     predicate-versus-scan consistency, with measured query cost. Define cadence from that cost
     instead of running full scans on every request. Record discrepancies and alert; no
     automatic deletion, reindexing or repair."

Two faults were found by hand and both are cheap to keep watching for:

  multiplicity      a child row present more times than the accepted payload declares;
  predicate_vs_scan a filtered read and a scan of the same table returning different
                    populations, which is what an index short of its table looks like.

What this tool will NOT do, by construction rather than by intention: it never writes, never
deletes, never reindexes and never asks for a repair. It has no code path that can. It reads
through the RUNNING service, so it needs no outage and cannot take the database lock.

**Cadence comes from measured cost, not from a guess.** Every run times its own queries and
writes a recommended interval derived from a duty-cycle budget: a check that costs a second
should not run every second. The default budget is one percent of wall clock, so a one-second
check earns a one-hundred-second interval, floored and capped at the bounds below.

Monitoring output is operational evidence and is written where operational evidence lives. It
is never a governed campaign, carries no terminal and produces no scientific result.

usage:
  olap_consistency_watch.py --service-url URL --state DIR [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

#: The governed child relations, and the column that ties each to its terminal.
CHILDREN = ("gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact")
PARENT = "gov_terminal"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Cadence bounds. A check that is cheap still should not be continuous, and one that is
#: expensive still should not be abandoned.
MIN_INTERVAL_SECONDS = 300
MAX_INTERVAL_SECONDS = 86_400
DEFAULT_DUTY_CYCLE = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Service:
    """Read-only access to the running warehouse. There is no write method on this class."""

    def __init__(self, url: str, token_env: str, schema: str):
        self.url, self.schema = url.rstrip("/"), schema
        self.token = os.environ.get(token_env)
        if not self.token:
            raise SystemExit(f"{token_env} is not set: the service token comes from the "
                             "environment, never from an argument")
        self.queries = 0
        self.seconds = 0.0

    def query(self, sql: str) -> list:
        request = urllib.request.Request(
            f"{self.url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
            headers={"Authorization": f"Bearer {self.token}"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=120) as handle:
            body = json.load(handle)
        self.seconds += time.monotonic() - started
        self.queries += 1
        return body["rows"]


def terminals(service: Service) -> list:
    rows = service.query(f'SELECT terminal_sha256 FROM "{service.schema}"."{PARENT}"'
                         " ORDER BY 1 LIMIT 5000")
    return [row["terminal_sha256"] for row in rows
            if isinstance(row.get("terminal_sha256"), str)
            and _HEX64.match(row["terminal_sha256"])]


def multiplicity(service: Service, relation: str, columns: list) -> list:
    """Child rows the cube holds more than once, as the cube itself groups them.

    This is one aggregate per relation rather than a row-by-row comparison against the accepted
    payloads: it cannot say whether a duplicate is legitimate, and it does not pretend to. It
    says where to look, and the reconciler - which does have the payloads - says what it means.
    """
    projection = ", ".join(f'"{column}"' for column in columns)
    rows = service.query(
        f'SELECT {projection}, count(*) AS n FROM "{service.schema}"."{relation}"'
        " GROUP BY ALL HAVING count(*) > 1 ORDER BY 1 LIMIT 1000")
    return rows


def predicate_vs_scan(service: Service, relation: str, keys: list) -> dict:
    """Does asking by predicate return what enumerating returns?

    Both sides are single aggregates over the whole relation, correlated against the parent
    table, so the cost is two queries per relation rather than two per terminal.
    """
    total = service.query(
        f'SELECT count(*) AS n FROM "{service.schema}"."{relation}" LIMIT 1')[0]["n"]
    inlined = ", ".join(f"'{key}'" for key in keys) or "''"
    by_predicate = service.query(
        f'SELECT count(*) AS n FROM "{service.schema}"."{relation}"'
        f" WHERE terminal_sha256 IN ({inlined}) LIMIT 1")[0]["n"]
    by_scan = service.query(
        f'SELECT count(*) AS n FROM (SELECT * FROM "{service.schema}"."{relation}" OFFSET 0)'
        f" WHERE terminal_sha256 IN ({inlined}) LIMIT 1")[0]["n"]
    return {"rows": total, "by_predicate": by_predicate, "by_scan": by_scan,
            "agrees": total == by_predicate == by_scan}


def recommended_interval(seconds: float, duty_cycle: float) -> int:
    """How often this check may run, given what it just cost. Measured, not assumed."""
    if duty_cycle <= 0:
        return MAX_INTERVAL_SECONDS
    interval = seconds / duty_cycle
    return int(min(MAX_INTERVAL_SECONDS, max(MIN_INTERVAL_SECONDS, interval)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--schema", default="main")
    parser.add_argument("--state", type=Path,
                        help="directory for the observation log and the latest report")
    parser.add_argument("--duty-cycle", type=float, default=DEFAULT_DUTY_CYCLE,
                        help="share of wall clock this check may consume; the cadence follows")
    parser.add_argument("--json", action="store_true", help="print the whole report")
    args = parser.parse_args(argv)

    service = Service(args.service_url, args.token_env, args.schema)
    started = time.monotonic()
    keys = terminals(service)
    findings, relations = [], {}
    for relation in CHILDREN:
        columns = sorted({
            row["column_name"] for row in service.query(
                "SELECT column_name FROM information_schema.columns WHERE table_name = "
                f"'{relation}' LIMIT 100")})
        duplicates = multiplicity(service, relation, columns) if columns else []
        agreement = predicate_vs_scan(service, relation, keys)
        relations[relation] = {"predicate_vs_scan": agreement,
                               "rows_present_more_than_once": len(duplicates),
                               "duplicate_groups": duplicates[:20]}
        if duplicates:
            findings.append({"relation": relation, "kind": "MULTIPLICITY",
                             "groups": len(duplicates),
                             "detail": "rows the cube holds more than once; the reconciler, "
                                       "which holds the accepted payloads, decides whether "
                                       "any of them is surplus"})
        if not agreement["agrees"]:
            findings.append({"relation": relation, "kind": "PREDICATE_VS_SCAN",
                             "detail": f"a filtered read reaches {agreement['by_predicate']} "
                                       f"rows and a scan reaches {agreement['by_scan']} of "
                                       f"{agreement['rows']}"})

    elapsed = time.monotonic() - started
    report = {
        "schema": "olap_consistency_watch.v1", "generated_utc": now(),
        "service": args.service_url, "schema_name": args.schema,
        "terminals": len(keys), "relations": relations,
        "findings": findings,
        "alert": bool(findings),
        "cost": {"queries": service.queries,
                 "query_seconds": round(service.seconds, 3),
                 "wall_seconds": round(elapsed, 3),
                 "duty_cycle": args.duty_cycle},
        "recommended_interval_seconds": recommended_interval(elapsed, args.duty_cycle),
        "actions_taken": [],
        "note": ("Read-only operational monitoring. Nothing here deletes, reindexes, repairs "
                 "or writes; a finding is evidence for an operator, never a trigger. This is "
                 "not a governed campaign and produces no scientific result."),
    }
    if args.state:
        args.state.mkdir(parents=True, exist_ok=True)
        (args.state / "LATEST.json").write_text(json.dumps(report, indent=1) + "\n",
                                                encoding="utf-8")
        with (args.state / "observations.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({key: report[key] for key in
                                  ("generated_utc", "terminals", "findings", "alert", "cost",
                                   "recommended_interval_seconds")},
                                 separators=(",", ":")) + "\n")
    print(json.dumps(report if args.json else
                     {"alert": report["alert"], "findings": findings,
                      "terminals": report["terminals"], "cost": report["cost"],
                      "recommended_interval_seconds":
                          report["recommended_interval_seconds"]}, indent=1))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
