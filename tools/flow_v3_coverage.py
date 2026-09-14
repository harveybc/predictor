#!/usr/bin/env python3
"""Flow v3 coverage per project from the cube, through data-gov's SELECT-only query.

Answers, for one project, how many terminals the cube holds by classification
and status, how many campaigns they belong to, and the latest terminals; the
same SQL can be pasted into Metabase (`--sql-only`). Terminal identity comes
from `gov_terminal` (one row per campaign/unit/generation).

usage: flow_v3_coverage.py --project predictor [--gov-url URL --api-key-file FILE]
                           [--lake olap_cube] [--limit 20] [--sql-only]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def coverage_sql(project: str) -> str:
    if not KEY_RE.match(project):
        raise ValueError("invalid project key")
    return (
        "SELECT project, classification, status, COUNT(*) AS terminals, "
        "COUNT(DISTINCT campaign_sha256) AS campaigns, COUNT(DISTINCT unit_id) AS units "
        f"FROM gov_terminal WHERE project = '{project}' "
        "GROUP BY project, classification, status ORDER BY project, classification, status"
    )


def latest_sql(project: str, limit: int) -> str:
    if not KEY_RE.match(project):
        raise ValueError("invalid project key")
    limit = int(limit)
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be within 1..1000")
    return (
        "SELECT campaign_key, unit_id, generation, classification, status, reason, finished_at, "
        f"terminal_sha256 FROM gov_terminal WHERE project = '{project}' "
        f"ORDER BY finished_at DESC LIMIT {limit}"
    )


def query(gov_url: str, api_key: str, lake: str, sql: str):
    url = gov_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"lake": lake, "sql": sql})
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode())
        except Exception:
            body = {"error": str(exc)}
        return exc.code, body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file")
    parser.add_argument("--lake", default="olap_cube")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sql-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        sql = {"coverage": coverage_sql(args.project), "latest": latest_sql(args.project, args.limit)}
    except ValueError as exc:
        print(f"flow_v3_coverage: {exc}", file=sys.stderr)
        return 2
    if args.sql_only:
        print(json.dumps(sql, indent=2))
        return 0
    key = None
    if args.api_key_file:
        with open(os.path.expanduser(args.api_key_file), encoding="utf-8") as handle:
            key = handle.read().strip()
    key = key or os.environ.get("DATA_GOV_API_KEY")
    if not key:
        print("flow_v3_coverage: no API key: pass --api-key-file or set DATA_GOV_API_KEY", file=sys.stderr)
        return 2
    out = {"project": args.project, "lake": args.lake}
    for name, text in sql.items():
        status, body = query(args.gov_url, key, args.lake, text)
        out[name] = {"http": status, "result": body}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if all(out[n]["http"] == 200 for n in sql) else 1


if __name__ == "__main__":
    raise SystemExit(main())
