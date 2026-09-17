#!/usr/bin/env python3
"""Select one D3 run's population in the cube by IDENTITY, never by campaign-key prefix (L3).

`d3mech-v2`'s envelope was emitted under the fixed `campaign_key d3-mechanics-v1`; the cube
distinguishes the runs by `run_id`, `design_sha256` and `envelope_sha256`, and so must every
query. This module builds the SQL for one run and, given a service, answers it; a selection is
refused when the identity resolves to zero or more than one envelope, or when the rows of two
identities overlap.

    python tools/df_d3_cube_select.py --run-id d3mech-v2 [--design-sha256 ...] --url ... --schema public
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def envelope_sql(schema: str, run_id: str, design_sha256: str | None = None) -> str:
    """The envelope(s) a run identity resolves to, from the units' own envelope digests."""
    where = f"run_id = '{run_id}'"
    if design_sha256:
        where += f" AND design_sha256 = '{design_sha256}'"
    return (f'SELECT r.campaign_key, r.run_id, r.design_sha256, r.code_identity, r.result_class, '
            f'f.envelope_sha256, count(*) AS unit_rows '
            f'FROM "{schema}"."dim_campaign_run" AS r '
            f'JOIN "{schema}"."fact_campaign_unit" AS f '
            f'ON f.campaign_key = r.campaign_key AND f.run_id = r.run_id '
            f'WHERE r.{where} GROUP BY 1,2,3,4,5,6 LIMIT 100')


#: the service caps LIMIT (max_rows, host-configured); cells are paged by key under it
PAGE = 1000


def units_sql(schema: str, envelope_sha256: str, after: str = "") -> str:
    return (f'SELECT cell_key, count(*) AS rows FROM "{schema}"."fact_campaign_unit" '
            f'WHERE envelope_sha256 = \'{envelope_sha256}\' AND cell_key > \'{after}\' '
            f'GROUP BY cell_key ORDER BY cell_key LIMIT {PAGE}')


def cells_of(url: str, token: str, schema: str, envelope_sha256: str) -> list:
    out, after = [], ""
    while True:
        page = query(url, token, units_sql(schema, envelope_sha256, after))
        out.extend(page)
        if len(page) < PAGE:
            return out
        after = page[-1]["cell_key"]


def query(url: str, token: str, sql: str) -> list:
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
                                     headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as answer:
            return json.loads(answer.read())["rows"]
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"REFUSED: query answered http {exc.code}: "
                         f"{exc.read()[:300]!r} for {sql[:200]}") from None


def select(url: str, token: str, *, schema: str, run_id: str, design_sha256: str | None = None) -> dict:
    """One run -> exactly one envelope and its unit rows; refused otherwise."""
    envelopes = query(url, token, envelope_sql(schema, run_id, design_sha256))
    if len(envelopes) != 1:
        raise SystemExit(f"REFUSED: identity run_id={run_id!r} design={design_sha256!r} resolves "
                         f"to {len(envelopes)} envelopes, not one: {envelopes}")
    env = envelopes[0]
    cells = cells_of(url, token, schema, env["envelope_sha256"])
    return {"identity": {"run_id": run_id, "design_sha256": env["design_sha256"],
                         "envelope_sha256": env["envelope_sha256"],
                         "campaign_key_as_stored": env["campaign_key"]},
            "unit_rows": sum(int(c["rows"]) for c in cells),
            "cells": sorted(c["cell_key"] for c in cells)}


def disjoint(a: dict, b: dict) -> bool:
    """Two selections mix populations iff they share an envelope or a unit row."""
    return a["identity"]["envelope_sha256"] != b["identity"]["envelope_sha256"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("WAREHOUSE_URL", "http://127.0.0.1:5057"))
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--design-sha256")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"REFUSED: {args.token_env} is not set")
    out = select(args.url, token, schema=args.schema, run_id=args.run_id,
                 design_sha256=args.design_sha256)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
