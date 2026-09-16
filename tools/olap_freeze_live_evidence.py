#!/usr/bin/env python3
"""Freeze what the RUNNING warehouse currently serves, without stopping it.

I1 of `docs/handoffs/MUSASHI_H1_H3_LIVE_REVIEW_AND_I1_I3_2026_09_16.md` asks for a consistent
evidence copy before any repair. The boundary-held snapshot path needs the owner stopped. This
is the other route, and it states its own limits rather than borrowing the stronger word:

* every governed relation is read through the service, in pages, with the row total checked
  against a separate `count(*)`;
* the service's OWN content digest for each relation is taken BEFORE and AFTER the read, and
  the copy's digest is recomputed independently and compared with both. Three agreements, or
  the copy is `UNSTABLE_DURING_READ` and carries no claim;
* the result is `CONSISTENT_LIVE_READ`, which is weaker than `VERIFIED_SNAPSHOT`: it proves
  the content did not move while it was read, not that no writer could have moved it.

The copy is a real cube, built through the provider, so a rehearsal on it exercises the same
constraints production has.

usage:
  olap_freeze_live_evidence.py --service-url URL --target FILE --out RECEIPT.json
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RELATIONS = ("gov_terminal", "gov_terminal_metric", "gov_terminal_dataset",
             "gov_terminal_artifact", "gov_availability_contract")
PAGE = 1000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Service:
    def __init__(self, url: str, token_env: str, schema: str):
        self.url, self.schema = url.rstrip("/"), schema
        self.token = os.environ.get(token_env)
        if not self.token:
            raise SystemExit(f"{token_env} is not set: the service token comes from the "
                             "environment, never from an argument")

    def _get(self, path: str, params: dict):
        request = urllib.request.Request(
            f"{self.url}{path}?{urllib.parse.urlencode(params)}",
            headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(request, timeout=120) as handle:
            return json.load(handle)

    def query(self, sql: str) -> list:
        return self._get("/api/v1/query", {"sql": sql})["rows"]

    def columns(self, relation: str) -> list:
        body = self._get("/api/v1/schema", {"relation": relation})
        return [column["name"] for column in body.get("columns") or []]

    def count(self, relation: str) -> int:
        return self.query(f'SELECT count(*) AS n FROM "{self.schema}"."{relation}" LIMIT 1'
                          )[0]["n"]

    def digest(self, relation: str, columns: list):
        """The service's own content digest: order-independent, multiplicity-sensitive."""
        projection = ", ".join(f'"{column}"' for column in columns)
        return self.query(
            "SELECT md5(string_agg(h, '' ORDER BY h)) AS d FROM (SELECT md5(CAST(ROW("
            f'{projection}) AS VARCHAR)) AS h FROM "{self.schema}"."{relation}") t LIMIT 1'
        )[0]["d"]

    def rows(self, relation: str, columns: list) -> list:
        expected = self.count(relation)
        order = ", ".join(str(index + 1) for index in range(len(columns)))
        projection = ", ".join(f'"{column}"' for column in columns)
        collected, offset = [], 0
        while True:
            page = self.query(
                f'SELECT {projection} FROM "{self.schema}"."{relation}" ORDER BY {order}'
                f" OFFSET {offset} LIMIT {PAGE}")
            collected.extend(page)
            if len(page) < PAGE:
                break
            offset += PAGE
        if len(collected) != expected:
            raise RuntimeError(f"{relation}: read {len(collected)} rows, the service counts "
                               f"{expected}")
        return collected


def local_digest(con, schema: str, relation: str, columns: list):
    projection = ", ".join(f'"{column}"' for column in columns)
    return con.execute(
        "SELECT md5(string_agg(h, '' ORDER BY h)) FROM (SELECT md5(CAST(ROW("
        f'{projection}) AS VARCHAR)) AS h FROM "{schema}"."{relation}") t').fetchone()[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--schema", default="main")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path,
                        help="the row multisets themselves, as data rather than as a database")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "olap" / "store" / "src"))
    sys.path.insert(0, str(repo / "olap" / "duckdb_store" / "src"))
    from predictor_duckdb_store.provider import PredictorDuckdbStore
    from sqlalchemy import text as sql

    service = Service(args.service_url, args.token_env, args.schema)
    if args.target.exists():
        raise SystemExit(f"{args.target} exists: an evidence copy is never written over")

    columns = {relation: service.columns(relation) for relation in RELATIONS}
    before = {relation: service.digest(relation, columns[relation]) for relation in RELATIONS}
    rows = {relation: service.rows(relation, columns[relation]) for relation in RELATIONS}
    after = {relation: service.digest(relation, columns[relation]) for relation in RELATIONS}

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(args.target), schema=args.schema, memory_limit="2GB",
                     threads=2, min_free_bytes=1)
    store.engine()
    with store.write_engine().begin() as conn:
        for relation in RELATIONS:
            if not rows[relation]:
                continue
            names = columns[relation]
            placeholders = ", ".join(f":{name}" for name in names)
            projection = ", ".join(f'"{name}"' for name in names)
            conn.execute(sql(f'INSERT INTO "{args.schema}"."{relation}" ({projection}) '
                             f"VALUES ({placeholders})"), rows[relation])
    with store.engine().connect() as conn:
        conn.execute(sql("CHECKPOINT"))
    raw = store.engine().raw_connection().driver_connection
    # An empty relation has no digest, and `None` is not the string "None": a receipt whose
    # fields disagree in type cannot be compared with the one taken beside it.
    copy = {relation: (lambda value: None if value is None else str(value))(
        local_digest(raw, args.schema, relation, columns[relation]))
        for relation in RELATIONS}
    store.engine().dispose()

    stable = [relation for relation in RELATIONS if before[relation] != after[relation]]
    faithful = [relation for relation in RELATIONS
                if (None if before[relation] is None else str(before[relation]))
                != copy[relation]]
    kind = ("CONSISTENT_LIVE_READ" if not stable and not faithful
            else "UNSTABLE_DURING_READ" if stable else "COPY_DIFFERS_FROM_SOURCE")

    receipt = {
        "schema": "olap_live_evidence.v1", "generated_utc": now(),
        "service": args.service_url, "target": str(args.target),
        "counts": {relation: len(rows[relation]) for relation in RELATIONS},
        "service_digests_before_read": before, "service_digests_after_read": after,
        "evidence_copy_digests": copy,
        "relations_that_moved_during_the_read": stable,
        "relations_whose_copy_differs": faithful,
        "kind": kind,
        "limit": ("A live read, not a transactional snapshot: it proves the content did not "
                  "move while it was read, not that no writer could have moved it. The "
                  "boundary-held snapshot path remains the stronger evidence and needs the "
                  "owning service stopped."),
    }
    if args.rows_out:
        args.rows_out.write_text(json.dumps(
            {"schema": "olap_live_rows.v1", "generated_utc": receipt["generated_utc"],
             "columns": columns, "rows": rows}, indent=1, default=str) + "\n",
            encoding="utf-8")
        receipt["rows_out"] = str(args.rows_out)
    args.out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"kind": kind, "counts": receipt["counts"]}, indent=1))
    return 0 if kind == "CONSISTENT_LIVE_READ" else 1


if __name__ == "__main__":
    raise SystemExit(main())
