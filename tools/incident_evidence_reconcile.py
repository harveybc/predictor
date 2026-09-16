#!/usr/bin/env python3
"""What governance says was accepted, against what the recovered cube actually holds.

F3 of `docs/handoffs/MUSASHI_E1_E6_REVIEW_AND_F1_F5_2026_09_16.md`:

    "Reconcile pre-incident accepted delivery/terminal/outbox identities and their child
     CONTENT against the recovered database; separate committed evidence from pending or
     replayable writes. Counts alone do not support no-loss claims. Do not presume the WAL
     contained only tests without supporting receipts."

I claimed nothing governed was lost, and supported it with three row counts. That is not the
same claim. What makes it checkable is: data-gov's OWN accounting is the record of what was
accepted, it is a separate database that the incident did not touch, and every terminal it
records must be present in the recovered cube with its children's CONTENT intact.

Three populations are kept apart, because they mean different things:

  committed    accepted by governance AND present in the cube — the evidence that survived;
  missing      accepted by governance and ABSENT from the cube — actual loss, if any;
  replayable   still held in an outbox, so its absence from the cube costs a retry, not a loss.

Nothing here opens the quarantined write-ahead log. Read-only throughout.

I2 adds the other direction. A child row the cube holds MORE times than the accepted payload
declares is surplus multiplicity, and `--repair-surplus` removes exactly that excess: expected
multiplicity is counted from the payload, never deduplicated globally, so a contract that
legitimately carries the same row twice keeps both. A row whose value merely differs is
missing-and-extra, not surplus, and is refused.

The cube can be read either from its file or through the RUNNING service, because a snapshot
describes a moment that has already passed and the live contents are the thing in question.

usage:
  incident_evidence_reconcile.py --accounting DB --cube FILE [--outbox DIR ...] --out R.json
  incident_evidence_reconcile.py --accounting DB --service-url URL --out R.json
  incident_evidence_reconcile.py --accounting DB --cube FILE --repair-surplus --evidence E.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CHILDREN = ("gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def accepted_terminals(accounting: Path) -> list[dict]:
    """What data-gov accepted, WITH the canonical payload it accepted.

    `body_json` is the terminal exactly as it was submitted: its metrics, artifacts, verified
    datasets, identity, costs and tags. It lives in a database the incident never touched, so
    it is an EXPECTATION that is not derived from the rows being checked — which is the whole
    point, and exactly what the previous version lacked.
    """
    con = sqlite3.connect(f"file:{accounting}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in con.execute(
            "SELECT terminal_sha256, campaign_sha256, unit_id, generation, status, body_json "
            "FROM governed_terminals ORDER BY terminal_sha256")]
    finally:
        con.close()
    for row in rows:
        try:
            row["body"] = json.loads(row.get("body_json") or "null")
        except ValueError:
            row["body"] = None
        row.pop("body_json", None)
    return rows


# --- the persistence contract, DERIVED rather than handpicked -----------------------------
# H1: the previous version compared a short list somebody chose, so altering `costs_json` or a
# dataset's availability contract link changed nothing it looked at. The mapping below is built
# from the store's own contract constants, and every stored column is either compared or
# explicitly excused - `field_coverage` in the report proves that claim rather than asserting it.

#: Payload field -> the column it is stored in, where the two names differ.
PARENT_COLUMN_OF = {"code_identity": "code_identity_json", "costs": "costs_json",
                    "tags": "tags_json"}
#: Payload fields that are deliberately NOT stored on the parent row, with the reason.
PARENT_NOT_PERSISTED = {
    "schema": "a constant of the contract; the same value for every terminal",
    "metrics": "stored as rows of gov_terminal_metric",
    "verified_datasets": "stored as rows of gov_terminal_dataset",
    "artifacts": "stored as rows of gov_terminal_artifact",
    "deliveries": "the delivery identities are carried by gov_terminal_dataset rows",
}
#: Columns the SERVICE generates, which no payload can be compared against.
SERVICE_GENERATED = {"gov_terminal": {"received_at"}}
#: Fields whose stored form is JSON text rather than a scalar.
JSON_FIELDS = {"code_identity", "costs", "tags"}

#: Child relation -> (payload key, column alias map). The compared field set is the store's own
#: dataset contract, so a field added there is compared here without anyone remembering to.
CHILD_PAYLOAD_KEY = {"gov_terminal_metric": "metrics",
                     "gov_terminal_artifact": "artifacts",
                     "gov_terminal_dataset": "verified_datasets"}
COLUMN_ALIASES = {"gov_terminal_dataset": {"state": "verification_state"}}


def _store_contract():
    """The terminal and dataset key sets the production store itself enforces."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "olap" / "store" / "src"))
    from predictor_olap_store.query import TERMINAL_DATASET_KEYS, TERMINAL_KEYS

    return set(TERMINAL_KEYS), set(TERMINAL_DATASET_KEYS)


TERMINAL_KEYS, TERMINAL_DATASET_KEYS = _store_contract()
#: Parent fields compared: every payload field that is persisted on the parent row.
PARENT_FIELDS = tuple(sorted(
    field for field in TERMINAL_KEYS
    if field not in PARENT_NOT_PERSISTED and field != "terminal_sha256"))
#: Child field sets, from the same source.
CHILD_SHAPES = {
    "gov_terminal_metric": ("metrics", ("metric", "split", "horizon", "unit", "value",
                                        "std_dev", "min_value", "max_value")),
    "gov_terminal_artifact": ("artifacts", ("role", "sha256", "bytes")),
    "gov_terminal_dataset": ("verified_datasets", tuple(sorted(TERMINAL_DATASET_KEYS))),
}


class FileCube:
    """The cube read from its file. What a snapshot or a stopped owner gives you."""

    def __init__(self, path: Path, schema: str, writable: bool = False):
        import duckdb

        self.schema = schema
        self.path = Path(path)
        self.con = duckdb.connect(str(path), read_only=not writable)

    def describe(self) -> str:
        return str(self.path)

    def columns(self, relation: str) -> list:
        try:
            return [row[1] for row in self.con.execute(
                f'PRAGMA table_info("{self.schema}"."{relation}")').fetchall()]
        except Exception:
            return []

    def rows(self, relation: str) -> list:
        columns = self.columns(relation)
        if not columns:
            return []
        projection = ", ".join(f'"{column}"' for column in columns)
        return [dict(zip(columns, row)) for row in self.con.execute(
            f'SELECT {projection} FROM "{self.schema}"."{relation}"').fetchall()]

    def content_digest(self, relation: str, columns: list):
        projection = ", ".join(f'"{column}"' for column in columns)
        return self.con.execute(
            "SELECT md5(string_agg(h, '' ORDER BY h)) FROM (SELECT md5(CAST(ROW("
            f'{projection}) AS VARCHAR)) AS h FROM "{self.schema}"."{relation}") t'
        ).fetchone()[0]

    def rows_with_rowid(self, relation: str, digest: str) -> list:
        """The physical rows of one terminal, each carrying the identity used to remove it."""
        columns = self.columns(relation)
        projection = ", ".join(f'"{column}"' for column in columns)
        rows = self.con.execute(
            f'SELECT rowid AS __rowid, {projection} FROM "{self.schema}"."{relation}"'
            " WHERE terminal_sha256 = ?", [digest]).fetchall()
        return [dict(zip(["__rowid", *columns], row)) for row in rows]

    def close(self) -> None:
        self.con.close()


class ServiceCube:
    """The cube read through the RUNNING service, which is the only live reading of it.

    A snapshot answers what the cube held at a moment that has already passed. When the
    question is what the deployed warehouse currently serves, the deployed warehouse has to be
    the one answering. Every relation is read in pages under the host's own `LIMIT` ceiling and
    the total is checked against a separate `count(*)`, so a truncated page cannot be mistaken
    for a complete population.
    """

    PAGE = 1000

    def __init__(self, url: str, token_env: str, schema: str):
        self.url = url.rstrip("/")
        self.schema = schema
        self.token = os.environ.get(token_env)
        if not self.token:
            raise SystemExit(f"{token_env} is not set: the service token comes from the "
                             "environment, never from an argument")

    def describe(self) -> str:
        return f"{self.url} (live service)"

    def _get(self, path: str, params: dict):
        request = urllib.request.Request(
            f"{self.url}{path}?{urllib.parse.urlencode(params)}",
            headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(request, timeout=120) as handle:
            return json.load(handle)

    def _query(self, sql: str) -> list:
        return self._get("/api/v1/query", {"sql": sql})["rows"]

    def columns(self, relation: str) -> list:
        try:
            body = self._get("/api/v1/schema", {"relation": relation})
        except Exception:
            return []
        return [column["name"] for column in body.get("columns") or []]

    def rows(self, relation: str) -> list:
        columns = self.columns(relation)
        if not columns:
            return []
        expected = self._query(
            f'SELECT count(*) AS n FROM "{self.schema}"."{relation}" LIMIT 1')[0]["n"]
        order = ", ".join(str(index + 1) for index in range(len(columns)))
        projection = ", ".join(f'"{column}"' for column in columns)
        collected, offset = [], 0
        while True:
            page = self._query(
                f'SELECT {projection} FROM "{self.schema}"."{relation}" ORDER BY {order}'
                f" OFFSET {offset} LIMIT {self.PAGE}")
            collected.extend(page)
            if len(page) < self.PAGE:
                break
            offset += self.PAGE
        if len(collected) != expected:
            # A page short of its own count is a truncated read, and a truncated read that is
            # reported as a population is exactly the mistake this whole tool exists to catch.
            raise RuntimeError(f"{relation}: read {len(collected)} rows, the service counts "
                               f"{expected}")
        return collected

    def content_digest(self, relation: str, columns: list):
        projection = ", ".join(f'"{column}"' for column in columns)
        return self._query(
            "SELECT md5(string_agg(h, '' ORDER BY h)) AS d FROM (SELECT md5(CAST(ROW("
            f'{projection}) AS VARCHAR)) AS h FROM "{self.schema}"."{relation}") t LIMIT 1'
        )[0]["d"]

    def close(self) -> None:
        return None


def source_content(source) -> dict:
    """The content this report was computed from, as counts and order-independent digests.

    A reconciliation receipt that does not say WHICH bytes it read cannot be checked against
    the snapshot receipt taken beside it. Mine could not: an H1 report stating 55 matches and
    a snapshot receipt stating 537 metric rows named the same file, and re-running the same
    code over content with that exact digest reports two differing terminals. Which step was
    wrong is not recoverable, because neither receipt carried the other's evidence. From here
    a report carries it.
    """
    content = {}
    for relation in ("gov_terminal", *CHILD_SHAPES):
        columns = source.columns(relation)
        if not columns:
            content[relation] = None
            continue
        try:
            content[relation] = {"rows": len(source.rows(relation)),
                                 "md5": str(source.content_digest(relation, columns))}
        except Exception as exc:
            content[relation] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return content


def field_coverage(source) -> dict:
    """Which stored columns are compared, and which are excused and why. Checkable, not claimed."""
    coverage = {}
    for relation in ("gov_terminal", *CHILD_SHAPES):
        columns = source.columns(relation)
        if relation == "gov_terminal":
            compared = {PARENT_COLUMN_OF.get(field, field) for field in PARENT_FIELDS}
            compared.add("terminal_sha256")
            excused = {"received_at": "generated by the service when the terminal is stored"}
        else:
            aliases = COLUMN_ALIASES.get(relation, {})
            compared = {aliases.get(field, field) for field in CHILD_SHAPES[relation][1]}
            compared.add("terminal_sha256")
            excused = {}
        coverage[relation] = {
            "compared": sorted(compared & set(columns)),
            "not_persisted_from_payload": sorted(excused),
            "excuses": excused,
            "stored_but_unrepresented": sorted(set(columns) - compared - set(excused)),
        }
    return coverage


def canonical_value(value):
    """One value rendered for comparison, typed.

    Integers keep their identity - an id of 1 is not the float 1.0 - while a float and an
    integer of equal magnitude compare equal ONLY where the store itself widens them. `None`,
    `True` and the strings "None"/"True" are all distinguishable.
    """
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("int", int(value)) if value.is_integer() else ("float", repr(value))
    if isinstance(value, (dict, list)):
        return ("json", json.dumps(value, sort_keys=True, separators=(",", ":")))
    return ("text", str(value))


def canonical_row(values: dict, fields) -> str:
    """One row rendered for comparison. Typed, so a changed number is a difference."""
    return json.dumps([canonical_value(values.get(field)) for field in fields],
                      separators=(",", ":"))


def payload_digest(body: dict) -> str:
    """The digest the store computes over a terminal's canonical body, reused not reinvented."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "olap" / "store" / "src"))
    from predictor_olap_store.query import canonical_text

    return hashlib.sha256(canonical_text(
        {key: value for key, value in body.items() if key != "terminal_sha256"}
    ).encode("ascii")).hexdigest()


def expected_children(body: dict) -> dict:
    """The child multisets the accepted payload says should exist."""
    out = {}
    for relation, (key, fields) in CHILD_SHAPES.items():
        rows = body.get(key) or []
        out[relation] = sorted(canonical_row(row, fields) for row in rows
                               if isinstance(row, dict))
    return out


def observed_children(con, schema: str, digest: str) -> dict:
    """The child multisets the cube actually holds for this terminal."""
    out = {}
    for relation, (_key, fields) in CHILD_SHAPES.items():
        aliases = COLUMN_ALIASES.get(relation, {})
        columns = [aliases.get(field, field) for field in fields]
        projection = ", ".join(f'"{column}"' for column in columns)
        rows = con.execute(
            f'SELECT {projection} FROM "{schema}"."{relation}" '
            "WHERE terminal_sha256 = ?", [digest]).fetchall()
        out[relation] = sorted(
            canonical_row(dict(zip(fields, row)), fields) for row in rows)
    return out


def children_by_terminal(source) -> dict:
    """Every child row the cube holds, canonicalised and grouped by its terminal.

    Read once per relation rather than once per terminal: the same rows, far fewer round trips
    against a live service, and one place where a truncated read is caught.
    """
    out = {}
    for relation, (_key, fields) in CHILD_SHAPES.items():
        aliases = COLUMN_ALIASES.get(relation, {})
        for row in source.rows(relation):
            values = {field: row.get(aliases.get(field, field)) for field in fields}
            out.setdefault(row.get("terminal_sha256"), {}).setdefault(relation, []).append(
                canonical_row(values, fields))
    for terminal in out:
        for relation in out[terminal]:
            out[terminal][relation].sort()
    return out


def surplus_plan(payload: dict, source, digest: str) -> dict:
    """Which physical rows are the EXCESS copies, per relation, with their before-images.

    Expected multiplicity is counted from the accepted payload. Two identical rows in a
    contract are two rows; the excess is `observed - expected` for that exact row and nothing
    else. `SELECT DISTINCT` and a global de-duplication rule are both wrong here: they would
    rewrite the first case as well as the second.
    """
    plan = {}
    for relation, (key, fields) in CHILD_SHAPES.items():
        expected = Counter(canonical_row(row, fields) for row in (payload.get(key) or [])
                           if isinstance(row, dict))
        aliases = COLUMN_ALIASES.get(relation, {})
        columns = [aliases.get(field, field) for field in fields]
        held = {}
        for row in source.rows_with_rowid(relation, digest):
            values = {field: row[aliases.get(field, field)] for field in fields}
            held.setdefault(canonical_row(values, fields), []).append(row)
        groups = []
        for canonical, rows in held.items():
            excess = len(rows) - expected.get(canonical, 0)
            if excess <= 0:
                continue
            # The copies kept are the earliest ones: identical rows are interchangeable, and a
            # deterministic choice is what makes a second invocation a no-op.
            doomed = sorted(rows, key=lambda row: row["__rowid"])[-excess:]
            groups.append({"relation": relation, "canonical": canonical,
                           "expected": expected.get(canonical, 0), "observed": len(rows),
                           "rowids": [row["__rowid"] for row in doomed],
                           "before_images": [
                               {"relation": relation, "terminal_sha256": digest,
                                "rowid": row["__rowid"],
                                "values": {column: row[column] for column in columns}}
                               for row in doomed]})
        if groups:
            plan[relation] = groups
    return plan


def delete_by_rowid(con, schema: str, relation: str, rowids: list) -> int:
    """Remove exactly these physical rows. Patched in tests to prove the operation is atomic."""
    placeholders = ", ".join("?" for _ in rowids)
    con.execute(f'DELETE FROM "{schema}"."{relation}" WHERE rowid IN ({placeholders})',
                list(rowids))
    return len(rowids)


def compare_multisets(expected: list, observed: list, fields=()) -> dict:
    """What is missing and what is extra, and WHICH FIELDS changed where that is answerable.

    Multiplicity is preserved: two identical rows are two rows. Where a row is missing and
    another is extra, the two are paired by position and the differing fields are NAMED — a
    report that renders values without their column tells an operator that something moved but
    not what, which is barely better than a count.
    """
    from collections import Counter

    missing = list((Counter(expected) - Counter(observed)).elements())
    extra = list((Counter(observed) - Counter(expected)).elements())
    changed_fields = {}
    for left, right in zip(sorted(missing), sorted(extra)):
        try:
            before, after = json.loads(left), json.loads(right)
        except ValueError:
            continue
        for index, field in enumerate(fields):
            if index < len(before) and index < len(after) and before[index] != after[index]:
                changed_fields.setdefault(field, []).append(
                    {"expected": before[index], "in_cube": after[index]})
    return {"missing": len(missing), "extra": len(extra),
            "changed_fields": changed_fields,
            "missing_examples": missing[:3], "extra_examples": extra[:3]}


def outbox_identities(roots: list) -> dict:
    """Envelope slots still held anywhere, keyed by (campaign, unit).

    Recoverability is part of the record, because absence from the cube is only a retry rather
    than a loss when an outbox holds THIS generation with a payload that could actually be
    replayed. A filename, or the same campaign and unit, is not that.
    """
    held = {}
    for root in roots:
        for state in ("pending", "sent", "adjudicated"):
            directory = Path(root) / state
            if not directory.is_dir():
                continue
            for item in sorted(directory.iterdir()):
                if not item.is_file():
                    continue
                try:
                    body = json.loads(item.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(body, dict):
                    continue
                terminal = body.get("terminal") if isinstance(body.get("terminal"), dict) else {}
                key = (body.get("campaign_sha256"), body.get("unit_id"))
                if key == (None, None):
                    continue
                held.setdefault(key, []).append({
                    "root": str(root), "state": state, "file": item.name,
                    "generation": terminal.get("generation"),
                    "status": terminal.get("status"),
                    "recoverable": bool(terminal.get("schema") and terminal.get("status"))})
    return held


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--cube", type=Path,
                        help="the cube file: a snapshot, a rehearsal copy, or the production "
                             "file with its owner stopped. Required for any repair.")
    parser.add_argument("--service-url",
                        help="read the cube through the RUNNING service instead of its file, "
                             "which is the only reading of what it currently serves")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN",
                        help="environment variable holding the store token; never a literal")
    parser.add_argument("--schema", default="main")
    parser.add_argument("--outbox", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repair", action="store_true",
                        help="restore child rows the accepted payload says should exist and "
                             "the cube does not hold. Only ADDS what is missing under a "
                             "matching identity; never edits or deletes anything.")
    parser.add_argument("--repair-surplus", action="store_true",
                        help="remove the EXCESS copies of child rows the cube holds more times "
                             "than the accepted payload declares. Multiplicity comes from the "
                             "payload; nothing is deduplicated globally.")
    parser.add_argument("--evidence", type=Path,
                        help="where the surplus rows are preserved BEFORE they are removed")
    args = parser.parse_args(argv)

    if not args.cube and not args.service_url:
        parser.error("one of --cube or --service-url is required")
    if args.service_url and (args.repair or args.repair_surplus):
        parser.error("a repair writes to the cube file, with its owner stopped; the service "
                     "route is read-only")
    if args.repair_surplus and not args.evidence:
        # Removing a row without first preserving it is not a repair, whatever it is called.
        parser.error("--repair-surplus requires --evidence: the surplus rows are preserved "
                     "before they leave the operational projection")

    accepted = accepted_terminals(args.accounting)
    outboxes = outbox_identities([root.expanduser() for root in args.outbox])

    writable = bool(args.repair or args.repair_surplus)
    source = (ServiceCube(args.service_url, args.token_env, args.schema) if args.service_url
              else FileCube(args.cube, args.schema, writable=writable))
    in_cube = {row["terminal_sha256"]: row for row in source.rows("gov_terminal")}
    grouped = children_by_terminal(source)
    observed = {digest: {relation: grouped.get(digest, {}).get(relation, [])
                         for relation in CHILD_SHAPES} for digest in in_cube}
    coverage = field_coverage(source)
    read_content = source_content(source)

    matches, differs, missing, replayable, unverifiable = [], [], [], [], []
    for row in accepted:
        digest = row["terminal_sha256"]
        found = in_cube.get(digest)
        record = {"terminal_sha256": digest, "campaign_sha256": row["campaign_sha256"],
                  "unit_id": row["unit_id"], "generation": row["generation"],
                  "status_in_accounting": row["status"]}
        if not found:
            key = (row["campaign_sha256"], row["unit_id"])
            candidates = [entry for entry in outboxes.get(key, [])
                          if entry.get("generation") == row["generation"]
                          and entry.get("recoverable")]
            if candidates:
                # An outbox only counts when it holds THIS generation and a payload that can
                # actually be replayed. A filename, or the same campaign and unit, is not that.
                record["outbox"] = candidates
                replayable.append(record)
            else:
                record["outbox_candidates_rejected"] = outboxes.get(key, [])
                missing.append(record)
            continue

        body = row.get("body")
        if not isinstance(body, dict) or not body:
            # No independently retained expectation: this terminal's CONTENT cannot be checked
            # either way. Reporting it as preserved would be the original defect.
            record["reason"] = "no canonical payload retained in the accounting"
            unverifiable.append(record)
            continue
        missing_fields = sorted(TERMINAL_KEYS - set(body))
        if missing_fields:
            record["reason"] = ("the retained payload is partial and cannot serve as an "
                                f"expectation; missing {missing_fields}")
            unverifiable.append(record)
            continue
        recomputed = payload_digest(body)
        if recomputed != digest:
            # An expectation that does not hash to the identity it is filed under is not an
            # expectation. Skipping it silently would be the original defect in another place.
            record["reason"] = ("the retained payload does not match its recorded digest: "
                                f"recomputed {recomputed}")
            record["recomputed_payload_digest"] = recomputed
            unverifiable.append(record)
            continue

        differences = {}
        for field in PARENT_FIELDS:
            if field not in body:
                differences.setdefault("parent", {})[field] = {
                    "expected": "ABSENT_FROM_PAYLOAD", "in_cube": found.get(
                        PARENT_COLUMN_OF.get(field, field))}
                continue
            column = PARENT_COLUMN_OF.get(field, field)
            expected_value = body[field]
            actual = found.get(column)
            if field in JSON_FIELDS:
                # stored as JSON text: compare the parsed values, so key order and spacing
                # are not mistaken for tampering, and tampering is not mistaken for spacing
                try:
                    actual = json.loads(actual) if isinstance(actual, str) else actual
                except ValueError:
                    pass
            if canonical_value(expected_value) != canonical_value(actual):
                differences.setdefault("parent", {})[field] = {
                    "column": column, "expected": expected_value, "in_cube": actual}
        expected = expected_children(body)
        seen = observed.get(digest, {})
        for relation, rows in expected.items():
            outcome = compare_multisets(rows, seen.get(relation, []),
                                        CHILD_SHAPES[relation][1])
            if outcome["missing"] or outcome["extra"]:
                differences.setdefault("children", {})[relation] = outcome
        record["expected_child_counts"] = {relation: len(rows)
                                           for relation, rows in expected.items()}
        record["observed_child_counts"] = {relation: len(rows)
                                           for relation, rows in seen.items()}
        if differences:
            record["differences"] = differences
            differs.append(record)
        else:
            matches.append(record)

    repaired = []
    if args.repair and differs:
        # Restore from the INDEPENDENT record: the payload governance accepted. Only rows that
        # are missing are added, only under an identity whose parent already matches, and
        # nothing is edited or removed - a repair that could overwrite would be a rewrite.
        con = source.con
        for record in differs:
            digest = record["terminal_sha256"]
            payload = next(row["body"] for row in accepted
                           if row["terminal_sha256"] == digest)
            if record["differences"].get("parent"):
                record["repair"] = "REFUSED_PARENT_DIFFERS"
                continue
            added = {}
            con.execute("BEGIN TRANSACTION")
            try:
                for relation, outcome in record["differences"]["children"].items():
                    if outcome["extra"]:
                        raise RuntimeError(
                            f"{relation} holds rows the payload does not: not a repair")
                    key, fields = CHILD_SHAPES[relation]
                    aliases = COLUMN_ALIASES.get(relation, {})
                    columns = ["terminal_sha256"] + [aliases.get(f, f) for f in fields]
                    held = set(observed_children(con, args.schema, digest)[relation])
                    rows = [row for row in (payload.get(key) or [])
                            if canonical_row(row, fields) not in held]
                    for row in rows:
                        placeholders = ", ".join("?" for _ in columns)
                        con.execute(
                            f'INSERT INTO "{args.schema}"."{relation}" '
                            f'({", ".join(chr(34) + c + chr(34) for c in columns)}) '
                            f"VALUES ({placeholders})",
                            [digest] + [row.get(field) for field in fields])
                    added[relation] = len(rows)
                con.execute("COMMIT")
            except Exception as exc:
                con.execute("ROLLBACK")
                record["repair"] = f"FAILED: {type(exc).__name__}: {exc}"
                continue
            record["repair"] = "RESTORED"
            record["rows_restored"] = added
            repaired.append(record)

    # --- I2: the other direction, bounded to proven surplus multiplicity --------------------
    surplus_removed, surplus_repaired, before_images = 0, [], []
    if args.repair_surplus:
        # The evidence file is written BEFORE anything is removed, and it names every physical
        # row. Preserving the rows after the fact would preserve whatever survived the removal.
        plans = {}
        for record in differs:
            if record.get("repair") == "RESTORED":
                # An additive repair in the same invocation already changed this terminal, so
                # the difference report above no longer describes the cube. A refused or
                # failed additive attempt changed nothing, and its read still stands.
                continue
            digest = record["terminal_sha256"]
            payload = next(row["body"] for row in accepted
                           if row["terminal_sha256"] == digest)
            if record["differences"].get("parent"):
                record["repair"] = "REFUSED_PARENT_DIFFERS"
                continue
            if any(outcome["missing"]
                   for outcome in record["differences"].get("children", {}).values()):
                # Missing AND extra is a CHANGED value, not a surplus copy. Removing either
                # side would pick a winner between two contents, which is a rewrite.
                record["repair"] = "REFUSED_NOT_PURE_SURPLUS"
                continue
            plan = surplus_plan(payload, source, digest)
            if not plan:
                record["repair"] = "NOTHING_TO_REMOVE"
                continue
            plans[digest] = (record, plan)
            for groups in plan.values():
                for group in groups:
                    before_images.extend(group["before_images"])
        args.evidence.write_text(json.dumps(
            {"schema": "surplus_multiplicity_evidence.v1", "generated_utc": now(),
             "cube": source.describe(),
             "planned": before_images,
             "note": ("Every physical row below is a copy the cube holds beyond the "
                      "multiplicity the accepted payload declares. It is recorded here before "
                      "it leaves the operational projection.")}, indent=1) + "\n",
            encoding="utf-8")
        for digest, (record, plan) in plans.items():
            removed_here = {}
            source.con.execute("BEGIN TRANSACTION")
            try:
                for relation, groups in plan.items():
                    for group in groups:
                        removed_here[relation] = removed_here.get(relation, 0) + delete_by_rowid(
                            source.con, args.schema, relation, group["rowids"])
                source.con.execute("COMMIT")
            except Exception as exc:
                source.con.execute("ROLLBACK")
                record["repair"] = f"FAILED: {type(exc).__name__}: {exc}"
                continue
            record["repair"] = "SURPLUS_REMOVED"
            record["rows_removed"] = removed_here
            record["removed_rows"] = [image for groups in plan.values()
                                      for group in groups
                                      for image in group["before_images"]]
            surplus_removed += sum(removed_here.values())
            surplus_repaired.append(record)
        if surplus_repaired:
            args.evidence.write_text(json.dumps(
                {"schema": "surplus_multiplicity_evidence.v1", "generated_utc": now(),
                 "cube": source.describe(),
                 "planned": before_images,
                 "removed": [image for record in surplus_repaired
                             for image in record["removed_rows"]],
                 "terminals": sorted(record["terminal_sha256"]
                                     for record in surplus_repaired),
                 "note": ("Every physical row below is a copy the cube held beyond the "
                          "multiplicity the accepted payload declares, preserved here before "
                          "it left the operational projection.")}, indent=1) + "\n",
                encoding="utf-8")
        # A terminal whose only difference WAS the surplus now matches: it stops being a
        # difference, so the verdict is measured on what is left rather than on what was found.
        settled = {record["terminal_sha256"] for record in surplus_repaired}
        for record in surplus_repaired:
            record.pop("differences", None)
        differs = [record for record in differs if record["terminal_sha256"] not in settled]
        matches.extend(surplus_repaired)

    # A repair writes from a process that is NOT the service. If it leaves a write-ahead log
    # beside the file, that log can later be replayed onto a base that already contains it, and
    # every row it carries appears twice - which is reproducibly how identical child rows
    # double. The log is folded in here, and the receipt states that it was.
    log_after, content_after = None, None
    if writable:
        source.con.execute("CHECKPOINT")
        # `source_content` describes what the report was COMPUTED from; after a write the
        # cube is no longer that, and a receipt that states only the first is unfalsifiable
        # against the cube it left behind.
        content_after = source_content(source)
        log = Path(str(args.cube) + ".wal")
        log_after = log.stat().st_size if log.exists() else 0
    described = source.describe()
    source.close()

    accepted_digests = {row["terminal_sha256"] for row in accepted}
    orphans = sorted(set(in_cube) - accepted_digests)

    verdict = "NO_LOSS_FOR_THE_COMPARED_POPULATION"
    if missing or differs:
        verdict = "LOSS_OR_ALTERATION_FOUND"
    elif unverifiable or orphans or not accepted:
        verdict = "INCONCLUSIVE"

    body = {"schema": "incident_evidence_reconcile.v2", "generated_utc": now(),
            "accounting": str(args.accounting), "cube": described,
            "expectation_source": ("data-gov governed_terminals.body_json — the canonical "
                                   "payload as accepted, in a database the incident never "
                                   "touched. Expectations are NOT derived from the cube."),
            "field_coverage": coverage,
            "source_content": read_content,
            "content_after_repair": content_after,
            "write_ahead_log_bytes_after_checkpoint": log_after,
            "counts": {"accepted_by_governance": len(accepted),
                       "content_matches": len(matches),
                       "content_differs": len(differs),
                       "missing_from_cube": len(missing),
                       "replayable_from_outbox": len(replayable),
                       "content_unverifiable": len(unverifiable),
                       "repaired": len(repaired),
                       "surplus_rows_removed": surplus_removed,
                       "terminals_with_surplus_removed": len(surplus_repaired),
                       "terminals_in_cube": len(in_cube),
                       "cube_rows_without_an_accepted_record": len(orphans)},
            "repaired": repaired,
            "surplus_removed": surplus_repaired,
            "content_differs": differs, "missing_from_cube": missing,
            "replayable_from_outbox": replayable,
            "content_unverifiable": unverifiable,
            "cube_rows_without_an_accepted_record": orphans,
            "verdict": verdict,
            "note": ("The quarantined write-ahead log was not opened. Absence from the cube is "
                     "LOSS only when no outbox holds this generation with a recoverable "
                     "payload. A population with no retained expectation is UNVERIFIABLE, "
                     "never preserved by inference.")}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, **body["counts"]}, indent=1))
    return 0 if verdict == "NO_LOSS_FOR_THE_COMPARED_POPULATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
