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

usage:
  incident_evidence_reconcile.py --accounting DB --cube FILE [--outbox DIR ...] --out R.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
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


#: Which child relation carries which part of a terminal payload, and the fields that identify
#: a row of it. Comparison is a MULTISET comparison over these fields: an absence, an addition
#: and a changed number are all differences, and all three were invisible before.
CHILD_SHAPES = {
    "gov_terminal_metric": ("metrics",
                            ("metric", "split", "horizon", "unit", "value", "std_dev",
                             "min_value", "max_value")),
    "gov_terminal_artifact": ("artifacts", ("role", "sha256", "bytes")),
    "gov_terminal_dataset": ("verified_datasets",
                             ("delivery_id", "lake_id", "resource_id", "role", "sha256",
                              "bytes", "state")),
}
#: Parent fields compared against the payload. Identity is not just the digest: a terminal
#: recorded under the wrong campaign, unit or generation is not the terminal that was accepted.
PARENT_FIELDS = ("campaign_sha256", "unit_id", "generation", "status", "actor", "project",
                 "classification", "config_sha256")
#: The cube stores `state` as `verification_state`.
COLUMN_ALIASES = {"gov_terminal_dataset": {"state": "verification_state"}}


def canonical_row(values: dict, fields) -> str:
    """One row rendered for comparison, with numbers normalised so 1 and 1.0 agree."""
    rendered = []
    for field in fields:
        value = values.get(field)
        if isinstance(value, bool) or value is None:
            rendered.append(repr(value))
        elif isinstance(value, (int, float)):
            rendered.append(repr(float(value)))
        else:
            rendered.append(str(value))
    return "\u0001".join(rendered)


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


def compare_multisets(expected: list, observed: list) -> dict:
    """What is missing and what is extra. Both are differences; neither is a count."""
    from collections import Counter

    missing = Counter(expected) - Counter(observed)
    extra = Counter(observed) - Counter(expected)
    return {"missing": sum(missing.values()), "extra": sum(extra.values()),
            "missing_examples": [row.replace("\u0001", " | ")
                                 for row in list(missing.elements())[:3]],
            "extra_examples": [row.replace("\u0001", " | ")
                               for row in list(extra.elements())[:3]]}


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
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--schema", default="main")
    parser.add_argument("--outbox", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repair", action="store_true",
                        help="restore child rows the accepted payload says should exist and "
                             "the cube does not hold. Only ADDS what is missing under a "
                             "matching identity; never edits or deletes anything.")
    args = parser.parse_args(argv)

    import duckdb

    accepted = accepted_terminals(args.accounting)
    outboxes = outbox_identities([root.expanduser() for root in args.outbox])

    con = duckdb.connect(str(args.cube), read_only=not args.repair)
    try:
        parent_columns = [row[1] for row in con.execute(
            f'PRAGMA table_info("{args.schema}"."gov_terminal")').fetchall()]
        in_cube = {}
        for row in con.execute(
                f'SELECT * FROM "{args.schema}".gov_terminal').fetchall():
            record = dict(zip(parent_columns, row))
            in_cube[record["terminal_sha256"]] = record
        observed = {digest: observed_children(con, args.schema, digest) for digest in in_cube}
    finally:
        con.close()

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

        differences = {}
        for field in PARENT_FIELDS:
            if field not in body:
                continue
            expected_value = body[field]
            actual = found.get(field)
            if str(expected_value) != str(actual):
                differences.setdefault("parent", {})[field] = {
                    "expected": expected_value, "in_cube": actual}
        expected = expected_children(body)
        seen = observed.get(digest, {})
        for relation, rows in expected.items():
            outcome = compare_multisets(rows, seen.get(relation, []))
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
        con = duckdb.connect(str(args.cube))
        try:
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
        finally:
            con.close()

    accepted_digests = {row["terminal_sha256"] for row in accepted}
    orphans = sorted(set(in_cube) - accepted_digests)

    verdict = "NO_LOSS_FOR_THE_COMPARED_POPULATION"
    if missing or differs:
        verdict = "LOSS_OR_ALTERATION_FOUND"
    elif unverifiable or orphans or not accepted:
        verdict = "INCONCLUSIVE"

    body = {"schema": "incident_evidence_reconcile.v2", "generated_utc": now(),
            "accounting": str(args.accounting), "cube": str(args.cube),
            "expectation_source": ("data-gov governed_terminals.body_json — the canonical "
                                   "payload as accepted, in a database the incident never "
                                   "touched. Expectations are NOT derived from the cube."),
            "counts": {"accepted_by_governance": len(accepted),
                       "content_matches": len(matches),
                       "content_differs": len(differs),
                       "missing_from_cube": len(missing),
                       "replayable_from_outbox": len(replayable),
                       "content_unverifiable": len(unverifiable),
                       "repaired": len(repaired),
                       "terminals_in_cube": len(in_cube),
                       "cube_rows_without_an_accepted_record": len(orphans)},
            "repaired": repaired,
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
