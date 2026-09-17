#!/usr/bin/env python3
"""Apply the D2-R6 coverage selection through the adoption route, rehearsed and verified.

R6 was written, rehearsed on a throwaway database and then left unapplied. It is the one
measured block on D3: `df_fact_coverage` holds 440,694 rows over ONE run id and TWO code
digests, so a coverage figure read from the cube today cannot say which matrix it counted.

What R6 does and does not do:

* it **selects**, it does not delete. `df_coverage_version_selection` names one
  `(table, run_id, code_sha256)` with a reason, `df_coverage_current` shows exactly that
  matrix, and `df_coverage_history` keeps every version with its provenance;
* the selection is a governed write with an explicit reason, never an implicit default;
* the denominator view exists so a coverage ratio cannot quietly be computed over a doubled
  population.

The SQL is `docs/audits/evidence/repro_runs/d2_support_r1/R6_COVERAGE_VIEWS.sql` and is used
verbatim. It was written for PostgreSQL; the cube is now DuckDB, so the statements are
translated by a table that is printed in the receipt rather than applied silently.

Every application rehearses first on a copy of the very file it is about to change, and
refuses to touch the original if the rehearsal does not come out right. The checks are
content, not exit codes: the current view must equal the selected matrix, the history view
must equal v1 plus v2, and the row counts of both fact tables must be unchanged.

usage:
  df_r6_apply.py --cube FILE --out RECEIPT.json [--rehearse-only]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SQL = REPO / "docs" / "audits" / "evidence" / "repro_runs" / "d2_support_r1" / \
    "R6_COVERAGE_VIEWS.sql"

#: PostgreSQL spellings the cube's engine does not share. Printed in the receipt, so a reader
#: sees what was changed rather than trusting that nothing was.
TRANSLATIONS = (
    ("TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"),
    ("::TEXT", "::VARCHAR"),
)

CREATED = ("df_coverage_version_selection", "df_coverage_current", "df_coverage_history",
           "df_coverage_current_denominator")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def split_statements(text: str) -> list:
    """Split on semicolons that are NOT inside a quoted string.

    The selection's own reason reads "(9 states + applicability); supersedes v1 c140", so a
    naive split on `;` cuts a statement in half and the parser then complains about an
    unterminated string rather than about the split. A `--` inside quotes is likewise not a
    comment, so both are handled in one pass over the characters.
    """
    out, current, quoted = [], [], False
    index = 0
    while index < len(text):
        char = text[index]
        if quoted:
            current.append(char)
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    current.append(text[index + 1])
                    index += 2
                    continue
                quoted = False
            index += 1
            continue
        if char == "'":
            quoted = True
            current.append(char)
            index += 1
            continue
        if char == "-" and text[index:index + 2] == "--":
            newline = text.find("\n", index)
            index = len(text) if newline == -1 else newline
            continue
        if char == ";":
            out.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    out.append("".join(current))
    return [chunk.strip() for chunk in out if chunk.strip()]


def statements() -> list:
    """The SQL as written, comments removed, split into statements."""
    return split_statements(SQL.read_text(encoding="utf-8"))


def translate(statement: str) -> str:
    for postgres, duckdb_form in TRANSLATIONS:
        statement = statement.replace(postgres, duckdb_form)
    return statement


def _counts(con, names) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        except Exception:
            out[name] = None
    return out


def apply_to(path: Path) -> dict:
    """Run the statements against one cube and measure what they produced."""
    import duckdb

    con = duckdb.connect(str(path))
    try:
        before = _counts(con, ("df_fact_coverage", "df_fact_coverage_v2"))
        applied = []
        for statement in statements():
            translated = translate(statement)
            con.execute(translated)
            applied.append(translated.splitlines()[0][:90])
        after = _counts(con, ("df_fact_coverage", "df_fact_coverage_v2"))
        selection = con.execute(
            "SELECT table_name, run_id, code_sha256, reason"
            " FROM df_coverage_version_selection ORDER BY selected_at DESC LIMIT 1"
        ).fetchall()
        chosen = selection[0] if selection else None
        measured = _counts(con, CREATED)
        expected_current = None
        if chosen:
            expected_current = con.execute(
                f"SELECT count(*) FROM {chosen[0]} WHERE run_id = ? AND code_sha256 = ?",
                [chosen[1], chosen[2]]).fetchone()[0]
        con.execute("CHECKPOINT")
        log = Path(str(path) + ".wal")
        return {"applied_statements": applied,
                "selection": None if not chosen else
                {"table_name": chosen[0], "run_id": chosen[1], "code_sha256": chosen[2],
                 "reason": chosen[3]},
                "fact_rows_before": before, "fact_rows_after": after,
                "relation_rows": measured,
                "expected_current_rows": expected_current,
                "write_ahead_log_bytes_after_checkpoint":
                    log.stat().st_size if log.exists() else 0}
    finally:
        con.close()


def verify(outcome: dict) -> list:
    """Content checks. A statement that ran is not the same as a selection that is right."""
    reasons = []
    if outcome["fact_rows_before"] != outcome["fact_rows_after"]:
        reasons.append("a fact table changed size: R6 selects, it never deletes")
    if not outcome["selection"]:
        reasons.append("no selection row was written, so the current view has no authority")
    current = outcome["relation_rows"].get("df_coverage_current")
    if current is None:
        reasons.append("df_coverage_current was not created")
    elif current != outcome["expected_current_rows"]:
        reasons.append(f"df_coverage_current holds {current} rows and the selected matrix has "
                       f"{outcome['expected_current_rows']}")
    history = outcome["relation_rows"].get("df_coverage_history")
    total = sum(value or 0 for value in outcome["fact_rows_after"].values())
    if history is None:
        reasons.append("df_coverage_history was not created")
    elif history != total:
        reasons.append(f"df_coverage_history holds {history} rows and the two fact tables hold "
                       f"{total}: history must keep every version")
    if outcome["relation_rows"].get("df_coverage_current_denominator") is None:
        reasons.append("df_coverage_current_denominator was not created")
    if outcome["write_ahead_log_bytes_after_checkpoint"]:
        reasons.append("a write-ahead log was left beside the cube")
    return reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rehearse-only", action="store_true",
                        help="rehearse on a copy and stop, without touching the cube")
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="r6-rehearsal-"))
    copy = work / args.cube.name
    shutil.copy2(args.cube, copy)
    log = Path(str(args.cube) + ".wal")
    if log.exists():
        shutil.copy2(log, Path(str(copy) + ".wal"))
    rehearsal = apply_to(copy)
    rehearsal_reasons = verify(rehearsal)

    receipt = {"schema": "df_r6_apply.v1", "generated_utc": now(),
               "cube": str(args.cube), "sql": str(SQL.relative_to(REPO)),
               "translations": [{"postgres": a, "engine": b} for a, b in TRANSLATIONS],
               "rehearsal": rehearsal, "rehearsal_refused_because": rehearsal_reasons,
               "applied": None, "applied_refused_because": None,
               "note": ("R6 selects a coverage version and keeps every other one. Nothing is "
                        "deleted or deduplicated, and the selection carries an explicit "
                        "reason rather than being an implicit default.")}

    if rehearsal_reasons:
        receipt["outcome"] = "REFUSED_ON_REHEARSAL"
    elif args.rehearse_only:
        receipt["outcome"] = "REHEARSED_ONLY"
    else:
        applied = apply_to(args.cube)
        applied_reasons = verify(applied)
        receipt["applied"] = applied
        receipt["applied_refused_because"] = applied_reasons
        receipt["outcome"] = "APPLIED" if not applied_reasons else "APPLIED_WITH_PROBLEMS"
    shutil.rmtree(work, ignore_errors=True)

    args.out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": receipt["outcome"],
                      "selection": rehearsal["selection"],
                      "relation_rows": (receipt["applied"] or rehearsal)["relation_rows"],
                      "refused_because": rehearsal_reasons
                      or receipt["applied_refused_because"] or []}, indent=1))
    return 0 if receipt["outcome"] in ("APPLIED", "REHEARSED_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
