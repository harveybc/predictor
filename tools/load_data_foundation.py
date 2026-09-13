#!/usr/bin/env python3
"""C139 (order 2026-09-12): the additive OLAP grains of the data foundation.

The schema is generated from TABLES, so the SQL file and the loader cannot
disagree (`olap/c139_data_foundation.sql` is this module's `ddl()` output,
checked by a test). Nothing here alters or deletes an existing table or row:
every statement is CREATE TABLE IF NOT EXISTS, and every insert is
ON CONFLICT (row_sha256) DO NOTHING, so a second load writes nothing.

Every fact binds its run, its dataset or unit and content digest, its
variable, partition, code and estimator. Every outcome is kept:
COMPLETED, FAILED, INCONCLUSIVE, REFUSED, REJECTED, UNAVAILABLE, NOT_RUN.
A row that does not validate is refused and listed in the load receipt;
it is never coerced. Lab decisions are only the four laboratory states, and
no column may carry PUBLICLY_ELIGIBLE or LIVE_ELIGIBLE: the database checks
that too.

This loader connects directly and additively. The running outbox loader is
untouched: it only knows campaign envelopes and events, and restarting it
to teach it new kinds is not allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SQL_FILE = REPO / "olap/c139_data_foundation.sql"
SCHEMA = "public"

STATUSES = ("COMPLETED", "FAILED", "INCONCLUSIVE", "REFUSED", "REJECTED", "UNAVAILABLE", "NOT_RUN")
LAB_DECISIONS = ("LAB_CALIBRATED", "REGIME_LIMITED", "NOT_IDENTIFIABLE", "LAB_REJECTED")
COVERAGE_STATES = ("NOT_RUN", "UNAVAILABLE", "FAILED", "INCONCLUSIVE", "RESULT")
BANKS = ("PUBLIC", "FINANCIAL", "SYNTHETIC")
LICENSE_STATES = ("OPEN_ATTRIBUTION", "OPEN_PUBLIC_DOMAIN", "RESTRICTED_NO_DERIVATIVES",
                  "RESTRICTED_NON_COMMERCIAL", "TERMS_REQUIRE_REVIEW",
                  "INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE", "NOT_APPLICABLE_GENERATED", "UNKNOWN")
COMPONENTS = ("RAW", "DENOISED", "RESIDUAL", "COMPARISON")
SUBJECT_KINDS = ("VARIABLE", "MATRIX")
FORBIDDEN = ("PUBLICLY_ELIGIBLE", "LIVE_ELIGIBLE")

# column types
TEXT, INT, BOOL, JSON = "text", "int", "bool", "json"
NUM_OR_NULL, INT_OR_NULL, TEXT_OR_NULL = "num?", "int?", "text?"


def enum(values):
    return ("enum", tuple(values))


def _metric(**extra):
    base = {"run_id": TEXT, "dataset_id": TEXT, "content_sha256": TEXT, "partition": TEXT,
            "metric": TEXT, "estimator": TEXT, "estimator_params": JSON,
            "value": NUM_OR_NULL, "value_text": TEXT_OR_NULL,
            "status": enum(STATUSES), "reason": TEXT, "code_sha256": TEXT, "cpu_seconds": NUM_OR_NULL}
    base.update(extra)
    return base


TABLES = {
    "df_dim_dataset": {"run_id": TEXT, "dataset_id": TEXT, "contract_sha256": TEXT, "content_sha256": TEXT,
                       "bank": enum(BANKS), "license_state": enum(LICENSE_STATES), "n_variables": INT,
                       "contract_json": JSON},
    "df_dim_variable": {"run_id": TEXT, "variable_id": TEXT, "dataset_id": TEXT, "contract_sha256": TEXT,
                        "name": TEXT, "unit": TEXT, "semantics_type": TEXT, "role": TEXT,
                        "license_state": enum(LICENSE_STATES)},
    "df_dim_run": {"run_id": TEXT, "module": TEXT, "code_sha256": TEXT, "inputs_sha256": TEXT,
                   "status": enum(STATUSES), "cpu_seconds": NUM_OR_NULL, "details": JSON},
    "df_fact_sampling_quality": _metric(variable_id=TEXT),
    "df_fact_variable_profile": _metric(variable_id=TEXT),
    "df_fact_information_metric": _metric(subject_kind=enum(SUBJECT_KINDS), subject_id=TEXT),
    "df_fact_pair_relation": _metric(variable_id_a=TEXT, variable_id_b=TEXT, lag=INT_OR_NULL),
    "df_fact_group_relation": _metric(group_id=TEXT, members=JSON),
    "df_fact_operator_run": {"run_id": TEXT, "subject_id": TEXT, "content_sha256": TEXT, "variable_id": TEXT,
                             "regime": JSON, "operator_kind": TEXT, "operator_params": JSON, "spec_sha256": TEXT,
                             "fitted_sha256": TEXT_OR_NULL, "status": enum(STATUSES), "reason": TEXT,
                             "code_sha256": TEXT, "cpu_seconds": NUM_OR_NULL, "peak_memory_bytes": INT_OR_NULL},
    "df_fact_operator_signal_metric": {"run_id": TEXT, "operator_run_sha256": TEXT, "component": enum(COMPONENTS),
                                       "partition": TEXT, "metric": TEXT, "estimator": TEXT,
                                       "estimator_params": JSON, "value": NUM_OR_NULL, "value_text": TEXT_OR_NULL,
                                       "status": enum(STATUSES), "reason": TEXT, "code_sha256": TEXT},
    "df_fact_operator_delay_cost": {"run_id": TEXT, "operator_kind": TEXT, "operator_params": JSON,
                                    "spec_sha256": TEXT, "metric": TEXT, "frequency": NUM_OR_NULL,
                                    "value": NUM_OR_NULL, "value_text": TEXT_OR_NULL, "status": enum(STATUSES),
                                    "reason": TEXT, "code_sha256": TEXT},
    "df_fact_lab_decision": {"run_id": TEXT, "operator_kind": TEXT, "operator_params": JSON, "spec_sha256": TEXT,
                             "regime": JSON, "decision": enum(LAB_DECISIONS), "rule_sha256": TEXT,
                             "evidence": JSON, "failure_regions": JSON, "externally_reviewed": BOOL,
                             "code_sha256": TEXT},
    "df_fact_snr_calibration": {"run_id": TEXT, "unit_id": TEXT, "content_sha256": TEXT, "variable_index": INT,
                                "partition": TEXT, "estimator": TEXT, "metric": TEXT, "value": NUM_OR_NULL,
                                "value_text": TEXT_OR_NULL, "status": enum(STATUSES), "reason": TEXT,
                                "code_sha256": TEXT},
    "df_fact_coverage": {"run_id": TEXT, "dataset_id": TEXT, "variable_id": TEXT, "metric": TEXT,
                         "operator": TEXT, "state": enum(COVERAGE_STATES), "code_sha256": TEXT},
    "df_fact_load_receipt": {"run_id": TEXT, "table_name": TEXT, "rows_offered": INT, "rows_inserted": INT,
                             "rows_already_present": INT, "rows_refused": INT, "refusals": JSON},
}


class LoadRefusal(ValueError):
    pass


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def row_sha256(table: str, row: dict) -> str:
    return hashlib.sha256(canonical({"table": table, "row": row})).hexdigest()


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# --------------------------------------------------------------------- DDL
def _sql_type(t) -> str:
    if isinstance(t, tuple):
        return "TEXT NOT NULL"
    return {TEXT: "TEXT NOT NULL", INT: "BIGINT NOT NULL", BOOL: "BOOLEAN NOT NULL", JSON: "JSONB NOT NULL",
            NUM_OR_NULL: "DOUBLE PRECISION", INT_OR_NULL: "BIGINT", TEXT_OR_NULL: "TEXT"}[t]


def _quote(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def ddl() -> str:
    parts = ["-- C139 (order 2026-09-12): additive data-foundation grains.",
             "-- GENERATED by tools/load_data_foundation.py ddl(); do not edit by hand.",
             "-- Only CREATE ... IF NOT EXISTS: nothing existing is altered or deleted.", ""]
    for table, cols in TABLES.items():
        lines = ["    row_sha256 TEXT PRIMARY KEY"]
        for c, t in cols.items():
            lines.append(f"    {c} {_sql_type(t)}")
        checks = []
        for c, t in cols.items():
            if isinstance(t, tuple):
                checks.append(f"CHECK ({c} IN ({', '.join(_quote(v) for v in t[1])}))")
            if t in (TEXT, TEXT_OR_NULL) or isinstance(t, tuple):
                checks.append(f"CHECK ({c} IS NULL OR {c} NOT IN ({', '.join(_quote(v) for v in FORBIDDEN)}))")
        if "status" in cols and "value" in cols:
            checks.append("CHECK (status <> 'COMPLETED' OR value IS NOT NULL OR value_text IS NOT NULL)")
            checks.append("CHECK (status = 'COMPLETED' OR (value IS NULL AND value_text IS NULL))")
        if table == "df_fact_lab_decision":
            checks.append("CHECK (externally_reviewed = FALSE)")
        lines += [f"    {c}" for c in checks]
        lines.append("    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()")
        parts.append(f"CREATE TABLE IF NOT EXISTS {SCHEMA}.{table} (\n" + ",\n".join(lines) + "\n);")
        parts.append("")
    return "\n".join(parts)


def ensure_schema(engine) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        for stmt in ddl().split(";\n"):
            body = "\n".join(ln for ln in stmt.splitlines() if not ln.strip().startswith("--")).strip()
            if body:
                conn.execute(text(body))


# -------------------------------------------------------------- validation
def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


def validate_row(table: str, row: dict) -> list[str]:
    if table not in TABLES:
        return [f"unknown table {table!r}"]
    spec = TABLES[table]
    if not isinstance(row, dict) or set(row) != set(spec):
        got = sorted(row) if isinstance(row, dict) else type(row).__name__
        return [f"keys differ: expected {sorted(spec)}, got {got}"]
    p = []
    for c, t in spec.items():
        v = row[c]
        if isinstance(t, tuple):
            if v not in t[1]:
                p.append(f"{c}: {v!r} not in {list(t[1])}")
        elif t == TEXT and not isinstance(v, str):
            p.append(f"{c}: expected text")
        elif t == TEXT_OR_NULL and not (v is None or isinstance(v, str)):
            p.append(f"{c}: expected text or null")
        elif t == INT and type(v) is not int:
            p.append(f"{c}: expected an integer")
        elif t == INT_OR_NULL and not (v is None or type(v) is int):
            p.append(f"{c}: expected an integer or null")
        elif t == BOOL and type(v) is not bool:
            p.append(f"{c}: expected a boolean")
        elif t == NUM_OR_NULL and not (v is None or (type(v) in (int, float) and math.isfinite(v))):
            p.append(f"{c}: expected a finite number or null")
        elif t == JSON and not isinstance(v, (dict, list)):
            p.append(f"{c}: expected an object or list")
    if any(s in FORBIDDEN for s in _walk_strings(row)):
        p.append("a data-foundation row never carries PUBLICLY_ELIGIBLE or LIVE_ELIGIBLE")
    if "status" in spec and "value" in spec:
        if row["status"] == "COMPLETED" and row["value"] is None and row["value_text"] is None:
            p.append("a COMPLETED row needs a value")
        if row["status"] != "COMPLETED" and (row["value"] is not None or row["value_text"] is not None):
            p.append("only a COMPLETED row carries a value")
        if row["status"] != "COMPLETED" and not row["reason"].strip():
            p.append("a row that did not complete needs a reason")
    if table == "df_fact_lab_decision" and row.get("externally_reviewed") is not False:
        p.append("a lab decision is loaded unreviewed; review is recorded by the reviewer, not here")
    for c in ("content_sha256", "code_sha256", "spec_sha256", "contract_sha256", "rule_sha256"):
        if c in spec and isinstance(row.get(c), str) and not (len(row[c]) == 64 and all(ch in "0123456789abcdef" for ch in row[c])):
            p.append(f"{c}: expected a sha256 hex digest")
    try:
        canonical(row)
    except (TypeError, ValueError) as exc:
        p.append(f"not strict JSON: {exc}")
    return p


# -------------------------------------------------------------------- load
def load(engine, table: str, rows: list, run_id: str) -> dict:
    """Insert the valid rows additively; refuse and list the rest."""
    from sqlalchemy import text
    if table == "df_fact_load_receipt":
        raise LoadRefusal("receipts are written by the loader itself")
    spec = TABLES[table]
    accepted, refusals = [], []
    for i, r in enumerate(rows):
        problems = validate_row(table, r)
        if problems:
            refusals.append({"index": i, "problems": problems[:5]})
        else:
            accepted.append(r)
    cols = ["row_sha256", *spec]
    placeholders = ", ".join(f"CAST(:{c} AS JSONB)" if spec.get(c) == JSON else f":{c}" for c in cols)
    stmt = text(f"INSERT INTO {SCHEMA}.{table} ({', '.join(cols)}) VALUES ({placeholders}) "
                "ON CONFLICT (row_sha256) DO NOTHING")
    with engine.begin() as conn:
        count = text(f"SELECT count(*) FROM {SCHEMA}.{table}")
        before = conn.execute(count).scalar()
        batch = []
        for r in accepted:
            params = {c: (json.dumps(r[c], sort_keys=True) if spec[c] == JSON else r[c]) for c in spec}
            params["row_sha256"] = row_sha256(table, r)
            batch.append(params)
            if len(batch) >= 5000:
                conn.execute(stmt, batch)
                batch = []
        if batch:
            conn.execute(stmt, batch)
        # Rows actually written, counted inside the same transaction; this
        # loader is the only writer of the data-foundation tables.
        inserted = conn.execute(count).scalar() - before
        receipt = {"run_id": run_id, "table_name": table, "rows_offered": len(rows),
                   "rows_inserted": inserted, "rows_already_present": len(accepted) - inserted,
                   "rows_refused": len(refusals), "refusals": refusals}
        rspec = TABLES["df_fact_load_receipt"]
        rcols = ["row_sha256", *rspec]
        rparams = {c: (json.dumps(receipt[c], sort_keys=True) if rspec[c] == JSON else receipt[c]) for c in rspec}
        rparams["row_sha256"] = row_sha256("df_fact_load_receipt", receipt)
        conn.execute(text(
            f"INSERT INTO {SCHEMA}.df_fact_load_receipt ({', '.join(rcols)}) VALUES "
            f"({', '.join(f'CAST(:{c} AS JSONB)' if rspec.get(c) == JSON else f':{c}' for c in rcols)}) "
            "ON CONFLICT (row_sha256) DO NOTHING"), rparams)
    return receipt


# ---------------------------------------------------------------- adapters
def dataset_rows(contract: dict, run_id: str) -> tuple[dict, list[dict]]:
    ds = {"run_id": run_id, "dataset_id": contract["dataset_id"], "contract_sha256": contract["contract_sha256"],
          "content_sha256": contract["content_sha256"], "bank": contract["bank"],
          "license_state": contract["license"]["state"], "n_variables": len(contract["variables"]),
          "contract_json": contract}
    vs = [{"run_id": run_id, "variable_id": v["variable_id"], "dataset_id": contract["dataset_id"],
           "contract_sha256": contract["contract_sha256"], "name": v["name"], "unit": v["unit"]["value"],
           "semantics_type": v["semantics"]["type"], "role": v["role"], "license_state": v["license_state"]}
          for v in contract["variables"]]
    return ds, vs


def metric_row(module_row: dict, *, run_id: str, content_sha256: str, **grain) -> dict:
    """A profile-module row ({dataset_id, partition, metric, estimator{name, params, assumptions},
    value, status, reason, code_sha256, cpu_seconds}) as a table row; `grain` adds the table's
    own keys (variable_id, pair, group ...). A text value goes to value_text."""
    est = module_row["estimator"]
    value = module_row.get("value")
    out = {"run_id": run_id, "dataset_id": module_row["dataset_id"], "content_sha256": content_sha256,
           "partition": module_row["partition"], "metric": module_row["metric"], "estimator": est["name"],
           "estimator_params": {"params": est.get("params", {}), "assumptions": est.get("assumptions", {})},
           "value": value if not isinstance(value, str) else None,
           "value_text": value if isinstance(value, str) else None,
           "status": module_row["status"], "reason": module_row.get("reason") or "",
           "code_sha256": module_row["code_sha256"], "cpu_seconds": module_row.get("cpu_seconds")}
    out.update(grain)
    return out


def _engine_from_env():
    from sqlalchemy import create_engine
    e = os.environ
    return create_engine(f"postgresql://{e['PGUSER']}:{e['PGPASSWORD']}@{e.get('PGHOST', '127.0.0.1')}:"
                         f"{e.get('PGPORT', '5432')}/{e.get('PGDATABASE', 'predictor_olap')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print-ddl", action="store_true")
    ap.add_argument("--ensure-schema", action="store_true")
    ap.add_argument("--table")
    ap.add_argument("--rows", type=Path, help="JSON list of table rows")
    ap.add_argument("--run-id")
    a = ap.parse_args(argv)
    if a.print_ddl:
        print(ddl())
        return 0
    engine = _engine_from_env()
    if a.ensure_schema:
        ensure_schema(engine)
    if a.table:
        rows = json.loads(a.rows.read_text())
        print(json.dumps(load(engine, a.table, rows, a.run_id), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
