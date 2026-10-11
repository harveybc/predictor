"""DOIN archive projection: gov_report/gov_metric rows plus versioned DOIN metadata.

One implementation serves the packaged backends (``query.Plugin`` and the DuckDB provider)
through the warehouse host's ``/api/v2/doin/projections*`` routes and the predictor ETL's
in-process sink (throwaway tests).

What is stored, additively (``CREATE ... IF NOT EXISTS`` only; nothing dropped or altered):

* the analytic content as an ordinary governed report (``gov_report`` / ``gov_metric``). The
  report carries no block hash, archive mode or locator, so the same receipt row committed in
  ``full_payload`` or ``external_payload`` mode, or in two blocks, is ONE report;
* ``gov_doin_projection``: one row per (row key, block hash, finality state), the projection
  key, with the typed receipt identity and the chain/archive metadata
  (``doin_metadata_schema = i6n.doin_projection.v1``);
* ``gov_doin_projection_event``: append-only events PROJECTED / DUPLICATE / REVOKED; the
  current state of a projection is its latest event. A reorg appends REVOKED and never deletes;
* views ``gov_doin_projection_current`` and ``gov_doin_metric_canonical`` (metrics of reports
  whose projection is currently PROJECTED). At most one PROJECTED projection per row key is
  admitted, so the canonical view never counts a duplicate twice.

Only FINALIZED references are projected; TEST splits are refused. The caller (the ETL) has
verified the chain, the bytes and the schema before it calls; this store re-verifies the
report digest and the row key, not the chain.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

from predictor_olap_store.fs4_store import _run, _run_many, list_relations

SCHEMA_WRITE = "i6n.doin_projection.write.v1"
SCHEMA_REVOKE = "i6n.doin_projection.revoke.v1"
SCHEMA_RECEIPT = "i6n.doin_projection.receipt.v1"
DOIN_METADATA_SCHEMA = "i6n.doin_projection.v1"
MIGRATION_ID = "doin_0001"

TABLE = "gov_doin_projection"
EVENT_TABLE = "gov_doin_projection_event"
ALL_TABLES = (TABLE, EVENT_TABLE)
VIEWS = ("gov_doin_projection_current", "gov_doin_metric_canonical")
ALL_RELATIONS = ALL_TABLES + VIEWS

IDENTITY_FIELDS = ("experiment", "candidate_identity_sha256", "fold", "horizon_hours",
                   "head_offset_hours", "role", "population", "split", "metric", "unit",
                   "schema_version")
EXTRA_FIELDS = ("search_candidate_sha256", "head_role", "comparison", "status")
CHAIN_FIELDS = ("block_hash", "block_height", "tx_id", "body_digest", "manifest_digest", "locator",
                "archive_mode", "finality_state", "confirmation_depth", "depth")
SPLITS = ("TRAIN", "INNER", "VALIDATION")
POPULATIONS = ("S", "U")
ARCHIVE_MODES = ("external_payload", "full_payload")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INT_FIELDS = ("horizon_hours", "head_offset_hours", "block_height", "confirmation_depth", "depth")

COLUMNS = ("projection_key", "row_key_sha256", "report_sha256", *IDENTITY_FIELDS,
           *EXTRA_FIELDS, *CHAIN_FIELDS, "doin_metadata_schema", "created_at")


class Refusal(ValueError):
    """The request is wrong, not the store. Answers 400 at the host."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def row_key_sha256(identity: dict) -> str:
    """Digest of the receipt row identity (the default shared keys, nothing else)."""
    return sha256({k: identity[k] for k in IDENTITY_FIELDS})


def projection_key(row_key: str, block_hash: str, finality_state: str) -> str:
    return sha256({"row_key_sha256": row_key, "block_hash": block_hash, "finality_state": finality_state})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- DDL
def ddl(qualified: Callable[[str], str] = lambda n: n, dialect: str = "duckdb") -> list[str]:
    t = qualified
    create_view = "CREATE VIEW IF NOT EXISTS" if dialect == "sqlite" else "CREATE OR REPLACE VIEW"
    int_cols = set(_INT_FIELDS)
    cols = [" projection_key TEXT PRIMARY KEY", " row_key_sha256 TEXT NOT NULL",
            " report_sha256 TEXT NOT NULL"]
    for name in (*IDENTITY_FIELDS, *EXTRA_FIELDS, *CHAIN_FIELDS):
        kind = "BIGINT" if name in int_cols else "TEXT"
        nullable = "" if name in ("locator", "search_candidate_sha256", "head_role", "comparison",
                                  "status", "depth") else " NOT NULL"
        cols.append(f" {name} {kind}{nullable}")
    cols += [" doin_metadata_schema TEXT NOT NULL", " created_at TEXT NOT NULL",
             " CHECK (split IN ('TRAIN','INNER','VALIDATION'))",
             " CHECK (population IN ('S','U'))",
             " CHECK (archive_mode IN ('external_payload','full_payload'))",
             " CHECK (finality_state = 'FINALIZED')",
             " CHECK (archive_mode = 'full_payload' OR locator IS NOT NULL)",
             " UNIQUE (row_key_sha256, block_hash, finality_state)"]
    current = (
        f"SELECT p.*, e.event AS state, e.seq AS state_seq, e.recorded_at AS state_at"
        f" FROM {t(TABLE)} p JOIN {t(EVENT_TABLE)} e ON e.projection_key = p.projection_key"
        f" WHERE e.seq = (SELECT MAX(e2.seq) FROM {t(EVENT_TABLE)} e2"
        " WHERE e2.projection_key = p.projection_key)"
    )
    canonical_view = (
        "SELECT c.experiment, c.search_candidate_sha256, c.candidate_identity_sha256, c.fold,"
        " c.horizon_hours, c.head_offset_hours, c.head_role, c.comparison, c.role, c.population,"
        " c.split, c.metric AS receipt_metric, c.unit AS receipt_unit, c.schema_version,"
        " c.row_key_sha256, c.report_sha256, c.projection_key, c.block_hash, c.block_height,"
        " c.tx_id, c.body_digest, c.manifest_digest, c.archive_mode, c.finality_state,"
        " m.metric AS derived_metric, m.unit AS derived_unit, m.value"
        f" FROM {t('gov_doin_projection_current')} c"
        f" JOIN {t('gov_metric')} m ON m.report_sha256 = c.report_sha256"
        " WHERE c.state = 'PROJECTED'"
    )
    return [
        f"CREATE TABLE IF NOT EXISTS {t(TABLE)} ({','.join(cols)})",
        f"CREATE TABLE IF NOT EXISTS {t(EVENT_TABLE)} ("
        " event_id TEXT PRIMARY KEY, seq BIGINT NOT NULL UNIQUE, projection_key TEXT NOT NULL,"
        " event TEXT NOT NULL CHECK (event IN ('PROJECTED','DUPLICATE','REVOKED')),"
        " reason TEXT, observed_tip_hash TEXT, observed_tip_height BIGINT, recorded_at TEXT NOT NULL)",
        f"CREATE INDEX IF NOT EXISTS gov_doin_projection_row_idx ON {t(TABLE)} (row_key_sha256)",
        f"CREATE INDEX IF NOT EXISTS gov_doin_projection_block_idx ON {t(TABLE)} (block_hash)",
        f"CREATE INDEX IF NOT EXISTS gov_doin_event_key_idx ON {t(EVENT_TABLE)} (projection_key)",
        f"{create_view} {t('gov_doin_projection_current')} AS {current}",
        f"{create_view} {t('gov_doin_metric_canonical')} AS {canonical_view}",
    ]


def render_sql() -> str:
    head = (f"-- {MIGRATION_ID}: additive schema for the DOIN archive projection.\n"
            "-- GENERATED from predictor_olap_store.doin_projection_store.ddl(); do not edit.\n"
            "-- Every statement is CREATE ... IF NOT EXISTS (or a view). Nothing is removed.\n\n")
    return head + ";\n\n".join(ddl()) + ";\n"


def apply_migration(conn, qualified: Callable[[str], str] = lambda n: n, dialect: str = "duckdb") -> dict:
    before = set(list_relations(conn))
    for statement in ddl(qualified, dialect):
        _run(conn, statement)
    after = set(list_relations(conn))
    missing = [r for r in ALL_RELATIONS if r not in after]
    if missing:
        raise RuntimeError(f"migration applied but relations missing: {missing}")
    return {"migration_id": MIGRATION_ID, "created": sorted(after - before), "relations": list(ALL_RELATIONS)}


# --------------------------------------------------------------------------- validation
def _validate(document: Any) -> tuple[dict, dict, dict]:
    from predictor_olap_store.query import canonical_body, canonical_text, normalise_report

    if not isinstance(document, dict) or document.get("schema") != SCHEMA_WRITE:
        raise Refusal("schema must be " + SCHEMA_WRITE)
    report, projection = document.get("report"), document.get("projection")
    if not isinstance(report, dict) or not isinstance(projection, dict):
        raise Refusal("report and projection objects are required")
    normalised = normalise_report(report)
    digest = hashlib.sha256(canonical_text(canonical_body(normalised)).encode()).hexdigest()
    if report.get("report_sha256") != digest:
        raise Refusal("report_sha256 mismatch")
    expected = set(IDENTITY_FIELDS) | set(EXTRA_FIELDS) | set(CHAIN_FIELDS) | {"row_key_sha256", "report_sha256"}
    if set(projection) != expected:
        raise Refusal(f"projection keys differ: missing {sorted(expected - set(projection))}, "
                      f"extra {sorted(set(projection) - expected)}")
    for name in IDENTITY_FIELDS:
        value = projection[name]
        if name in _INT_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Refusal(f"{name} must be a non-negative integer")
        elif not isinstance(value, str) or not value:
            raise Refusal(f"{name} must be a non-empty string")
    if projection["split"] not in SPLITS:
        raise Refusal(f"split {projection['split']!r} is not projected (TEST is never opened)")
    if projection["population"] not in POPULATIONS:
        raise Refusal("population must be S or U")
    if projection["row_key_sha256"] != row_key_sha256(projection):
        raise Refusal("row_key_sha256 does not match the receipt identity")
    if report.get("tags", {}).get("row_key_sha256") != projection["row_key_sha256"]:
        raise Refusal("report tags do not carry this row_key_sha256")
    if projection["report_sha256"] != digest:
        raise Refusal("projection.report_sha256 differs from the report")
    for name in ("block_hash", "tx_id", "body_digest", "manifest_digest", "candidate_identity_sha256"):
        if not isinstance(projection[name], str) or not _HEX64.match(projection[name]):
            raise Refusal(f"{name} must be a lowercase sha256")
    if projection["body_digest"] == projection["manifest_digest"] or projection["block_hash"] in (
            projection["body_digest"], projection["manifest_digest"]):
        raise Refusal("body, manifest and block hashes must be distinct")
    for name in ("block_height", "confirmation_depth"):
        value = projection[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Refusal(f"{name} must be a non-negative integer")
    if projection["finality_state"] != "FINALIZED":
        raise Refusal("only FINALIZED references are projected")
    depth = projection["depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < projection["confirmation_depth"]:
        raise Refusal("depth below confirmation_depth is not FINALIZED")
    if projection["archive_mode"] not in ARCHIVE_MODES:
        raise Refusal("archive_mode must be external_payload or full_payload")
    locator = projection["locator"]
    if projection["archive_mode"] == "external_payload":
        if not isinstance(locator, str) or not locator.endswith("/sha256/" + projection["body_digest"]):
            raise Refusal("external_payload requires the body locator")
    elif locator is not None:
        raise Refusal("full_payload carries no locator")
    if not isinstance(projection["comparison"], bool) and projection["comparison"] is not None:
        raise Refusal("comparison must be a boolean")
    return report, projection, normalised


# --------------------------------------------------------------------------- write
def _next_seq(conn, t) -> int:
    rows = _run(conn, f"SELECT COALESCE(MAX(seq), 0) FROM {t(EVENT_TABLE)}")
    return int(rows[0][0]) + 1


def _append_event(conn, t, key: str, event: str, reason=None, tip_hash=None, tip_height=None) -> int:
    seq = _next_seq(conn, t)
    _run(conn, f"INSERT INTO {t(EVENT_TABLE)} (event_id, seq, projection_key, event, reason,"
               " observed_tip_hash, observed_tip_height, recorded_at) VALUES (?,?,?,?,?,?,?,?)",
         (sha256({"projection_key": key, "seq": seq}), seq, key, event, reason, tip_hash, tip_height, _now()))
    return seq


def _current(conn, t, where: str, params: tuple) -> list[dict]:
    rows = _run(conn, f"SELECT projection_key, row_key_sha256, report_sha256, block_hash, state"
                      f" FROM {t('gov_doin_projection_current')} WHERE {where}", params)
    return [dict(zip(("projection_key", "row_key_sha256", "report_sha256", "block_hash", "state"), r))
            for r in rows]


def _store_report(conn, t, report: dict, normalised: dict) -> int:
    digest = report["report_sha256"]
    if _run(conn, f"SELECT 1 FROM {t('gov_report')} WHERE report_sha256 = ?", (digest,)):
        return 0
    from predictor_olap_store.query import canonical_text

    _run(conn, f"INSERT INTO {t('gov_report')} (report_sha256, experiment_key, experiment_set_key,"
               " actor, lake_id, received_at, lineage, config_sha256, code_commit, project, phase,"
               " tags_json, n_metrics, n_datasets) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
         (digest, normalised["experiment_key"], normalised["experiment_set_key"], normalised["actor"],
          normalised["lake"], normalised["received_at"] or _now(), normalised["lineage"],
          normalised["config_sha256"], normalised["code_commit"], normalised["project"],
          normalised["phase"], canonical_text(normalised["tags"]), len(normalised["metrics"]),
          len(normalised["datasets"])))
    _run_many(conn, f"INSERT INTO {t('gov_metric')} (report_sha256, experiment_key, metric, value,"
                    " split, horizon, std_dev, min_value, max_value, unit) VALUES (?,?,?,?,?,?,?,?,?,?)",
              [(digest, normalised["experiment_key"], m["metric"], m["value"], m["split"], m["horizon"],
                m["std_dev"], m["min_value"], m["max_value"], m["unit"]) for m in normalised["metrics"]])
    if normalised["datasets"]:
        raise Refusal("DOIN projection reports carry no datasets")
    return len(normalised["metrics"])


def write_document(conn, document: dict, *, qualified: Callable[[str], str]) -> dict:
    """Atomic in the caller's transaction: report (once), projection row, event."""
    t = qualified
    report, projection, normalised = _validate(document)
    row_key = projection["row_key_sha256"]
    key = projection_key(row_key, projection["block_hash"], projection["finality_state"])
    receipt = {"schema": SCHEMA_RECEIPT, "projection_key": key, "row_key_sha256": row_key,
               "report_sha256": report["report_sha256"], "block_hash": projection["block_hash"],
               "metrics_inserted": 0}
    active = _current(conn, t, "row_key_sha256 = ? AND state = 'PROJECTED'", (row_key,))
    existing = _current(conn, t, "projection_key = ?", (key,))
    for other in active:
        if other["report_sha256"] != report["report_sha256"]:
            raise Refusal("CONFLICT: an active projection of this row key carries different metrics")
    if existing:
        if existing[0]["report_sha256"] != report["report_sha256"]:
            raise Refusal("CONFLICT: projection key already bound to another report")
        if existing[0]["state"] == "PROJECTED":
            return dict(receipt, state="ALREADY_PROJECTED")
        if active:
            return dict(receipt, state="DUPLICATE_NOT_PROJECTED", active_projection_key=active[0]["projection_key"])
        _append_event(conn, t, key, "PROJECTED", reason="REINSTATED")
        return dict(receipt, state="REINSTATED")
    inserted = _store_report(conn, t, report, normalised)
    row = dict(projection, projection_key=key, doin_metadata_schema=DOIN_METADATA_SCHEMA, created_at=_now())
    row["comparison"] = None if row["comparison"] is None else ("true" if row["comparison"] else "false")
    _run(conn, f"INSERT INTO {t(TABLE)} ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})",
         tuple(row[c] for c in COLUMNS))
    if active:
        _append_event(conn, t, key, "DUPLICATE", reason="ROW_KEY_ALREADY_PROJECTED")
        return dict(receipt, state="DUPLICATE_NOT_PROJECTED", metrics_inserted=inserted,
                    active_projection_key=active[0]["projection_key"])
    _append_event(conn, t, key, "PROJECTED")
    return dict(receipt, state="PROJECTED", metrics_inserted=inserted)


def revoke_document(conn, document: dict, *, qualified: Callable[[str], str]) -> dict:
    """Append REVOKED to every PROJECTED projection of one block. Nothing is deleted."""
    t = qualified
    if not isinstance(document, dict) or document.get("schema") != SCHEMA_REVOKE:
        raise Refusal("schema must be " + SCHEMA_REVOKE)
    block = document.get("block_hash")
    if not isinstance(block, str) or not _HEX64.match(block):
        raise Refusal("block_hash must be a lowercase sha256")
    reason = document.get("reason")
    if reason not in ("REORG", "NOT_CANONICAL", "OPERATOR"):
        raise Refusal("reason must be REORG, NOT_CANONICAL or OPERATOR")
    revoked = []
    for row in _current(conn, t, "block_hash = ? AND state IN ('PROJECTED','DUPLICATE')", (block,)):
        _append_event(conn, t, row["projection_key"], "REVOKED", reason=reason,
                      tip_hash=document.get("observed_tip_hash"),
                      tip_height=document.get("observed_tip_height"))
        revoked.append(row["projection_key"])
    return {"schema": SCHEMA_RECEIPT, "block_hash": block, "revoked": len(revoked),
            "projection_keys": sorted(revoked)}


def read_document(conn, document: dict, *, qualified: Callable[[str], str], limit: int = 5000) -> dict:
    """Projections (current state), their events and the gov_metric rows of their reports."""
    t = qualified
    if not isinstance(document, dict):
        raise Refusal("read document must be an object")
    clauses, params = [], []
    for name, column in (("report_sha256", "report_sha256"), ("projection_key", "projection_key"),
                         ("row_key_sha256", "row_key_sha256"), ("block_hash", "block_hash"),
                         ("experiment", "experiment")):
        value = document.get(name)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(v, str) for v in values):
            raise Refusal(f"{name} must be a string or a list of strings")
        clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
        params.extend(values)
    if not clauses:
        raise Refusal("one selector is required")
    names = (*COLUMNS, "state", "state_seq")
    rows = _run(conn, f"SELECT {', '.join(names)} FROM {t('gov_doin_projection_current')}"
                      f" WHERE {' AND '.join(clauses)} ORDER BY state_seq LIMIT {int(limit)}", tuple(params))
    projections = [dict(zip(names, r)) for r in rows]
    keys = [p["projection_key"] for p in projections]
    reports = sorted({p["report_sha256"] for p in projections})
    events, metrics = [], []
    if keys:
        ev = _run(conn, f"SELECT seq, projection_key, event, reason FROM {t(EVENT_TABLE)}"
                        f" WHERE projection_key IN ({', '.join('?' for _ in keys)}) ORDER BY seq", tuple(keys))
        events = [dict(zip(("seq", "projection_key", "event", "reason"), r)) for r in ev]
        mt = _run(conn, f"SELECT report_sha256, metric, value, unit, split, horizon FROM {t('gov_metric')}"
                        f" WHERE report_sha256 IN ({', '.join('?' for _ in reports)})"
                        " ORDER BY report_sha256, metric", tuple(reports))
        metrics = [dict(zip(("report_sha256", "metric", "value", "unit", "split", "horizon"), r)) for r in mt]
    return {"schema": "i6n.doin_projection.read.v1", "projections": projections, "events": events,
            "metrics": metrics}
