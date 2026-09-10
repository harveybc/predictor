"""The common campaign envelope — one shape every producer emits.

The cube already holds inventory and profiles. What it does not
hold is the campaigns: T1/T2, B4, M3/M4 and the RL work all
finish, adjudicate and then live only in their own state roots.
This module defines the envelope they share and the additive
tables that receive it.

Three rules shape every decision here:

  * **nothing is invented.** A field a producer cannot supply is
    written `UNAVAILABLE`; it is never guessed, defaulted or left
    to a nullable column that a later reader will misread as
    zero.
  * **result classes never mix.** A mechanical rehearsal, a
    development probe, a calibration run, a confirmation and a
    result its own producer marked non-governing are different
    kinds of thing. They are stored with their class, and a query
    that wants confirmations gets confirmations.
  * **identity survives.** Every envelope carries the producer's
    own identifiers and digests unchanged, so a row in the cube
    can always be traced back to the bytes it came from.

The DDL is additive: it creates tables, never drops or alters
one, and this module has no delete path at all.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

SCHEMA = "public"
ENVELOPE_SCHEMA = "crispdm.campaign_envelope.v1"

UNAVAILABLE = "UNAVAILABLE"

# The five kinds of result, kept apart on purpose.
RESULT_CLASSES = (
    "NON_GOVERNING",   # the producer itself withdrew it
    "MECHANICAL",      # mechanics/rehearsal, no scientific claim
    "DEVELOPMENT",     # exploratory, never confirmatory
    "CALIBRATION",     # sets a rule, does not test it
    "CONFIRMATION",    # tests a pre-registered rule
)

REQUIRED_TOP = (
    "schema", "campaign_key", "producer", "result_class",
    "identity", "data_consumed", "partitions", "budget",
    "terminal", "artifacts",
)
REQUIRED_IDENTITY = ("run_id", "code_identity", "design_sha256")
REQUIRED_PARTITIONS = ("exposure", "splits")
REQUIRED_BUDGET = ("device", "wall_seconds", "cost_units")
REQUIRED_TERMINAL = ("state", "adjudication")


class EnvelopeRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _sha(doc: dict, key: str) -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------
# additive DDL
# --------------------------------------------------------------

ENVELOPE_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_campaign (
  campaign_key      TEXT PRIMARY KEY,
  producer          TEXT NOT NULL,
  result_class      TEXT NOT NULL,
  design_sha256     TEXT NOT NULL,
  code_identity     TEXT NOT NULL,
  run_id            TEXT NOT NULL,
  first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_campaign_unit (
  envelope_sha256   TEXT NOT NULL,
  campaign_key      TEXT NOT NULL
      REFERENCES {SCHEMA}.dim_campaign(campaign_key)
      ON DELETE RESTRICT,
  cell_key          TEXT NOT NULL,
  candidate_key     TEXT NOT NULL,
  result_class      TEXT NOT NULL,
  terminal_state    TEXT NOT NULL,
  adjudication      TEXT NOT NULL,
  metric_name       TEXT NOT NULL,
  metric_value      DOUBLE PRECISION,
  uncertainty_kind  TEXT NOT NULL,
  uncertainty_low   DOUBLE PRECISION,
  uncertainty_high  DOUBLE PRECISION,
  exposure          TEXT NOT NULL,
  device            TEXT NOT NULL,
  wall_seconds      DOUBLE PRECISION,
  checkpoint_count  INTEGER,
  epoch_count       INTEGER,
  envelope_json     JSONB NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (envelope_sha256, cell_key, candidate_key,
               metric_name)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_campaign_consumption (
  envelope_sha256   TEXT NOT NULL,
  campaign_key      TEXT NOT NULL
      REFERENCES {SCHEMA}.dim_campaign(campaign_key)
      ON DELETE RESTRICT,
  subject_kind      TEXT NOT NULL,
  subject_id        TEXT NOT NULL,
  subject_digest    TEXT NOT NULL,
  eligibility_state TEXT NOT NULL,
  PRIMARY KEY (envelope_sha256, subject_kind, subject_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_campaign_unit_class
  ON {SCHEMA}.fact_campaign_unit (result_class, campaign_key);
CREATE INDEX IF NOT EXISTS idx_fact_campaign_unit_metric
  ON {SCHEMA}.fact_campaign_unit (metric_name);
CREATE INDEX IF NOT EXISTS idx_fact_campaign_consumption_subject
  ON {SCHEMA}.fact_campaign_consumption (subject_kind,
                                         subject_id);
"""


# --------------------------------------------------------------
# building and validating
# --------------------------------------------------------------

def build_envelope(*, campaign_key: str, producer: str,
                   result_class: str, identity: dict,
                   data_consumed: dict, partitions: dict,
                   budget: dict, terminal: dict,
                   artifacts: dict,
                   units: list[dict] | None = None) -> dict:
    doc = {
        "schema": ENVELOPE_SCHEMA,
        "campaign_key": campaign_key,
        "producer": producer,
        "result_class": result_class,
        "identity": identity,
        "data_consumed": data_consumed,
        "partitions": partitions,
        "budget": budget,
        "terminal": terminal,
        "artifacts": artifacts,
        "units": units or [],
    }
    doc["envelope_sha256"] = _sha(doc, "envelope_sha256")
    validate_envelope(doc)
    return doc


def validate_envelope(doc: dict) -> dict:
    if doc.get("schema") != ENVELOPE_SCHEMA:
        raise EnvelopeRefusal(
            f"envelope schema is {doc.get('schema')!r}")
    missing = [k for k in REQUIRED_TOP if k not in doc]
    if missing:
        raise EnvelopeRefusal(
            f"envelope is missing {missing} — an ETL that "
            "accepts a partial identity cannot be traced back "
            "to bytes")
    if doc["result_class"] not in RESULT_CLASSES:
        raise EnvelopeRefusal(
            f"unknown result class {doc['result_class']!r} — a "
            "result whose kind is unknown is not stored")
    for group, required in (("identity", REQUIRED_IDENTITY),
                            ("partitions", REQUIRED_PARTITIONS),
                            ("budget", REQUIRED_BUDGET),
                            ("terminal", REQUIRED_TERMINAL)):
        block = doc[group]
        if not isinstance(block, dict):
            raise EnvelopeRefusal(f"{group} must be an object")
        gaps = [k for k in required if k not in block]
        if gaps:
            raise EnvelopeRefusal(
                f"{group} is missing {gaps} — write "
                f"{UNAVAILABLE} rather than omitting a field")
    for k, v in _walk(doc):
        if v is None:
            raise EnvelopeRefusal(
                f"{k} is null — an unknown value is written "
                f"{UNAVAILABLE}, never null")
    declared = doc.get("envelope_sha256")
    if declared and _sha(doc, "envelope_sha256") != declared:
        raise EnvelopeRefusal(
            "envelope self digest does not re-derive — the "
            "artifact was mutated after it was produced")
    for u in doc.get("units", []):
        for f in ("cell_key", "candidate_key", "metric_name",
                  "terminal_state"):
            if f not in u:
                raise EnvelopeRefusal(
                    f"unit is missing {f!r}")
    return doc


def _walk(node, prefix=""):
    """Yield (path, value) for scalars, so a null anywhere is
    caught rather than only at the top level."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


# --------------------------------------------------------------
# loading — additive and idempotent
# --------------------------------------------------------------

def ensure_envelope_tables(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(ENVELOPE_DDL)


def load_envelope(engine, doc: dict) -> dict:
    """Insert an envelope. Re-loading the same envelope changes
    nothing; loading a mutated one refuses before any write."""
    validate_envelope(doc)
    from sqlalchemy import text

    ident = doc["identity"]
    counts = {"campaigns": 0, "units": 0, "consumption": 0,
              "skipped_existing": 0}
    with engine.begin() as conn:
        conn.exec_driver_sql(ENVELOPE_DDL)
        existing = conn.execute(text(
            f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit "
            "WHERE envelope_sha256 = :e"),
            {"e": doc["envelope_sha256"]}).scalar()
        conn.execute(text(f"""
            INSERT INTO {SCHEMA}.dim_campaign
              (campaign_key, producer, result_class,
               design_sha256, code_identity, run_id)
            VALUES (:k, :p, :rc, :d, :c, :r)
            ON CONFLICT (campaign_key) DO NOTHING
        """), {"k": doc["campaign_key"],
               "p": doc["producer"],
               "rc": doc["result_class"],
               "d": str(ident["design_sha256"]),
               "c": str(ident["code_identity"]),
               "r": str(ident["run_id"])})
        counts["campaigns"] = 1
        if existing:
            counts["skipped_existing"] = int(existing)
        for u in doc.get("units", []):
            res = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.fact_campaign_unit
                  (envelope_sha256, campaign_key, cell_key,
                   candidate_key, result_class, terminal_state,
                   adjudication, metric_name, metric_value,
                   uncertainty_kind, uncertainty_low,
                   uncertainty_high, exposure, device,
                   wall_seconds, checkpoint_count, epoch_count,
                   envelope_json)
                VALUES (:e, :k, :cell, :cand, :rc, :ts, :adj,
                        :mn, :mv, :uk, :ul, :uh, :ex, :dev,
                        :ws, :cc, :ec, CAST(:j AS JSONB))
                ON CONFLICT (envelope_sha256, cell_key,
                             candidate_key, metric_name)
                DO NOTHING
            """), {
                "e": doc["envelope_sha256"],
                "k": doc["campaign_key"],
                "cell": u["cell_key"],
                "cand": u["candidate_key"],
                "rc": doc["result_class"],
                "ts": u["terminal_state"],
                "adj": str(doc["terminal"]["adjudication"]),
                "mn": u["metric_name"],
                "mv": _num(u.get("metric_value")),
                "uk": str(u.get("uncertainty_kind",
                                UNAVAILABLE)),
                "ul": _num(u.get("uncertainty_low")),
                "uh": _num(u.get("uncertainty_high")),
                "ex": str(doc["partitions"]["exposure"]),
                "dev": str(doc["budget"]["device"]),
                "ws": _num(doc["budget"].get("wall_seconds")),
                "cc": _int(u.get("checkpoint_count")),
                "ec": _int(u.get("epoch_count")),
                "j": json.dumps(u, sort_keys=True)})
            counts["units"] += res.rowcount or 0
        for kind, items in (
                ("variable",
                 doc["data_consumed"].get("variables", [])),
                ("operator",
                 doc["data_consumed"].get("operators", [])),
                ("dataset",
                 doc["data_consumed"].get("datasets", []))):
            for item in items:
                res = conn.execute(text(f"""
                    INSERT INTO
                      {SCHEMA}.fact_campaign_consumption
                      (envelope_sha256, campaign_key,
                       subject_kind, subject_id, subject_digest,
                       eligibility_state)
                    VALUES (:e, :k, :sk, :si, :sd, :es)
                    ON CONFLICT (envelope_sha256, subject_kind,
                                 subject_id) DO NOTHING
                """), {
                    "e": doc["envelope_sha256"],
                    "k": doc["campaign_key"],
                    "sk": kind,
                    "si": str(item.get("id", UNAVAILABLE)),
                    "sd": str(item.get("digest", UNAVAILABLE)),
                    "es": str(item.get("eligibility_state",
                                       UNAVAILABLE))})
                counts["consumption"] += res.rowcount or 0
    return counts


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not \
        isinstance(v, bool) else None


def _int(v):
    return int(v) if isinstance(v, int) and not \
        isinstance(v, bool) else None


def ingestion_receipt(doc: dict, *, loaded_counts: dict,
                      as_of: str | None = None) -> dict:
    rec = {
        "schema": "crispdm.campaign_ingestion_receipt.v1",
        "envelope_sha256": doc["envelope_sha256"],
        "campaign_key": doc["campaign_key"],
        "producer": doc["producer"],
        "result_class": doc["result_class"],
        "as_of": as_of or datetime.now(
            timezone.utc).replace(microsecond=0).isoformat(),
        "loaded": loaded_counts,
        "grants_nothing": "ingestion records a result; it never "
                          "changes the result's authority",
    }
    rec["receipt_sha256"] = _sha(rec, "receipt_sha256")
    return rec
