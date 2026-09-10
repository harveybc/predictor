"""C15: the census and the common index, inside the cube.

Additive tables for the rows the index now carries — physical
appearances, conceptual variables, public series, synthetic
generators and eligibility decisions. They are kept apart from
the 99 historical variable profiles already in the cube, because
those describe two locally registered datasets and these describe
a 1,965-variable lake: mixing them would make one look like the
other.

`event_time` and `available_time` stay separate columns, the
authority class travels with every row, and the metadata state is
explicit — so a query can always ask "what is actually declared
here?" and get an honest answer.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "public"


def observation_sha256(payload: dict) -> str:
    """C25: the identity of ONE OBSERVATION of a logical id.

    Two loads of the same observation collide on this digest and
    are idempotent; a changed digest, availability, semantics or
    membership produces a DIFFERENT observation, which is stored
    as a new version beside the old one instead of being dropped.
    """
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()

INVENTORY_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_lake_appearance (
  appearance_id     TEXT NOT NULL,
  entity            TEXT NOT NULL,
  source_class      TEXT NOT NULL,
  frequency         TEXT NOT NULL,
  period_start      TEXT NOT NULL,
  period_end        TEXT NOT NULL,
  physical_sha256   TEXT NOT NULL,
  digest_state      TEXT NOT NULL,
  authority_class   TEXT NOT NULL,
  census_sha256     TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (appearance_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_lake_variable (
  variable_id       TEXT NOT NULL,
  entity            TEXT NOT NULL,
  concept_name      TEXT NOT NULL,
  source_class      TEXT NOT NULL,
  unit              TEXT NOT NULL,
  event_time        TEXT NOT NULL,
  available_time    TEXT NOT NULL,
  semantics_declared BOOLEAN NOT NULL,
  appearance_count  INTEGER NOT NULL,
  authority_class   TEXT NOT NULL,
  census_sha256     TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (variable_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_public_series (
  series_id         TEXT NOT NULL,
  family            TEXT NOT NULL,
  dataset_id        TEXT NOT NULL,
  frequency         TEXT NOT NULL,
  digest            TEXT NOT NULL,
  authority_class   TEXT NOT NULL,
  index_sha256      TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (series_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_synthetic_generator (
  generator_id      TEXT NOT NULL,
  producer          TEXT NOT NULL,
  family            TEXT NOT NULL,
  role              TEXT NOT NULL,
  mechanism_json    JSONB NOT NULL,
  authority_class   TEXT NOT NULL,
  index_sha256      TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (generator_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_eligibility_decision (
  subject_id        TEXT NOT NULL,
  subject_kind      TEXT NOT NULL,
  scope             TEXT NOT NULL,
  decision          TEXT NOT NULL,
  decision_reason   TEXT NOT NULL,
  reviewer          TEXT NOT NULL,
  reviewed_at       TEXT NOT NULL,
  manifest_sha256   TEXT NOT NULL,
  authority_class   TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (subject_id, scope, manifest_sha256)
);

-- C25: a deterministic CURRENT view per inventory table. The
-- newest observation of a logical id wins, ties broken by the
-- observation digest so two readers always agree. No version is
-- ever deleted; history is the point.
CREATE OR REPLACE VIEW {SCHEMA}.v_lake_appearance_current AS
SELECT DISTINCT ON (appearance_id) *
FROM {SCHEMA}.dim_lake_appearance
ORDER BY appearance_id, loaded_at DESC, observation_sha256 DESC;

CREATE OR REPLACE VIEW {SCHEMA}.v_lake_variable_current AS
SELECT DISTINCT ON (variable_id) *
FROM {SCHEMA}.dim_lake_variable
ORDER BY variable_id, loaded_at DESC, observation_sha256 DESC;

CREATE OR REPLACE VIEW {SCHEMA}.v_public_series_current AS
SELECT DISTINCT ON (series_id) *
FROM {SCHEMA}.dim_public_series
ORDER BY series_id, loaded_at DESC, observation_sha256 DESC;

CREATE OR REPLACE VIEW {SCHEMA}.v_synthetic_generator_current AS
SELECT DISTINCT ON (generator_id) *
FROM {SCHEMA}.dim_synthetic_generator
ORDER BY generator_id, loaded_at DESC, observation_sha256 DESC;

CREATE INDEX IF NOT EXISTS idx_dim_lake_variable_entity
  ON {SCHEMA}.dim_lake_variable (entity);
CREATE INDEX IF NOT EXISTS idx_dim_lake_variable_available
  ON {SCHEMA}.dim_lake_variable (available_time);
CREATE INDEX IF NOT EXISTS idx_dim_lake_appearance_entity
  ON {SCHEMA}.dim_lake_appearance (entity);
CREATE INDEX IF NOT EXISTS idx_dim_public_series_family
  ON {SCHEMA}.dim_public_series (family);
"""

# The historical tables this loader must never touch.
HISTORICAL_TABLES = ("dim_experiment", "fact_performance",
                     "fact_results_summary", "dim_dataset",
                     "dim_series", "dim_variable",
                     "fact_dataset_inventory",
                     "fact_variable_profile")

NEW_TABLES = ("dim_lake_appearance", "dim_lake_variable",
              "dim_public_series", "dim_synthetic_generator",
              "fact_eligibility_decision")


class InventoryLoadRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def ensure_inventory_tables(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(INVENTORY_DDL)


def load_index(engine, index: dict) -> dict:
    """Load the common index additively. Idempotent by id."""
    from sqlalchemy import text

    if index.get("schema") != "crispdm.bank_index.v1":
        raise InventoryLoadRefusal(
            f"not a bank index: {index.get('schema')!r}")
    index_sha = index["index_sha256"]
    banks = index["banks"]
    counts = {t: 0 for t in NEW_TABLES}
    with engine.begin() as conn:
        conn.exec_driver_sql(INVENTORY_DDL)
        fin = banks.get("financial_domain", {})
        census_sha = fin.get("binding", {}).get(
            "census_document_sha256", "UNAVAILABLE")
        auth_fin = fin.get("authority_class", "UNAVAILABLE")
        for a in fin.get("appearances", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_lake_appearance
                  (appearance_id, entity, source_class,
                   frequency, period_start, period_end,
                   physical_sha256, digest_state,
                   authority_class, census_sha256,
                   observation_sha256)
                VALUES (:i, :e, :sc, :f, :ps, :pe, :d, :ds, :a,
                        :c, :o)
                ON CONFLICT (appearance_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "entity": a["entity"],
                       "source_class": a["source_class"],
                       "frequency": a["frequency"],
                       "period_start": str(a["period_start"]),
                       "period_end": str(a["period_end"]),
                       "physical_sha256": a["physical_sha256"],
                       "digest_state": a["digest_state"],
                       "authority_class": auth_fin,
                       "census_sha256": census_sha}),
                   "i": a["appearance_id"], "e": a["entity"],
                   "sc": a["source_class"],
                   "f": a["frequency"],
                   "ps": str(a["period_start"]),
                   "pe": str(a["period_end"]),
                   "d": a["physical_sha256"],
                   "ds": a["digest_state"], "a": auth_fin,
                   "c": census_sha})
            counts["dim_lake_appearance"] += r.rowcount or 0
        for v in fin.get("variables", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_lake_variable
                  (variable_id, entity, concept_name,
                   source_class, unit, event_time,
                   available_time, semantics_declared,
                   appearance_count, authority_class,
                   census_sha256, observation_sha256)
                VALUES (:i, :e, :cn, :sc, :u, :et, :at, :sd,
                        :ac, :a, :c, :o)
                ON CONFLICT (variable_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "entity": v["entity"],
                       "concept_name": v["concept_name"],
                       "source_class": v["source_class"],
                       "unit": v["unit"],
                       "event_time": v["event_time"],
                       "available_time": v["available_time"],
                       "semantics_declared":
                           bool(v["semantics_declared"]),
                       "appearance_count":
                           int(v["appearance_count"]),
                       "authority_class": auth_fin,
                       "census_sha256": census_sha}),
                   "i": v["variable_id"], "e": v["entity"],
                   "cn": v["concept_name"],
                   "sc": v["source_class"], "u": v["unit"],
                   "et": v["event_time"],
                   "at": v["available_time"],
                   "sd": bool(v["semantics_declared"]),
                   "ac": int(v["appearance_count"]),
                   "a": auth_fin, "c": census_sha})
            counts["dim_lake_variable"] += r.rowcount or 0

        pub = banks.get("public_forecasting", {})
        auth_pub = pub.get("authority_class", "UNAVAILABLE")
        for s in pub.get("series", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_public_series
                  (series_id, family, dataset_id, frequency,
                   digest, authority_class, index_sha256,
                   observation_sha256)
                VALUES (:i, :f, :d, :fr, :dg, :a, :x, :o)
                ON CONFLICT (series_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "family": s["family"],
                       "dataset_id": s["dataset_id"],
                       "frequency": str(s.get("frequency",
                                              "UNKNOWN")),
                       "digest": s.get("digest", "UNAVAILABLE"),
                       "authority_class": auth_pub}),
                   "i": s["series_id"], "f": s["family"],
                   "d": s["dataset_id"],
                   "fr": str(s.get("frequency", "UNKNOWN")),
                   "dg": s.get("digest", "UNAVAILABLE"),
                   "a": auth_pub, "x": index_sha})
            counts["dim_public_series"] += r.rowcount or 0

        syn = banks.get("synthetic_known_mechanism", {})
        auth_syn = syn.get("authority_class", "UNAVAILABLE")
        for g in syn.get("generators", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_synthetic_generator
                  (generator_id, producer, family, role,
                   mechanism_json, authority_class,
                   index_sha256, observation_sha256)
                VALUES (:i, :p, :f, :r, CAST(:m AS JSONB), :a,
                        :x, :o)
                ON CONFLICT (generator_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "producer": g["producer"],
                       "mechanism": g["mechanism"],
                       "role": g["role"],
                       "authority_class": auth_syn}),
                   "i": g["generator_id"], "p": g["producer"],
                   "f": str(g["mechanism"].get("family",
                                               "UNKNOWN")),
                   "r": g["role"],
                   "m": json.dumps(g["mechanism"],
                                   sort_keys=True),
                   "a": auth_syn, "x": index_sha})
            counts["dim_synthetic_generator"] += r.rowcount or 0
    return counts


def load_eligibility_decisions(engine, manifest: dict,
                               *, authority_class: str) -> dict:
    """Eligibility decisions are facts about subjects, and belong
    in the cube like any other decision."""
    from sqlalchemy import text
    loaded = 0
    with engine.begin() as conn:
        conn.exec_driver_sql(INVENTORY_DDL)
        for e in manifest.get("entries", []):
            scopes = e["decision_scope"]
            if isinstance(scopes, str):
                scopes = [scopes]
            for scope in scopes:
                r = conn.execute(text(f"""
                    INSERT INTO
                      {SCHEMA}.fact_eligibility_decision
                      (subject_id, subject_kind, scope,
                       decision, decision_reason, reviewer,
                       reviewed_at, manifest_sha256,
                       authority_class)
                    VALUES (:i, :k, :s, :d, :r, :rv, :ra, :m,
                            :a)
                    ON CONFLICT (subject_id, scope,
                                 manifest_sha256) DO NOTHING
                """), {"i": e["subject_id"],
                       "k": e["subject_kind"], "s": scope,
                       "d": e["decision"],
                       "r": e["decision_reason"],
                       "rv": e["reviewer"],
                       "ra": e["reviewed_at"],
                       "m": manifest["manifest_sha256"],
                       "a": authority_class})
                loaded += r.rowcount or 0
    return {"fact_eligibility_decision": loaded}
